#include "domain/RBF/mq.hpp"
#include "intSharp.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"
#include <fstream>
#include <PetscTime.h>


void ablate::finiteVolume::processes::IntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
    IntSharp::subDomain = solver.GetSubDomainPtr();
}
ablate::finiteVolume::processes::IntSharp::IntSharp(PetscReal Gamma, PetscReal epsilon, bool flipPhiTilde) : Gamma(Gamma), epsilon(epsilon), flipPhiTilde(flipPhiTilde) {}
ablate::finiteVolume::processes::IntSharp::~IntSharp() { DMDestroy(&vertexDM) >> utilities::PetscUtilities::checkError; }

void intSharpPreStageWrapper(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime, ablate::finiteVolume::processes::IntSharp* intSharpProcess) {
    intSharpProcess->PreStage(flowTs, solver, stagetime);
  }

void ablate::finiteVolume::processes::IntSharp::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    auto dim = flow.GetSubDomain().GetDimensions();
    auto dm = flow.GetSubDomain().GetDM();
    PetscFE fe_coords;
    PetscInt k = 1;

    DMClone(dm, &vertexDM) >> utilities::PetscUtilities::checkError;
    PetscFECreateLagrange(PETSC_COMM_SELF, dim, dim, PETSC_TRUE, k, PETSC_DETERMINE, &fe_coords) >> utilities::PetscUtilities::checkError;
    DMSetField(vertexDM, 0, nullptr, (PetscObject)fe_coords) >> utilities::PetscUtilities::checkError;
    PetscFEDestroy(&fe_coords) >> utilities::PetscUtilities::checkError;
    DMCreateDS(vertexDM) >> utilities::PetscUtilities::checkError;

    ablate::domain::Range cellRange; 
    auto fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver*>(&flow);

    if (!fvSolver) {
      return;
    }
    PetscReal h;
    DMPlexGetMinRadius(dm, &h);

    PetscInt cStart, cEnd; 
    DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
    for (PetscInt i = cStart; i < cEnd; ++i) {

        PetscInt cell = cellRange.GetPoint(i);
        PetscInt nNeighbors, *neighbors;
        PetscReal layers=3;
        DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        cellNeighbors[cell] = std::vector<PetscInt>(neighbors, neighbors + nNeighbors);
        //corresponding to each of the neighbors, get the weight of the neighbor which is calculated via (inverse distance) /(total weight) such that the sum of the weights is 1
        cellWeights[cell] = std::vector<PetscReal>(nNeighbors, 0);

        for (PetscInt j = 0; j < nNeighbors; ++j) {
            //define a centroid for the cell and store it via DMPlexComputeCellGeometryFVM
            PetscReal ccentroid[3];
            DMPlexComputeCellGeometryFVM(dm, cell, nullptr, ccentroid, nullptr);
            //define a neighbor centroid and store it in the same way
            PetscReal ncentroid[3];
            DMPlexComputeCellGeometryFVM(dm, neighbors[j], nullptr, ncentroid, nullptr);
            PetscReal d = std::sqrt(PetscSqr(ncentroid[0] - ccentroid[0]) + PetscSqr(ncentroid[1] - ccentroid[1]) + PetscSqr(ncentroid[2] - ccentroid[2]));
            
            if (d > 1e-10) {
                //h is the getminradius
                cellWeights[cell][j] = PetscExpReal( -d*d/ (2*PetscSqr(2 * h)) );
            }
            else {
                cellWeights[cell][j] = 1.0;
            }
        }

        PetscReal totalWeight = std::accumulate(cellWeights[cell].begin(), cellWeights[cell].end(), 0.0);
        for (PetscInt j = 0; j < nNeighbors; ++j) {
            cellWeights[cell][j] /= totalWeight;
        }

        //for the last neighbor, set the weight such that the sum of the weights is exactly 1
        // if (nNeighbors > 0) {
        //     cellWeights[cell][nNeighbors - 1] = 1.0 - std::accumulate(cellWeights[cell].begin(), cellWeights[cell].end() - 1, 0.0);
        // }
        DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);

    }
    
    PetscInt vStart, vEnd;
    DMPlexGetDepthStratum(vertexDM, 0, &vStart, &vEnd);
    for (PetscInt vertex = vStart; vertex < vEnd; ++vertex) {
        PetscInt nvn, *vertexneighbors;
        DMPlexVertexGetCells(dm, vertex, &nvn, &vertexneighbors);
        vertexNeighbors[vertex] = std::vector<PetscInt>(vertexneighbors, vertexneighbors + nvn);
        DMPlexVertexRestoreCells(dm, vertex, &nvn, &vertexneighbors);
    }

    auto intSharpPreStage = std::bind(intSharpPreStageWrapper, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, this);
    flow.RegisterPreStage(intSharpPreStage);
}

PetscErrorCode ablate::finiteVolume::processes::IntSharp::PreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime) {
    PetscFunctionBegin;

    const auto &fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver &>(solver);
    ablate::domain::Range cellRange; 
    fvSolver.GetCellRangeWithoutGhost(cellRange);
    PetscInt dim; 
    PetscCall(DMGetDimension(fvSolver.GetSubDomain().GetDM(), &dim));
    DM dm = fvSolver.GetSubDomain().GetDM();
    Vec globFlowVec; 
    PetscCall(TSGetSolution(flowTs, &globFlowVec));
    PetscScalar *flowArray; 
    PetscCall(VecGetArray(globFlowVec, &flowArray));
    Vec locFVec; PetscCall(DMGetLocalVector(dm, &locFVec)); 
    PetscCall(VecZeroEntries(locFVec));

    const auto &eulerOffset = fvSolver.GetSubDomain().GetField(CompressibleFlowFields::EULER_FIELD).offset;
    const auto &vfOffset = fvSolver.GetSubDomain().GetField(VOLUME_FRACTION_FIELD).offset;
    const auto &rhoAlphaOffset = fvSolver.GetSubDomain().GetField(DENSITY_VF_FIELD).offset;
    PetscInt uOff[3]; uOff[0] = vfOffset; uOff[1] = rhoAlphaOffset; uOff[2] = eulerOffset;

    Vec locX = solver.GetSubDomain().GetSolutionVector(); 
    ablate::finiteVolume::processes::IntSharp *process = this;

    std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
    subDomain->UpdateAuxLocalVector();

    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector(); //LOCAL aux vector, not global

    Vec vertexVec; DMGetLocalVector(process->vertexDM, &vertexVec);
    const PetscScalar *solArray; 
    VecGetArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError; //solution (cell centered) variables rho, rhoe, rhou
    PetscScalar *auxArray; 
    VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError; //aux (cell centered) variables phi, p, T, etc
    PetscScalar *vertexArray; 
    VecGetArray(vertexVec, &vertexArray); //vertex based info
    PetscScalar *fArray; 
    PetscCall(VecGetArray(locFVec, &fArray)); //rhs vector
    PetscInt vStart, vEnd; 
    DMPlexGetDepthStratum(process->vertexDM, 0, &vStart, &vEnd);
    PetscInt cStart, cEnd; 
    DMPlexGetHeightStratum(auxDM, 0, &cStart, &cEnd);

    //get the volumeFraction field
    const auto &phiField = subDomain->GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
    //get the euler field in the same manner
    // const auto &gasDensityField = subDomain->GetField("gasDensity");
    const auto &densityVFField = subDomain->GetField("densityvolumeFraction");
    //make a debug field
    const auto &ofield = subDomain->GetField("debug1");
    const auto &gasDensityField = subDomain->GetField("gasDensity");
    const auto &liquidDensityField = subDomain->GetField("liquidDensity");
    const auto &gasEnergyField = subDomain->GetField("gasEnergy");
    const auto &liquidEnergyField = subDomain->GetField("liquidEnergy");

    //let's 

    for (PetscInt cell = cStart; cell < cEnd; ++cell) {

        const PetscScalar *phic; 
        xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
        PetscScalar *rhophiSource; 
        xDMPlexPointLocalRef(dm, cell, densityVFField.id, fArray, &rhophiSource);

        //get the magnitude of the gradient of the volume fraction field using DMPlexCellGradFromCell
        PetscScalar gradphic[dim];
        DMPlexCellGradFromCell(dm, cell, vertexVec, -1, 0, gradphic);
        PetscReal normgradphi = 0.0;
        for (int k = 0; k < dim; ++k) {
            normgradphi += PetscSqr(gradphic[k]);
        }
        normgradphi = PetscSqrtReal(normgradphi);

        PetscScalar *fsharp;
        xDMPlexPointLocalRef(auxDM, cell, ofield.id, auxArray, &fsharp);
        //get the value of the debug field
        *fsharp = process->Gamma * ( (-1 * *phic) * (1 - *phic) * (1 - 2 * *phic) + process->epsilon * (1 - 2 * *phic) * normgradphi );

        // *rhophiSource += rhog * *fsharp;

        PetscScalar *allFields = nullptr; DMPlexPointLocalRef(dm, cell, flowArray, &allFields) >> utilities::PetscUtilities::checkError;
        auto density = allFields[ablate::finiteVolume::CompressibleFlowFields::RHO];
        PetscReal velocity[3]; 
        for (PetscInt d = 0; d < dim; d++) { 
            velocity[d] = allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density; 
        }
        PetscReal pseudoTime = 1e-3;
        PetscReal *densityG, *densityL, *eG, *eL;
        xDMPlexPointLocalRead(auxDM, cell, gasDensityField.id, auxArray, &densityG) >> utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(auxDM, cell, liquidDensityField.id, auxArray, &densityL) >> utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(auxDM, cell, gasEnergyField.id, auxArray, &eG) >> utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(auxDM, cell, liquidEnergyField.id, auxArray, &eL) >> utilities::PetscUtilities::checkError;
        const PetscScalar oldAlpha = allFields[vfOffset];
    
        // update corresponding euler field values based on new alpha
        if (oldAlpha > 1e-3 && oldAlpha < 1-1e-3) {
        allFields[vfOffset] += pseudoTime * *fsharp;
        if (allFields[vfOffset] < 0.0) { allFields[vfOffset] = 0.0; } 
        else if (allFields[vfOffset] > 1.0) { allFields[vfOffset] = 1.0; }
        allFields[rhoAlphaOffset] = *densityG * allFields[vfOffset];
        allFields[ablate::finiteVolume::CompressibleFlowFields::RHO] = allFields[vfOffset] * *densityG + (1 - allFields[vfOffset]) * *densityL;
        allFields[ablate::finiteVolume::CompressibleFlowFields::RHOE] = allFields[rhoAlphaOffset] * *eG + (allFields[ablate::finiteVolume::CompressibleFlowFields::RHO] - allFields[rhoAlphaOffset]) * *eL;
        for (PetscInt d = 0; d < dim; ++d) {
            allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = allFields[ablate::finiteVolume::CompressibleFlowFields::RHO] * velocity[d];
        }
      }

    }

    PetscCall(VecRestoreArray(globFlowVec, &flowArray)); 
    VecRestoreArrayRead(locX, &solArray);
    VecRestoreArray(auxVec, &auxArray);
    VecRestoreArray(vertexVec, &vertexArray);
    VecRestoreArray(locFVec, &fArray);
    solver.RestoreRange(cellRange);

    DMRestoreLocalVector(process->vertexDM, &vertexVec);
    VecDestroy(&vertexVec); 

    PetscFunctionReturn(0);
}



REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::IntSharp, "calculates interface regularization term",
         ARG(PetscReal, "Gamma", "Gamma, velocity scale parameter (approx. umax)"),
         ARG(PetscReal, "epsilon", "epsilon, interface thickness scale parameter (approx. h)"),
         ARG(bool, "flipPhiTilde", "if true: phiTilde-->1-phiTilde (set it to true if primary phase is phi=0 or false if phi=1)")
);
