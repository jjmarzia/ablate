#include "domain/RBF/mq.hpp"
#include "intSharp.hpp"
#include "finiteVolume/nPhaseFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"
#include <fstream>
#include <PetscTime.h>
#include "nPhaseNonconservativeRHS.hpp"

void ablate::finiteVolume::processes::NPhaseNonconservativeRHS::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
    NPhaseNonconservativeRHS::subDomain = solver.GetSubDomainPtr();
}
ablate::finiteVolume::processes::NPhaseNonconservativeRHS::NPhaseNonconservativeRHS() {}
ablate::finiteVolume::processes::NPhaseNonconservativeRHS::~NPhaseNonconservativeRHS() { DMDestroy(&vertexDM) >> utilities::PetscUtilities::checkError; }

void nPhaseNonconservativeRHSPreStageWrapper(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime, ablate::finiteVolume::processes::NPhaseNonconservativeRHS* nPhaseNonconservativeRHSProcess) {
    nPhaseNonconservativeRHSProcess->PreStage(flowTs, solver, stagetime);
  }

void ablate::finiteVolume::processes::NPhaseNonconservativeRHS::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    auto dim = flow.GetSubDomain().GetDimensions();
    auto dm = flow.GetSubDomain().GetDM();
    PetscFE fe_coords;
    PetscInt k = 1;

    DMClone(dm, &vertexDM) >> utilities::PetscUtilities::checkError;
    PetscFECreateLagrange(PETSC_COMM_SELF, dim, dim, PETSC_TRUE, k, PETSC_DETERMINE, &fe_coords) >> utilities::PetscUtilities::checkError;
    DMSetField(vertexDM, 0, nullptr, (PetscObject)fe_coords) >> utilities::PetscUtilities::checkError;
    PetscFEDestroy(&fe_coords) >> utilities::PetscUtilities::checkError;
    DMCreateDS(vertexDM) >> utilities::PetscUtilities::checkError;

    // ablate::domain::Range cellRange; 
    auto fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver*>(&flow);

    if (!fvSolver) {
      return;
    }
    PetscReal h;
    DMPlexGetMinRadius(dm, &h);

    // PetscInt cStart, cEnd; 
    // DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
    // for (PetscInt i = cStart; i < cEnd; ++i) {

    //     PetscInt cell = cellRange.GetPoint(i);
    //     PetscInt nNeighbors, *neighbors;
    //     PetscReal layers=1;
    //     DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
    //     cellNeighbors[cell] = std::vector<PetscInt>(neighbors, neighbors + nNeighbors);
    //     //corresponding to each of the neighbors, get the weight of the neighbor which is calculated via (inverse distance) /(total weight) such that the sum of the weights is 1
    //     cellWeights[cell] = std::vector<PetscReal>(nNeighbors, 0);

    //     for (PetscInt j = 0; j < nNeighbors; ++j) {
    //         //define a centroid for the cell and store it via DMPlexComputeCellGeometryFVM
    //         PetscReal ccentroid[3];
    //         DMPlexComputeCellGeometryFVM(dm, cell, nullptr, ccentroid, nullptr);
    //         //define a neighbor centroid and store it in the same way
    //         PetscReal ncentroid[3];
    //         DMPlexComputeCellGeometryFVM(dm, neighbors[j], nullptr, ncentroid, nullptr);
    //         PetscReal d = std::sqrt(PetscSqr(ncentroid[0] - ccentroid[0]) + PetscSqr(ncentroid[1] - ccentroid[1]) + PetscSqr(ncentroid[2] - ccentroid[2]));
            
    //         if (d > 1e-10) {
    //             //h is the getminradius
    //             cellWeights[cell][j] = PetscExpReal( -d*d/ (2*PetscSqr(2 * h)) );
    //         }
    //         else {
    //             cellWeights[cell][j] = 1.0;
    //         }
    //     }

    //     PetscReal totalWeight = std::accumulate(cellWeights[cell].begin(), cellWeights[cell].end(), 0.0);
    //     for (PetscInt j = 0; j < nNeighbors; ++j) {
    //         cellWeights[cell][j] /= totalWeight;
    //     }


    //     DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);

    // }
    
    // PetscInt vStart, vEnd;
    // DMPlexGetDepthStratum(vertexDM, 0, &vStart, &vEnd);
    // for (PetscInt vertex = vStart; vertex < vEnd; ++vertex) {
    //     PetscInt nvn, *vertexneighbors;
    //     DMPlexVertexGetCells(dm, vertex, &nvn, &vertexneighbors);
    //     vertexNeighbors[vertex] = std::vector<PetscInt>(vertexneighbors, vertexneighbors + nvn);
    //     DMPlexVertexRestoreCells(dm, vertex, &nvn, &vertexneighbors);
    // }

    auto nPhaseNonconservativeRHSPreStage = std::bind(nPhaseNonconservativeRHSPreStageWrapper, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, this);
    flow.RegisterPreStage(nPhaseNonconservativeRHSPreStage);
}





PetscErrorCode ablate::finiteVolume::processes::NPhaseNonconservativeRHS::PreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime) {
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

    const auto &allaireOffset = fvSolver.GetSubDomain().GetField(NPhaseFlowFields::ALLAIRE_FIELD).offset;
    const auto &alphakOffset = fvSolver.GetSubDomain().GetField(NPhaseFlowFields::ALPHAK).offset;
    const auto &alphakrhokOffset = fvSolver.GetSubDomain().GetField(NPhaseFlowFields::ALPHAKRHOK).offset;
    PetscInt uOff[3]; uOff[0] = alphakOffset; uOff[1] = alphakrhokOffset; uOff[2] = allaireOffset;

    Vec locX = solver.GetSubDomain().GetSolutionVector(); 
    ablate::finiteVolume::processes::NPhaseNonconservativeRHS *process = this;

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

    const auto &alphakField = subDomain->GetField(NPhaseFlowFields::ALPHAK);
    const auto &velocityField = subDomain->GetField(NPhaseFlowFields::UI);
    const auto &debugfield = subDomain->GetField("debug");

    for (PetscInt cell = cStart; cell < cEnd; ++cell) {

        const PetscScalar *alpha1; 
        xDMPlexPointLocalRead(dm, cell, alphakField.id, solArray, &alpha1);
        PetscScalar *term;
        xDMPlexPointLocalRef(auxDM, cell, debugfield.id, auxArray, &term);
        PetscReal divU = 0.0;

        //if dim = 1, then just do a cell centered div(u) based on the neighbors obtained from cellNeighbors;
        //loop over the neighbors and get the grad(u) at the cell center
        //then do a dot product with the unit normal vector of the cell
        //then multiply by alpha1 and add to the rhs

        if (dim == 1) {
            //if cell is near the boundary, then divU is 0
            if (PetscAbs(cell - cStart) < 4 || PetscAbs(cell - cEnd) < 4) {
                divU = 0.0 + 0*velocityField.id;
            }
            else {
                    PetscReal cp1[dim], cm1[dim];
                    PetscScalar *fp1, *fm1;
                    xDMPlexPointLocalRead(auxDM, cell+1, velocityField.id, auxArray, &fp1);
                    xDMPlexPointLocalRead(auxDM, cell-1, velocityField.id, auxArray, &fm1);
                    DMPlexComputeCellGeometryFVM(auxDM, cell+1, nullptr, cp1, nullptr);
                    DMPlexComputeCellGeometryFVM(auxDM, cell-1, nullptr, cm1, nullptr);
                    divU = (fp1[0] - fm1[0]) / (2 * (cp1[0] - cm1[0]));
                    // PetscPrintf(PETSC_COMM_WORLD, "cell %d with left coord %f and right coord %f has left vel %f and right vel %f\n", cell, cm1[0], cp1[0], fm1[0], fp1[0]);
            }
            
        }
        if (dim > 1) {
        for (PetscInt offset = 0; offset < dim; ++offset) {
            PetscScalar gradvel[dim];
            DMPlexCellGradFromCell(auxDM, cell, auxVec, velocityField.id, offset, gradvel);
            divU += gradvel[offset];
            }
        }
        //get the velocity at the cell center
        // PetscScalar *u;
        // xDMPlexPointLocalRead(auxDM, cell, velocityField.id, auxArray, &u);
        *term = *alpha1 * divU;

        PetscScalar *allFields = nullptr; DMPlexPointLocalRef(dm, cell, flowArray, &allFields) >> utilities::PetscUtilities::checkError;
        allFields[alphakOffset] += 0 * *term;
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

REGISTER_WITHOUT_ARGUMENTS(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::NPhaseNonconservativeRHS, "calculates nonconservative rhs term");

