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
ablate::finiteVolume::processes::IntSharp::IntSharp(PetscReal Gamma, PetscReal epsilon, bool flipPhiTilde, PetscReal boundaryLayerMultiplier) : Gamma(Gamma), epsilon(epsilon), flipPhiTilde(flipPhiTilde), boundaryLayerMultiplier(boundaryLayerMultiplier) {
    // Initialize boundary layer thickness as a multiple of minRadius (will be set in Setup)
    boundaryLayerThickness = 0.0;
    minRadius = 0.0;
    for (int i = 0; i < 6; ++i) {
        boundingBox[i] = 0.0;
    }
}
ablate::finiteVolume::processes::IntSharp::~IntSharp() { DMDestroy(&vertexDM) >> utilities::PetscUtilities::checkError; }

void intSharpPreStageWrapper(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime, ablate::finiteVolume::processes::IntSharp* intSharpProcess) {
    intSharpProcess->PreStage(flowTs, solver, stagetime);
  }

void ablate::finiteVolume::processes::IntSharp::ComputeBoundaryInformation(DM dm) {
    PetscInt dim;
    DMGetDimension(dm, &dim);
    
    // Get bounding box of the domain
    PetscReal xymin[3], xymax[3];
    DMGetBoundingBox(dm, xymin, xymax);
    
    // Store bounding box in the format [xmin, xmax, ymin, ymax, zmin, zmax]
    boundingBox[0] = xymin[0];  // xmin
    boundingBox[1] = xymax[0];  // xmax
    boundingBox[2] = xymin[1];  // ymin
    boundingBox[3] = xymax[1];  // ymax
    boundingBox[4] = xymin[2];  // zmin
    boundingBox[5] = xymax[2];  // zmax
    
    // Get minimum radius (characteristic mesh size)
    DMPlexGetMinRadius(dm, &minRadius);
    
    // Set boundary layer thickness as a multiple of minRadius (e.g., 3-5 cell layers)
    boundaryLayerThickness = boundaryLayerMultiplier * minRadius;
    
    PetscInt cStart, cEnd;
    DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
    
    // Compute boundary distance and weight for each cell
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        PetscReal centroid[3];
        DMPlexComputeCellGeometryFVM(dm, cell, nullptr, centroid, nullptr);
        
        // Compute minimum distance to any boundary
        PetscReal minDistToBoundary = PETSC_INFINITY;
        
        // Check distance to each boundary face
        for (int d = 0; d < dim; ++d) {
            // Distance to lower boundary
            PetscReal distToLower = centroid[d] - boundingBox[2*d];
            if (distToLower < minDistToBoundary) {
                minDistToBoundary = distToLower;
            }
            
            // Distance to upper boundary
            PetscReal distToUpper = boundingBox[2*d + 1] - centroid[d];
            if (distToUpper < minDistToBoundary) {
                minDistToBoundary = distToUpper;
            }
        }
        
        cellBoundaryDistances[cell] = minDistToBoundary;
        
        // Compute boundary weight: 1.0 for interior, 0.0 for boundary (binary)
        PetscReal weight = (minDistToBoundary >= boundaryLayerThickness) ? 1.0 : 0.0;
        cellBoundaryWeights[cell] = weight;
    }
    
    PetscPrintf(PETSC_COMM_WORLD, "[IntSharp] Binary boundary damping: thickness = %g (%.1f * minRadius)\n", 
                boundaryLayerThickness, boundaryLayerThickness / minRadius);
    PetscPrintf(PETSC_COMM_WORLD, "[IntSharp] Bounding box: [%g, %g] x [%g, %g] x [%g, %g]\n",
                boundingBox[0], boundingBox[1], boundingBox[2], boundingBox[3], boundingBox[4], boundingBox[5]);
}

PetscReal ablate::finiteVolume::processes::IntSharp::GetBoundaryWeight(PetscInt cell) const {
    auto it = cellBoundaryWeights.find(cell);
    if (it != cellBoundaryWeights.end()) {
        return it->second;
    }
    return 1.0;  // Default to interior weight if cell not found
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

    // Compute boundary information
    ComputeBoundaryInformation(dm);

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
        PetscReal layers=1;
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
    // const auto &phiField = subDomain->GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
    const auto &phiField = subDomain->GetField("volumeFraction");
    //get the euler field in the same manner
    // const auto &gasDensityField = subDomain->GetField("gasDensity");
    // const auto &densityVFField = subDomain->GetField("densityvolumeFraction");
    //make a debug field
    const auto &ofield = subDomain->GetField("debug1");
    const auto &gasDensityField = subDomain->GetField("gasDensity");
    const auto &liquidDensityField = subDomain->GetField("liquidDensity");
    const auto &gasEnergyField = subDomain->GetField("gasEnergy");
    const auto &liquidEnergyField = subDomain->GetField("liquidEnergy");

    // --- Determinism: Print checksum of solution vector at start ---
    PetscReal solNormStart = 0.0;
    VecNorm(locX, NORM_2, &solNormStart);
    PetscPrintf(PETSC_COMM_WORLD, "[IntSharp::PreStage] Start solution norm: %g\n", solNormStart);

    // Zero locFVec and vertexVec before use (already done)
    PetscCall(VecZeroEntries(locFVec));
    PetscCall(VecZeroEntries(vertexVec));

    // Synchronize ghost cells for input vectors (if needed)
    DMGlobalToLocalBegin(dm, globFlowVec, INSERT_VALUES, locX);
    DMGlobalToLocalEnd(dm, globFlowVec, INSERT_VALUES, locX);
    DMGlobalToLocalBegin(auxDM, auxVec, INSERT_VALUES, auxVec);
    DMGlobalToLocalEnd(auxDM, auxVec, INSERT_VALUES, auxVec);

    // Get the field offset and number of components for debug1 and debug2
    PetscInt offset = ofield.offset;
    PetscInt nComp = ofield.numberComponents;
    const auto &phitildeField = subDomain->GetField("debug2");
    PetscInt phitildeOffset = phitildeField.offset;
    PetscInt phitildeNComp = phitildeField.numberComponents;

    // Zero just the debug1 and debug2 fields for all cells
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        PetscScalar* auxCell = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, ofield.id, auxArray, &auxCell);
        if (auxCell) {
            for (PetscInt c = 0; c < nComp; ++c) {
                auxCell[offset + c] = 0.0;
            }
        }
        PetscScalar* phitildeCell = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, phitildeField.id, auxArray, &phitildeCell);
        if (phitildeCell) {
            for (PetscInt c = 0; c < phitildeNComp; ++c) {
                phitildeCell[phitildeOffset + c] = 0.0;
            }
        }
    }

    // --- Compute phitilde (smoothed volume fraction) for all cells and store in debug2 field ---
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        PetscReal sum = 0.0;
        const auto& neighbors = cellNeighbors[cell];
        const auto& weights = cellWeights[cell];
        for (std::size_t j = 0; j < neighbors.size(); ++j) {
            PetscInt neighbor = neighbors[j];
            PetscReal weight = weights[j];
            // Get phi value for neighbor
            const PetscScalar* neighborPhi = nullptr;
            xDMPlexPointLocalRead(dm, neighbor, phiField.id, solArray, &neighborPhi);
            sum += weight * (neighborPhi ? *neighborPhi : 0.0);
        }
        // Store phitilde in debug2 field
        PetscScalar* phitildeCell = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, phitildeField.id, auxArray, &phitildeCell);
        *phitildeCell = sum;
    }

    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        const PetscScalar *phic; 
        xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
        PetscScalar *fsharp; 
        xDMPlexPointLocalRef(auxDM, cell, ofield.id, auxArray, &fsharp);
        PetscScalar *phitildeCell = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, phitildeField.id, auxArray, &phitildeCell);
        PetscScalar *allFields = nullptr; 
        DMPlexPointLocalRef(dm, cell, flowArray, &allFields) >> utilities::PetscUtilities::checkError;
        // PetscReal phitilde = *phitildeCell;

        // Only process interfacial cells based on phitilde; set fsharp=0 and skip otherwise
        if (*phic <= 1e-3 || *phic >= 1.0 - 1e-3) {
            *fsharp = 0.0;
            continue;
        }

        //get the magnitude of the gradient of the phitilde field using DMPlexCellGradFromCell
        PetscScalar gradphic[dim];
        // DMPlexCellGradFromCell(auxDM, cell, auxVec, phitildeField.id, 0, gradphic);
        //take grad from phic field
        DMPlexCellGradFromCell(dm, cell, locX, phiField.id, 0, gradphic);
        // DMPlexCellGradFromCell(dm, cell, vertexVec, -1, 0, gradphic);
        PetscReal normgradphi = 0.0;
        for (int k = 0; k < dim; ++k) {
            normgradphi += PetscSqr(gradphic[k]);
        }
        normgradphi = PetscSqrtReal(normgradphi);

        // Compute fsharp for interfacial cells using phitilde
        // *fsharp = process->Gamma * ( (-1 * phitilde) * (1 - phitilde) * (1 - 2 * phitilde) + process->epsilon * (1 - 2 * phitilde) * normgradphi );
        //get it from phic instead of phitilde
        *fsharp = process->Gamma * ( (-1 * *phic) * (1 - *phic) * (1 - 2 * *phic) + process->epsilon * (1 - 2 * *phic) * normgradphi );
        
        // Apply boundary weight to fsharp (dampen near boundaries)
        PetscReal boundaryWeight = process->GetBoundaryWeight(cell);
        *fsharp *= boundaryWeight;


        // *rhophiSource += rhog * *fsharp;

        PetscReal velocity[3]; 
        for (PetscInt d = 0; d < dim; d++) { 
            velocity[d] = allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / allFields[ablate::finiteVolume::CompressibleFlowFields::RHO]; 
        }
        PetscReal pseudoTime = 1e-3;
        PetscReal *densityG, *densityL, *eG, *eL;
        xDMPlexPointLocalRead(auxDM, cell, gasDensityField.id, auxArray, &densityG) >> utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(auxDM, cell, liquidDensityField.id, auxArray, &densityL) >> utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(auxDM, cell, gasEnergyField.id, auxArray, &eG) >> utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(auxDM, cell, liquidEnergyField.id, auxArray, &eL) >> utilities::PetscUtilities::checkError;
    
        // update corresponding euler field values based on new alpha
        if (*phic > 1e-3 && *phic < 1-1e-3) {
            allFields[vfOffset] += pseudoTime * *fsharp;
            if (allFields[vfOffset] < 0.0) { allFields[vfOffset] = 0.0; } 
            else if (allFields[vfOffset] > 1.0) { allFields[vfOffset] = 1.0; }
            allFields[rhoAlphaOffset] = *densityG * allFields[vfOffset];
            allFields[ablate::finiteVolume::CompressibleFlowFields::RHO] = allFields[vfOffset] * *densityG + (1 - allFields[vfOffset]) * *densityL;
            // allFields[ablate::finiteVolume::CompressibleFlowFields::RHOE] = allFields[rhoAlphaOffset] * *eG + (allFields[ablate::finiteVolume::CompressibleFlowFields::RHO] - allFields[rhoAlphaOffset]) * *eL;
            for (PetscInt d = 0; d < dim; ++d) {
                allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = allFields[ablate::finiteVolume::CompressibleFlowFields::RHO] * velocity[d];
            }
        }
    }

    // Synchronize ghost cells for output vectors (if needed)
    DMLocalToGlobalBegin(dm, locFVec, ADD_VALUES, globFlowVec);
    DMLocalToGlobalEnd(dm, locFVec, ADD_VALUES, globFlowVec);
    DMLocalToGlobalBegin(auxDM, auxVec, INSERT_VALUES, auxVec);
    DMLocalToGlobalEnd(auxDM, auxVec, INSERT_VALUES, auxVec);

    // --- Determinism: Print checksum of solution vector at end ---
    PetscReal solNormEnd = 0.0;
    VecNorm(locX, NORM_2, &solNormEnd);
    PetscPrintf(PETSC_COMM_WORLD, "[IntSharp::PreStage] End solution norm: %g\n", solNormEnd);
    
    // Count cells with boundary damping (binary: 0 = damped, 1 = not damped)
    PetscInt boundaryCells = 0;
    for (const auto& pair : cellBoundaryWeights) {
        if (pair.second == 0.0) {
            boundaryCells++;
        }
    }
    if (boundaryCells > 0) {
        PetscPrintf(PETSC_COMM_WORLD, "[IntSharp::PreStage] Applied binary boundary damping to %d cells (fsharp = 0)\n", boundaryCells);
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
         ARG(bool, "flipPhiTilde", "if true: phiTilde-->1-phiTilde (set it to true if primary phase is phi=0 or false if phi=1)"),
         OPT(PetscReal, "boundaryLayerMultiplier", "multiplier for boundary layer thickness (default: 3.0)")
);
