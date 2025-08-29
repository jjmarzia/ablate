#include "domain/RBF/mq.hpp"
#include "issf.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"
#include <fstream>
#include <PetscTime.h>
#include <numeric>

// Helper functions from SurfaceForce
void PhiNeighborGaussWeight(PetscReal d, PetscReal s, PetscReal *weight) {
    PetscReal pi = 3.14159265358979323846264338327950288419716939937510;
    PetscReal Coeff = 1/(PetscSqrtReal(2*pi)*s);
    PetscReal g0 = Coeff*PetscExpReal(0/ (2*PetscSqr(s)));
    PetscReal gd = Coeff*PetscExpReal(-PetscSqr(d)/ (2*PetscSqr(s)));
    *weight = gd/g0;
}

void Get3DCoordinate(DM dm, PetscInt p, PetscReal *xp, PetscReal *yp, PetscReal *zp) {
    PetscReal vol;
    PetscReal centroid[3];
    DMPlexComputeCellGeometryFVM(dm, p, &vol, centroid, nullptr);
    *xp = centroid[0];
    *yp = centroid[1];
    *zp = centroid[2];
}

// Helper function to copy DM (from surfaceForce.cpp)
static void SF_CopyDM(DM oldDM, const PetscInt pStart, const PetscInt pEnd, const PetscInt nDOF, DM *newDM) {
    PetscSection section;
    // Create a sub auxDM
    DM coordDM;
    DMGetCoordinateDM(oldDM, &coordDM) >> ablate::utilities::PetscUtilities::checkError;
    DMClone(oldDM, newDM) >> ablate::utilities::PetscUtilities::checkError;
    // this is a hard coded "dmAux" that petsc looks for
    DMSetCoordinateDM(*newDM, coordDM) >> ablate::utilities::PetscUtilities::checkError;
    PetscSectionCreate(PetscObjectComm((PetscObject)(*newDM)), &section) >> ablate::utilities::PetscUtilities::checkError;
    PetscSectionSetChart(section, pStart, pEnd) >> ablate::utilities::PetscUtilities::checkError;
    for (PetscInt p = pStart; p < pEnd; ++p) PetscSectionSetDof(section, p, nDOF) >> ablate::utilities::PetscUtilities::checkError;
    PetscSectionSetUp(section) >> ablate::utilities::PetscUtilities::checkError;
    DMSetLocalSection(*newDM, section) >> ablate::utilities::PetscUtilities::checkError;
    PetscSectionDestroy(&section) >> ablate::utilities::PetscUtilities::checkError;
    DMSetUp(*newDM) >> ablate::utilities::PetscUtilities::checkError;
    // This builds the global section information based on the local section. It's necessary if we don't create a global vector
    //    right away.
    DMGetGlobalSection(*newDM, &section) >> ablate::utilities::PetscUtilities::checkError;
    /* Calling DMPlexComputeGeometryFVM() generates the value returned by DMPlexGetMinRadius() */
    Vec cellgeom = NULL;
    Vec facegeom = NULL;
    DMPlexComputeGeometryFVM(*newDM, &cellgeom, &facegeom);
    VecDestroy(&cellgeom);
    VecDestroy(&facegeom);
}

void ablate::finiteVolume::processes::IntSharpSurfaceForce::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
    IntSharpSurfaceForce::subDomain = solver.GetSubDomainPtr();
}

ablate::finiteVolume::processes::IntSharpSurfaceForce::IntSharpSurfaceForce(
    PetscReal Gamma, PetscReal epsilon, bool flipPhiTilde, PetscReal boundaryLayerMultiplier,
    PetscReal sigma, PetscReal C, PetscReal N, bool enableIntSharp, bool enableSurfaceForce) 
    : Gamma(Gamma), epsilon(epsilon), boundaryLayerMultiplier(boundaryLayerMultiplier),
      sigma(sigma), enableIntSharp(enableIntSharp), enableSurfaceForce(enableSurfaceForce) {
    
    // Initialize boundary layer thickness as a multiple of minRadius (will be set in Setup)
    boundaryLayerThickness = 0.0;
    minRadius = 0.0;
    for (int i = 0; i < 6; ++i) {
        boundingBox[i] = 0.0;
    }
}

void intSharpPreStageWrapper(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime, ablate::finiteVolume::processes::IntSharpSurfaceForce* intSharpProcess) {
    intSharpProcess->PreStage(flowTs, solver, stagetime);
}

void ablate::domain::SubDomain::UpdateAuxLocalVector() {
    if (auxDM) {
        DMLocalToGlobal(auxDM, auxLocalVec, INSERT_VALUES, auxGlobalVec) >> utilities::PetscUtilities::checkError;
//        DMLocalToGlobal(auxDM, auxLocalVec, ADD_VALUES, auxGlobalVec) >> utilities::PetscUtilities::checkError;
        DMGlobalToLocal(auxDM, auxGlobalVec, INSERT_VALUES, auxLocalVec) >> utilities::PetscUtilities::checkError;
    }
}

ablate::finiteVolume::processes::IntSharpSurfaceForce::~IntSharpSurfaceForce() { 
    if (enableSurfaceForce) {
        DMDestroy(&vertexDM) >> utilities::PetscUtilities::checkError; 
    }
}



void ablate::finiteVolume::processes::IntSharpSurfaceForce::ComputeBoundaryInformation(DM dm) {
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
    
    // Set boundary layer thickness as a multiple of minRadius
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
    
    PetscPrintf(PETSC_COMM_WORLD, "[IntSharpSurfaceForce] Binary boundary damping: thickness = %g (%.1f * minRadius)\n", 
                boundaryLayerThickness, boundaryLayerThickness / minRadius);
    PetscPrintf(PETSC_COMM_WORLD, "[IntSharpSurfaceForce] Bounding box: [%g, %g] x [%g, %g] x [%g, %g]\n",
                boundingBox[0], boundingBox[1], boundingBox[2], boundingBox[3], boundingBox[4], boundingBox[5]);
}

PetscReal ablate::finiteVolume::processes::IntSharpSurfaceForce::GetBoundaryWeight(PetscInt cell) const {
    auto it = cellBoundaryWeights.find(cell);
    if (it != cellBoundaryWeights.end()) {
        return it->second;
    }
    return 1.0;  // Default to interior weight if cell not found
}

void ablate::finiteVolume::processes::IntSharpSurfaceForce::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    auto dim = flow.GetSubDomain().GetDimensions();
    auto dm = flow.GetSubDomain().GetDM();
    
    // Only create vertexDM if SurfaceForce is enabled (it's needed for vertex operations)
    if (enableSurfaceForce) {
        PetscFE fe_coords;
        PetscInt k = 1;

        DMClone(dm, &vertexDM) >> utilities::PetscUtilities::checkError;
        PetscFECreateLagrange(PETSC_COMM_SELF, dim, dim, PETSC_TRUE, k, PETSC_DETERMINE, &fe_coords) >> utilities::PetscUtilities::checkError;
        DMSetField(vertexDM, 0, nullptr, (PetscObject)fe_coords) >> utilities::PetscUtilities::checkError;
        PetscFEDestroy(&fe_coords) >> utilities::PetscUtilities::checkError;
        DMCreateDS(vertexDM) >> utilities::PetscUtilities::checkError;
    }

    // Compute boundary information (needed for both IntSharp and SurfaceForce)
    if (enableIntSharp || enableSurfaceForce) {
        ComputeBoundaryInformation(dm);
    }

    auto fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver*>(&flow);

    if (!fvSolver) {
        return;
    }
    
    PetscReal h;
    DMPlexGetMinRadius(dm, &h);

    PetscInt cStart, cEnd; 
    DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
    
    // Setup shared neighbor data structures (needed for phitilde computation)
    if (enableIntSharp || enableSurfaceForce) {
        PetscPrintf(PETSC_COMM_WORLD, "[Setup] Setting up shared neighbor data structures for %d cells\n", cEnd - cStart);
        for (PetscInt i = cStart; i < cEnd; ++i) {
            PetscInt cell = i;
            PetscInt nNeighbors, *neighbors;
            PetscReal layers = 3;  // Match original IntSharp: use 3-layer neighbors
            DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
            cellNeighbors[cell] = std::vector<PetscInt>(neighbors, neighbors + nNeighbors);
            cellWeights[cell] = std::vector<PetscReal>(nNeighbors, 0);

            for (PetscInt j = 0; j < nNeighbors; ++j) {
                PetscReal ccentroid[3];
                DMPlexComputeCellGeometryFVM(dm, cell, nullptr, ccentroid, nullptr);
                PetscReal ncentroid[3];
                DMPlexComputeCellGeometryFVM(dm, neighbors[j], nullptr, ncentroid, nullptr);
                PetscReal d = std::sqrt(PetscSqr(ncentroid[0] - ccentroid[0]) + PetscSqr(ncentroid[1] - ccentroid[1]) + PetscSqr(ncentroid[2] - ccentroid[2]));
                
                if (d > 1e-10) {
                    cellWeights[cell][j] = PetscExpReal(-d*d / (2*PetscSqr(2 * h)));
                } else {
                    cellWeights[cell][j] = 1.0;
                }
            }

            PetscReal totalWeight = std::accumulate(cellWeights[cell].begin(), cellWeights[cell].end(), 0.0);
            for (PetscInt j = 0; j < nNeighbors; ++j) {
                cellWeights[cell][j] /= totalWeight;
            }
            
            DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
            
            // Debug: Print info for first few cells
            if (i < cStart + 5) {
                PetscPrintf(PETSC_COMM_WORLD, "[Setup] DEBUG: Cell %d - %d neighbors, totalWeight = %g\n", 
                           cell, nNeighbors, totalWeight);
            }
        }
        PetscPrintf(PETSC_COMM_WORLD, "[Setup] Completed neighbor setup. cellNeighbors size = %zu, cellWeights size = %zu\n", 
                   cellNeighbors.size(), cellWeights.size());
    }
        
        // Setup vertex neighbors (needed for SurfaceForce)
    if (enableSurfaceForce) {
        PetscInt vStart, vEnd;
        DMPlexGetDepthStratum(vertexDM, 0, &vStart, &vEnd);
        for (PetscInt vertex = vStart; vertex < vEnd; ++vertex) {
            PetscInt nvn, *vertexneighbors;
            DMPlexVertexGetCells(dm, vertex, &nvn, &vertexneighbors);
            vertexNeighbors[vertex] = std::vector<PetscInt>(vertexneighbors, vertexneighbors + nvn);
            DMPlexVertexRestoreCells(dm, vertex, &nvn, &vertexneighbors);
        }
    }
    
    // Setup SurfaceForce neighbor data structures
    if (enableSurfaceForce) {
        for (PetscInt i = cStart; i < cEnd; ++i) {
            PetscInt cell = i;
            PetscInt nNeighbors1, *neighbors1, nNeighbors3, *neighbors3;
            
            // 1-layer neighbors
            DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors1, &neighbors1);
            cellNeighbors1[cell] = std::vector<PetscInt>(neighbors1, neighbors1 + nNeighbors1);
            DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors1, &neighbors1);
            
            // 3-layer neighbors
            DMPlexGetNeighbors(dm, cell, 3, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors3, &neighbors3);
            cellNeighbors3[cell] = std::vector<PetscInt>(neighbors3, neighbors3 + nNeighbors3);
            DMPlexRestoreNeighbors(dm, cell, 3, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors3, &neighbors3);
        }
    }

    // Register PreStage function for both IntSharp and SurfaceForce (like original IntSharp)
    if (enableIntSharp || enableSurfaceForce) {
        auto intSharpPreStage = std::bind(intSharpPreStageWrapper, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, this);
        flow.RegisterPreStage(intSharpPreStage);
    }
}



PetscErrorCode ablate::finiteVolume::processes::IntSharpSurfaceForce::PreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime) {
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

    const auto &eulerOffset = fvSolver.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD).offset;
    const auto &vfOffset = fvSolver.GetSubDomain().GetField(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD).offset;
    const auto &rhoAlphaOffset = fvSolver.GetSubDomain().GetField(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD).offset;
    PetscInt uOff[3]; uOff[0] = vfOffset; uOff[1] = rhoAlphaOffset; uOff[2] = eulerOffset;

    Vec locX = solver.GetSubDomain().GetSolutionVector(); 
    ablate::finiteVolume::processes::IntSharpSurfaceForce *process = this;

    std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
    subDomain->UpdateAuxLocalVector();

    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector(); //LOCAL aux vector, not global

    Vec vertexVec = nullptr;
    if (process->enableSurfaceForce) {
        DMGetLocalVector(process->vertexDM, &vertexVec);
    }
    const PetscScalar *solArray; 
    VecGetArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError; //solution (cell centered) variables rho, rhoe, rhou
    PetscScalar *auxArray; 
    VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError; //aux (cell centered) variables phi, p, T, etc
    PetscScalar *vertexArray = nullptr; 
    if (process->enableSurfaceForce && vertexVec) {
        VecGetArray(vertexVec, &vertexArray); //vertex based info
    }
    PetscScalar *fArray; 
    PetscCall(VecGetArray(locFVec, &fArray)); //rhs vector
    PetscInt vStart, vEnd; 
    if (process->enableSurfaceForce) {
        DMPlexGetDepthStratum(process->vertexDM, 0, &vStart, &vEnd);
    }
    PetscInt cStart, cEnd; 
    DMPlexGetHeightStratum(auxDM, 0, &cStart, &cEnd);

    //get the volumeFraction field
    const auto &phiField = subDomain->GetField("volumeFraction");
    const auto &ofield = subDomain->GetField("debug1");
    const auto &phitildeField = subDomain->GetField("debug2");
    const auto &gasDensityField = subDomain->GetField("gasDensity");
    const auto &liquidDensityField = subDomain->GetField("liquidDensity");
    const auto &gasEnergyField = subDomain->GetField("gasEnergy");
    const auto &liquidEnergyField = subDomain->GetField("liquidEnergy");

    // --- Determinism: Print checksum of solution vector at start ---
    PetscReal solNormStart = 0.0;
    VecNorm(locX, NORM_2, &solNormStart);
    PetscPrintf(PETSC_COMM_WORLD, "[IntSharpSurfaceForce::PreStage] Start solution norm: %g\n", solNormStart);

    // Zero locFVec and vertexVec before use (already done)
    PetscCall(VecZeroEntries(locFVec));
    if (process->enableSurfaceForce && vertexVec) {
        PetscCall(VecZeroEntries(vertexVec));
    }

    // Synchronize ghost cells for input vectors (if needed)
    DMGlobalToLocalBegin(dm, globFlowVec, INSERT_VALUES, locX);
    DMGlobalToLocalEnd(dm, globFlowVec, INSERT_VALUES, locX);
    DMGlobalToLocalBegin(auxDM, auxVec, INSERT_VALUES, auxVec);
    DMGlobalToLocalEnd(auxDM, auxVec, INSERT_VALUES, auxVec);

    // Get the field offset and number of components for debug1 and debug2
    PetscInt offset = ofield.offset;
    PetscInt nComp = ofield.numberComponents;
    PetscInt phitildeOffset = phitildeField.offset;
    PetscInt phitildeNComp = phitildeField.numberComponents;

    // Only do shared operations if at least one process is enabled
    if (process->enableIntSharp || process->enableSurfaceForce) {
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

        // --- SHARED: Compute phitilde (smoothed volume fraction) for all cells ---
        // PetscPrintf(PETSC_COMM_WORLD, "[IntSharpSurfaceForce::PreStage] Computing shared phitilde\n");
        for (PetscInt cell = cStart; cell < cEnd; ++cell) {
            PetscReal sum = 0.0;
            const auto& neighbors = process->cellNeighbors[cell];
            const auto& weights = process->cellWeights[cell];
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
    }

    // --- SURFACE FORCE: Compute vertex normals and curvature (if enabled) ---
    DM nvDM = nullptr;
    Vec nvLocalVec = nullptr, nvGlobalVec = nullptr;
    PetscScalar *nvLocalArray = nullptr;
    
    if (process->enableSurfaceForce) {
        // PetscPrintf(PETSC_COMM_WORLD, "[IntSharpSurfaceForce::PreStage] Setting up SurfaceForce vertex operations\n");
        
        // Create and setup vertex normal field
        SF_CopyDM(process->vertexDM, vStart, vEnd, dim, &nvDM);
        DMCreateLocalVector(nvDM, &nvLocalVec);
        DMCreateGlobalVector(nvDM, &nvGlobalVec);
        VecZeroEntries(nvLocalVec);
        VecZeroEntries(nvGlobalVec);
        VecGetArray(nvLocalVec, &nvLocalArray);
        
        // Compute vertex normals
        for (PetscInt vertex = vStart; vertex < vEnd; vertex++) {
            PetscInt nCells, *cells;
            DMPlexVertexGetCells(dm, vertex, &nCells, &cells);
            
            // Check if adjacent to interface
            PetscBool isAdjToInterface = PETSC_FALSE;
            for (PetscInt k = 0; k < nCells && !isAdjToInterface; k++) {
                const PetscScalar *phic;
                xDMPlexPointLocalRead(dm, cells[k], phiField.id, solArray, &phic);
                isAdjToInterface = (*phic > 1e-4 && *phic < 1.0 - 1e-4) ? PETSC_TRUE : PETSC_FALSE;
            }
            
            PetscScalar *nv;
            xDMPlexPointLocalRef(nvDM, vertex, -1, nvLocalArray, &nv);
            
            if (isAdjToInterface) {
                // Compute and normalize gradient
                DMPlexVertexGradFromCell(auxDM, vertex, auxVec, phitildeField.id, 0, nv);
                
                PetscReal norm = 0.0;
                for (int d = 0; d < dim; ++d) norm += PetscSqr(nv[d]);
                norm = PetscSqrtReal(norm);
                
                if (norm > 1e-10) {
                    for (int d = 0; d < dim; ++d) nv[d] = -nv[d] / norm;  // Outward normal
                } else {
                    for (int d = 0; d < dim; ++d) nv[d] = 0.0;
                }
            } else {
                for (int d = 0; d < dim; ++d) nv[d] = 0.0;
            }
            
            DMPlexVertexRestoreCells(dm, vertex, &nCells, &cells);
        }
        
        // Synchronize vertex normals
        DMLocalToGlobal(nvDM, nvLocalVec, INSERT_VALUES, nvGlobalVec);
        DMGlobalToLocal(nvDM, nvGlobalVec, INSERT_VALUES, nvLocalVec);
    }

    // --- PHASE 1: SURFACE FORCE FIRST (if enabled) ---
    if (process->enableSurfaceForce) {
        // PetscPrintf(PETSC_COMM_WORLD, "[IntSharpSurfaceForce::PreStage] PHASE 1: Applying SurfaceForce\n");
        
        for (PetscInt cell = cStart; cell < cEnd; ++cell) {
            const PetscScalar *phic; 
            xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
            
            // Only process interfacial cells
            if (*phic <= 1e-3 || *phic >= 1.0 - 1e-3) {
                continue;
            }
            
            PetscScalar *allFields = nullptr; 
            DMPlexPointLocalRef(dm, cell, flowArray, &allFields) >> utilities::PetscUtilities::checkError;
            
            // Get boundary weight for this cell
            PetscReal boundaryWeight = process->GetBoundaryWeight(cell);
            
            // Compute curvature as trace of gradient of vertex normals
            PetscReal kappa = 0.0;
            for (int d = 0; d < dim; ++d) {
                PetscScalar nabla_n[dim];
                DMPlexCellGradFromVertex(nvDM, cell, nvLocalVec, -1, d, nabla_n);
                kappa += nabla_n[d];
            }
            
            // Apply boundary weight to curvature (zero near boundaries)
            kappa *= boundaryWeight;
            
            // Compute surface force using gradient of phitilde (shared field)
            PetscScalar gradphitilde[dim];
            DMPlexCellGradFromCell(auxDM, cell, auxVec, phitildeField.id, 0, gradphitilde);
            
            PetscReal normgrad = 0.0;
            for (int d = 0; d < dim; ++d) normgrad += PetscSqr(gradphitilde[d]);
            normgrad = PetscSqrtReal(normgrad);
            
            PetscReal surfaceForce[3] = {0.0, 0.0, 0.0};
            if (normgrad > 1e-10) {
                for (int d = 0; d < dim; ++d) {
                    surfaceForce[d] = -1 * process->sigma * kappa * (-gradphitilde[d] / normgrad);
                }
            }
            
            // Apply surface forces directly to solution fields
            auto density = allFields[ablate::finiteVolume::CompressibleFlowFields::RHO];
            PetscReal u[3] = {
                allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] / density,
                allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU + 1] / density,
                allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU + 2] / density
            };
            
            PetscReal pseudoTime = 1e-3;
            for (int d = 0; d < dim; ++d) {
                if (PetscAbs(surfaceForce[d]) > 1e-10) {
                    allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] += pseudoTime * surfaceForce[d];
                    allFields[ablate::finiteVolume::CompressibleFlowFields::RHOE] += pseudoTime * surfaceForce[d] * u[d];
                }
            }
            
            // Update aux fields for visualization (only if SurfaceForce fields exist)
            try {
                const auto &ofield3 = subDomain->GetField("phitilde_surfaceforce");
                const auto &ofield4 = subDomain->GetField("sf_magnitude");
                const auto &ofield6 = subDomain->GetField("kappa");
                
                PetscScalar *optr3, *optr4, *optr6;
                xDMPlexPointLocalRef(auxDM, cell, ofield3.id, auxArray, &optr3);
                xDMPlexPointLocalRef(auxDM, cell, ofield4.id, auxArray, &optr4);
                xDMPlexPointLocalRef(auxDM, cell, ofield6.id, auxArray, &optr6);
                
                PetscScalar *phitildeCell = nullptr;
                xDMPlexPointLocalRef(auxDM, cell, phitildeField.id, auxArray, &phitildeCell);
                *optr3 = *phitildeCell;
                *optr4 = (dim == 3) ? 
                    PetscSqrtReal(PetscSqr(surfaceForce[0]) + PetscSqr(surfaceForce[1]) + PetscSqr(surfaceForce[2])) :
                    PetscSqrtReal(PetscSqr(surfaceForce[0]) + PetscSqr(surfaceForce[1]));
                *optr6 = kappa;
            } catch (const std::exception&) {
                // SurfaceForce fields not defined, skip visualization update
            }
        }
        
        // Synchronize the updated solution after SurfaceForce
        DMLocalToGlobalBegin(dm, locFVec, ADD_VALUES, globFlowVec);
        DMLocalToGlobalEnd(dm, locFVec, ADD_VALUES, globFlowVec);
        DMGlobalToLocalBegin(dm, globFlowVec, INSERT_VALUES, locX);
        DMGlobalToLocalEnd(dm, globFlowVec, INSERT_VALUES, locX);
    }
    
    // --- PHASE 2: INTSHARP SECOND (if enabled) ---
    if (process->enableIntSharp) {
        PetscPrintf(PETSC_COMM_WORLD, "[IntSharpSurfaceForce::PreStage] PHASE 2: Applying IntSharp\n");
        
        for (PetscInt cell = cStart; cell < cEnd; ++cell) {
            const PetscScalar *phic; 
            xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
            PetscScalar *fsharp; 
            xDMPlexPointLocalRef(auxDM, cell, ofield.id, auxArray, &fsharp);
            PetscScalar *allFields = nullptr; 
            DMPlexPointLocalRef(dm, cell, flowArray, &allFields) >> utilities::PetscUtilities::checkError;

            // Only process interfacial cells based on phic; set fsharp=0 and skip otherwise
            if (*phic <= 1e-3 || *phic >= 1.0 - 1e-3) {
                *fsharp = 0.0;
                continue;
            }

            // Get the magnitude of the gradient of the phic field
            PetscScalar gradphic[dim];
            DMPlexCellGradFromCell(dm, cell, locX, phiField.id, 0, gradphic);
            PetscReal normgradphi = 0.0;
            for (int k = 0; k < dim; ++k) {
                normgradphi += PetscSqr(gradphic[k]);
            }
            normgradphi = PetscSqrtReal(normgradphi);

            // Compute fsharp for interfacial cells using phic
            *fsharp = process->Gamma * ( (-1 * *phic) * (1 - *phic) * (1 - 2 * *phic) + process->epsilon * (1 - 2 * *phic) * normgradphi );
            
            // Apply boundary weight to fsharp (dampen near boundaries)
            PetscReal boundaryWeight = process->GetBoundaryWeight(cell);
            *fsharp *= boundaryWeight;

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
        
            // Update solution fields directly with IntSharp corrections
            if (*phic > 1e-3 && *phic < 1-1e-3 && PetscAbs(*fsharp) > 1e-10) {
                allFields[vfOffset] += pseudoTime * *fsharp;
                if (allFields[vfOffset] < 0.0) { allFields[vfOffset] = 0.0; } 
                else if (allFields[vfOffset] > 1.0) { allFields[vfOffset] = 1.0; }
                allFields[rhoAlphaOffset] = *densityG * allFields[vfOffset];
                // allFields[ablate::finiteVolume::CompressibleFlowFields::RHO] = allFields[vfOffset] * *densityG + (1 - allFields[vfOffset]) * *densityL;
                // for (PetscInt d = 0; d < dim; ++d) {
                //     allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = allFields[ablate::finiteVolume::CompressibleFlowFields::RHO] * velocity[d];
                // }
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
    PetscPrintf(PETSC_COMM_WORLD, "[IntSharpSurfaceForce::PreStage] End solution norm: %g\n", solNormEnd);

    PetscCall(VecRestoreArray(globFlowVec, &flowArray)); 
    VecRestoreArrayRead(locX, &solArray);
    VecRestoreArray(auxVec, &auxArray);
    VecRestoreArray(locFVec, &fArray);
    if (process->enableSurfaceForce && vertexVec) {
        VecRestoreArray(vertexVec, &vertexArray);
        DMRestoreLocalVector(process->vertexDM, &vertexVec);
        VecDestroy(&vertexVec);
    }
    
    // Cleanup SurfaceForce resources
    if (process->enableSurfaceForce) {
        VecRestoreArray(nvLocalVec, &nvLocalArray);
        DMRestoreLocalVector(nvDM, &nvLocalVec);
        DMRestoreGlobalVector(nvDM, &nvGlobalVec);
        DMDestroy(&nvDM);
    }
    
    solver.RestoreRange(cellRange);

    PetscFunctionReturn(0);
}



REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::IntSharpSurfaceForce, "combined interface sharpening and surface force process",
         ARG(PetscReal, "Gamma", "IntSharp: velocity scale parameter (approx. umax)"),
         ARG(PetscReal, "epsilon", "IntSharp: interface thickness scale parameter (approx. h)"),
         ARG(bool, "flipPhiTilde", "IntSharp: if true: phiTilde-->1-phiTilde (not used in current implementation)"),
         OPT(PetscReal, "boundaryLayerMultiplier", "IntSharp: multiplier for boundary layer thickness (default: 3.0)"),
         ARG(PetscReal, "sigma", "SurfaceForce: surface tension coefficient"),
         ARG(PetscReal, "C", "SurfaceForce: stdev length with respect to grid spacing magnitude (not used in current implementation)"),
         ARG(PetscReal, "N", "SurfaceForce: number of stdevs that the convolution integral captures (not used in current implementation)"),
         OPT(bool, "enableIntSharp", "whether to enable IntSharp functionality (default: true)"),
         OPT(bool, "enableSurfaceForce", "whether to enable SurfaceForce functionality (default: true)")
); 