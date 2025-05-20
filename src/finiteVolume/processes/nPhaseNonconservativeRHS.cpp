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
ablate::finiteVolume::processes::NPhaseNonconservativeRHS::NPhaseNonconservativeRHS(const double mInf, const std::shared_ptr<ablate::finiteVolume::processes::PressureGradientScaling> pgs) : pgs(pgs), mInf(mInf) {}
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

    // Get cell and face ranges
    DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;  // Cells
    DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd) >> utilities::PetscUtilities::checkError;  // Faces

    // Only compute mesh relationships in Setup
    cellToFaces.resize(cEnd - cStart);
    faceToCells.resize(fEnd - fStart);

    // Populate cell-face relationships
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        PetscInt nFaces;
        const PetscInt* faces;
        DMPlexGetConeSize(dm, cell, &nFaces) >> utilities::PetscUtilities::checkError;
        DMPlexGetCone(dm, cell, &faces) >> utilities::PetscUtilities::checkError;
        cellToFaces[cell - cStart].assign(faces, faces + nFaces);
    }

    // Populate face-cell relationships
    for (PetscInt face = fStart; face < fEnd; ++face) {
        PetscInt nCells;
        const PetscInt* cells;
        DMPlexGetSupportSize(dm, face, &nCells) >> utilities::PetscUtilities::checkError;
        DMPlexGetSupport(dm, face, &cells) >> utilities::PetscUtilities::checkError;
        faceToCells[face - fStart].assign(cells, cells + nCells);
    }

    // Register prestage
    // auto nPhaseNonconservativeRHSPreStage = std::bind(nPhaseNonconservativeRHSPreStageWrapper, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, this);
    // flow.RegisterPreStage(nPhaseNonconservativeRHSPreStage);

    flow.RegisterRHSFunction(ComputeNonconservativeRHS, this);
    
    // Register post-step function to enforce bounds
    RegisterPostStep(flow);
}

static inline PetscReal MagVector(PetscInt dim, const PetscReal *in) {
    PetscReal mag = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        mag += in[d] * in[d];
    }
    return PetscSqrtReal(mag);
}

void ablate::finiteVolume::processes::NPhaseNonconservativeRHS::ComputeBoundaryDistances() {
    // Initialize all distances to a large number
    cellBoundaryDistance.resize(cEnd - cStart, std::numeric_limits<PetscInt>::max());
    
    // First pass: mark boundary cells (distance = 0)
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        const auto& faces = cellToFaces[cell - cStart];
        for (PetscInt f = 0; f < (PetscInt)faces.size(); ++f) {
            PetscInt face = faces[f];
            const auto& cells = faceToCells[face - fStart];
            if (cells.size() == 1) {  // This is a boundary face
                cellBoundaryDistance[cell - cStart] = 0;
                break;
            }
        }
    }
    
    // Iteratively compute distances to boundary
    bool changed;
    do {
        changed = false;
        for (PetscInt cell = cStart; cell < cEnd; ++cell) {
            if (cellBoundaryDistance[cell - cStart] == std::numeric_limits<PetscInt>::max()) {
                continue;  // Skip cells that haven't been reached yet
            }
            
            // Check all neighboring cells
            const auto& faces = cellToFaces[cell - cStart];
            for (PetscInt f = 0; f < (PetscInt)faces.size(); ++f) {
                PetscInt face = faces[f];
                const auto& cells = faceToCells[face - fStart];
                if (cells.size() == 2) {  // Interior face
                    PetscInt neighborCell = (cells[0] == cell) ? cells[1] : cells[0];
                    if (cellBoundaryDistance[neighborCell - cStart] > cellBoundaryDistance[cell - cStart] + 1) {
                        cellBoundaryDistance[neighborCell - cStart] = cellBoundaryDistance[cell - cStart] + 1;
                        changed = true;
                    }
                }
            }
        }
    } while (changed);
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
    Vec locFVec; 
    PetscCall(DMGetLocalVector(dm, &locFVec)); 
    PetscCall(VecZeroEntries(locFVec));

    const auto &allaireOffset = fvSolver.GetSubDomain().GetField(NPhaseFlowFields::ALLAIRE_FIELD).offset;
    const auto &alphakOffset = fvSolver.GetSubDomain().GetField(NPhaseFlowFields::ALPHAK).offset;
    const auto &alphakrhokOffset = fvSolver.GetSubDomain().GetField(NPhaseFlowFields::ALPHAKRHOK).offset;
    const auto &alphakField = fvSolver.GetSubDomain().GetField(NPhaseFlowFields::ALPHAK);
    const auto &velocityField = fvSolver.GetSubDomain().GetField(NPhaseFlowFields::UI);
    const auto &debugfield = fvSolver.GetSubDomain().GetField("debug");
    const auto &densityField = fvSolver.GetSubDomain().GetField(NPhaseFlowFields::RHO);
    const auto &soskField = fvSolver.GetSubDomain().GetField(NPhaseFlowFields::SOSK);
    const auto &pressureField = fvSolver.GetSubDomain().GetField(NPhaseFlowFields::PRESSURE);

    PetscInt uOff[3]; 
    uOff[0] = alphakOffset; 
    uOff[1] = alphakrhokOffset; 
    uOff[2] = allaireOffset;

    Vec locX = solver.GetSubDomain().GetSolutionVector(); 
    std::shared_ptr<ablate::domain::SubDomain> subDomain = this->subDomain;
    subDomain->UpdateAuxLocalVector();

    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector();

    const PetscScalar *solArray; 
    VecGetArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError;
    PetscScalar *auxArray; 
    VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
    PetscScalar *fArray; 
    PetscCall(VecGetArray(locFVec, &fArray));

    // Resize cellValues vector to hold all cells
    cellValues.resize(cEnd - cStart);
    nPhases = alphakField.numberComponents;

    //compute boundary distance (in order to inform divU=0 near boundaries)
    ComputeBoundaryDistances();

    //load cell values to be called in the face loop
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        auto& cellVal = cellValues[cell - cStart];
        cellVal.alphak.resize(nPhases);
        cellVal.sosk.resize(nPhases);

        // Read cell values
        const PetscScalar *alpha1;
        xDMPlexPointLocalRead(dm, cell, alphakField.id, solArray, &alpha1);
        PetscReal *rho, *p, *u, *sosk;
        xDMPlexPointLocalRead(auxDM, cell, densityField.id, auxArray, &rho);
        xDMPlexPointLocalRead(auxDM, cell, pressureField.id, auxArray, &p);
        xDMPlexPointLocalRead(auxDM, cell, velocityField.id, auxArray, &u);
        xDMPlexPointLocalRead(auxDM, cell, soskField.id, auxArray, &sosk);

        // Store values
        if (alpha1) {
            for (PetscInt k = 0; k < nPhases; k++) {
                cellVal.alphak[k] = alpha1[k];
                cellVal.sosk[k] = sosk[k];
            }
        }
        cellVal.rho = *rho;
        cellVal.p = *p;
        for (PetscInt d = 0; d < dim; d++) {
            cellVal.u[d] = u[d];
        }
        cellVal.divU = 0.0; // Will be computed in next loop
    }

    // Second loop: Compute velocity divergence using pre-computed values
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        auto& cellVal = cellValues[cell - cStart];
        
        // Check if cell is within 5 cells of boundary (petsc ghost cell depth?)
        if (cellBoundaryDistance[cell - cStart] <= 5) {
            // For cells near boundary, set divU to zero
            cellVal.divU = 0.0;
            continue;  // Skip to next cell
        }

        // For interior cells, compute divergence as before
        PetscReal ujj = 0.0;
        
        for (PetscInt f = 0; f < (PetscInt)cellToFaces[cell - cStart].size(); ++f) {
            PetscInt face = cellToFaces[cell - cStart][f];
            
            PetscReal faceNormal[dim];
            PetscReal faceSign = 1.0;  
            PetscReal faceAreaMag = 1.0;
            const auto& cells = faceToCells[face - fStart];
            if (dim == 1) {
                // In 1D, determine if this is a left or right face
                    
                faceSign = (cells[0] == cell) ? 1.0 : -1.0;  // +1 for right face, -1 for left face
                faceNormal[0] = faceSign;  

            } else {
                DMPlexFaceCentroidOutwardAreaNormal(auxDM, cell, face, nullptr, faceNormal);
                faceAreaMag = MagVector(dim, faceNormal);
            }
            

            if (cells.size() != 2) {
                throw std::runtime_error("Face " + std::to_string(face) + " has " + std::to_string(cells.size()) + " cells, expected 2");
            }

            // Get pre-computed values for both cells
            const auto& cellL = cellValues[cells[0] - cStart];
            const auto& cellR = cellValues[cells[1] - cStart];

            PetscReal rho12 = 0.5 * (cellL.rho + cellR.rho);

            // Compute effective speed of sound
            PetscReal sosL = 0.0, sosR = 0.0;
            for (PetscInt k = 0; k < nPhases; k++) {
                if (cellL.alphak[k] > PETSC_SMALL) {
                    sosL += cellL.alphak[k] / cellL.sosk[k];
                }
                if (cellR.alphak[k] > PETSC_SMALL) {
                    sosR += cellR.alphak[k] / cellR.sosk[k];
                }
            }
            sosL = 1.0/sosL;
            sosR = 1.0/sosR;
            PetscReal sos12 = 0.5 * (sosL + sosR);

            // Compute normal velocities
            PetscReal unL = 0.0, unR = 0.0;
            for (PetscInt d = 0; d < dim; d++) {
                unL += cellL.u[d] * faceNormal[d];
                unR += cellR.u[d] * faceNormal[d];
            }

            PetscReal mL = unL / sos12;
            PetscReal mR = unR / sos12;

            PetscReal mBar2 = (PetscSqr(unL) + PetscSqr(unR)) / (2.0 * PetscSqr(sos12));
            PetscReal fa = 1.0;

            if (mInf > 0) {
                PetscReal mInf2 = PetscSqr(mInf);
                PetscReal mO2 = PetscMin(1.0, PetscMax(mBar2, mInf2));
                PetscReal mO = PetscSqrtReal(mO2);
                fa = mO * (2.0 - mO);
            }

            PetscReal m12 = M4Plus(mL) + M4Minus(mR) - (Kp / fa) * PetscMax(1.0 - (sigma * mBar2), 0) * (cellR.p - cellL.p) / (rho12 * sos12 * sos12 * pgsAlpha * pgsAlpha);
            PetscReal vRiem = sos12 * m12;

            //if vriem is nan (due to being unable to be computed near the boundary), just set to zero
            if (PetscIsNanReal(vRiem)) {
                vRiem = 0.0;
            }

            ujj += vRiem * faceAreaMag * faceSign;  

            // Debug print for all faces IF vriem > petscsmall
            // if (vRiem > PETSC_SMALL) {
            //     PetscPrintf(PETSC_COMM_WORLD, "nonconservative vriem %f, alphaL %f, alphaR %f, rhoL %f, rhoR %f, pL %f, pR %f, uL %f, uR %f, soskL %f, soskR %f\n", vRiem, cellL.alphak[0], cellR.alphak[0], cellL.rho, cellR.rho, cellL.p, cellR.p, cellL.u[0], cellR.u[0], cellL.sosk[0], cellR.sosk[0]);
            // //     PetscPrintf(PETSC_COMM_WORLD, "distance %d, faceareaMag %f, face %d (cell %d), normal sign %f, alphaL %f, alphaR %f, rhoL %f, rhoR %f, pL %f, pR %f, uL %f, uR %f, soskL %f, soskR %f, vRiem %f\n", 
            // //     cellBoundaryDistance[cell - cStart], faceAreaMag, face, cell, faceSign, cellL.alphak[0], cellR.alphak[0], cellL.rho, cellR.rho, cellL.p, cellR.p, cellL.u[0], cellR.u[0], cellL.sosk[0], cellR.sosk[0], vRiem);
            // }
            
            
        }

        cellVal.divU = ujj;

        

        // Update the debug field with the computed divergence
        PetscScalar *term;
        xDMPlexPointLocalRef(auxDM, cell, debugfield.id, auxArray, &term);
        *term = cellVal.alphak[0] * cellVal.divU; 

        // if ujj > petscsmall, print it 
        if (PetscAbs(*term) > PETSC_SMALL) {
            PetscPrintf(PETSC_COMM_WORLD, "rhs term %f\n", *term);
        }

        // Update the solution vector if needed
        PetscScalar *allFields = nullptr;
        DMPlexPointLocalRef(dm, cell, flowArray, &allFields) >> utilities::PetscUtilities::checkError;
        if (allFields) {
            allFields[alphakOffset] += 0 * *term;
        }
    }

    // Clean up
    PetscCall(VecRestoreArray(globFlowVec, &flowArray)); 
    VecRestoreArrayRead(locX, &solArray);
    VecRestoreArray(auxVec, &auxArray);
    VecRestoreArray(locFVec, &fArray);
    solver.RestoreRange(cellRange);

    PetscFunctionReturn(0);
}

PetscReal ablate::finiteVolume::processes::NPhaseNonconservativeRHS::M1Plus(PetscReal m) { return 0.5 * (m + PetscAbs(m)); }

PetscReal ablate::finiteVolume::processes::NPhaseNonconservativeRHS::M2Plus(PetscReal m) { return 0.25 * PetscSqr(m + 1); }

PetscReal ablate::finiteVolume::processes::NPhaseNonconservativeRHS::M1Minus(PetscReal m) { return 0.5 * (m - PetscAbs(m)); }
PetscReal ablate::finiteVolume::processes::NPhaseNonconservativeRHS::M2Minus(PetscReal m) { return -0.25 * PetscSqr(m - 1); }

PetscReal ablate::finiteVolume::processes::NPhaseNonconservativeRHS::M4Plus(PetscReal m) {
    if (PetscAbs(m) >= 1.0) {
        return M1Plus(m);
    } else {
        PetscReal beta = 0.125;
        return M2Plus(m) * (1.0 - 16.0 * beta * M2Minus(m));
    }
}
PetscReal ablate::finiteVolume::processes::NPhaseNonconservativeRHS::M4Minus(PetscReal m) {
    if (PetscAbs(m) >= 1.0) {
        return M1Minus(m);
    } else {
        PetscReal beta = 0.125;
        return M2Minus(m) * (1.0 + 16.0 * beta * M2Plus(m));
    }
}
PetscReal ablate::finiteVolume::processes::NPhaseNonconservativeRHS::P5Plus(PetscReal m, double fa) {
    if (PetscAbs(m) >= 1.0) {
        return (M1Plus(m) / (m + 1E-30));
    } else {
        // compute alpha
        double alpha = 3.0 / 16.0 * (-4.0 + 5 * fa * fa);

        return (M2Plus(m) * ((2.0 - m) - 16. * alpha * m * M2Minus(m)));
    }
}
PetscReal ablate::finiteVolume::processes::NPhaseNonconservativeRHS::P5Minus(PetscReal m, double fa) {
    if (PetscAbs(m) >= 1.0) {
        return (M1Minus(m) / (m + 1E-30));
    } else {
        double alpha = 3.0 / 16.0 * (-4.0 + 5 * fa * fa);
        return (M2Minus(m) * ((-2.0 - m) + 16. * alpha * m * M2Plus(m)));
    }
}

PetscErrorCode ablate::finiteVolume::processes::NPhaseNonconservativeRHS::ComputeNonconservativeRHS(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locXVec, Vec locFVec, void *ctx) {
    PetscFunctionBegin;
    
    auto nPhaseNonconservativeRHSProcess = (NPhaseNonconservativeRHS *)ctx;
    
    // Get the subdomain from the solver - use non-const version since we need to modify it
    auto subDomain = const_cast<FiniteVolumeSolver&>(solver).GetSubDomainPtr();
    if (!subDomain) {
        throw std::runtime_error("SubDomain not set in solver");
    }
    
    // Get dimensions
    const PetscInt dim = subDomain->GetDimensions();
    
    // Get cell range
    ablate::domain::Range cellRange;
    solver.GetCellRangeWithoutGhost(cellRange);
    
    // Get field offsets
    const auto &alphakOffset = subDomain->GetField(ALPHAK).offset;
    
    // Get arrays
    PetscScalar *flowArray;
    VecGetArray(locXVec, &flowArray) >> utilities::PetscUtilities::checkError;
    PetscScalar *fArray;
    VecGetArray(locFVec, &fArray) >> utilities::PetscUtilities::checkError;
    
    // Get aux fields
    subDomain->UpdateAuxLocalVector();
    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector();
    PetscScalar *auxArray;
    VecGetArray(auxVec, &auxArray) >> utilities::PetscUtilities::checkError;
    
    // Get field information
    const auto &debugfield = subDomain->GetField("debug");
    const auto &velocityField = subDomain->GetField(NPhaseFlowFields::UI);
    const auto &densityField = subDomain->GetField(NPhaseFlowFields::RHO);
    const auto &pressureField = subDomain->GetField(NPhaseFlowFields::PRESSURE);
    const auto &soskField = subDomain->GetField(NPhaseFlowFields::SOSK);
    
    // Compute boundary distances if not already done
    if (nPhaseNonconservativeRHSProcess->cellBoundaryDistance.empty()) {
        nPhaseNonconservativeRHSProcess->ComputeBoundaryDistances();
    }
    
    // Resize cellValues if needed
    if (nPhaseNonconservativeRHSProcess->cellValues.empty()) {
        nPhaseNonconservativeRHSProcess->cellValues.resize(nPhaseNonconservativeRHSProcess->cEnd - nPhaseNonconservativeRHSProcess->cStart);
        nPhaseNonconservativeRHSProcess->nPhases = subDomain->GetField(ALPHAK).numberComponents;
    }
    
    // Load cell values
    for (PetscInt cell = nPhaseNonconservativeRHSProcess->cStart; cell < nPhaseNonconservativeRHSProcess->cEnd; ++cell) {
        auto& cellVal = nPhaseNonconservativeRHSProcess->cellValues[cell - nPhaseNonconservativeRHSProcess->cStart];
        cellVal.alphak.resize(nPhaseNonconservativeRHSProcess->nPhases);
        cellVal.sosk.resize(nPhaseNonconservativeRHSProcess->nPhases);
        
        // Read cell values
        const PetscScalar *alpha1;
        xDMPlexPointLocalRead(dm, cell, subDomain->GetField(ALPHAK).id, flowArray, &alpha1);
        PetscReal *rho, *p, *u, *sosk;
        xDMPlexPointLocalRead(auxDM, cell, densityField.id, auxArray, &rho);
        xDMPlexPointLocalRead(auxDM, cell, pressureField.id, auxArray, &p);
        xDMPlexPointLocalRead(auxDM, cell, velocityField.id, auxArray, &u);
        xDMPlexPointLocalRead(auxDM, cell, soskField.id, auxArray, &sosk);
        
        // Store values
        if (alpha1) {
            for (PetscInt k = 0; k < nPhaseNonconservativeRHSProcess->nPhases; k++) {
                cellVal.alphak[k] = alpha1[k];
                cellVal.sosk[k] = sosk[k];
            }
        }
        cellVal.rho = *rho;
        cellVal.p = *p;
        for (PetscInt d = 0; d < dim; d++) {
            cellVal.u[d] = u[d];
        }
        cellVal.divU = 0.0; // Will be computed in next loop
    }
    
    // Compute velocity divergence
    for (PetscInt cell = nPhaseNonconservativeRHSProcess->cStart; cell < nPhaseNonconservativeRHSProcess->cEnd; ++cell) {
        auto& cellVal = nPhaseNonconservativeRHSProcess->cellValues[cell - nPhaseNonconservativeRHSProcess->cStart];
        
        // Check if cell is near boundary
        if (nPhaseNonconservativeRHSProcess->cellBoundaryDistance[cell - nPhaseNonconservativeRHSProcess->cStart] <= 5) {
            cellVal.divU = 0.0;
            continue;
        }
        
        // Compute divergence as before
        PetscReal ujj = 0.0;
        for (PetscInt f = 0; f < (PetscInt)nPhaseNonconservativeRHSProcess->cellToFaces[cell - nPhaseNonconservativeRHSProcess->cStart].size(); ++f) {
            PetscInt face = nPhaseNonconservativeRHSProcess->cellToFaces[cell - nPhaseNonconservativeRHSProcess->cStart][f];
            
            PetscReal faceNormal[dim];
            PetscReal faceSign = 1.0;  
            PetscReal faceAreaMag = 1.0;
            const auto& cells = nPhaseNonconservativeRHSProcess->faceToCells[face - nPhaseNonconservativeRHSProcess->fStart];
            if (dim == 1) {
                // In 1D, determine if this is a left or right face
                    
                faceSign = (cells[0] == cell) ? 1.0 : -1.0;  // +1 for right face, -1 for left face
                faceNormal[0] = faceSign;  

            } else {
                DMPlexFaceCentroidOutwardAreaNormal(auxDM, cell, face, nullptr, faceNormal);
                faceAreaMag = MagVector(dim, faceNormal);
            }
            

            if (cells.size() != 2) {
                throw std::runtime_error("Face " + std::to_string(face) + " has " + std::to_string(cells.size()) + " cells, expected 2");
            }

            // Get pre-computed values for both cells
            const auto& cellL = nPhaseNonconservativeRHSProcess->cellValues[cells[0] - nPhaseNonconservativeRHSProcess->cStart];
            const auto& cellR = nPhaseNonconservativeRHSProcess->cellValues[cells[1] - nPhaseNonconservativeRHSProcess->cStart];

            PetscReal rho12 = 0.5 * (cellL.rho + cellR.rho);

            // Compute effective speed of sound
            PetscReal sosL = 0.0, sosR = 0.0;
            for (PetscInt k = 0; k < nPhaseNonconservativeRHSProcess->nPhases; k++) {
                if (cellL.alphak[k] > PETSC_SMALL) {
                    sosL += cellL.alphak[k] / cellL.sosk[k];
                }
                if (cellR.alphak[k] > PETSC_SMALL) {
                    sosR += cellR.alphak[k] / cellR.sosk[k];
                }
            }
            sosL = 1.0/sosL;
            sosR = 1.0/sosR;
            PetscReal sos12 = 0.5 * (sosL + sosR);

            // Compute normal velocities
            PetscReal unL = 0.0, unR = 0.0;
            for (PetscInt d = 0; d < dim; d++) {
                unL += cellL.u[d] * faceNormal[d];
                unR += cellR.u[d] * faceNormal[d];
            }

            PetscReal mL = unL / sos12;
            PetscReal mR = unR / sos12;

            PetscReal mBar2 = (PetscSqr(unL) + PetscSqr(unR)) / (2.0 * PetscSqr(sos12));
            PetscReal fa = 1.0;

            if (nPhaseNonconservativeRHSProcess->mInf > 0) {
                PetscReal mInf2 = PetscSqr(nPhaseNonconservativeRHSProcess->mInf);
                PetscReal mO2 = PetscMin(1.0, PetscMax(mBar2, mInf2));
                PetscReal mO = PetscSqrtReal(mO2);
                fa = mO * (2.0 - mO);
            }

            PetscReal m12 = M4Plus(mL) + M4Minus(mR) - (Kp / fa) * PetscMax(1.0 - (sigma * mBar2), 0) * (cellR.p - cellL.p) / (rho12 * sos12 * sos12 * pgsAlpha * pgsAlpha);
            PetscReal vRiem = sos12 * m12;

            //if vriem is nan (due to being unable to be computed near the boundary), just set to zero
            if (PetscIsNanReal(vRiem)) {
                vRiem = 0.0;
            }

            ujj += vRiem * faceAreaMag * faceSign;  

            // Debug print for all faces IF vriem > petscsmall
            // if (vRiem > PETSC_SMALL) {
            //     PetscPrintf(PETSC_COMM_WORLD, "nonconservative vriem %f, alphaL %f, alphaR %f, rhoL %f, rhoR %f, pL %f, pR %f, uL %f, uR %f, soskL %f, soskR %f\n", vRiem, cellL.alphak[0], cellR.alphak[0], cellL.rho, cellR.rho, cellL.p, cellR.p, cellL.u[0], cellR.u[0], cellL.sosk[0], cellR.sosk[0]);
            // //     PetscPrintf(PETSC_COMM_WORLD, "distance %d, faceareaMag %f, face %d (cell %d), normal sign %f, alphaL %f, alphaR %f, rhoL %f, rhoR %f, pL %f, pR %f, uL %f, uR %f, soskL %f, soskR %f, vRiem %f\n", 
            // //     cellBoundaryDistance[cell - cStart], faceAreaMag, face, cell, faceSign, cellL.alphak[0], cellR.alphak[0], cellL.rho, cellR.rho, cellL.p, cellR.p, cellL.u[0], cellR.u[0], cellL.sosk[0], cellR.sosk[0], vRiem);
            // }
            
            
        }

        cellVal.divU = ujj;


        
        // Update the debug field
        PetscScalar *term;
        xDMPlexPointLocalRef(auxDM, cell, debugfield.id, auxArray, &term);
        *term = cellVal.alphak[0] * cellVal.divU;

        //if term is greater than petscsmall, print it
        if (PetscAbs(*term) > PETSC_SMALL) {
            PetscPrintf(PETSC_COMM_WORLD, "nonconservative term %f\n", *term);
        }
        
        // Update the solution vector
        PetscScalar *allFields = nullptr;
        DMPlexPointLocalRef(dm, cell, fArray, &allFields) >> utilities::PetscUtilities::checkError;
        //if allfields and NOT allfields[alphakOffset] is greater than 1, add it
        if (allFields) {
            allFields[alphakOffset] += 1 * *term/0.01;
        }
    }
    
    // Clean up
    VecRestoreArray(locXVec, &flowArray);
    VecRestoreArray(locFVec, &fArray);
    VecRestoreArray(auxVec, &auxArray);
    solver.RestoreRange(cellRange);
    
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::NPhaseNonconservativeRHS::EnforceAlphaKBounds(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locXVec, void *ctx) {
    PetscFunctionBegin;
    
    // auto nPhaseNonconservativeRHSProcess = (NPhaseNonconservativeRHS *)ctx;
    
    // Get the subdomain from the solver
    auto subDomain = const_cast<FiniteVolumeSolver&>(solver).GetSubDomainPtr();
    if (!subDomain) {
        throw std::runtime_error("SubDomain not set in solver");
    }
    
    // Get cell range
    ablate::domain::Range cellRange;
    solver.GetCellRangeWithoutGhost(cellRange);
    
    // Get field information
    const auto &alphakField = subDomain->GetField(ALPHAK);
    const auto &alphakOffset = alphakField.offset;
    
    // Get array
    PetscScalar *xArray;
    VecGetArray(locXVec, &xArray) >> utilities::PetscUtilities::checkError;
    
    // Loop over all cells
    for (PetscInt cell = cellRange.start; cell < cellRange.end; ++cell) {
        PetscScalar *allFields = nullptr;
        DMPlexPointLocalRef(dm, cell, xArray, &allFields) >> utilities::PetscUtilities::checkError;
        
        if (allFields) {
            // Loop over all phases
            for (PetscInt k = 0; k < alphakField.numberComponents; k++) {
                // Enforce bounds: 0 <= alpha_k <= 1
                allFields[alphakOffset + k] = PetscMax(0.0, PetscMin(1.0, allFields[alphakOffset + k]));
            }
        }
    }
    
    // Clean up
    VecRestoreArray(locXVec, &xArray);
    solver.RestoreRange(cellRange);
    
    PetscFunctionReturn(0);
}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::NPhaseNonconservativeRHS, "calculates nonconservative rhs term",
         OPT(double, "mInf", "must be same as mInf in ausmpUp"),
         OPT(ablate::finiteVolume::processes::PressureGradientScaling, "pgs", "must be same as pgs in ausmpUp"));