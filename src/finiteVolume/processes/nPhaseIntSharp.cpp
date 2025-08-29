#include "nPhaseIntSharp.hpp"
#include "eos/kthStiffenedGas.hpp"
#include "eos/nPhase.hpp"
#include "finiteVolume/nPhaseFlowFields.hpp"
#include "nPhaseAllaireAdvection.hpp"
#include "utilities/petscUtilities.hpp"
#include <petsc/private/dmpleximpl.h>

namespace ablate::finiteVolume::processes {

    void ablate::finiteVolume::processes::NPhaseIntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
        NPhaseIntSharp::subDomain = solver.GetSubDomainPtr();
    }

    ablate::finiteVolume::processes::NPhaseIntSharp::NPhaseIntSharp(const std::vector<PetscReal>& Gammak, const std::vector<PetscReal>& epsilonk, const std::vector<PetscInt>& flipPhiTildek, PetscReal boundaryLayerMultiplier) : Gammak(Gammak), epsilonk(epsilonk), flipPhiTildek(flipPhiTildek), boundaryLayerMultiplier(boundaryLayerMultiplier) {
        // Initialize boundary layer thickness as a multiple of minRadius (will be set in Setup)
        boundaryLayerThickness = 0.0;
        minRadius = 0.0;
        for (int i = 0; i < 6; ++i) {
            boundingBox[i] = 0.0;
        }
    }

    ablate::finiteVolume::processes::NPhaseIntSharp::~NPhaseIntSharp() { DMDestroy(&vertexDM) >> utilities::PetscUtilities::checkError; }

    void ablate::finiteVolume::processes::NPhaseIntSharp::ComputeBoundaryInformation(DM dm) {
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
    }

    PetscReal ablate::finiteVolume::processes::NPhaseIntSharp::GetBoundaryWeight(PetscInt cell) const {
        auto it = cellBoundaryWeights.find(cell);
        if (it != cellBoundaryWeights.end()) {
            return it->second;
        }
        return 1.0;  // Default to interior weight if cell not found
    }

    void nPhaseIntSharpPreStageWrapper(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime, ablate::finiteVolume::processes::NPhaseIntSharp* nPhaseIntSharpProcess) {
        nPhaseIntSharpProcess->PreStage(flowTs, solver, stagetime);
    }



    void NPhaseIntSharp::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] Starting Setup\n");

        NPhaseIntSharp::subDomain = flow.GetSubDomainPtr();
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] Got subDomain\n");

        auto dim = flow.GetSubDomain().GetDimensions();
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] Got dimensions: %d\n", dim);
        
        auto dm = flow.GetSubDomain().GetDM();
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] Got DM\n");
        
        PetscFE fe_coords;
        PetscInt k = 1;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to clone DM\n");
        DMClone(dm, &vertexDM) >> utilities::PetscUtilities::checkError;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] DM cloned successfully\n");
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to create FE\n");
        PetscFECreateLagrange(PETSC_COMM_SELF, dim, dim, PETSC_TRUE, k, PETSC_DETERMINE, &fe_coords) >> utilities::PetscUtilities::checkError;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] FE created\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to set field\n");
        DMSetField(vertexDM, 0, nullptr, (PetscObject)fe_coords) >> utilities::PetscUtilities::checkError;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] Field set\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to destroy FE\n");
        PetscFEDestroy(&fe_coords) >> utilities::PetscUtilities::checkError;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] FE destroyed\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to create DS\n");
        DMCreateDS(vertexDM) >> utilities::PetscUtilities::checkError;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] DS created\n");

        // Compute boundary information
        ComputeBoundaryInformation(dm);

                //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to get cell range\n");
        ablate::domain::Range cellRange; 
        auto fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver*>(&flow);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] Dynamic cast done\n");

        if (!fvSolver) {
          //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] fvSolver cast failed, returning\n");
          return;
        }
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] fvSolver cast successful\n");

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to get height stratum\n");
        PetscInt cStart, cEnd; DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] Height stratum: %d to %d\n", cStart, cEnd);
        cellRange.start = cStart; cellRange.end = cEnd;

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to get depth stratum\n");
        PetscInt vStart, vEnd;
        DMPlexGetDepthStratum(vertexDM, 0, &vStart, &vEnd);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] Depth stratum: %d to %d\n", vStart, vEnd);

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to register PreStage\n");
        // flow.RegisterRHSFunction(ComputeTerm, this);
        auto nPhaseIntSharpPreStage = std::bind(nPhaseIntSharpPreStageWrapper, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, this);
        flow.RegisterPreStage(nPhaseIntSharpPreStage);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] PreStage registered successfully\n");
        
    }

    PetscErrorCode ablate::finiteVolume::processes::NPhaseIntSharp::PreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime) {
        PetscFunctionBegin;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Starting PreStage\n");

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to do dynamic cast\n");
        const auto &fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver &>(solver);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Dynamic cast successful\n");
        
        ablate::domain::Range cellRange; 
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get cell range\n");
        fvSolver.GetCellRangeWithoutGhost(cellRange);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell range: %d to %d\n", cellRange.start, cellRange.end);
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get dimension\n");
        PetscInt dim; 
        PetscCall(DMGetDimension(fvSolver.GetSubDomain().GetDM(), &dim));
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got dimension: %d\n", dim);
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get DM\n");
        DM dm = fvSolver.GetSubDomain().GetDM();
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got DM\n");
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get solution vector\n");
        Vec globFlowVec; 
        PetscCall(TSGetSolution(flowTs, &globFlowVec));
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got solution vector\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get flow array\n");
        PetscScalar *flowArray; 
        PetscCall(VecGetArray(globFlowVec, &flowArray));
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got flow array\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get local vector\n");
        Vec locFVec; PetscCall(DMGetLocalVector(dm, &locFVec)); 
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got local vector\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to zero entries\n");
        PetscCall(VecZeroEntries(locFVec));
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Zeroed entries\n");

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get solution vector from solver\n");
        Vec locX = solver.GetSubDomain().GetSolutionVector(); 
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got solution vector from solver\n");
        
        ablate::finiteVolume::processes::NPhaseIntSharp *process = this;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got process pointer\n");

        std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got subDomain\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get aux DM\n");
        DM auxDM = subDomain->GetAuxDM();
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got aux DM\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get aux vector\n");
        Vec auxVec = subDomain->GetAuxVector();
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got aux vector\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get height stratum from aux DM\n");
        PetscInt cStart, cEnd; DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Aux height stratum: %d to %d\n", cStart, cEnd);
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get vertex vector\n");
        Vec vertexVec; 
        DMGetLocalVector(process->vertexDM, &vertexVec);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got vertex vector\n");

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get arrays\n");
        const PetscScalar *solArray;
        PetscScalar *auxArray;
        PetscScalar *vertexArray;
        PetscScalar *fArray;

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get sol array\n");
        VecGetArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got sol array\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get aux array\n");
        VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got aux array\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get vertex array\n");
        VecGetArray(vertexVec, &vertexArray);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got vertex array\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get f array\n");
        VecGetArray(locFVec, &fArray);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got f array\n");

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get fields\n");
        // ablate::domain::Range cellRange;
        // solver.GetCellRangeWithoutGhost(cellRange);

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get alphak field\n");
        const auto &alphakField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got alphak field\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get alphakrhok field\n");
        const auto &alphakrhokField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got alphakrhok field\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get allaire field\n");
        // const auto &allaireField = solver.GetSubDomain().GetField(ablate::finiteVolume::NPhaseFlowFields::ALLAIRE_FIELD);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got allaire field\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get fsharpk field\n");
        const auto &fsharpkField = subDomain->GetField("fsharpk");
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got fsharpk field\n");

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get phases\n");
        // const auto alphakOffset = alphakField.offset;
        // const auto alphakrhokOffset = alphakrhokField.offset;
        // const auto allaireOffset = allaireField.offset;
        // const auto fsharpkOffset = fsharpkField.offset;

        std::size_t phases = alphakField.numberComponents;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Number of phases: %zu\n", phases);

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get min radius\n");
        // PetscReal h;
        // DMPlexGetMinRadius(auxDM, &h);
        // //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Min radius: %f\n", h);



        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to start cell loop\n");
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell range: %d to %d\n", cStart, cEnd);
        
        for (PetscInt cell = cStart; cell < cEnd; ++cell) {
            if (cell % 100 == 0) {  // Print every 100th cell to avoid spam
                // PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Processing cell %d\n", cell);
            }

            // Check boundary weight first - skip entire calculation if cell is near boundary
            PetscReal boundaryWeight = process->GetBoundaryWeight(cell);
            if (boundaryWeight < 0.5) {  // Skip cells with boundary weight < 0.5
                //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: Skipping boundary cell (weight = %f)\n", cell, boundaryWeight);
                continue;  // Skip to next cell
            }

            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: About to get old values\n", cell);
            //keep old values
            std::vector<PetscReal> alphakold(phases);
            std::vector<PetscReal> alphakrhokold(phases);
            PetscReal rhoold = 0.0;
            std::vector<PetscReal> uiold(dim);
            std::vector<PetscReal> rhokold(phases);
            
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: About to get allFields\n", cell);
            PetscScalar *allFields = nullptr;
            DMPlexPointLocalRef(dm, cell, flowArray, &allFields) >> utilities::PetscUtilities::checkError;
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: Got allFields\n", cell);
            
            //coordinates of this cell:
            PetscReal centroid[dim];
            DMPlexComputeCellGeometryFVM(dm, cell, nullptr, centroid, nullptr);
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: Centroid: %g, %g, %g\n", cell, centroid[0], centroid[1], centroid[2]);
            
            // Get field pointers ONCE outside the loop
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: About to get field pointers\n", cell);
            const PetscScalar *alphakFieldPtr;
            const PetscScalar *alphakrhokFieldPtr;
            xDMPlexPointLocalRead(dm, cell, alphakField.id, solArray, &alphakFieldPtr);
            xDMPlexPointLocalRead(dm, cell, alphakrhokField.id, solArray, &alphakrhokFieldPtr);
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: Got field pointers\n", cell);
            
            // Now access the components correctly
            for (std::size_t k = 0; k < phases; ++k) {
                alphakold[k] = alphakFieldPtr[k];
                alphakrhokold[k] = alphakrhokFieldPtr[k];
                rhoold += alphakrhokFieldPtr[k];
                
                // Avoid division by zero
                if (alphakFieldPtr[k] > PETSC_SMALL) {
                    rhokold[k] = alphakrhokFieldPtr[k] / alphakFieldPtr[k];
                } else {
                    rhokold[k] = 0.0;
                    //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: Warning - alphak[%zu] = %g (very small)\n", cell, k, alphakFieldPtr[k]);
                }
            }
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: About to get uiold\n", cell);
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: Current uiold: %g, %g, %g\n", cell, uiold[0], uiold[1], uiold[2]);
            if (rhoold > PETSC_SMALL) {
                // PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: Current rhoold: %g\n", cell, rhoold);
            }
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Current alphakrhok for all k: ");
            for (std::size_t k = 0; k < phases; ++k) {
                //PetscPrintf(MPI_COMM_WORLD, "%g ", alphakrhokFieldPtr[k]);
            }
            //PetscPrintf(MPI_COMM_WORLD, "\n");
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: Current allFields: %g, %g, %g\n", cell, allFields[ablate::finiteVolume::NPhaseFlowFields::RHOU], allFields[ablate::finiteVolume::NPhaseFlowFields::RHOU + 1], allFields[ablate::finiteVolume::NPhaseFlowFields::RHOU + 2]);
            
            // Check for zero density to avoid division by zero
            if (rhoold < PETSC_SMALL) {
                //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: Zero density detected, skipping cell\n", cell);
                continue;  // Skip this cell entirely
            }
            
            for (PetscInt d = 0; d < dim; ++d) {
                uiold[d] = allFields[ablate::finiteVolume::NPhaseFlowFields::RHOU + d] / rhoold;
            }

            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: About to compute fsharpk\n", cell);

            //compute fsharpk for all k
            for (std::size_t k = 0; k < phases; ++k) {
                // Use the field pointers we already got, or get them if needed
                PetscScalar *fsharpk;
                xDMPlexPointLocalRef(auxDM, cell, fsharpkField.id, auxArray, &fsharpk);
                
                if (alphakold[k] <= 1e-3 || alphakold[k] >= 1.0 - 1e-3) {
                    fsharpk[k] = 0.0;
                    continue;
                }

                PetscScalar gradalphak[dim];
                DMPlexCellGradFromCell(auxDM, cell, auxVec, alphakField.id, k, gradalphak);

                PetscReal normgradalphak = 0.0;
                for (PetscInt d = 0; d < dim; ++d) {
                    normgradalphak += PetscSqr(gradalphak[d]);
                }
                normgradalphak = PetscSqrtReal(normgradalphak);

                PetscReal Gammak = process->Gammak[k];
                PetscReal epsilonk = process->epsilonk[k];

                PetscReal alphaktilde = alphakold[k];  // Use the stored old value
                if (process->flipPhiTildek[k] == 1) {
                    alphaktilde = 1 - alphakold[k];
                }

                fsharpk[k] = Gammak * ( (-1 * alphaktilde) * (1 - alphaktilde) * (1 - 2 * alphaktilde) + epsilonk * (1 - 2 * alphaktilde) * normgradalphak );
                // Boundary weight already checked at start of cell loop - no need to multiply here
                // PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: fsharpk[%zu] = %g\n", cell, k, fsharpk[k]);
            }
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: fsharpk computed\n", cell);

            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: About to update alphak, alphakrhok=alphak*rhokold for all k\n", cell);
            //update alphak, alphakrhok=alphak*rhokold for all k
            PetscReal dt = 1e-4;
            PetscReal rho = 0.0;
            for (std::size_t k = 0; k < phases; ++k) {
                // const PetscScalar *alphak;
                // xDMPlexPointLocalRef(dm, cell, alphakField.id, solArray, &alphak);
                // const PetscScalar *alphakrhok;
                // xDMPlexPointLocalRef(dm, cell, alphakrhokField.id, solArray, &alphakrhok);
                PetscScalar *fsharpk;
                xDMPlexPointLocalRef(auxDM, cell, fsharpkField.id, auxArray, &fsharpk);

                allFields[alphakField.offset + k] += dt * fsharpk[k];
                if (allFields[alphakField.offset + k] < 0.0) { allFields[alphakField.offset + k] = 0.0; }
                else if (allFields[alphakField.offset + k] > 1.0) { allFields[alphakField.offset + k] = 1.0; }

                allFields[alphakrhokField.offset + k] = allFields[alphakField.offset + k] * rhokold[k];
                rho += allFields[alphakrhokField.offset + k];
            }
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: alphak, alphakrhok=alphak*rhokold for all k updated\n", cell);

            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: About to update rhou = rho*uiold, rhoe = rho*eold\n", cell);
            //update rhou = rho*uiold, rhoe = rho*eold
            for (std::size_t k = 0; k < phases; ++k) {
                for (PetscInt d = 0; d < dim; ++d) {
                    allFields[ablate::finiteVolume::NPhaseFlowFields::RHOU + d] = rho * uiold[d];
                }
            }
        }

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to restore arrays\n");
        VecRestoreArrayRead(locX, &solArray);
        VecRestoreArray(auxVec, &auxArray);
        VecRestoreArray(vertexVec, &vertexArray);
        VecRestoreArray(locFVec, &fArray);
        solver.RestoreRange(cellRange);

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to restore vertex vector\n");
        DMRestoreLocalVector(process->vertexDM, &vertexVec);
        VecDestroy(&vertexVec); 

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] PreStage completed successfully\n");
        PetscFunctionReturn(0);
    }

}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, 
    ablate::finiteVolume::processes::NPhaseIntSharp, 
    "N-phase interface regularization term",
    ARG(std::vector<PetscReal>, "Gammak", "Gamma, velocity scale parameter (approx. umax)"),
    ARG(std::vector<PetscReal>, "epsilonk", "epsilon, interface thickness scale parameter (approx. h)"),
    ARG(std::vector<PetscInt>, "flipPhiTildek", "if 1: phiTilde-->1-phiTilde, if 0: keep phiTilde (set to 1 if primary phase is phi=0 or 0 if phi=1)"),
    ARG(PetscReal, "boundaryLayerMultiplier", "multiplier for boundary layer thickness (default: 3.0)"));
