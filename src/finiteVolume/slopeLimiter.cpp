#include "slopeLimiter.hpp"
#include <petsc/private/dmpleximpl.h>
#include <utilities/petscUtilities.hpp>
#include <cmath>

namespace ablate::finiteVolume {

void SlopeLimiter::Setup(DM dm, const domain::Range& cellRange) {
    if (isSetup) return;

    PetscInt cStart, cEnd, fStart, fEnd;
    PetscInt dim;
    
    // Get dimensions and ranges
    DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;
    DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;
    DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd) >> utilities::PetscUtilities::checkError;
    
    //PetscPrintf(PETSC_COMM_WORLD, "SlopeLimiter Setup:\n");
    //PetscPrintf(PETSC_COMM_WORLD, "  Dimension: %d\n", dim);
    //PetscPrintf(PETSC_COMM_WORLD, "  Cell range: %d to %d\n", cStart, cEnd);
    //PetscPrintf(PETSC_COMM_WORLD, "  Face range: %d to %d\n", fStart, fEnd);
    //PetscPrintf(PETSC_COMM_WORLD, "  CellRange: %d to %d\n", cellRange.start, cellRange.end);
    if (cellRange.points) {
        //PetscPrintf(PETSC_COMM_WORLD, "  Using custom cell points\n");
    }

    // Check if we're in 1D
    is1D = (dim == 1);
    //PetscPrintf(PETSC_COMM_WORLD, "  Is 1D: %s\n", is1D ? "true" : "false");

    // Resize cellToFaces to hold faces for each cell
    cellToFaces.resize(cEnd - cStart);
    //PetscPrintf(PETSC_COMM_WORLD, "  Resized cellToFaces to %d cells\n", cEnd - cStart);

    // Get minimum cell radius for 1D calculations
    DMPlexGetMinRadius(dm, &minCellRadius) >> utilities::PetscUtilities::checkError;

    // For 1D case, store cell and face centers
    if (is1D) {
        cellCenters.resize(cEnd - cStart);
        faceCenters.resize(fEnd - fStart);
        //PetscPrintf(PETSC_COMM_WORLD, "  Resized cellCenters to %d cells\n", cEnd - cStart);
        //PetscPrintf(PETSC_COMM_WORLD, "  Resized faceCenters to %d faces\n", fEnd - fStart);

        // Get cell geometry
        DM cellDM;
        // Get the cell DM directly from the input DM
        cellDM = dm;
        //PetscPrintf(PETSC_COMM_WORLD, "using input dm as cell dm\n");
        
        // Check if cellDM is valid
        if (!cellDM) {
            //PetscPrintf(PETSC_COMM_WORLD, "ERROR: cellDM is NULL\n");
            return;
        }
        
        // Compute the cell geometry vector
        DMPlexComputeGeometryFVM(cellDM, &this->cellGeomVec, &this->faceGeomVec) >> utilities::PetscUtilities::checkError;
        //PetscPrintf(PETSC_COMM_WORLD, "dmplexcomputedgeometryfvm\n");
        
        // Check if cellGeomVec is valid
        if (!this->cellGeomVec) {
            //PetscPrintf(PETSC_COMM_WORLD, "ERROR: cellGeomVec is NULL\n");
            return;
        }
        
        // Get vector size and local size
        PetscInt vecSize, localSize;
        VecGetSize(this->cellGeomVec, &vecSize) >> utilities::PetscUtilities::checkError;
        VecGetLocalSize(this->cellGeomVec, &localSize) >> utilities::PetscUtilities::checkError;
        //PetscPrintf(PETSC_COMM_WORLD, "cellGeomVec info - global size: %d, local size: %d\n", vecSize, localSize);
        
        const PetscScalar* cellGeomArray;
        const PetscScalar* faceGeomArray;
        //PetscPrintf(PETSC_COMM_WORLD, "vecgetarrayread\n");
        VecGetArrayRead(this->cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
        VecGetArrayRead(this->faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;
        //PetscPrintf(PETSC_COMM_WORLD, "vecgetarrayread complete\n");

        // Get the cell and face ranges
        //PetscPrintf(PETSC_COMM_WORLD, "cell range: %d to %d, face range: %d to %d\n", cStart, cEnd, fStart, fEnd);

        // Get the cell centers and face centers
        for (PetscInt c = cStart; c < cEnd; c++) {
            const PetscFVCellGeom* cg;
            DMPlexPointLocalRead(cellDM, c, cellGeomArray, &cg) >> utilities::PetscUtilities::checkError;
            if (!cg) {
                //PetscPrintf(PETSC_COMM_WORLD, "WARNING: No geometry data for cell %d\n", c);
                continue;
            }
            if (is1D) {
                cellCenters[c - cStart] = cg->centroid[0];
            }
        }

        for (PetscInt f = fStart; f < fEnd; f++) {
            const PetscFVFaceGeom* fg;
            DMPlexPointLocalRead(dm, f, faceGeomArray, &fg) >> utilities::PetscUtilities::checkError;
            if (!fg) {
                PetscPrintf(PETSC_COMM_WORLD, "WARNING: No geometry data for face %d\n", f);
                continue;
            }
            if (is1D) {
                if (!fg) {
                    PetscPrintf(PETSC_COMM_WORLD, "WARNING: No geometry data for face %d\n", f);
                    continue;
                }
                else{
                    faceCenters[f - fStart] = fg->centroid[0];
                }
                
            }
        }

        // Build the cell to face connectivity
        for (PetscInt f = fStart; f < fEnd; f++) {
            const PetscInt* faceCells;
            DMPlexGetSupport(dm, f, &faceCells) >> utilities::PetscUtilities::checkError;
            if (!faceCells) {
                //PetscPrintf(PETSC_COMM_WORLD, "WARNING: No support for face %d\n", f);
                continue;
            }

            // Add this face to each of its cells
            for (PetscInt c = 0; c < 2; c++) {
                PetscInt cell = faceCells[c];
                if (cell >= cStart && cell < cEnd) {
                    cellToFaces[cell - cStart].push_back(f);
                }
            }
        }

        // Verify the connectivity
        for (PetscInt c = cStart; c < cEnd; c++) {
            if (cellToFaces[c - cStart].empty()) {
                //PetscPrintf(PETSC_COMM_WORLD, "WARNING: Cell %d has no faces\n", c);
            }
        }

        // Clean up
        VecRestoreArrayRead(this->cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
        VecRestoreArrayRead(this->faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;

        isSetup = true;
        //PetscPrintf(PETSC_COMM_WORLD, "SlopeLimiter setup complete\n");
    }

    // Build cell-to-face and face-to-cell connectivity
    //PetscPrintf(PETSC_COMM_WORLD, "  Building connectivity maps...\n");
    for (PetscInt c = cStart; c < cEnd; c++) {
        const PetscInt* faces;
        PetscInt numFaces;
        DMPlexGetCone(dm, c, &faces) >> utilities::PetscUtilities::checkError;
        DMPlexGetConeSize(dm, c, &numFaces) >> utilities::PetscUtilities::checkError;

        if (numFaces == 0) {
            //PetscPrintf(PETSC_COMM_WORLD, "  WARNING: Cell %d has no faces\n", c);
            continue;
        }

        // Store faces for this cell
        cellToFaces[c - cStart].assign(faces, faces + numFaces);

        // Store cells for each face
        for (PetscInt f = 0; f < numFaces; f++) {
            const PetscInt face = faces[f];
            const PetscInt* cells;
            PetscInt numCells;
            DMPlexGetSupport(dm, face, &cells) >> utilities::PetscUtilities::checkError;
            DMPlexGetSupportSize(dm, face, &numCells) >> utilities::PetscUtilities::checkError;

            if (numCells != 2) {
                //PetscPrintf(PETSC_COMM_WORLD, "  WARNING: Face %d has %d cells (expected 2)\n", face, numCells);
                continue;
            }

            faceToCells[face] = std::vector<PetscInt>(cells, cells + 2);

            if (c > cStart + 5 && c < cEnd - 5) {  // Print connectivity for first and last few cells
                //PetscPrintf(PETSC_COMM_WORLD, "  Cell %d -> Face %d -> Cells [%d, %d]\n", 
                        //    c, face, cells[0], cells[1]);
            }
        }
    }

    // Verify connectivity maps
    //PetscPrintf(PETSC_COMM_WORLD, "  Verifying connectivity maps...\n");
    for (PetscInt c = cStart; c < cEnd; c++) {
        if (cellToFaces[c - cStart].empty()) {
            //PetscPrintf(PETSC_COMM_WORLD, "  ERROR: Cell %d has no faces in cellToFaces map\n", c);
        }
    }

    for (const auto& [face, cells] : faceToCells) {
        if (cells.size() != 2) {
            //PetscPrintf(PETSC_COMM_WORLD, "  ERROR: Face %d has %d cells in faceToCells map\n", 
                    //    face, (int)cells.size());
        }
    }
}

PetscInt SlopeLimiter::GetNeighborCell(PetscInt cell, PetscInt face) const {
    const auto& cells = faceToCells.at(face);
    return (cells[0] == cell) ? cells[1] : cells[0];
}

PetscReal SlopeLimiter::GetCellToFaceVector1D(PetscInt cell, PetscInt face) const {
    // Get the two faces for this cell
    const auto& faces = cellToFaces[cell];
    if (faces.size() != 2) {
        throw std::runtime_error("Cell " + std::to_string(cell) + " does not have exactly 2 faces in 1D");
    }

    // Determine if this is the left (-h) or right (+h) face
    PetscInt leftFace = PetscMin(faces[0], faces[1]);
    // PetscInt rightFace = PetscMax(faces[0], faces[1]);

    // Return -h for left face, +h for right face
    return (face == leftFace) ? -minCellRadius : minCellRadius;
}

void SlopeLimiter::GetNeighborMinMax1D(PetscInt cell, const PetscScalar* cellValues, const domain::Field& field,
                                       PetscInt component, PetscReal& minVal, PetscReal& maxVal, PetscInt totalComponents) const {
    // Get current cell value
    PetscReal cellVal = cellValues[cell*totalComponents + field.offset + component];
    minVal = maxVal = cellVal;

    // Check left neighbor (cell-1)
    if (cell > 0) {
        PetscReal leftVal = cellValues[(cell-1)*totalComponents + field.offset + component];
        minVal = PetscMin(minVal, leftVal);
        maxVal = PetscMax(maxVal, leftVal);
    }

    // Check right neighbor (cell+1)
    if (cell < (PetscInt)cellToFaces.size() - 1) {
        PetscReal rightVal = cellValues[(cell+1)*totalComponents + field.offset + component];
        minVal = PetscMin(minVal, rightVal);
        maxVal = PetscMax(maxVal, rightVal);
    }
}

void SlopeLimiter::ApplyLimiter(DM dm, PetscInt dim, const domain::Field& field, const domain::Range& cellRange,
                               const PetscScalar* cellValues, PetscScalar* gradients) {
    if (!isSetup) {
        throw std::runtime_error("SlopeLimiter must be set up before applying limiter");
    }

    // Get the total solution array size
    PetscInt cStart, cEnd;
    DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;
    
    // Get the total number of fields and their components
    PetscDS ds;
    DMGetDS(dm, &ds) >> utilities::PetscUtilities::checkError;
    PetscInt nf;
    PetscDSGetNumFields(ds, &nf) >> utilities::PetscUtilities::checkError;
    
    // Print info for each field
    PetscInt totalComponents = 0;
    for (PetscInt f = 0; f < nf; f++) {
        PetscInt fieldSize;
        PetscDSGetFieldSize(ds, f, &fieldSize) >> utilities::PetscUtilities::checkError;
        PetscPrintf(PETSC_COMM_WORLD, "Field %d: %d components\n", f, fieldSize);
        totalComponents += fieldSize;
    }
    PetscPrintf(PETSC_COMM_WORLD, "Total components: %d\n", totalComponents);

    PetscPrintf(PETSC_COMM_WORLD, "Starting Barth-Jespersen limiter application for field %s with %d components\n", field.name.c_str(), field.numberComponents);

    //for the size of cellvalues, print the values
    // for (PetscInt i = 0; i < 100; i++) {
    //     for (PetscInt j = 0; j < field.numberComponents; j++) {
    //         // PetscInt idx = i*4 + field.offset + j;
    //         PetscInt idx = i*totalComponents + field.offset + j;
    //         PetscPrintf(PETSC_COMM_WORLD, "cellvalues[%d]: %g\n", idx, cellValues[idx]);
    //     }
    // }

    // for (PetscInt i = 0; i < 300; i++) {
    //     for (PetscInt j = 0; j < field.numberComponents; j++) {
    //         PetscInt idx = i*field.numberComponents + j;
    //         PetscPrintf(PETSC_COMM_WORLD, "gradient[%d]: %g\n", idx, gradients[idx]);
    //     }
    // }




    // Get cell geometry for non-1D cases
    Vec cellGeomVec = nullptr;
    Vec faceGeomVec = nullptr;
    const PetscScalar* cellGeomArray = nullptr;
    const PetscScalar* faceGeomArray = nullptr;
    
    if (!is1D) {
        DM cellDM, faceDM;
        DMGetCoordinateDM(dm, &cellDM) >> utilities::PetscUtilities::checkError;
        DMGetCoordinateDM(dm, &faceDM) >> utilities::PetscUtilities::checkError;
        DMGetCoordinates(cellDM, &cellGeomVec) >> utilities::PetscUtilities::checkError;
        DMGetCoordinates(faceDM, &faceGeomVec) >> utilities::PetscUtilities::checkError;
        VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
        VecGetArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;
    }

    // Loop over each cell
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscInt cell = cellRange.points ? cellRange.points[c] : c;
        //PetscPrintf(PETSC_COMM_WORLD, "\ncell %d\n", cell);

        // Skip cells outside our range
        if (c < cellRange.start + 5 || c > cellRange.end - 5) {
            //PetscPrintf(PETSC_COMM_WORLD, "Skipping cell %d\n", cell);
            // Zero out gradients for cells outside range
            for (PetscInt comp = 0; comp < field.numberComponents; ++comp) {
                PetscInt gradidx = cellRange.start + field.numberComponents*cell + comp; //2D?? (idx)dim + d? no that's not right but figure out what it is 

                for (PetscInt d = 0; d < dim; ++d) {
                    gradients[gradidx] = 0.0;
                }
            }
            continue;
        }
        //PetscPrintf(PETSC_COMM_WORLD, "Processing cell %d\n", cell);

        // For each component in the field
        for (PetscInt comp = 0; comp < field.numberComponents; ++comp) {
            // Print field information and indices
            //PetscPrintf(PETSC_COMM_WORLD, "Field info: offset=%d, numberComponents=%d\n", field.offset, field.numberComponents);
            PetscInt cellValIndex = cell*totalComponents + field.offset + comp; //i*totalComponents + field.offset + j;
            //PetscPrintf(PETSC_COMM_WORLD, "Accessing cell value at index %d (offset + cell*components + comp)\n", cellValIndex);
            
            // Get current cell value
            const PetscReal cellVal = cellValues[cellValIndex];
            //PetscPrintf(PETSC_COMM_WORLD, "Cell %d, component %d: cellVal = %g (at index %d)\n", cell, comp, cellVal, cellValIndex);

            // Print a few values around this cell to verify indexing
            //PetscPrintf(PETSC_COMM_WORLD, "Values around cell %d:\n", cell);
            // for (PetscInt i = -2; i <= 2; i++) {
            //     PetscInt idx = cell + i;
            //     if (idx >= 0) {
            //         PetscInt arrayIdx = idx*totalComponents + field.offset + comp;
            //         PetscPrintf(PETSC_COMM_WORLD, "  Cell %d: value = %g (at index %d)\n", idx, cellValues[arrayIdx], arrayIdx);
            //     }
            // }

            // Get total solution size
            PetscDS ds;
            DMGetDS(dm, &ds) >> utilities::PetscUtilities::checkError;
            PetscInt totDim;
            PetscDSGetTotalDimension(ds, &totDim) >> utilities::PetscUtilities::checkError;

            // Find min/max values from neighbors
            PetscReal minVal, maxVal;
            if (is1D) {
                GetNeighborMinMax1D(cell, cellValues, field, comp, minVal, maxVal, totDim);
            } else {
                // For non-1D, find min/max from all face neighbors
                minVal = maxVal = cellVal;
                for (const PetscInt& face : cellToFaces[cell]) {
                    const PetscInt neighbor = GetNeighborCell(cell, face);
                    PetscInt neighborValIndex = neighbor*totalComponents + field.offset + comp;
                    const PetscReal neighborVal = cellValues[neighborValIndex];
                    //PetscPrintf(PETSC_COMM_WORLD, "  Neighbor cell %d (from face %d): value = %g (at index %d)\n", 
                            //    neighbor, face, neighborVal, neighborValIndex);
                    minVal = PetscMin(minVal, neighborVal);
                    maxVal = PetscMax(maxVal, neighborVal);
                }
            }

            PetscPrintf(PETSC_COMM_WORLD, "cellval %g minval %g maxval %g\n", cellVal, minVal, maxVal);

            // Compute limiting factor for each face
            PetscReal alpha = 1.0;  // Start with no limiting
            for (const PetscInt& face : cellToFaces[cell]) {
                // Get vector from cell center to face center
                PetscReal r[3] = {0.0, 0.0, 0.0};
                if (is1D) {
                    r[0] = GetCellToFaceVector1D(cell, face);
                } else {
                    const PetscFVCellGeom* cg;
                    const PetscFVFaceGeom* fg;
                    DMPlexPointLocalRead(dm, cell, cellGeomArray, &cg) >> utilities::PetscUtilities::checkError;
                    DMPlexPointLocalRead(dm, face, faceGeomArray, &fg) >> utilities::PetscUtilities::checkError;
                    for (PetscInt d = 0; d < dim; ++d) {
                        r[d] = fg->centroid[d] - cg->centroid[d];
                        
                    }
                }

                // Compute unlimited extrapolated increment
                PetscReal delta = 0.0;
                for (PetscInt d = 0; d < dim; ++d) {
                    PetscInt gradidx = cellRange.start + field.numberComponents*cell + comp;
                    delta += gradients[gradidx] * r[d]; //cellRange.start + totalComponents*cell + comp
                }

                // Compute face-specific limiter factor
                if (PetscAbs(delta) > 1e-10) {  // Avoid division by very small numbers
                    PetscReal alpha_f;
                    if (delta > 0) {
                        alpha_f = PetscMin(1.0, (maxVal - cellVal) / delta);
                    } else {
                        alpha_f = PetscMin(1.0, (minVal - cellVal) / delta);
                    }
                    alpha = PetscMin(alpha, alpha_f);
                }
            }

            // Apply limiting to gradient
            PetscInt gradidx = cellRange.start + field.numberComponents*cell + comp;
            for (PetscInt d = 0; d < dim; ++d) {
                
                gradients[gradidx] *= alpha; //cellRange.start + totalComponents*cell + comp
            }
            PetscPrintf(PETSC_COMM_WORLD, "alphacell %f comp %d gradients[%d] %g\n", alpha, comp, gradidx, gradients[gradidx]);
        }
    }

    // Cleanup
    if (!is1D) {
        VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
        VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;
    }
}

SlopeLimiter::~SlopeLimiter() {
    if (cellGeomVec) {
        VecDestroy(&cellGeomVec) >> utilities::PetscUtilities::checkError;
    }
    if (faceGeomVec) {
        VecDestroy(&faceGeomVec) >> utilities::PetscUtilities::checkError;
    }
}

}  // namespace ablate::finiteVolume 