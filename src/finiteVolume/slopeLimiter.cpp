#include "slopeLimiter.hpp"
#include "utilities/petscUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include <petscdmplex.h>
#include <cmath>

namespace ablate::finiteVolume {

PetscReal SlopeLimiter::SuperbeeLimiter(PetscReal r) {
    // Superbee limiter: phi(r) = max(0, min(1, 2r), min(2, r))
    if (r <= 0) {
        return 0.0;
    } else if (r <= 0.5) {
        return 2.0 * r;
    } else if (r <= 1.0) {
        return 1.0;
    } else if (r <= 2.0) {
        return r;
    } else {
        return 2.0;
    }
}

void SlopeLimiter::Setup(DM dm, const domain::Range& cellRange) {
    // PetscPrintf(PETSC_COMM_WORLD, "=== Entering SlopeLimiter::Setup ===\n");

    //PetscPrintf(PETSC_COMM_WORLD, "=== Entering SlopeLimiter::Setup ===\n");
    //PetscPrintf(PETSC_COMM_WORLD, "Cell range: %d to %d\n", cellRange.start, cellRange.end);

    if (isSetup) {
        //PetscPrintf(PETSC_COMM_WORLD, "Already setup, returning\n");
        return;  // Already setup
    }

    // Get face range
    PetscInt fStart, fEnd;
    DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd) >> utilities::PetscUtilities::checkError;
    //PetscPrintf(PETSC_COMM_WORLD, "Face range: %d to %d\n", fStart, fEnd);

    // Get dimensions
    PetscInt dim;
    DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;
    //PetscPrintf(PETSC_COMM_WORLD, "Dimension: %d\n", dim);

    // Resize the cellToFaces vector to hold all cells
    cellToFaces.resize(cellRange.end);
    faceToCells.clear();
    //PetscPrintf(PETSC_COMM_WORLD, "Resized cellToFaces to %d and cleared faceToCells\n", cellRange.end);

    // For each cell, get and store its faces and build face-to-cell connectivity
    for (PetscInt c = cellRange.start; c < cellRange.end; c++) {
        //PetscPrintf(PETSC_COMM_WORLD, "Processing cell %d\n", c);
        
        // Get faces connected to this cell using DMPlexGetCone
        PetscInt numFaces;
        const PetscInt* faces;
        DMPlexGetConeSize(dm, c, &numFaces) >> utilities::PetscUtilities::checkError;
        DMPlexGetCone(dm, c, &faces) >> utilities::PetscUtilities::checkError;
        //PetscPrintf(PETSC_COMM_WORLD, "  Cell %d has %d faces\n", c, numFaces);

        // Store the faces for this cell
        cellToFaces[c].assign(faces, faces + numFaces);
        //PetscPrintf(PETSC_COMM_WORLD, "  Stored faces for cell %d\n", c);

        // Build face-to-cell connectivity
        for (PetscInt f = 0; f < numFaces; f++) {
            PetscInt face = faces[f];
            PetscInt nCells;
            const PetscInt* cells;
            DMPlexGetSupportSize(dm, face, &nCells) >> utilities::PetscUtilities::checkError;
            DMPlexGetSupport(dm, face, &cells) >> utilities::PetscUtilities::checkError;
            
            // Store cells for this face
            faceToCells[face] = std::vector<PetscInt>(cells, cells + nCells);
            //PetscPrintf(PETSC_COMM_WORLD, "  Added cells [");
            for (PetscInt i = 0; i < nCells; i++) {
                //PetscPrintf(PETSC_COMM_WORLD, "%d%s", cells[i], (i < nCells-1) ? ", " : "");
            }
            //PetscPrintf(PETSC_COMM_WORLD, "] to face %d\n", face);
        }
    }

    // Compute boundary distances after setting up connectivity
    ComputeBoundaryDistances(dm, cellRange);
    
    isSetup = true;
    //PetscPrintf(PETSC_COMM_WORLD, "=== Exiting SlopeLimiter::Setup ===\n");
}

PetscInt SlopeLimiter::GetNeighborCell(PetscInt cell, PetscInt face) const {
    //PetscPrintf(PETSC_COMM_WORLD, "GetNeighborCell: cell=%d, face=%d\n", cell, face);
    
    // Check if face exists in our map
    auto it = faceToCells.find(face);
    if (it == faceToCells.end()) {
        //PetscPrintf(PETSC_COMM_WORLD, "  Face %d not found in faceToCells map\n", face);
        return -1;
    }

    const auto& cells = it->second;
    //PetscPrintf(PETSC_COMM_WORLD, "  Face %d has %zu cells\n", face, cells.size());

    // Find the cell that is not the current cell
    for (PetscInt neighborCell : cells) {
        if (neighborCell != cell) {
            //PetscPrintf(PETSC_COMM_WORLD, "  Found neighbor cell %d\n", neighborCell);
            return neighborCell;
        }
    }

    //PetscPrintf(PETSC_COMM_WORLD, "  No neighbor cell found\n");
    return -1;
}

PetscReal SlopeLimiter::ComputeRatio(DM dm, PetscInt dim, PetscInt cell, PetscInt face, const domain::Field& field,
                                    const PetscScalar* cellValues, const PetscScalar* gradients, PetscInt component) const {
    //PetscPrintf(PETSC_COMM_WORLD, "ComputeRatio: cell=%d, face=%d, component=%d\n", cell, face, component);

    // Get the neighbor cell
    PetscInt neighborCell = GetNeighborCell(cell, face);
    if (neighborCell == -1) {
        //PetscPrintf(PETSC_COMM_WORLD, "  No neighbor cell found, returning 0.0\n");
        return 0.0;
    }

    // Get cell values
    PetscReal cellValue = cellValues[field.offset + component];
    const PetscScalar* neighborValues = nullptr;
    DMPlexPointGlobalRead(dm, neighborCell, cellValues, &neighborValues) >> utilities::PetscUtilities::checkError;
    PetscReal neighborValue = neighborValues[field.offset + component];

    // Get cell centers
    PetscReal cellCenter[3], neighborCenter[3];
    DMPlexComputeCellGeometryFVM(dm, cell, nullptr, cellCenter, nullptr) >> utilities::PetscUtilities::checkError;
    DMPlexComputeCellGeometryFVM(dm, neighborCell, nullptr, neighborCenter, nullptr) >> utilities::PetscUtilities::checkError;

    // Get face normal and sign
    PetscReal faceNormal[3];
    PetscReal faceSign = 1.0;
    if (dim == 1) {
        // In 1D, use the face-to-cell connectivity to determine sign
        const auto& cells = faceToCells.at(face);
        faceSign = (cells[0] == cell) ? 1.0 : -1.0;  // +1 for right face, -1 for left face
        faceNormal[0] = faceSign;
    } else {
        // In 2D/3D, compute actual face normal
        DMPlexFaceCentroidOutwardAreaNormal(dm, cell, face, nullptr, faceNormal) >> utilities::PetscUtilities::checkError;
    }

    // Compute differences
    PetscReal dx[3];
    for (PetscInt d = 0; d < dim; d++) {
        dx[d] = neighborCenter[d] - cellCenter[d];
    }

    // Compute gradient projection
    PetscReal gradProj = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        gradProj += gradients[d] * dx[d];
    }

    // Compute ratio
    PetscReal ratio = 0.0;
    if (PetscAbsReal(gradProj) > PETSC_SMALL) {
        ratio = (neighborValue - cellValue) / gradProj;
    }

    //PetscPrintf(PETSC_COMM_WORLD, "  Computed ratio: %g\n", ratio);
    return ratio;
}

void SlopeLimiter::ComputeBoundaryDistances(DM dm, const domain::Range& cellRange) {
    // Initialize distances to a large number
    cellBoundaryDistance.resize(cellRange.end - cellRange.start, PETSC_MAX_INT);

    // First pass: identify boundary cells (distance = 0)
    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
        PetscInt cell = cellRange.points ? cellRange.points[i] : i;
        PetscInt cStart, cEnd;
        DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;

        // Get faces of this cell
        const auto& faces = cellToFaces[cell - cStart];
        bool isBoundaryCell = false;

        // Check if any face is a boundary face (has only one cell)
        for (PetscInt face : faces) {
            const auto& cells = faceToCells.at(face);
            if (cells.size() == 1) {
                isBoundaryCell = true;
                break;
            }
        }

        if (isBoundaryCell) {
            cellBoundaryDistance[cell - cStart] = 0;
        }
    }

    // Iteratively compute distances
    bool changed;
    do {
        changed = false;
        for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
            PetscInt cell = cellRange.points ? cellRange.points[i] : i;
            PetscInt cStart, cEnd;
            DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;
            PetscInt cellIdx = cell - cStart;

            // Skip if we already know this cell's distance
            if (cellBoundaryDistance[cellIdx] < PETSC_MAX_INT) continue;

            // Get faces of this cell
            const auto& faces = cellToFaces[cellIdx];
            PetscInt minNeighborDist = PETSC_MAX_INT;

            // Check all neighbors through faces
            for (PetscInt face : faces) {
                const auto& cells = faceToCells.at(face);
                for (PetscInt neighborCell : cells) {
                    if (neighborCell != cell) {
                        PetscInt neighborIdx = neighborCell - cStart;
                        if (cellBoundaryDistance[neighborIdx] < minNeighborDist) {
                            minNeighborDist = cellBoundaryDistance[neighborIdx];
                        }
                    }
                }
            }

            // Update distance if we found a closer path to boundary
            if (minNeighborDist < PETSC_MAX_INT && minNeighborDist + 1 < cellBoundaryDistance[cellIdx]) {
                cellBoundaryDistance[cellIdx] = minNeighborDist + 1;
                changed = true;
            }
        }
    } while (changed);
}

void SlopeLimiter::ApplyLimiter(DM dm, PetscInt dim, const domain::Field& field, const domain::Range& cellRange,
                               const PetscScalar* cellValues, PetscScalar* gradients) {
    // Check if we need to apply limiting based on boundary distance
    if (!isSetup) {
        throw std::runtime_error("SlopeLimiter must be set up before use");
    }

    PetscPrintf(PETSC_COMM_WORLD, "=== Entering SlopeLimiter::ApplyLimiter ===\n");

    // Special handling for volume fraction fields (alphak)
    bool isVolumeFraction = (field.name == "alphak");

    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
        const PetscInt cell = cellRange.GetPoint(i);
        
        // Skip cells that are too far from boundary
        if (cellBoundaryDistance[cell] > maxBoundaryDistance) {
            // Zero out gradients for cells far from boundary
            for (PetscInt c = 0; c < field.numberComponents; ++c) {
                for (PetscInt d = 0; d < dim; ++d) {
                    gradients[(cell * field.numberComponents + c) * dim + d] = 0.0;
                }
            }
            continue;
        }

        // Get the faces for this cell
        const auto& faces = cellToFaces[cell];
        
        for (PetscInt c = 0; c < field.numberComponents; ++c) {
            // For volume fractions, we need to ensure the limited value stays between 0 and 1
            if (isVolumeFraction) {
                PetscReal minGrad = 0.0;
                PetscReal maxGrad = 0.0;
                
                // First pass: compute min/max gradients that would keep values in bounds
                for (const auto& face : faces) {
                    // PetscReal ratio = ComputeRatio(dm, dim, cell, face, field, cellValues, gradients, c);
                    // PetscReal limiter = SuperbeeLimiter(ratio);
                    
                    // Get face normal and distance
                    PetscReal faceNormal[3];
                    PetscReal faceCentroid[3];
                    DMPlexFaceCentroidOutwardAreaNormal(dm, cell, face, faceCentroid, faceNormal) >> utilities::PetscUtilities::checkError;
                    
                    // Compute gradient component in face normal direction
                    PetscReal gradComponent = 0.0;
                    for (PetscInt d = 0; d < dim; ++d) {
                        gradComponent += gradients[(cell * field.numberComponents + c) * dim + d] * faceNormal[d];
                    }
                    
                    // Update min/max gradients to keep values in bounds
                    PetscReal cellValue = cellValues[cell * field.numberComponents + c];
                    if (gradComponent > 0) {
                        maxGrad = PetscMin(maxGrad, (1.0 - cellValue) / gradComponent);
                    } else if (gradComponent < 0) {
                        minGrad = PetscMax(minGrad, -cellValue / gradComponent);
                    }
                }
                
                // Second pass: apply limiting while respecting bounds
                for (PetscInt d = 0; d < dim; ++d) {
                    PetscReal grad = gradients[(cell * field.numberComponents + c) * dim + d];
                    if (grad > 0) {
                        gradients[(cell * field.numberComponents + c) * dim + d] *= PetscMin(1.0, maxGrad);
                    } else if (grad < 0) {
                        gradients[(cell * field.numberComponents + c) * dim + d] *= PetscMax(1.0, minGrad);
                    }
                }
            } else {
                // Standard limiting for non-volume fraction fields
                for (const auto& face : faces) {
                    PetscReal ratio = ComputeRatio(dm, dim, cell, face, field, cellValues, gradients, c);
                    PetscReal limiter = SuperbeeLimiter(ratio);
                    
                    // Apply limiter to each gradient component
                    for (PetscInt d = 0; d < dim; ++d) {
                        gradients[(cell * field.numberComponents + c) * dim + d] *= limiter;
                    }
                }
            }
        }
    }
}

}  // namespace ablate::finiteVolume 