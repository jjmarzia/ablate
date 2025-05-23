#pragma once

#include <petsc.h>
#include <domain/domain.hpp>
#include <domain/range.hpp>
#include <vector>
#include <memory>
#include <unordered_map>

namespace ablate::finiteVolume {

class SlopeLimiter {
   private:
    // Helper function to compute Superbee limiter
    static PetscReal SuperbeeLimiter(PetscReal r);

    // Storage for mesh connectivity
    std::vector<std::vector<PetscInt>> cellToFaces;  // For each cell, list of its faces
    std::unordered_map<PetscInt, std::vector<PetscInt>> faceToCells;  // For each face, list of its cells (should always be 2)
    std::vector<PetscInt> cellBoundaryDistance;  // Store distance to boundary for each cell
    bool isSetup = false;
    PetscInt maxBoundaryDistance = 5;  // Maximum distance from boundary to apply limiting

    void ComputeBoundaryDistances(DM dm, const domain::Range& cellRange);

   public:
    SlopeLimiter() = default;
    ~SlopeLimiter() = default;

    // Check if the limiter has been set up
    bool IsSetup() const { return isSetup; }

    // Setup function to initialize mesh connectivity
    void Setup(DM dm, const domain::Range& cellRange);

    // Helper function to get the neighbor cell across a face
    PetscInt GetNeighborCell(PetscInt cell, PetscInt face) const;

    /**
     * Compute the ratio of consecutive differences for slope limiting
     * @param dm The mesh DM
     * @param dim The dimension
     * @param cell The current cell
     * @param face The face to compute ratio for
     * @param field The field being limited
     * @param cellValues The cell values array
     * @param gradients The gradients array
     * @param component The component to compute ratio for
     * @return The ratio for limiting
     */
    PetscReal ComputeRatio(DM dm, PetscInt dim, PetscInt cell, PetscInt face, const domain::Field& field,
                          const PetscScalar* cellValues, const PetscScalar* gradients, PetscInt component) const;

    /**
     * Apply the slope limiter to the gradients
     * @param dm The DM object
     * @param dim The dimension of the problem
     * @param field The field being limited
     * @param cellRange The range of cells to process
     * @param cellValues The cell-centered values
     * @param gradients The gradients to limit (modified in place)
     */
    void ApplyLimiter(DM dm, PetscInt dim, const domain::Field& field, const domain::Range& cellRange,
                     const PetscScalar* cellValues, PetscScalar* gradients);

    // Set the maximum distance from boundary to apply limiting
    void SetMaxBoundaryDistance(PetscInt distance) { maxBoundaryDistance = distance; }
};

}  // namespace ablate::finiteVolume 