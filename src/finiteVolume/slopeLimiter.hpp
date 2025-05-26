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
    //! store the dmGrad, these are specific to this finite volume solver
    std::vector<DM> gradientCellDms;

    //! store the cell geometry vector
    Vec cellGeomVec = nullptr;

    //! store the face geometry vector
    Vec faceGeomVec = nullptr;

    //! store the cell to face connectivity
    std::vector<std::vector<PetscInt>> cellToFaces;

    //! store the face to cell connectivity
    std::unordered_map<PetscInt, std::vector<PetscInt>> faceToCells;

    //! store the cell centers
    std::vector<PetscReal> cellCenters;

    //! store the face centers
    std::vector<PetscReal> faceCenters;

    //! flag to indicate if we are in 1D mode
    bool is1D = false;

    //! flag to indicate if we are set up
    bool isSetup = false;

    // Store minimum cell radius for 1D calculations
    PetscReal minCellRadius = 0.0;

   public:
    SlopeLimiter() = default;
    ~SlopeLimiter();

    // Check if the limiter has been set up
    bool IsSetup() const { return isSetup; }

    // Setup function to initialize mesh connectivity
    void Setup(DM dm, const domain::Range& cellRange);

    // Helper function to get the neighbor cell across a face
    PetscInt GetNeighborCell(PetscInt cell, PetscInt face) const;

    /**
     * Apply the Barth-Jespersen slope limiter to the gradients
     * @param dm The DM object
     * @param dim The dimension of the problem
     * @param field The field being limited
     * @param cellRange The range of cells to process
     * @param cellValues The cell-centered values
     * @param gradients The gradients to limit (modified in place)
     */
    void ApplyLimiter(DM dm, PetscInt dim, const domain::Field& field, const domain::Range& cellRange,
                     const PetscScalar* cellValues, PetscScalar* gradients);

    /**
     * For 1D cases, compute the vector from cell center to face center
     * @param cell The cell index
     * @param face The face index
     * @return The vector from cell center to face center
     */
    PetscReal GetCellToFaceVector1D(PetscInt cell, PetscInt face) const;

    /**
     * Get min/max values from neighbors in 1D
     */
    void GetNeighborMinMax1D(PetscInt cell, const PetscScalar* cellValues, const domain::Field& field,
                           PetscInt component, PetscReal& minVal, PetscReal& maxVal, PetscInt totalComponents) const;
};

}  // namespace ablate::finiteVolume 