#ifndef ABLATELIBRARY_FINITEVOLUME_ISSHARP_SURFACEFORCE_HPP
#define ABLATELIBRARY_FINITEVOLUME_ISSHARP_SURFACEFORCE_HPP

#include <petsc.h>
#include <memory>
#include <vector>
#include <map>
#include "domain/range.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/solver.hpp"
#include "twoPhaseEulerAdvection.hpp"

namespace ablate::finiteVolume::processes {

class IntSharpSurfaceForce : public Process {

   private:
    // IntSharp parameters
    PetscReal Gamma;
    PetscReal epsilon;
    PetscReal boundaryLayerMultiplier;
    
    // SurfaceForce parameters
    PetscReal sigma;
    
    // Control flags
    bool enableIntSharp;
    bool enableSurfaceForce;
    
    //mesh for vertex information
    DM vertexDM{};
    std::shared_ptr<ablate::domain::SubDomain> subDomain;
    
    // IntSharp data structures
    std::map<PetscInt, std::vector<PetscInt>> cellNeighbors;
    std::map<PetscInt, std::vector<PetscReal>> cellWeights;
    std::map<PetscInt, std::vector<PetscInt>> vertexNeighbors;
    
    // Boundary information for IntSharp
    PetscReal boundingBox[6];  // [xmin, xmax, ymin, ymax, zmin, zmax]
    PetscReal minRadius;
    PetscReal boundaryLayerThickness;
    std::map<PetscInt, PetscReal> cellBoundaryDistances;
    std::map<PetscInt, PetscReal> cellBoundaryWeights;
    
    // SurfaceForce data structures
    std::map<PetscInt, std::vector<PetscInt>> cellNeighbors1;  // 1-layer neighbors
    std::map<PetscInt, std::vector<PetscInt>> cellNeighbors3;  // 3-layer neighbors

   public:
    /**
     * Combined IntSharp and SurfaceForce process
     * @param Gamma IntSharp velocity scale parameter (approx. umax)
     * @param epsilon IntSharp interface thickness scale parameter (approx. h)
     * @param flipPhiTilde IntSharp: if true: phiTilde-->1-phiTilde (not used in current implementation)
     * @param boundaryLayerMultiplier IntSharp: multiplier for boundary layer thickness (default: 3.0)
     * @param sigma SurfaceForce: surface tension coefficient
     * @param C SurfaceForce: stdev length with respect to grid spacing magnitude (not used in current implementation)
     * @param N SurfaceForce: number of stdevs that the convolution integral captures (not used in current implementation)
     * @param enableIntSharp Whether to enable IntSharp functionality (default: true)
     * @param enableSurfaceForce Whether to enable SurfaceForce functionality (default: true)
     */
    explicit IntSharpSurfaceForce(PetscReal Gamma, PetscReal epsilon, bool flipPhiTilde, 
                                 PetscReal boundaryLayerMultiplier, PetscReal sigma, 
                                 PetscReal C, PetscReal N, bool enableIntSharp = true, 
                                 bool enableSurfaceForce = true);

    ~IntSharpSurfaceForce();

    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver);
    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow);
    PetscErrorCode PreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime);

   private:
    void ComputeBoundaryInformation(DM dm);
    PetscReal GetBoundaryWeight(PetscInt cell) const;
};

}  // namespace ablate::finiteVolume::processes

#endif  // ABLATELIBRARY_FINITEVOLUME_ISSHARP_SURFACEFORCE_HPP 