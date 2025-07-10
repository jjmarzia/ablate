#ifndef ABLATELIBRARY_FINITEVOLUME_INTSHARP_HPP
#define ABLATELIBRARY_FINITEVOLUME_INTSHARP_HPP

#include <petsc.h>
#include <memory>
#include <vector>
#include "domain/range.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/solver.hpp"
#include "twoPhaseEulerAdvection.hpp"

namespace ablate::finiteVolume::processes {

class IntSharp : public Process {

   private:
    //coeffs
    PetscReal Gamma;
    PetscReal epsilon;
    bool flipPhiTilde;
    //mesh for vertex information
    DM vertexDM{};
    std::shared_ptr<ablate::domain::SubDomain> subDomain;
    
    // Boundary information
    PetscReal boundingBox[6];  // [xmin, xmax, ymin, ymax, zmin, zmax]
    PetscReal minRadius;
    PetscReal boundaryLayerThickness;
    PetscReal boundaryLayerMultiplier;
    std::map<PetscInt, PetscReal> cellBoundaryDistances;
    std::map<PetscInt, PetscReal> cellBoundaryWeights;

   public:
    /**
     *
     * @param Gamma
     * @param epsilon
     * @param flipPhiTilde
     * @param boundaryLayerMultiplier Optional multiplier for boundary layer thickness (default: 3.0)
     */
    explicit IntSharp(PetscReal Gamma, PetscReal epsilon, bool flipPhiTilde, PetscReal boundaryLayerMultiplier = 3.0);

    /**
     * Clean up the dm created
     */
    ~IntSharp() override;

    /**
     * Setup the process to define the vertex dm
     * @param flow
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;

    /**
     * Compute boundary distances and weights for all cells
     * @param dm The domain mesh
     */
    void ComputeBoundaryInformation(DM dm);

    /**
     * Get boundary weight for a cell (1.0 for interior, 0.0 for boundary, smooth transition in between)
     * @param cell The cell index
     * @return Boundary weight between 0.0 and 1.0
     */
    PetscReal GetBoundaryWeight(PetscInt cell) const;

    /**
     * static function private function to compute interface regularization term and add source to eulerset
     * @param solver
     * @param dm
     * @param time
     * @param locX
     * @param fVec
     * @param ctx
     * @return
     */
    static PetscErrorCode ComputeTerm(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx);

    //intsharp prestage stuff
    inline const static std::string VOLUME_FRACTION_FIELD = eos::TwoPhase::VF;
    inline const static std::string DENSITY_VF_FIELD = ablate::finiteVolume::CompressibleFlowFields::CONSERVED + VOLUME_FRACTION_FIELD;
    PetscErrorCode PreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime);
    std::map<PetscInt, std::vector<PetscInt>> cellNeighbors;
    std::map<PetscReal, std::vector<PetscReal>> cellWeights;
    std::map<PetscInt, std::vector<PetscInt>> vertexNeighbors;
};
}  // namespace ablate::finiteVolume::processes
#endif
