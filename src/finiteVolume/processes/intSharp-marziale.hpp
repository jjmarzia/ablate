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
    bool addtoRHS;
    //mesh for vertex information
    DM vertexDM{};
    std::shared_ptr<ablate::domain::SubDomain> subDomain;

   public:
    /**
     *
     * @param Gamma
     * @param epsilon
     */
    explicit IntSharp(PetscReal Gamma, PetscReal epsilon, bool flipPhiTilde, bool addtoRHS = false);

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
    std::vector<std::vector<PetscScalar>> fluxGradValues;
    std::map<PetscInt, std::vector<PetscInt>> cellNeighbors;
    std::map<PetscInt, std::vector<PetscInt>> vertexNeighbors;
};
}  // namespace ablate::finiteVolume::processes
#endif
