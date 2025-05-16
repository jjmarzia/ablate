#ifndef ABLATELIBRARY_FINITEVOLUME_NPHASENONCONSERVATIVE_RHS_HPP
#define ABLATELIBRARY_FINITEVOLUME_NPHASENONCONSERVATIVE_RHS_HPP

#include <petsc.h>
#include <memory>
#include <vector>
#include "domain/range.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/solver.hpp"
#include "nPhaseAllaireAdvection.hpp"

namespace ablate::finiteVolume::processes {

class NPhaseNonconservativeRHS : public Process {

   private:
    //mesh for vertex information
    DM vertexDM{};
    std::shared_ptr<ablate::domain::SubDomain> subDomain;

   public:
    /**
     */
    explicit NPhaseNonconservativeRHS();

    /**
     * Clean up the dm created
     */
    ~NPhaseNonconservativeRHS() override;

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
    inline const static std::string ALPHAK_FIELD = "alphak"; //eos::NPhase::ALPHAK; //VOLUME_FRACTION_FIELD = eos::TwoPhase::VF;
    // inline const static std::string DENSITY_VF_FIELD = ablate::finiteVolume::CompressibleFlowFields::CONSERVED + VOLUME_FRACTION_FIELD;
    PetscErrorCode PreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime);
    // std::map<PetscInt, std::vector<PetscInt>> cellNeighbors;
    // std::map<PetscReal, std::vector<PetscReal>> cellWeights;
    // std::map<PetscInt, std::vector<PetscInt>> vertexNeighbors;
};
}  // namespace ablate::finiteVolume::processes
#endif
