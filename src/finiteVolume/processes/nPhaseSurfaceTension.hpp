#pragma once

#include <petsc.h>
#include <memory>
#include <vector>
#include "domain/range.hpp"
#include "domain/RBF/rbf.hpp"
#include "domain/reverseRange.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/solver.hpp"
// #include "twoPhaseEulerAdvection.hpp"
#include "nPhaseAllaireAdvection.hpp"

namespace ablate::finiteVolume::processes {

class NPhaseSurfaceTension : public Process {
   private:
    DM vertexDM{};
    std::shared_ptr<ablate::domain::SubDomain> subDomain;

    std::vector<PetscReal> sigmaij;

    static PetscErrorCode ComputeSource(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx);

    // const std::shared_ptr<eos::EOS> eosNPhase;
    // std::vector<std::shared_ptr<eos::EOS>> eosk;

    // Connectivity containers
    std::map<PetscInt, std::vector<PetscInt>> cellNeighbors;   // multi-layer neighbors
    std::map<PetscInt, std::vector<PetscInt>> cellNeighbors1;  // 1-layer neighbors
    std::map<PetscInt, std::vector<PetscInt>> vertexNeighbors; // vertex -> cells

   public:
    explicit NPhaseSurfaceTension(const std::vector<PetscReal>& surfaceTensionCoeffs);

    void Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) override;
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver& flow) override;
};

}  // namespace ablate::finiteVolume::processes


