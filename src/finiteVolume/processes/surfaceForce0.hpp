#ifndef ABLATELIBRARY_FINITEVOLUME_SURFACEFORCE_HPP
#define ABLATELIBRARY_FINITEVOLUME_SURFACEFORCE_HPP

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
#include "twoPhaseEulerAdvection.hpp"

namespace ablate::finiteVolume::processes {

class SurfaceForce : public Process {


   private:
    //surface tension coefficient
    PetscReal sigma;
    PetscReal C;
    PetscReal N;
    bool flipPhiTilde;
    //mesh for vertex information
    DM vertexDM{};
    std::shared_ptr<ablate::domain::SubDomain> subDomain;

   public:

    /**
     *
     * @param sigma
     */
    explicit SurfaceForce(PetscReal sigma, PetscReal C, PetscReal N, bool flipPhiTilde);

    /**
     * Clean up the dm created
     */
    ~SurfaceForce() override;

    /**
     * Setup the process to define the vertex dm
     * @param flow
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;

    /**
     * static function private function to compute surface force and add source to eulerset
     * @param solver
     * @param dm
     * @param time
     * @param locX
     * @param fVec
     * @param ctx
     * @return
     */
    static PetscErrorCode ComputeSource(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx);
};
}  // namespace ablate::finiteVolume::processes
#endif
