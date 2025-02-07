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
#include "finiteVolume/stencils/gaussianConvolution.hpp"


namespace ablate::finiteVolume::processes {

class IntSharp : public Process {

   private:
    //coeffs
    PetscReal Gamma;
    PetscReal epsilon;
    DM cellDM = nullptr;
    DM fluxDM = nullptr;
    DM vertDM = nullptr;
    std::shared_ptr<ablate::finiteVolume::stencil::GaussianConvolution> gaussianConv = nullptr;
    enum VecLoc { LOCAL , GLOBAL };
    const PetscReal phiRange[2] = {1.e-4, 1.0 - 1.e-4};

    void ClearData();



   public:
    /**
     *
     * @param Gamma
     * @param epsilon
     */
    explicit IntSharp(PetscReal Gamma, PetscReal epsilon);

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
};
}  // namespace ablate::finiteVolume::processes
#endif
