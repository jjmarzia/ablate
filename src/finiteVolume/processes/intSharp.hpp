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
#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"
#include "finiteVolume/stencils/gaussianConvolution.hpp"

namespace ablate::finiteVolume::processes {

class IntSharp : public Process {

   private:
    //coeffs
    PetscReal Gamma;
    PetscReal epsilon;
    //need a more permanent fix since the user wouldn't know how to use this (or we can just do default true) but
    //optionally add the intsharp term to the RHS of the densityVFField equation 
    bool addToRHS;


    DM cellDM = nullptr;
    DM fluxDM = nullptr;
    DM vertDM = nullptr;
    std::shared_ptr<ablate::finiteVolume::stencil::GaussianConvolution> cellGaussianConv = nullptr;
    std::shared_ptr<ablate::finiteVolume::stencil::GaussianConvolution> vertexGaussianConv = nullptr;
    enum VecLoc { LOCAL , GLOBAL };
    const PetscReal phiRange[2] = {1.e-4, 1.0 - 1.e-4};

    void ClearData();

    void SetMasks();

    struct vecData {
      DM dm;
      Vec vec;
      PetscScalar *array;
    };

    std::vector<struct vecData> localVecList = {};
    std::vector<struct vecData> globalVecList = {};

    void MemoryHelper(DM dm, VecLoc loc, Vec *vec, PetscScalar **array);
    void MemoryHelper();

    void SetMasks(ablate::domain::Range &cellRange, DM phiDM, Vec phiVec, PetscInt phiID, Vec cellMaskVec[2], PetscScalar *cellMaskArray[2], PetscScalar *vertMaskArray);



   public:
    /**
     *
     * @param Gamma
     * @param epsilon
     */
    explicit IntSharp(PetscReal Gamma, PetscReal epsilon, bool addToRHS = true);

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
    static PetscErrorCode PreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime);

    // std::shared_ptr<ablate::finiteVolume::processes::TwoPhaseEulerAdvection::TwoPhaseDecoder> decoder; //"error: no member named 'TwoPhaseEulerAdvection'...??"
    
    //public fluxgradvalues to be accessible to prestage
    std::vector<std::vector<PetscScalar>> fluxGradValues;

};
}  // namespace ablate::finiteVolume::processes
#endif
