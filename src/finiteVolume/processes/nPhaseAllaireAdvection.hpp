#ifndef ABLATELIBRARY_NPHASEALLAIREADVECTION_HPP
#define ABLATELIBRARY_NPHASEALLAIREADVECTION_HPP

#include <petsc.h>
#include "eos/stiffenedGas.hpp"
#include "eos/nPhase.hpp"
#include "finiteVolume/nPhaseFlowFields.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "process.hpp"

// #include "finiteVolume/process.hpp"
#include <memory>
#include <vector>
#include "domain/range.hpp"
#include "eos/eos.hpp"
#include "parameters/parameters.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "domain/field.hpp"
#include "domain/region.hpp"
#include "domain/subDomain.hpp"
#include "utilities/petscUtilities.hpp"
#include "finiteVolume/processes/intSharp.hpp"
#include "finiteVolume/stencils/gaussianConvolution.hpp"

namespace ablate::finiteVolume::processes {

class NPhaseAllaireAdvection : public Process {
   public:

    inline const static std::string ALPHAK = eos::NPhase::ALPHAK;
    inline const static std::string ALPHAKRHOK = eos::NPhase::ALPHAKRHOK;



    /**
     * General two phase decoder interface
     */
    class NPhaseDecoder {
        public:
         virtual void DecodeNPhaseAllaireState(PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues, const PetscReal *normal, PetscReal *density, std::vector<PetscReal> *densityk,
            PetscReal *normalVelocity, PetscReal *velocity, PetscReal *internalEnergy, std::vector<PetscReal> *internalEnergyk, std::vector<PetscReal> *ak,
            std::vector<PetscReal> *Mk, PetscReal *p, std::vector<PetscReal> *Tk, std::vector<PetscReal> *alphak) = 0;
         virtual ~NPhaseDecoder() = default;
     };
     

    struct TimeStepData {
        PetscReal cfl;
        eos::ThermodynamicFunction computeSpeedOfSound;
    };
    TimeStepData timeStepData;


   private:

    // struct DecodeDataStructStiff {
        // PetscReal etot;
        // PetscReal rhotot;
        // // PetscReal Yg;
        // // PetscReal Yl;
        // // PetscReal gam1;
        // // PetscReal gam2;
        // // PetscReal cpg;
        // // PetscReal cpl;
        // // PetscReal p0g;
        // // PetscReal p0l;
        // //instead create a vector of gammak, Yk, Cpk, pik
        // std::vector<PetscReal> gammak;
        // std::vector<PetscReal> Cpk;
        // std::vector<PetscReal> pik;
    // };
    // static PetscErrorCode FormFunctionGas(SNES snes, Vec x, Vec F, void *ctx);
    // static PetscErrorCode FormJacobianGas(SNES snes, Vec x, Mat J, Mat P, void *ctx);
    // static PetscErrorCode FormFunctionStiff(SNES snes, Vec x, Vec F, void *ctx);
    // static PetscErrorCode FormJacobianStiff(SNES snes, Vec x, Mat J, Mat P, void *ctx);

    PetscErrorCode MultiphaseFlowPreStage(TS flowTs, ablate::solver::Solver &flow, PetscReal stagetime);



    /**
     * Implementation for two stiffened gases
     */
    class NStiffDecoder : public NPhaseDecoder {

        const std::vector<std::shared_ptr<eos::StiffenedGas>> eosk;


        /**
         * Store a scratch euler field for use with the eos
         */
        std::vector<std::vector<PetscReal>> kAllaireFieldScratch;

        /**
         * Get the compute functions using a fake field with only euler
         */
        std::vector<eos::ThermodynamicFunction> kComputeTemperature;
        std::vector<eos::ThermodynamicFunction> kComputeInternalEnergy;
        std::vector<eos::ThermodynamicFunction> kComputeSpeedOfSound;
        std::vector<eos::ThermodynamicFunction> kComputePressure;

       public:
        NStiffDecoder(PetscInt dim, const std::vector<std::shared_ptr<eos::StiffenedGas>> &eosk);

        void DecodeNPhaseAllaireState(PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues, const PetscReal *normal, PetscReal *density, std::vector<PetscReal> *densityk,
            PetscReal *normalVelocity, PetscReal *velocity, PetscReal *internalEnergy, std::vector<PetscReal> *internalEnergyk, std::vector<PetscReal> *ak,
            std::vector<PetscReal> *Mk, PetscReal *p, std::vector<PetscReal> *Tk, std::vector<PetscReal> *alphak) override;
    };

    const std::shared_ptr<eos::EOS> eosNPhase;
    std::vector<std::shared_ptr<eos::EOS>> eosk;
    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorNStiff;

    /**
     * Create and store the decoder
     */
    std::shared_ptr<NPhaseDecoder> decoder;

    std::vector<std::string> auxUpdateFields = {};

   public:

    static PetscErrorCode UpdateAuxFieldsNPhase(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[], const PetscScalar *conservedValues, const PetscInt aOff[],
                                                     PetscScalar *auxField, void *ctx);



    NPhaseAllaireAdvection(std::shared_ptr<eos::EOS> eosNPhase, const std::shared_ptr<parameters::Parameters> &parameters, 
                           std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorNStiff);
    ~NPhaseAllaireAdvection();
    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;

   private:
    // static function to compute time step for twoPhase euler advection
    static double ComputeCflTimeStep(TS ts, ablate::finiteVolume::FiniteVolumeSolver &flow, void *ctx);

    static PetscErrorCode NPhaseFlowComputeAllaireFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[],
                                                           const PetscInt aOff[], const PetscScalar auxL[], const PetscScalar auxR[], PetscScalar *flux, void *ctx);
    static PetscErrorCode NPhaseFlowComputeAlphakRhokFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscInt aOff[],
                                                        const PetscScalar auxL[], const PetscScalar auxR[], PetscScalar *flux, void *ctx);

    //this will be necessary later
    // static PetscErrorCode NPhaseFlowComputeAlphakFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscInt aOff[],
    //                                                     const PetscScalar auxL[], const PetscScalar auxR[], PetscScalar *flux, void *ctx);

    // Compute the Euler and density-volume fraction fluxes
    static PetscErrorCode NPhaseFlowCompleteFlux(const ablate::finiteVolume::FiniteVolumeSolver &flow, DM dm, PetscReal time, Vec locXVec, Vec locFVec, void* ctx);

   public:
    /**
     * static call to create a NPhaseDecoder based upon eos
     * @param dim
     * @param eosk
     * @return
     */
    static std::shared_ptr<NPhaseDecoder> CreateNPhaseDecoder(PetscInt dim, const std::vector<std::shared_ptr<eos::EOS>> &eosk);
};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_NPHASEALLAIREADVECTION_HPP
