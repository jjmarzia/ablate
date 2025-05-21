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
#include "finiteVolume/processes/pressureGradientScaling.hpp"
#include "nPhaseAllaireAdvection.hpp"
#include <functional>

namespace ablate::finiteVolume::processes {

// Struct to hold face values needed for vRiem calculation
struct FaceValues {
    std::vector<PetscReal> alphaL;  // Volume fractions for each phase
    std::vector<PetscReal> alphaR;
    std::vector<PetscReal> soskL;   // Speed of sound for each phase
    std::vector<PetscReal> soskR;
    PetscReal rhoL;                 // Density
    PetscReal rhoR;
    PetscReal pL;                   // Pressure
    PetscReal pR;
    PetscReal uL[3];                // Velocity (up to 3D)
    PetscReal uR[3];
};

// Add new struct for cell values
struct CellValues {
    std::vector<PetscReal> alphak;  // Volume fractions for each phase
    std::vector<PetscReal> sosk;    // Speed of sound for each phase
    PetscReal rho;                  // Density
    PetscReal p;                    // Pressure
    PetscReal u[3];                 // Velocity (up to 3D)
    PetscReal divU;                 // Velocity divergence
};

class NPhaseNonconservativeRHS : public Process {

   private:
    //mesh for vertex information
    DM vertexDM{};
    std::shared_ptr<ablate::domain::SubDomain> subDomain;

    // Store cell-face and face-cell relationships
    std::vector<std::vector<PetscInt>> cellToFaces;  // For each cell, list of its faces
    std::vector<std::vector<PetscInt>> faceToCells;  // For each face, list of its cells (should always be 2)
    PetscInt cStart, cEnd;  // Cell range
    PetscInt fStart, fEnd;  // Face range

    // Store pre-computed face values
    std::vector<FaceValues> faceValues;  // Indexed by face number

    // Add vector to store cell values
    std::vector<CellValues> cellValues;  // Indexed by cell number
    std::vector<PetscInt> cellBoundaryDistance;  // Store distance to boundary for each cell
    PetscInt nPhases;  // Number of phases

    // Helper function to compute boundary distances
    void ComputeBoundaryDistances();

    // Add declaration for the RHS function
    static PetscErrorCode ComputeNonconservativeRHS(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscInt aOff[],
                                                  const PetscScalar auxL[], const PetscScalar auxR[], PetscScalar *flux, void *ctx);

    // Add constants for field names
    inline const static std::string ALPHAK = eos::NPhase::ALPHAK;
    inline const static std::string ALPHAKRHOK = eos::NPhase::ALPHAKRHOK;

    /**
     * Wrapper function to match the expected post-step signature
     */
    static void EnforceAlphaKBoundsWrapper(TS ts, ablate::solver::Solver &solver, void *ctx) {
        auto nPhaseNonconservativeRHSProcess = (NPhaseNonconservativeRHS *)ctx;
        if (!nPhaseNonconservativeRHSProcess) {
            throw std::runtime_error("Context not set in solver");
        }

        // Get the solution vector
        Vec xVec;
        TSGetSolution(ts, &xVec) >> utilities::PetscUtilities::checkError;

        // Get the DM
        DM dm = solver.GetSubDomain().GetDM();

        // Call the actual function
        EnforceAlphaKBounds(dynamic_cast<const FiniteVolumeSolver&>(solver), dm, 0.0, xVec, nPhaseNonconservativeRHSProcess);
    }

   public:
    /**
     */
    explicit NPhaseNonconservativeRHS(double mInf, std::shared_ptr<ablate::finiteVolume::processes::PressureGradientScaling> pgs);

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
    // PetscErrorCode PreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime);
    // Add new RHS function declaration
    static PetscErrorCode ComputeNonconservativeRHS(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locXVec, Vec locFVec, void *ctx);
    // std::map<PetscInt, std::vector<PetscInt>> cellNeighbors;
    // std::map<PetscReal, std::vector<PetscReal>> cellWeights;
    // std::map<PetscInt, std::vector<PetscInt>> vertexNeighbors;


    const std::shared_ptr<ablate::finiteVolume::processes::PressureGradientScaling> pgs;

    // Make these static to match their usage in ComputeNonconservativeRHS
    static PetscReal M1Plus(PetscReal m);
    static PetscReal M2Plus(PetscReal m);
    static PetscReal M1Minus(PetscReal m);
    static PetscReal M2Minus(PetscReal m);
    static PetscReal M4Plus(PetscReal m);
    static PetscReal M4Minus(PetscReal m);
    static PetscReal P5Plus(PetscReal m, double fa);
    static PetscReal P5Minus(PetscReal m, double fa);

    static constexpr PetscReal beta = 1.e+0 / 8.e+0;
    static constexpr PetscReal Kp = 0.25;
    static constexpr PetscReal Ku = 0.75;
    static constexpr PetscReal sigma = 0.25;
    static constexpr PetscReal pgsAlpha = 1.0;

    const double mInf;
    
    /**
     * Register the RHS function with the solver
     * @param solver The solver to register with
     */
    void RegisterRHSFunction(FiniteVolumeSolver &solver) {
        solver.RegisterRHSFunction(ComputeNonconservativeRHS, this);
    }

    /**
     * Post-step function to enforce alpha_k bounds
     */
    static PetscErrorCode EnforceAlphaKBounds(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locXVec, void *ctx);

    /**
     * Register the post-step function with the solver
     */
    void RegisterPostStep(FiniteVolumeSolver &solver) {
        auto wrapper = std::bind(EnforceAlphaKBoundsWrapper, std::placeholders::_1, std::placeholders::_2, this);
        solver.RegisterPostStep(wrapper);
    }
};
}  // namespace ablate::finiteVolume::processes
#endif
