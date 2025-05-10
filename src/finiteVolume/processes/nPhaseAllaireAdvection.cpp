#include "nPhaseAllaireAdvection.hpp"

#include <utility>
// #include "eos/stiffenedGas.hpp"
#include "eos/kthStiffenedGas.hpp"
#include "eos/nPhase.hpp"
#include "finiteVolume/nPhaseFlowFields.hpp"
#include "flowProcess.hpp"
#include "domain/region.hpp"
#include "domain/subDomain.hpp"
#include "parameters/emptyParameters.hpp"
#include "utilities/petscSupport.hpp"

#include "intSharp.hpp"

#include <signal.h>

#define NOTE0EXIT(S, ...) {PetscFPrintf(MPI_COMM_WORLD, stderr,                                     \
  "\x1b[1m(%s:%d, %s)\x1b[0m\n  \x1b[1m\x1b[90mexiting:\x1b[0m " S "\n",    \
  __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); exit(0);}

static inline void NormVector(PetscInt dim, const PetscReal *in, PetscReal *out) {
    PetscReal mag = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        mag += in[d] * in[d];
    }
    mag = PetscSqrtReal(mag);
    for (PetscInt d = 0; d < dim; d++) {
        out[d] = in[d] / mag;
    }
}
static inline PetscReal MagVector(PetscInt dim, const PetscReal *in) {
    PetscReal mag = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        mag += in[d] * in[d];
    }
    return PetscSqrtReal(mag);
}

//two phase precedent
//eosTwoPhase encapsulates thermo properties of the two phases (eos::TwoPhase --> eos::NPhase)
//parametersIn contains the cfl number
//std move transfers ownership of eosTwoPhase, fluxCalculatorXX to this class

//parameters checks parametersIn. If null then it returns an empty set
 

ablate::finiteVolume::processes::NPhaseAllaireAdvection::NPhaseAllaireAdvection(std::shared_ptr<eos::EOS> eosNPhase, const std::shared_ptr<parameters::Parameters> &parametersIn,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorNStiff)
    : eosNPhase(std::move(eosNPhase)),
      fluxCalculatorNStiff(std::move(fluxCalculatorNStiff)) {
    auto parameters = ablate::parameters::EmptyParameters::Check(parametersIn);
    // check that eos is nPhase
    if (!this->eosNPhase) {
        throw std::invalid_argument("EOS cannot be null");
    }

    auto nPhaseEOS = std::dynamic_pointer_cast<eos::NPhase>(this->eosNPhase);
    if (!nPhaseEOS) {
        throw std::invalid_argument("EOS must be of type NPhase");
    }

    // populate component eoses
    std::size_t phases = nPhaseEOS->GetNumberOfPhases();
    //PetscPrintf(MPI_COMM_WORLD, "phases = %lu\n", phases);  
    eosk.resize(phases);
    
    for (std::size_t k=0; k<phases; k++) {
        auto phaseEOS = nPhaseEOS->GetEOSk(k);
        auto kthEOS = std::dynamic_pointer_cast<eos::KthStiffenedGas>(phaseEOS);
        if (!kthEOS) {
            throw std::invalid_argument("Each phase EOS must be of type KthStiffenedGas");
        }
        eosk[k] = kthEOS;
        //PetscPrintf(MPI_COMM_WORLD, "eosk[%lu] initialized\n", k);
    }

    // If there is a flux calculator assumed advection
    if (this->fluxCalculatorNStiff) {
        // cfl
        timeStepData.cfl = parameters->Get<PetscReal>("cfl", 0.5);
    }

    //PetscPrintf(MPI_COMM_WORLD, "end of constructor\n");
}


ablate::finiteVolume::processes::NPhaseAllaireAdvection::~NPhaseAllaireAdvection() {
    // Destructor implementation
}

void ablate::finiteVolume::processes::NPhaseAllaireAdvection::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    //PetscPrintf(MPI_COMM_WORLD, "Starting Setup\n");
    //PetscPrintf(MPI_COMM_WORLD, "eosk.size() in Setup = %lu\n", eosk.size());
    
    // Before each step, compute the alpha
    auto multiphasePreStage = std::bind(&ablate::finiteVolume::processes::NPhaseAllaireAdvection::MultiphaseFlowPreStage, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    flow.RegisterPreStage(multiphasePreStage);

    ablate::domain::SubDomain& subDomain = flow.GetSubDomain();
    //PetscPrintf(MPI_COMM_WORLD, "Creating decoder with %lu phases\n", eosk.size());

    // Create the decoder based upon the eoses
    decoder = CreateNPhaseDecoder(subDomain.GetDimensions(), eosk);
    //PetscPrintf(MPI_COMM_WORLD, "Decoder created\n");

    // Currently, no option for species advection
//    flow.RegisterRHSFunction(CompressibleFlowCompleteFlux, this);
    // flow.RegisterRHSFunction(NPhaseFlowCompleteFlux, this); //necessary?

    flow.RegisterRHSFunction(NPhaseFlowComputeAllaireFlux, this, NPhaseFlowFields::ALLAIRE_FIELD, {ALPHAK, ALPHAKRHOK, NPhaseFlowFields::ALLAIRE_FIELD}, {});
    flow.RegisterRHSFunction(NPhaseFlowComputeAlphakRhokFlux, this, ALPHAKRHOK, {ALPHAK, ALPHAKRHOK, NPhaseFlowFields::ALLAIRE_FIELD}, {});
    flow.RegisterComputeTimeStepFunction(ComputeCflTimeStep, &timeStepData, "cfl");
    timeStepData.computeSpeedOfSound = eosNPhase->GetThermodynamicFunction(eos::ThermodynamicProperty::SpeedOfSound, subDomain.GetFields());



    if (subDomain.ContainsField(NPhaseFlowFields::PRESSURE) && (subDomain.GetField(NPhaseFlowFields::PRESSURE).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(NPhaseFlowFields::PRESSURE);
    }

    if (subDomain.ContainsField(NPhaseFlowFields::UI) && (subDomain.GetField(NPhaseFlowFields::UI).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(NPhaseFlowFields::UI);
    }

    if (subDomain.ContainsField(NPhaseFlowFields::TK) && (subDomain.GetField(NPhaseFlowFields::TK).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(NPhaseFlowFields::TK);
    }
    if (subDomain.ContainsField(NPhaseFlowFields::RHO) && (subDomain.GetField(NPhaseFlowFields::RHO).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(NPhaseFlowFields::RHO);
    }
    if (subDomain.ContainsField(NPhaseFlowFields::RHOK) && (subDomain.GetField(NPhaseFlowFields::RHOK).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(NPhaseFlowFields::RHOK);
    }
    if (subDomain.ContainsField(NPhaseFlowFields::E) && (subDomain.GetField(NPhaseFlowFields::E).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(NPhaseFlowFields::E);
    }
    if (subDomain.ContainsField(NPhaseFlowFields::EK) && (subDomain.GetField(NPhaseFlowFields::EK).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(NPhaseFlowFields::EK);
    }

    if (subDomain.ContainsField(ALPHAK) && (subDomain.GetField(ALPHAK).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(ALPHAK);
    }

    // if (subDomain.ContainsField(ALPHAKRHOK) && (subDomain.GetField(ALPHAKRHOK).location == ablate::domain::FieldLocation::AUX)) {
    //   auxUpdateFields.push_back(ALPHAKRHOK);
    // }

    if (auxUpdateFields.size() > 0) {
      flow.RegisterAuxFieldUpdate(
            UpdateAuxFieldsNPhase, this, auxUpdateFields, {ALPHAK, ALPHAKRHOK, NPhaseFlowFields::ALLAIRE_FIELD});
    }

    //initialize intsharp instance in setup;
    // we will then compute the intsharp term in the prestage and iteratively add it to the volume fraction
    // until a sufficient volume fraction gradient/sharpness is achieved
    // std::shared_ptr<ablate::finiteVolume::processes::IntSharp> intSharpProcess;
    // auto intSharpProcess = std::make_shared<ablate::finiteVolume::processes::IntSharp>(0, 0.001, false);
    // intSharpProcess->Initialize(flow);

    //PetscPrintf(MPI_COMM_WORLD, "Setup complete\n");

}
#include <signal.h>
// Update the volume fraction, velocity, temperature, pressure fields, and gas density fields (if they exist).
PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvection::UpdateAuxFieldsNPhase(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[],
                                                                                                   const PetscScalar *conservedValues, const PetscInt aOff[], PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;
    //PetscPrintf(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Starting function\n");

    if (!auxField) {
        //PetscPrintf(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: auxField is null, returning\n");
        PetscFunctionReturn(0);
    }

    auto nPhaseAllaireAdvection = (NPhaseAllaireAdvection *)ctx;
    //PetscPrintf(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Got context\n");

    // For cell center, the norm is unity
    PetscReal norm[3];
    norm[0] = 1;
    norm[1] = 1;
    norm[2] = 1;

    PetscReal density = 1.0;
    std::vector<PetscReal> densityk;
    PetscReal normalVelocity = 0.0;  // uniform velocity in cell
    PetscReal velocity[3] = {0.0, 0.0, 0.0};
    PetscReal internalEnergy = 0.0;
    std::vector<PetscReal> internalEnergyk;
    std::vector<PetscReal> ak;
    std::vector<PetscReal> Mk;
    PetscReal p = 0.0;  // pressure equilibrium
    std::vector<PetscReal> Tk;  
    std::vector<PetscReal> alphak;

    if (conservedValues) {
        //PetscPrintf(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: About to decode state\n");
        try {
            nPhaseAllaireAdvection->decoder->DecodeNPhaseAllaireState(
                dim, uOff, conservedValues, norm, &density, &densityk, &normalVelocity, velocity, &internalEnergy, &internalEnergyk, &ak,
                &Mk, &p, &Tk, &alphak);
            //PetscPrintf(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Successfully decoded state\n");
        } catch (const std::exception& e) {
            //PetscPrintf(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Error in DecodeNPhaseAllaireState: %s\n", e.what());
            throw;
        }

        for (PetscInt d = 0; d < dim; d++) {
            velocity[d] = conservedValues[NPhaseFlowFields::RHOU + d] / density;
        }
        //PetscPrintf(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Computed velocities\n");
    }

    auto fields = nPhaseAllaireAdvection->auxUpdateFields.data();
    //PetscPrintf(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Starting field updates, number of fields: %zu\n", nPhaseAllaireAdvection->auxUpdateFields.size());

    for (std::size_t f = 0; f < nPhaseAllaireAdvection->auxUpdateFields.size(); ++f) {
        //PetscPrintf(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Processing field %s\n", fields[f].c_str());
        
        if (fields[f] == NPhaseFlowFields::UI) {
            for (PetscInt d = 0; d < dim; d++) {
                auxField[aOff[f] + d] = velocity[d];
            }
        }
        else if (fields[f] == NPhaseFlowFields::PRESSURE) {
            auxField[aOff[f]] = p;
        }
        else if (fields[f] == NPhaseFlowFields::TK) {
            for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
                auxField[aOff[f] + k] = Tk[k];
            }
        }
        else if (fields[f] == NPhaseFlowFields::RHO) {
            auxField[aOff[f]] = density;
        }
        else if (fields[f] == NPhaseFlowFields::RHOK) {
            for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
                auxField[aOff[f] + k] = densityk[k];
            }
        }
        else if (fields[f] == NPhaseFlowFields::E) {
            auxField[aOff[f]] = internalEnergy;
        }
        else if (fields[f] == NPhaseFlowFields::EK) {
            for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
                auxField[aOff[f] + k] = internalEnergyk[k];
            }
        }
        else if (fields[f] == NPhaseFlowFields::ALPHAKRHOK) {
            for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
                auxField[aOff[f] + k] = densityk[k];
            }
        }
        else if (fields[f] == NPhaseFlowFields::ALPHAK) {
            for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
                auxField[aOff[f] + k] = ak[k];
            }
        }
        //PetscPrintf(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Completed field %s\n", fields[f].c_str());
    }

    //PetscPrintf(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Completed all field updates\n");
    PetscFunctionReturn(0);
}
#include <iostream>


PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvection::MultiphaseFlowPreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime) {
    PetscFunctionBegin;
    //PetscPrintf(MPI_COMM_WORLD, "1. Entering MultiphaseFlowPreStage\n");
    
    // Get flow field data
    //PetscPrintf(MPI_COMM_WORLD, "2. About to dynamic_cast solver\n");
    const auto &fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver &>(solver);
    //PetscPrintf(MPI_COMM_WORLD, "3. Cast successful\n");

    ablate::domain::Range cellRange;
    //PetscPrintf(MPI_COMM_WORLD, "4. About to GetCellRangeWithoutGhost\n");
    fvSolver.GetCellRangeWithoutGhost(cellRange);
    //PetscPrintf(MPI_COMM_WORLD, "5. Got cell range\n");

    PetscInt dim;
    //PetscPrintf(MPI_COMM_WORLD, "6. About to get dimension\n");
    PetscCall(DMGetDimension(fvSolver.GetSubDomain().GetDM(), &dim));
    //PetscPrintf(MPI_COMM_WORLD, "7. Got dimension\n");

    //PetscPrintf(MPI_COMM_WORLD, "8. About to get field offsets\n");
    const auto &allaireOffset = fvSolver.GetSubDomain().GetField(NPhaseFlowFields::ALLAIRE_FIELD).offset;
    const auto &alphakOffset = fvSolver.GetSubDomain().GetField(ALPHAK).offset;
    const auto &alphakRhokOffset = fvSolver.GetSubDomain().GetField(ALPHAKRHOK).offset;
    // PetscPrintf(MPI_COMM_WORLD, "alphakOffset: %d, alphakRhokOffset: %d, allaireOffset: %d\n", alphakOffset, alphakRhokOffset, allaireOffset);
    //PetscPrintf(MPI_COMM_WORLD, "9. Got field offsets\n");

    //PetscPrintf(MPI_COMM_WORLD, "10. About to get DM\n");
    DM dm = fvSolver.GetSubDomain().GetDM();
    //PetscPrintf(MPI_COMM_WORLD, "11. Got DM\n");

    //PetscPrintf(MPI_COMM_WORLD, "12. About to get solution\n");
    Vec globFlowVec;
    PetscCall(TSGetSolution(flowTs, &globFlowVec));
    //PetscPrintf(MPI_COMM_WORLD, "13. Got solution\n");

    //PetscPrintf(MPI_COMM_WORLD, "14. About to get array\n");
    PetscScalar *flowArray;
    PetscCall(VecGetArray(globFlowVec, &flowArray));
    //PetscPrintf(MPI_COMM_WORLD, "15. Got array\n");

    PetscInt uOff[3];
    uOff[0] = alphakOffset;
    uOff[1] = alphakRhokOffset;
    uOff[2] = allaireOffset;

    //get the rhs vector
    Vec locFVec;
    PetscCall(DMGetLocalVector(dm, &locFVec));
    PetscCall(VecZeroEntries(locFVec));

    // For cell center, the norm is unity
    PetscReal norm[3];
    norm[0] = 1;
    norm[1] = 1;
    norm[2] = 1;

    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
        const PetscInt cell = cellRange.GetPoint(i);
        PetscScalar *allFields = nullptr;
        DMPlexPointLocalRef(dm, cell, flowArray, &allFields) >> utilities::PetscUtilities::checkError;
        auto density = allFields[ablate::finiteVolume::CompressibleFlowFields::RHO];
        PetscReal velocity[3];
        for (PetscInt d = 0; d < dim; d++) {
            velocity[d] = allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density;
        }

        // Decode state
        std::vector<PetscReal> densityk;
        PetscReal normalVelocity;
        PetscReal internalEnergy;
        std::vector<PetscReal> internalEnergyk;
        std::vector<PetscReal> ak;
        std::vector<PetscReal> Mk;
        PetscReal p;
        std::vector<PetscReal> tk;
        std::vector<PetscReal> alphak;

        decoder->DecodeNPhaseAllaireState(
            dim, uOff, allFields, norm, &density, &densityk, &normalVelocity, velocity, &internalEnergy, &internalEnergyk, &ak, &Mk, &p, &tk, &alphak);

        // update all alphak
        for (std::size_t k = 0; k < eosk.size(); k++) {
            allFields[uOff[0] + k] = alphak[k];
        }

    }

    //restore
    PetscCall(DMRestoreLocalVector(dm, &locFVec));
    PetscCall(VecRestoreArray(globFlowVec, &flowArray));

    // clean up
    fvSolver.RestoreRange(cellRange);
    PetscFunctionReturn(0);

}
double ablate::finiteVolume::processes::NPhaseAllaireAdvection::ComputeCflTimeStep(TS ts, ablate::finiteVolume::FiniteVolumeSolver &flow, void *ctx) {
    // Get the dm and current solution vector
    DM dm;
    TSGetDM(ts, &dm) >> utilities::PetscUtilities::checkError;
    Vec v;
    TSGetSolution(ts, &v) >> utilities::PetscUtilities::checkError;

    // Get the flow param
    auto timeStepData = (TimeStepData *)ctx;

    // Get the fv geom
    PetscReal minCellRadius;
    DMPlexGetGeometryFVM(dm, NULL, NULL, &minCellRadius) >> utilities::PetscUtilities::checkError;

    // Get the valid cell range over this region
    ablate::domain::Range cellRange;
    flow.GetCellRange(cellRange);

    const PetscScalar *x;
    VecGetArrayRead(v, &x) >> utilities::PetscUtilities::checkError;

    // Get the dim from the dm
    PetscInt dim;
    DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;

    // assume the smallest cell is the limiting factor for now
    const PetscReal dx = 2.0 * minCellRadius;

    // Get field location for euler and densityYi
    auto allaireID = flow.GetSubDomain().GetField(ablate::finiteVolume::NPhaseFlowFields::ALLAIRE_FIELD).id;

    // March over each cell
    PetscReal dtMin = 1000.0;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.points ? cellRange.points[c] : c;

        const PetscReal *allaire;
        const PetscReal *conserved = NULL;
        DMPlexPointGlobalFieldRead(dm, cell, allaireID, x, &allaire) >> utilities::PetscUtilities::checkError;
        DMPlexPointGlobalRead(dm, cell, x, &conserved) >> utilities::PetscUtilities::checkError;

        if (allaire) {  // must be real cell and not ghost
            PetscReal rho = 1000.0; //fix later
            // for (std::size_t k = 0; k < timeStepData->eosk.size(); k++) {
            //     rho += allaire[CompressibleFlowFields::ALPHAKRHOK + k];
            // }

            // Get the speed of sound from the eos
            PetscReal a;
            timeStepData->computeSpeedOfSound.function(conserved, &a, timeStepData->computeSpeedOfSound.context.get()) >> utilities::PetscUtilities::checkError;

            PetscReal velSum = 0.0;
            for (PetscInt d = 0; d < dim; d++) {
                velSum += PetscAbsReal(allaire[CompressibleFlowFields::RHOU + d]) / rho;
            }

            PetscReal dt = timeStepData->cfl * dx / (a + velSum);

            dtMin = PetscMin(dtMin, dt);
        }
    }
    VecRestoreArrayRead(v, &x) >> utilities::PetscUtilities::checkError;
    flow.RestoreRange(cellRange);
    return dtMin;
}
PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvection::NPhaseFlowCompleteFlux(const ablate::finiteVolume::FiniteVolumeSolver &flow, DM dm, PetscReal time, Vec locXVec, Vec locFVec, void* ctx) {

  PetscFunctionBeginUser;
//  auto flow = (ablate::finiteVolume::FiniteVolumeSolver *)ctx;
//  auto twoPhaseEulerAdvection = (TwoPhaseEulerAdvection *)ctx;
  NOTE0EXIT("");

  PetscFunctionReturn(0);

}
PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvection::NPhaseFlowComputeAllaireFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscScalar *fieldL,
                                                                                                         const PetscScalar *fieldR, const PetscInt *aOff, const PetscScalar *auxL,
                                                                                                         const PetscScalar *auxR, PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    //PetscPrintf(MPI_COMM_WORLD, "Starting NPhaseFlowComputeAllaireFlux\n");
    
    auto nPhaseAllaireAdvection = (NPhaseAllaireAdvection *)ctx;
    //PetscPrintf(MPI_COMM_WORLD, "Got nPhaseAllaireAdvection context\n");

    // Compute the norm of cell face
    PetscReal norm[3];
    NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = MagVector(dim, fg->normal);
    //PetscPrintf(MPI_COMM_WORLD, "Computed norm and area magnitude\n");


    // Decode left and right states
    PetscReal densityL = 0.0;
    std::vector<PetscReal> densityk_L;
    PetscReal normalVelocityL = 0.0; 
    PetscReal velocityL[3] = {0.0, 0.0, 0.0};
    PetscReal internalEnergyL = 0.0;
    std::vector<PetscReal> internalEnergyk_L;
    std::vector<PetscReal> ak_L;
    std::vector<PetscReal> Mk_L;
    PetscReal pL = 0.0; 
    std::vector<PetscReal> tk_L;  
    std::vector<PetscReal> alphak_L;
    nPhaseAllaireAdvection->decoder->DecodeNPhaseAllaireState(dim,
                                                              uOff,
                                                              fieldL,
                                                              norm,
                                                              &densityL,
                                                              &densityk_L,
                                                              &normalVelocityL,
                                                              velocityL,
                                                              &internalEnergyL,
                                                              &internalEnergyk_L,
                                                              &ak_L,
                                                              &Mk_L,
                                                              &pL,
                                                              &tk_L,
                                                              &alphak_L);

    //print all resulting values
//     PetscPrintf(MPI_COMM_WORLD, "densityL densityk_L normalVelocityL velocityL internalEnergyL internalEnergyk_L ak_L Mk_L pL tk_L alphak_L\n");
// PetscPrintf(MPI_COMM_WORLD, "%f %f %f %f %f %f %f %f %f %f %f %f %f\n", 
    // densityL, densityk_L[0], normalVelocityL, velocityL[0], velocityL[1], velocityL[2], 
    // internalEnergyL, internalEnergyk_L[0], ak_L[0], Mk_L[0], pL, tk_L[0], alphak_L[0]);
    //do the same for _R instead of _L
    PetscReal densityR = 0.0;
    std::vector<PetscReal> densityk_R;
    PetscReal normalVelocityR = 0.0;
    PetscReal velocityR[3] = {0.0, 0.0, 0.0}; 
    PetscReal internalEnergyR = 0.0;
    std::vector<PetscReal> internalEnergyk_R;
    std::vector<PetscReal> ak_R;
    std::vector<PetscReal> Mk_R;
    PetscReal pR = 0.0;
    std::vector<PetscReal> tk_R;
    std::vector<PetscReal> alphak_R;
    nPhaseAllaireAdvection->decoder->DecodeNPhaseAllaireState(dim,
                                                              uOff,
                                                              fieldR,
                                                              norm,
                                                              &densityR,
                                                              &densityk_R,
                                                              &normalVelocityR,
                                                              velocityR,
                                                              &internalEnergyR,
                                                              &internalEnergyk_R,
                                                              &ak_R,
                                                              &Mk_R,
                                                              &pR,
                                                              &tk_R,
                                                              &alphak_R);

      //print all resulting values
      // PetscPrintf(MPI_COMM_WORLD, "densityR densityk_R normalVelocityR velocityR internalEnergyR internalEnergyk_R ak_R Mk_R pR tk_R alphak_R\n");
      // PetscPrintf(MPI_COMM_WORLD, "%f %f %f %f %f %f %f %f %f %f %f %f %f\n", 
        // densityR, densityk_R[0], normalVelocityR, velocityR[0], velocityR[1], velocityR[2], 
        // internalEnergyR, internalEnergyk_R[0], ak_R[0], Mk_R[0], pR, tk_R[0], alphak_R[0]);

      // a_L = 1/ (sumk alphak_L / ak_L)
      // a_R = 1/ (sumk alphak_R / ak_R)
    PetscReal aL=0.0;
    for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
      //PetscPrintf(MPI_COMM_WORLD, "alphak_L[%zu]: %f, ak_L[%zu]: %f\n", k, alphak_L[k], k, ak_L[k]);
      //only add if alphak_L[k] is not zero
      if (alphak_L[k] > PETSC_SMALL) {
        aL += alphak_L[k] / ak_L[k];
      }
    }
    aL = 1.0 / aL;
    PetscReal aR=0.0;
    for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
      //PetscPrintf(MPI_COMM_WORLD, "alphak_R[%zu]: %f, ak_R[%zu]: %f\n", k, alphak_R[k], k, ak_R[k]);
      //only add if alphak_R[k] is not zero
      if (alphak_R[k] > PETSC_SMALL) {
        aR += alphak_R[k] / ak_R[k];
      }
    }
    aR = 1.0 / aR;

    PetscReal massFlux = 0.0, p12 = 0.0;

    //print all inputs
    //PetscPrintf(MPI_COMM_WORLD, "normalVelocityL: %f, aL: %f, densityL: %f, pL: %f, normalVelocityR: %f, aR: %f, densityR: %f, pR: %f\n", normalVelocityL, aL, densityL, pL, normalVelocityR, aR, densityR, pR);
    fluxCalculator::Direction direction = nPhaseAllaireAdvection->fluxCalculatorNStiff->GetFluxCalculatorFunction()(
      nPhaseAllaireAdvection->fluxCalculatorNStiff->GetFluxCalculatorContext(), normalVelocityL, aL, densityL, pL, normalVelocityR, aR, densityR, pR, &massFlux, &p12);

//print the flux results
//PetscPrintf(MPI_COMM_WORLD, "massFlux: %f, p12: %f\n", massFlux, p12);
//PetscPrintf(MPI_COMM_WORLD, "direction: %d\n", direction);

    // Calculate total flux
    PetscReal vel[3] = {0.0, 0.0, 0.0};
    PetscReal internalEnergy = 0.0, density = 0.0;

    if (direction == fluxCalculator::LEFT) {
        internalEnergy = internalEnergyL;
        density = densityL;
        PetscArraycpy(vel, velocityL, dim);
    } else if (direction == fluxCalculator::RIGHT) {
        internalEnergy = internalEnergyR;
        density = densityR;
        PetscArraycpy(vel, velocityR, dim);
    } else {
        internalEnergy = 0.5*(internalEnergyL + internalEnergyR);
        density = 0.5*(densityL + densityR);
        for (PetscInt d = 0; d < dim; d++) vel[d] = 0.5*(velocityL[d] + velocityR[d]);
    }

    PetscReal velMag = MagVector(dim, vel);
    PetscReal H = internalEnergy + 0.5 * velMag * velMag + p12 / density;

    // flux[CompressibleFlowFields::RHO] = massFlux * areaMag;
    flux[NPhaseFlowFields::RHOE] = H * massFlux * areaMag;
    for (PetscInt n = 0; n < dim; n++) {
        flux[NPhaseFlowFields::RHOU + n] = vel[n] * areaMag * massFlux + p12 * fg->normal[n];
    }

    //print rhoe, rhou, rhov
    //PetscPrintf(MPI_COMM_WORLD, "rhoe: %f, rhou: %f, rhov: %f\n", flux[NPhaseFlowFields::RHOE], flux[NPhaseFlowFields::RHOU], flux[NPhaseFlowFields::RHOV]);



    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvection::NPhaseFlowComputeAlphakRhokFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscScalar *fieldL,
                                                                                                      const PetscScalar *fieldR, const PetscInt *aOff, const PetscScalar *auxL, const PetscScalar *auxR,
                                                                                                      PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    //PetscPrintf(MPI_COMM_WORLD, "Starting NPhaseFlowComputeAlphakRhokFlux\n");
    
    auto nPhaseAllaireAdvection = (NPhaseAllaireAdvection *)ctx;
    //PetscPrintf(MPI_COMM_WORLD, "Got nPhaseAllaireAdvection context\n");

    // Compute the norm
    PetscReal norm[3];
    NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = MagVector(dim, fg->normal);
    //PetscPrintf(MPI_COMM_WORLD, "Computed norm and area magnitude\n");

    PetscReal densityL = 0.0;
    std::vector<PetscReal> densityk_L;
    PetscReal normalVelocityL = 0.0; 
    PetscReal velocityL[3] = {0.0, 0.0, 0.0};
    PetscReal internalEnergyL = 0.0;
    std::vector<PetscReal> internalEnergyk_L;
    std::vector<PetscReal> ak_L;
    std::vector<PetscReal> Mk_L;
    PetscReal pL = 0.0; 
    std::vector<PetscReal> tk_L;  
    std::vector<PetscReal> alphak_L;
    nPhaseAllaireAdvection->decoder->DecodeNPhaseAllaireState(dim,
                                                              uOff,
                                                              fieldL,
                                                              norm,
                                                              &densityL,
                                                              &densityk_L,
                                                              &normalVelocityL,
                                                              velocityL,
                                                              &internalEnergyL,
                                                              &internalEnergyk_L,
                                                              &ak_L,
                                                              &Mk_L,
                                                              &pL,
                                                              &tk_L,
                                                              &alphak_L);

    //do the same for _R instead of _L
    PetscReal densityR = 0.0;
    std::vector<PetscReal> densityk_R;
    PetscReal normalVelocityR = 0.0;
    PetscReal velocityR[3] = {0.0, 0.0, 0.0}; 
    PetscReal internalEnergyR = 0.0;
    std::vector<PetscReal> internalEnergyk_R;
    std::vector<PetscReal> ak_R;
    std::vector<PetscReal> Mk_R;
    PetscReal pR = 0.0;
    std::vector<PetscReal> tk_R;
    std::vector<PetscReal> alphak_R;
    nPhaseAllaireAdvection->decoder->DecodeNPhaseAllaireState(dim,
                                                              uOff,
                                                              fieldR,
                                                              norm,
                                                              &densityR,
                                                              &densityk_R,
                                                              &normalVelocityR,
                                                              velocityR,
                                                              &internalEnergyR,
                                                              &internalEnergyk_R,
                                                              &ak_R,
                                                              &Mk_R,
                                                              &pR,
                                                              &tk_R,
                                                              &alphak_R);

    // get the face values
    PetscReal massFlux;
    PetscReal p12;

    const int ALPHAKRHOK_FIELD = 1;
    const int ALPHAK_FIELD = 0;
    const PetscInt ALPHAKRHOK_OFFSET = uOff[ALPHAKRHOK_FIELD]; 
    const PetscInt ALPHAK_OFFSET = uOff[ALPHAK_FIELD];

    //flux[0] is flawed, it should be NPHASEFLOWFIELDS::ALPHAKRHOK(k) or variant
    for (size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
      // PetscPrintf(MPI_COMM_WORLD, "alphakrhokoffset + k: %lu\n", ALPHAKRHOK_OFFSET + k);
        fluxCalculator::Direction directionk = nPhaseAllaireAdvection->fluxCalculatorNStiff->GetFluxCalculatorFunction()(
            nPhaseAllaireAdvection->fluxCalculatorNStiff->GetFluxCalculatorContext(), normalVelocityL, ak_L[k], densityk_L[k], pL, normalVelocityR, ak_R[k], densityk_R[k], pR, &massFlux, &p12);
        if (directionk == fluxCalculator::LEFT) {
            flux[ALPHAKRHOK_OFFSET + k] = massFlux * areaMag * alphak_L[k];
            flux[ALPHAK_OFFSET + k] = massFlux/densityk_L[k] * areaMag * alphak_L[k];
        } else if (directionk == fluxCalculator::RIGHT) {
            flux[ALPHAKRHOK_OFFSET + k] = massFlux * areaMag * alphak_R[k]; //massflux is rho(u dot n) but should this be rhok(u dot n) ?
            flux[ALPHAK_OFFSET + k] = massFlux/densityk_R[k] * areaMag * alphak_R[k];
        } else {
            flux[ALPHAKRHOK_OFFSET + k] = massFlux * areaMag * 0.5 * (alphak_L[k] + alphak_R[k]);
            flux[ALPHAK_OFFSET + k] = massFlux/densityk_R[k] * areaMag * 0.5 * (alphak_L[k] + alphak_R[k]);
        }
    }

    PetscFunctionReturn(0);
}

//need a PetscErrorCode NPhaseFlowComputeAlphakFlux 

std::shared_ptr<ablate::finiteVolume::processes::NPhaseAllaireAdvection::NPhaseDecoder> ablate::finiteVolume::processes::NPhaseAllaireAdvection::CreateNPhaseDecoder(
    PetscInt dim, const std::vector<std::shared_ptr<eos::EOS>> &eosk) {

    //PetscPrintf(MPI_COMM_WORLD, "Entering CreateNPhaseDecoder\n");
    //PetscPrintf(MPI_COMM_WORLD, "Input eosk size: %lu\n", eosk.size());

    // return std::make_shared<NStiffDecoder>(dim, eosk);
    std::vector<std::shared_ptr<ablate::eos::KthStiffenedGas>> stiffGases;
    //PetscPrintf(MPI_COMM_WORLD, "Created kthStiffenedGas vector\n");

    for (const auto& eos : eosk) {
      //PetscPrintf(MPI_COMM_WORLD, "Attempting to cast EOS to kthStiffenedGas\n");
      auto stiffGas = std::dynamic_pointer_cast<ablate::eos::KthStiffenedGas>(eos);
      if (!stiffGas) {
        //PetscPrintf(MPI_COMM_WORLD, "Failed to cast EOS to kthStiffenedGas\n");
        throw std::invalid_argument("All EOSs must be kthStiffenedGas for NPhaseAllaireAdvection");
      }
      //PetscPrintf(MPI_COMM_WORLD, "Successfully cast EOS to kthStiffenedGas\n");
      stiffGases.push_back(stiffGas);
    }
    //PetscPrintf(MPI_COMM_WORLD, "Created stiffGases vector, size: %lu\n", stiffGases.size());
    auto decoder = std::make_shared<NStiffDecoder>(dim, stiffGases); //this is where the error is
    //PetscPrintf(MPI_COMM_WORLD, "Successfully created NStiffDecoder\n");
    return decoder;
}


#include <signal.h>

ablate::finiteVolume::processes::NPhaseAllaireAdvection::NStiffDecoder::NStiffDecoder(PetscInt dim, const std::vector<std::shared_ptr<eos::KthStiffenedGas>> &eosk)
    : eosk(eosk) {
    //PetscPrintf(MPI_COMM_WORLD, "Starting NStiffDecoder constructor\n");

    std::size_t phases = eosk.size();
    //PetscPrintf(MPI_COMM_WORLD, "Input eosk size: %lu\n", phases);

    // Create the fake euler field
    //PetscPrintf(MPI_COMM_WORLD, "Creating fake Allaire field\n");
    auto fakeAllaireField = ablate::domain::Field{.name = NPhaseFlowFields::ALLAIRE_FIELD,
                                                .numberComponents = 1 + dim,
                                                .components = {},
                                                .id = PETSC_DEFAULT,
                                                .subId = PETSC_DEFAULT,
                                                .offset = 0,
                                                .location = ablate::domain::FieldLocation::SOL,
                                                .type = ablate::domain::FieldType::FVM,
                                                .tags = {}};

    // Initialize all vectors to the correct size first
    //PetscPrintf(MPI_COMM_WORLD, "Initializing vectors\n");
    kAllaireFieldScratch.resize(phases);
    kComputeTemperature.resize(phases);
    kComputeInternalEnergy.resize(phases);
    kComputeSpeedOfSound.resize(phases);
    kComputePressure.resize(phases);

    // Now initialize each phase
    //PetscPrintf(MPI_COMM_WORLD, "Initializing phase data\n");
    for (std::size_t k = 0; k < phases; k++) {
        if (!eosk[k]) {
            throw std::invalid_argument("EOS for phase " + std::to_string(k) + " is null");
        }
        //PetscPrintf(MPI_COMM_WORLD, "Initializing phase %lu\n", k);
        kAllaireFieldScratch[k].resize(1 + dim);
        //PetscPrintf(MPI_COMM_WORLD, "Getting thermodynamic functions for phase %lu\n", k);
        kComputeTemperature[k] = eosk[k]->GetThermodynamicFunction(eos::ThermodynamicProperty::Temperature, {fakeAllaireField});
        kComputeInternalEnergy[k] = eosk[k]->GetThermodynamicFunction(eos::ThermodynamicProperty::InternalSensibleEnergy, {fakeAllaireField});
        kComputeSpeedOfSound[k] = eosk[k]->GetThermodynamicFunction(eos::ThermodynamicProperty::SpeedOfSound, {fakeAllaireField});
        kComputePressure[k] = eosk[k]->GetThermodynamicFunction(eos::ThermodynamicProperty::Pressure, {fakeAllaireField});
        //PetscPrintf(MPI_COMM_WORLD, "Finished initializing phase %lu\n", k);
    }
    //PetscPrintf(MPI_COMM_WORLD, "Finished NStiffDecoder constructor\n");
}

void ablate::finiteVolume::processes::NPhaseAllaireAdvection::NStiffDecoder::DecodeNPhaseAllaireState(PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues,
                                                                                                                        const PetscReal *normal, PetscReal *densityOut, std::vector<PetscReal> *densitykOut, PetscReal *normalVelocityOut, PetscReal *velocityOut,
                                                                                                                        PetscReal *internalEnergyOut, std::vector<PetscReal> *internalEnergykOut, std::vector<PetscReal> *akOut, std::vector<PetscReal> *MkOut,
                                                                                                                         PetscReal *pOut, std::vector<PetscReal> *TkOut, std::vector<PetscReal> *alphakOut) {

    

 

    //PetscPrintf(MPI_COMM_WORLD, "Start DecodeNPhaseAllaireState\n");

    std::size_t phases = eosk.size();
    densitykOut->resize(phases);
    internalEnergykOut->resize(phases);
    akOut->resize(phases);
    MkOut->resize(phases);
    TkOut->resize(phases);
    alphakOut->resize(phases);

    
    const int ALPHAK_FIELD = 0;
    const int ALPHAKRHOK_FIELD = 1; //these are correct
    const int ALLAIRE_FIELD = 2;

    
    const PetscInt ALPHAK_OFFSET = uOff[ALPHAK_FIELD];
    const PetscInt ALPHAKRHOK_OFFSET = uOff[ALPHAKRHOK_FIELD]; 
    const PetscInt ALLAIRE_OFFSET = uOff[ALLAIRE_FIELD];

    //PetscPrintf(MPI_COMM_WORLD, "field offsets: allaire=%d, alphak=%d, alphakrhok=%d\n", ALLAIRE_OFFSET, ALPHAK_OFFSET, ALPHAKRHOK_OFFSET);
    
    //check to make sure the offsets are correct
    //loop over phases
    for (std::size_t k = 0; k < phases; k++) {
        //PetscPrintf(MPI_COMM_WORLD, "conservedValues[ALPHAKRHOK_OFFSET + %lu]: %f\n", k, conservedValues[ALPHAKRHOK_OFFSET + k]);
        //PetscPrintf(MPI_COMM_WORLD, "conservedValues[ALPHAK_OFFSET + %lu]: %f\n", k, conservedValues[ALPHAK_OFFSET + k]);
    }
    for (PetscInt i = 0; i < dim + 1; i++) {
        //PetscPrintf(MPI_COMM_WORLD, "conservedValues[ALLAIRE_OFFSET + %d]: %f\n", i, conservedValues[ALLAIRE_OFFSET + i]);
    }

    // Declare all needed vectors and variables
    std::vector<PetscReal> rhok(phases);
    std::vector<PetscReal> ek(phases);
    std::vector<PetscReal> Tk(phases);
    std::vector<PetscReal> alphak(phases);
    std::vector<PetscReal> Cpk(phases);
    std::vector<PetscReal> gammak(phases);
    std::vector<PetscReal> pik(phases);
    std::vector<PetscReal> ck(phases);
    std::vector<PetscReal> Mk(phases);  
    PetscReal velocity[3] = {0.0, 0.0, 0.0};  // Initialize velocity array

    // Get phase-specific parameters
    for (std::size_t k = 0; k < phases; k++) {
        Cpk[k] = eosk[k]->GetSpecificHeatCp();
        gammak[k] = eosk[k]->GetSpecificHeatRatio();
        pik[k] = eosk[k]->GetReferencePressure();
    }

    // First compute alpha_k and alpha_k*rho_k from conserved values
    for (std::size_t k = 0; k < phases; k++) {
        alphak[k] = conservedValues[uOff[0] + k];  // alpha_k
        
        // Only compute phase density if alpha_k is non-zero
        if (alphak[k] > PETSC_SMALL) {
            rhok[k] = conservedValues[uOff[1] + k] / alphak[k];  // rho_k = (alpha_k*rho_k)/alpha_k
        } else {
            rhok[k] = 0.0;
            ek[k] = 0.0;
            Tk[k] = 0.0;
            continue;  // Skip further calculations for this phase
        }
    }

    // Compute total density rho = sum_k (alpha_k*rho_k)
    PetscReal rho = 0.0;
    for (std::size_t k = 0; k < phases; k++) {
        rho += conservedValues[uOff[1] + k];  // sum of alpha_k*rho_k
    }

    // Compute velocity components u_i = (rho*u_i)/(rho + PETSC_SMALL)
    PetscReal uiui = 0.0;  // u_i*u_i
    for (PetscInt d = 0; d < dim; d++) {
        velocity[d] = conservedValues[uOff[2] + ablate::finiteVolume::NPhaseFlowFields::RHOU + d] / (rho + PETSC_SMALL);  // u_i = (rho*u_i)/(rho + PETSC_SMALL)
        uiui += velocity[d] * velocity[d];
    }

    // Compute total energy per unit mass e = (rho*e)/(rho + PETSC_SMALL)
    PetscReal e = conservedValues[ablate::finiteVolume::NPhaseFlowFields::RHOE] / (rho + PETSC_SMALL);  // e = (rho*e)/(rho + PETSC_SMALL)

    //PetscPrintf(MPI_COMM_WORLD, "e: %f\n", e);
    //PetscPrintf(MPI_COMM_WORLD, "rho: %f\n", rho);
    //PetscPrintf(MPI_COMM_WORLD, "uiui: %f\n", uiui);
    //PetscPrintf(MPI_COMM_WORLD, "velocity: %f, %f, %f\n", velocity[0], velocity[1], velocity[2]);
    

    // Compute pressure using the given equation
    PetscReal numerator = conservedValues[uOff[2] + ablate::finiteVolume::NPhaseFlowFields::RHOE];  // rho*e
    numerator -= 0.5 * rho * uiui;  // subtract kinetic energy term
    
    // Subtract sum_k (alpha_k*gamma_k*pi_k)/(gamma_k-1)
    for (std::size_t k = 0; k < phases; k++) {
        if (alphak[k] > PETSC_SMALL) {  // Only include non-zero phases in pressure calculation
            numerator -= alphak[k] * gammak[k] * pik[k] / (gammak[k] - 1.0);
        }
    }

    // Compute denominator sum_k (alpha_k)/(gamma_k-1)
    PetscReal denominator = 0.0;
    for (std::size_t k = 0; k < phases; k++) {
        if (alphak[k] > PETSC_SMALL) {  // Only include non-zero phases in denominator
            denominator += alphak[k] / (gammak[k] - 1.0);
        }
    }

    // Final pressure calculation
    PetscReal p = numerator / (denominator + PETSC_SMALL);

    // Compute internal energy per unit mass epsilon = e - u_i*u_i/2
    PetscReal epsilon = e - 0.5 * uiui;

    // Compute phase-specific quantities only for non-zero phases
    for (std::size_t k = 0; k < phases; k++) {
        if (alphak[k] > PETSC_SMALL) {
            // Compute internal energy per unit mass for phase k
            ek[k] = (p + gammak[k] * pik[k]) / ((gammak[k] - 1.0) * rhok[k]);
            // Compute temperature for phase k
            Tk[k] = gammak[k] * (ek[k] - pik[k]/rhok[k]) / Cpk[k];
            // Compute speed of sound for phase k
            Cpk[k] = eosk[k]->GetSpecificHeatCp();
            gammak[k] = eosk[k]->GetSpecificHeatRatio();
            pik[k] = eosk[k]->GetReferencePressure();
            ck[k] = PetscSqrtReal(gammak[k] * (p + pik[k]) / rhok[k]);
            //compute mach number
        } else {
            // Set all phase-specific quantities to zero for zero-phase-fraction phases
            rhok[k] = 0.0;
            ek[k] = 0.0;
            Tk[k] = 0.0;
        }
    }

    // Set output values
    *densityOut = rho;
    *normalVelocityOut = 0.0; 
    for (PetscInt d = 0; d < dim; d++) {
        velocityOut[d] = velocity[d];
        *normalVelocityOut += velocity[d] * normal[d];
    }
    *internalEnergyOut = epsilon;

    //*ak = (PetscReal)PetscSqrtReal((gammak - 1.0)*Cpk*Tk);
    //*Mk = normalVelocity / ak;


    for (std::size_t k = 0; k < phases; k++) {
        (*densitykOut)[k] = rhok[k];
        (*internalEnergykOut)[k] = ek[k];
        (*akOut)[k] = PetscSqrtReal((gammak[k] - 1.0)*Cpk[k]*Tk[k]);
        (*MkOut)[k] = *normalVelocityOut / (*akOut)[k];
        (*TkOut)[k] = Tk[k];
        (*alphakOut)[k] = alphak[k];
    }
    *pOut = p + ALLAIRE_OFFSET*0 + ALPHAK_OFFSET*0 + ALPHAKRHOK_OFFSET*0;

    // Debug print
    for (std::size_t k = 0; k < phases; k++) {
        // PetscPrintf(MPI_COMM_WORLD, "Phase %zu: rhok=%g, ek=%g, Tk=%g, pk=%g, alphak=%g\n", 
        //    k, rhok[k], ek[k], Tk[k], p, alphak[k]);
    }


    //PetscPrintf(MPI_COMM_WORLD, "finished DecodeNPhaseAllaireState\n");

}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::NPhaseAllaireAdvection, "", ARG(ablate::eos::EOS, "eos", "must be nPhase"),
         OPT(ablate::parameters::Parameters, "parameters", "the parameters used by advection: cfl(.5)"), 
         ARG(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculatorNStiff", ""));
