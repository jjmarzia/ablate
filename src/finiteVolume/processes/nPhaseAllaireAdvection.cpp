#include "nPhaseAllaireAdvection.hpp"

#include <utility>
#include "eos/perfectGas.hpp"
#include "eos/stiffenedGas.hpp"
#include "eos/twoPhase.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
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
 

ablate::finiteVolume::processes::TwoPhaseEulerAdvection::TwoPhaseEulerAdvection(std::shared_ptr<eos::EOS> eosTwoPhase, const std::shared_ptr<parameters::Parameters> &parametersIn,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasGas,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasLiquid,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidGas,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidLiquid)
    : eosTwoPhase(std::move(eosTwoPhase)),
      fluxCalculatorGasGas(std::move(fluxCalculatorGasGas)),
      fluxCalculatorGasLiquid(std::move(fluxCalculatorGasLiquid)),
      fluxCalculatorLiquidGas(std::move(fluxCalculatorLiquidGas)),
      fluxCalculatorLiquidLiquid(std::move(fluxCalculatorLiquidLiquid)) {
    auto parameters = ablate::parameters::EmptyParameters::Check(parametersIn);
    // check that eos is twoPhase
    if (this->eosTwoPhase) {
        auto twoPhaseEOS = std::dynamic_pointer_cast<eos::TwoPhase>(this->eosTwoPhase);
        // populate component eoses
        if (twoPhaseEOS) {
            eosGas = twoPhaseEOS->GetEOSGas();
            eosLiquid = twoPhaseEOS->GetEOSLiquid();
        } else {
            throw std::invalid_argument("invalid EOS. twoPhaseEulerAdvection requires TwoPhase equation of state.");
        }
    }

    // If there is a flux calculator assumed advection
    if (this->fluxCalculatorGasGas) {
        // cfl
        timeStepData.cfl = parameters->Get<PetscReal>("cfl", 0.5);
    }
}


ablate::finiteVolume::processes::TwoPhaseEulerAdvection::~TwoPhaseEulerAdvection() {

}
void ComputeFieldGradientDM(ablate::finiteVolume::FiniteVolumeSolver &flow, Vec faceGeomVec, Vec cellGeomVec, const std::string fieldName, DM *gradDM) {

  ablate::domain::Field field = flow.GetSubDomain().GetField(fieldName);
  PetscObject petscField = flow.GetSubDomain().GetPetscFieldObject(field);
  PetscFV petscFieldFV = (PetscFV)petscField;

  DMLabel regionLabel = nullptr;
  PetscInt regionValue = PETSC_DECIDE;
  ablate::domain::Region::GetLabel(flow.GetRegion(), flow.GetSubDomain().GetDM(), regionLabel, regionValue);

  ComputeGradientFVM(flow.GetSubDomain().GetFieldDM(field), regionLabel, regionValue, petscFieldFV, faceGeomVec, cellGeomVec, gradDM) >> ablate::utilities::PetscUtilities::checkError;
}

void ablate::finiteVolume::processes::TwoPhaseEulerAdvection::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    // Before each step, compute the alpha
    auto multiphasePreStage = std::bind(&ablate::finiteVolume::processes::TwoPhaseEulerAdvection::MultiphaseFlowPreStage, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    flow.RegisterPreStage(multiphasePreStage);

    ablate::domain::SubDomain& subDomain = flow.GetSubDomain();

    // Create the decoder based upon the eoses
    decoder = CreateTwoPhaseDecoder(subDomain.GetDimensions(), eosGas, eosLiquid);

    // Currently, no option for species advection
//    flow.RegisterRHSFunction(CompressibleFlowCompleteFlux, this);
    flow.RegisterRHSFunction(CompressibleFlowComputeEulerFlux, this, CompressibleFlowFields::EULER_FIELD, {VOLUME_FRACTION_FIELD, DENSITY_VF_FIELD, CompressibleFlowFields::EULER_FIELD}, {});
    flow.RegisterRHSFunction(CompressibleFlowComputeVFFlux, this, DENSITY_VF_FIELD, {VOLUME_FRACTION_FIELD, DENSITY_VF_FIELD, CompressibleFlowFields::EULER_FIELD}, {});
    flow.RegisterComputeTimeStepFunction(ComputeCflTimeStep, &timeStepData, "cfl");
    timeStepData.computeSpeedOfSound = eosTwoPhase->GetThermodynamicFunction(eos::ThermodynamicProperty::SpeedOfSound, subDomain.GetFields());

    if (subDomain.ContainsField(CompressibleFlowFields::VELOCITY_FIELD) && (subDomain.GetField(CompressibleFlowFields::VELOCITY_FIELD).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(CompressibleFlowFields::VELOCITY_FIELD);
    }

    if (subDomain.ContainsField(CompressibleFlowFields::TEMPERATURE_FIELD) && (subDomain.GetField(CompressibleFlowFields::TEMPERATURE_FIELD).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(CompressibleFlowFields::TEMPERATURE_FIELD);
    }

    if (subDomain.ContainsField(CompressibleFlowFields::PRESSURE_FIELD) && (subDomain.GetField(CompressibleFlowFields::PRESSURE_FIELD).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(CompressibleFlowFields::PRESSURE_FIELD);
    }

    if (subDomain.ContainsField(CompressibleFlowFields::GASDENSITY_FIELD) && (subDomain.GetField(CompressibleFlowFields::GASDENSITY_FIELD).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(CompressibleFlowFields::GASDENSITY_FIELD);
    }

    //we also need liquid density and mixture energy in intsharp prestage, not sure if it needs to be here though
    if (subDomain.ContainsField(CompressibleFlowFields::LIQUIDDENSITY_FIELD) && (subDomain.GetField(CompressibleFlowFields::LIQUIDDENSITY_FIELD).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(CompressibleFlowFields::LIQUIDDENSITY_FIELD);
    }
    if (subDomain.ContainsField(CompressibleFlowFields::MIXTUREENERGY_FIELD) && (subDomain.GetField(CompressibleFlowFields::MIXTUREENERGY_FIELD).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(CompressibleFlowFields::MIXTUREENERGY_FIELD);
    }

    if (subDomain.ContainsField(CompressibleFlowFields::GASENERGY_FIELD) && (subDomain.GetField(CompressibleFlowFields::GASENERGY_FIELD).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(CompressibleFlowFields::GASENERGY_FIELD);
    }

    if (subDomain.ContainsField(CompressibleFlowFields::LIQUIDENERGY_FIELD) && (subDomain.GetField(CompressibleFlowFields::LIQUIDENERGY_FIELD).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(CompressibleFlowFields::LIQUIDENERGY_FIELD);
    }

    // There's more work that needs to be done before VOLUME_FRACTION_FIELD can be in the AUX field.
    if (subDomain.ContainsField(VOLUME_FRACTION_FIELD) && (subDomain.GetField(VOLUME_FRACTION_FIELD).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(VOLUME_FRACTION_FIELD);
    }

    if (auxUpdateFields.size() > 0) {
      flow.RegisterAuxFieldUpdate(
            UpdateAuxFieldsTwoPhase, this, auxUpdateFields, {VOLUME_FRACTION_FIELD, DENSITY_VF_FIELD, CompressibleFlowFields::EULER_FIELD});
    }

    //initialize intsharp instance in setup;
    // we will then compute the intsharp term in the prestage and iteratively add it to the volume fraction
    // until a sufficient volume fraction gradient/sharpness is achieved
    // std::shared_ptr<ablate::finiteVolume::processes::IntSharp> intSharpProcess;
    // auto intSharpProcess = std::make_shared<ablate::finiteVolume::processes::IntSharp>(0, 0.001, false);
    // intSharpProcess->Initialize(flow);



}
#include <signal.h>
// Update the volume fraction, velocity, temperature, pressure fields, and gas density fields (if they exist).
PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::UpdateAuxFieldsTwoPhase(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[],
                                                                                                   const PetscScalar *conservedValues, const PetscInt aOff[], PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;

    if (!auxField) PetscFunctionReturn(0);

    auto twoPhaseEulerAdvection = (TwoPhaseEulerAdvection *)ctx;

    // For cell center, the norm is unity
    PetscReal norm[3];
    norm[0] = 1;
    norm[1] = 1;
    norm[2] = 1;

    PetscReal density = 1.0;
    PetscReal densityG = 0.0;
    PetscReal densityL = 0.0;
    PetscReal normalVelocity = 0.0;  // uniform velocity in cell
    PetscReal velocity[3] = {0.0, 0.0, 0.0};
    PetscReal internalEnergy = 0.0;
    PetscReal internalEnergyG = 0.0;
    PetscReal internalEnergyL = 0.0;
    PetscReal aG = 0.0;
    PetscReal aL = 0.0;
    PetscReal MG = 0.0;
    PetscReal ML = 0.0;
    PetscReal p = 0.0;  // pressure equilibrium
    PetscReal T = 0.0;  // temperature equilibrium, Tg = TL
    PetscReal alpha = 0.0;

    if (conservedValues) {
      twoPhaseEulerAdvection->decoder->DecodeTwoPhaseEulerState(
        dim, uOff, conservedValues, norm, &density, &densityG, &densityL, &normalVelocity, velocity, &internalEnergy, &internalEnergyG, &internalEnergyL, &aG, &aL, &MG, &ML, &p, &T, &alpha);

      density = conservedValues[CompressibleFlowFields::RHO];
      for (PetscInt d = 0; d < dim; d++) velocity[d] = conservedValues[CompressibleFlowFields::RHOU + d] / density;

    }

    auto fields = twoPhaseEulerAdvection->auxUpdateFields.data();

    for (std::size_t f = 0; f < twoPhaseEulerAdvection->auxUpdateFields.size(); ++f) {
      if (fields[f] == CompressibleFlowFields::VELOCITY_FIELD) {
        for (PetscInt d = 0; d < dim; d++) {
          auxField[aOff[f] + d] = velocity[d];
        }
      }
      else if (fields[f] == CompressibleFlowFields::TEMPERATURE_FIELD) {
        auxField[aOff[f]] = T;
      }
      else if (fields[f] == CompressibleFlowFields::PRESSURE_FIELD) {
        auxField[aOff[f]] = p;
      }
      else if (fields[f] == CompressibleFlowFields::GASDENSITY_FIELD) {
        auxField[aOff[f]] = densityG;
      }
      else if (fields[f] == CompressibleFlowFields::LIQUIDDENSITY_FIELD) {
        auxField[aOff[f]] = densityL;
      }
      else if (fields[f] == CompressibleFlowFields::MIXTUREENERGY_FIELD) {
        auxField[aOff[f]] = internalEnergy;
      }
      else if (fields[f] == CompressibleFlowFields::GASENERGY_FIELD) {
        auxField[aOff[f]] = internalEnergyG;
      }
      else if (fields[f] == CompressibleFlowFields::LIQUIDENERGY_FIELD) {
        auxField[aOff[f]] = internalEnergyL;
      }
      else if (fields[f] == VOLUME_FRACTION_FIELD) { // In case it's ever moved to the AUX vector
        auxField[aOff[f]] = alpha;
      }
    }

    PetscFunctionReturn(0);
}
#include <iostream>
PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::MultiphaseFlowPreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime) {
    PetscFunctionBegin;
    // Get flow field data
    const auto &fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver &>(solver);
    ablate::domain::Range cellRange;
    fvSolver.GetCellRangeWithoutGhost(cellRange);
    PetscInt dim;
    PetscCall(DMGetDimension(fvSolver.GetSubDomain().GetDM(), &dim));
    const auto &eulerOffset = fvSolver.GetSubDomain().GetField(CompressibleFlowFields::EULER_FIELD).offset;  // need this to get uOff
    const auto &vfOffset = fvSolver.GetSubDomain().GetField(VOLUME_FRACTION_FIELD).offset;
    const auto &rhoAlphaOffset = fvSolver.GetSubDomain().GetField(DENSITY_VF_FIELD).offset;

    DM dm = fvSolver.GetSubDomain().GetDM();
    Vec globFlowVec;
    PetscCall(TSGetSolution(flowTs, &globFlowVec));

    PetscScalar *flowArray;
    PetscCall(VecGetArray(globFlowVec, &flowArray));

    PetscInt uOff[3];
    uOff[0] = vfOffset;
    uOff[1] = rhoAlphaOffset;
    uOff[2] = eulerOffset;

    //get the rhs vector
    Vec locFVec;
    PetscCall(DMGetLocalVector(dm, &locFVec));
    PetscCall(VecZeroEntries(locFVec));
    

    //new stuff
    

    //compute the term for all cells
    // auto intSharpProcess = std::make_shared<ablate::finiteVolume::processes::IntSharp>(0, 0.001, false);
    // std::cout << "Debug: intSharpProcess created" << std::endl; // this is successful
    // intSharpProcess->ComputeTerm(fvSolver, dm, stagetime, globFlowVec, locFVec, intSharpProcess.get()); // const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx
    // std::cout << "Debug: intSharpProcess->ComputeTerm called" << std::endl; // this is not successful

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
        PetscReal densityG, densityL, normalVelocity, internalEnergy, internalEnergyG, internalEnergyL, aG, aL, MG, ML, p, t, alpha;
        decoder->DecodeTwoPhaseEulerState(
            dim, uOff, allFields, norm, &density, &densityG, &densityL, &normalVelocity, velocity, &internalEnergy, &internalEnergyG, &internalEnergyL, &aG, &aL, &MG, &ML, &p, &t, &alpha);
        // maybe save other values for use later, would interpolation to the face be the same as calculating at face?

        allFields[uOff[0]] = alpha;  // sets volumeFraction field, does every iteration of time step (euler=1, rk=4)


    }

    //restore
    PetscCall(DMRestoreLocalVector(dm, &locFVec));
    PetscCall(VecRestoreArray(globFlowVec, &flowArray));

    // clean up
    fvSolver.RestoreRange(cellRange);
    PetscFunctionReturn(0);

}
double ablate::finiteVolume::processes::TwoPhaseEulerAdvection::ComputeCflTimeStep(TS ts, ablate::finiteVolume::FiniteVolumeSolver &flow, void *ctx) {
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
    auto eulerId = flow.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD).id;

    // March over each cell
    PetscReal dtMin = 1000.0;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.points ? cellRange.points[c] : c;

        const PetscReal *euler;
        const PetscReal *conserved = NULL;
        DMPlexPointGlobalFieldRead(dm, cell, eulerId, x, &euler) >> utilities::PetscUtilities::checkError;
        DMPlexPointGlobalRead(dm, cell, x, &conserved) >> utilities::PetscUtilities::checkError;

        if (euler) {  // must be real cell and not ghost
            PetscReal rho = euler[CompressibleFlowFields::RHO];

            // Get the speed of sound from the eos
            PetscReal a;
            timeStepData->computeSpeedOfSound.function(conserved, &a, timeStepData->computeSpeedOfSound.context.get()) >> utilities::PetscUtilities::checkError;

            PetscReal velSum = 0.0;
            for (PetscInt d = 0; d < dim; d++) {
                velSum += PetscAbsReal(euler[CompressibleFlowFields::RHOU + d]) / rho;
            }
            PetscReal dt = timeStepData->cfl * dx / (a + velSum);

            dtMin = PetscMin(dtMin, dt);
        }
    }
    VecRestoreArrayRead(v, &x) >> utilities::PetscUtilities::checkError;
    flow.RestoreRange(cellRange);
    return dtMin;
}
PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::CompressibleFlowCompleteFlux(const ablate::finiteVolume::FiniteVolumeSolver &flow, DM dm, PetscReal time, Vec locXVec, Vec locFVec, void* ctx) {

  PetscFunctionBeginUser;
//  auto flow = (ablate::finiteVolume::FiniteVolumeSolver *)ctx;
//  auto twoPhaseEulerAdvection = (TwoPhaseEulerAdvection *)ctx;
  NOTE0EXIT("");

  PetscFunctionReturn(0);

}
PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::CompressibleFlowComputeEulerFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscScalar *fieldL,
                                                                                                         const PetscScalar *fieldR, const PetscInt *aOff, const PetscScalar *auxL,
                                                                                                         const PetscScalar *auxR, PetscScalar *flux, void *ctx) {


    PetscFunctionBeginUser;
    auto twoPhaseEulerAdvection = (TwoPhaseEulerAdvection *)ctx;

    // Compute the norm of cell face
    PetscReal norm[3];
    NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = MagVector(dim, fg->normal);


    // Decode left and right states
    PetscReal densityL = 0.0;
    PetscReal densityG_L = 0.0;
    PetscReal densityL_L = 0.0;
    PetscReal normalVelocityL = 0.0;  // uniform velocity in cell
    PetscReal velocityL[3] = {0.0, 0.0, 0.0};
    PetscReal internalEnergyL = 0.0;
    PetscReal internalEnergyG_L = 0.0;
    PetscReal internalEnergyL_L = 0.0;
    PetscReal aG_L = 0.0;
    PetscReal aL_L = 0.0;
    PetscReal MG_L = 0.0;
    PetscReal ML_L = 0.0;
    PetscReal pL = 0.0;  // pressure equilibrium
    PetscReal tL = 0.0;
    PetscReal alphaL = 0.0;
    twoPhaseEulerAdvection->decoder->DecodeTwoPhaseEulerState(dim,
                                                              uOff,
                                                              fieldL,
                                                              norm,
                                                              &densityL,
                                                              &densityG_L,
                                                              &densityL_L,
                                                              &normalVelocityL,
                                                              velocityL,
                                                              &internalEnergyL,
                                                              &internalEnergyG_L,
                                                              &internalEnergyL_L,
                                                              &aG_L,
                                                              &aL_L,
                                                              &MG_L,
                                                              &ML_L,
                                                              &pL,
                                                              &tL,
                                                              &alphaL);

    PetscReal densityR = 0.0;
    PetscReal densityG_R = 0.0;
    PetscReal densityL_R = 0.0;
    PetscReal normalVelocityR = 0.0;
    PetscReal velocityR[3] = {0.0, 0.0, 0.0};
    PetscReal internalEnergyR = 0.0;
    PetscReal internalEnergyG_R = 0.0;
    PetscReal internalEnergyL_R = 0.0;
    PetscReal aG_R = 0.0;
    PetscReal aL_R = 0.0;
    PetscReal MG_R = 0.0;
    PetscReal ML_R = 0.0;
    PetscReal pR = 0.0;
    PetscReal tR = 0.0;
    PetscReal alphaR = 0.0;

    twoPhaseEulerAdvection->decoder->DecodeTwoPhaseEulerState(dim,
                                                              uOff,
                                                              fieldR,
                                                              norm,
                                                              &densityR,
                                                              &densityG_R,
                                                              &densityL_R,
                                                              &normalVelocityR,
                                                              velocityR,
                                                              &internalEnergyR,
                                                              &internalEnergyG_R,
                                                              &internalEnergyL_R,
                                                              &aG_R,
                                                              &aL_R,
                                                              &MG_R,
                                                              &ML_R,
                                                              &pR,
                                                              &tR,
                                                              &alphaR);


    // Blended speed of sound
//    const PetscInt aR = 1/(alphaR/aG_R + (1-alphaR)/aL_R);
//    const PetscInt aL = 1/(alphaL/aG_L + (1-alphaL)/aL_L);
    // These are from Eq. (23) of Change and Liou
    PetscReal aR = alphaR/(densityG_R*aG_R*aG_R) + (1 - alphaR)/(densityL_R*aL_R*aL_R);
    aR = PetscSqrtReal((alphaR/densityG_R + (1 - alphaR)/densityL_R)/aR);
    PetscReal aL = alphaL/(densityG_L*aG_L*aG_L) + (1 - alphaL)/(densityL_L*aL_L*aL_L);
    aL = PetscSqrtReal((alphaL/densityG_L + (1 - alphaL)/densityL_L)/aL);


    PetscReal massFlux = 0.0, p12 = 0.0;
    fluxCalculator::Direction direction = twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorFunction()(
        twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorContext(), normalVelocityL, aL, densityL, pL, normalVelocityR, aR, densityR, pR, &massFlux, &p12);

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

    flux[CompressibleFlowFields::RHO] = massFlux * areaMag;
    flux[CompressibleFlowFields::RHOE] = H * massFlux * areaMag;
    for (PetscInt n = 0; n < dim; n++) {
        flux[CompressibleFlowFields::RHOU + n] = vel[n] * areaMag * massFlux + p12 * fg->normal[n];
    }

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::CompressibleFlowComputeVFFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscScalar *fieldL,
                                                                                                      const PetscScalar *fieldR, const PetscInt *aOff, const PetscScalar *auxL, const PetscScalar *auxR,
                                                                                                      PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    auto twoPhaseEulerAdvection = (TwoPhaseEulerAdvection *)ctx;

    // Compute the norm
    PetscReal norm[3];
    NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = MagVector(dim, fg->normal);

    //     Decode left and right states
    PetscReal densityL;
    PetscReal densityG_L;
    PetscReal densityL_L;
    PetscReal normalVelocityL;  // uniform velocity in cell
    PetscReal velocityL[3];
    PetscReal internalEnergyL;
    PetscReal internalEnergyG_L;
    PetscReal internalEnergyL_L;
    PetscReal aG_L;
    PetscReal aL_L;
    PetscReal MG_L;
    PetscReal ML_L;
    PetscReal pL;  // pressure equilibrium
    PetscReal tL;
    PetscReal alphaL;
    twoPhaseEulerAdvection->decoder->DecodeTwoPhaseEulerState(dim,
                                                              uOff,
                                                              fieldL,
                                                              norm,
                                                              &densityL,
                                                              &densityG_L,
                                                              &densityL_L,
                                                              &normalVelocityL,
                                                              velocityL,
                                                              &internalEnergyL,
                                                              &internalEnergyG_L,
                                                              &internalEnergyL_L,
                                                              &aG_L,
                                                              &aL_L,
                                                              &MG_L,
                                                              &ML_L,
                                                              &pL,
                                                              &tL,
                                                              &alphaL);

    PetscReal densityR;
    PetscReal densityG_R;
    PetscReal densityL_R;
    PetscReal normalVelocityR;
    PetscReal velocityR[3];
    PetscReal internalEnergyR;
    PetscReal internalEnergyG_R;
    PetscReal internalEnergyL_R;
    PetscReal aG_R;
    PetscReal aL_R;
    PetscReal MG_R;
    PetscReal ML_R;
    PetscReal pR;
    PetscReal tR;
    PetscReal alphaR;
    twoPhaseEulerAdvection->decoder->DecodeTwoPhaseEulerState(dim,
                                                              uOff,
                                                              fieldR,
                                                              norm,
                                                              &densityR,
                                                              &densityG_R,
                                                              &densityL_R,
                                                              &normalVelocityR,
                                                              velocityR,
                                                              &internalEnergyR,
                                                              &internalEnergyG_R,
                                                              &internalEnergyL_R,
                                                              &aG_R,
                                                              &aL_R,
                                                              &MG_R,
                                                              &ML_R,
                                                              &pR,
                                                              &tR,
                                                              &alphaR);

    // get the face values
    PetscReal massFlux;
    PetscReal p12;
    // calculate gas sub-area of face (stratified flow model)
    fluxCalculator::Direction directionG = twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorFunction()(
        twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorContext(), normalVelocityL, aG_L, densityG_L, pL, normalVelocityR, aG_R, densityG_R, pR, &massFlux, &p12);

    if (directionG == fluxCalculator::LEFT) {
        flux[0] = massFlux * areaMag * alphaL;
    } else if (directionG == fluxCalculator::RIGHT) {
        flux[0] = massFlux * areaMag * alphaR;
    } else {
        flux[0] = massFlux * areaMag * 0.5 * (alphaL + alphaR);
    }

    PetscFunctionReturn(0);
}

std::shared_ptr<ablate::finiteVolume::processes::TwoPhaseEulerAdvection::TwoPhaseDecoder> ablate::finiteVolume::processes::TwoPhaseEulerAdvection::CreateTwoPhaseDecoder(
    PetscInt dim, const std::shared_ptr<eos::EOS> &eosGas, const std::shared_ptr<eos::EOS> &eosLiquid) {
    // check if both perfect gases, use analytical solution
    auto perfectGasEos1 = std::dynamic_pointer_cast<eos::PerfectGas>(eosGas);
    auto perfectGasEos2 = std::dynamic_pointer_cast<eos::PerfectGas>(eosLiquid);
    // check if stiffened gas
    auto stiffenedGasEos1 = std::dynamic_pointer_cast<eos::StiffenedGas>(eosGas);
    auto stiffenedGasEos2 = std::dynamic_pointer_cast<eos::StiffenedGas>(eosLiquid);

    if (perfectGasEos1 && perfectGasEos2) {
        return std::make_shared<PerfectGasPerfectGasDecoder>(dim, perfectGasEos1, perfectGasEos2);
    } else if (perfectGasEos1 && stiffenedGasEos2) {
        return std::make_shared<PerfectGasStiffenedGasDecoder>(dim, perfectGasEos1, stiffenedGasEos2);
    } else if (perfectGasEos2 && stiffenedGasEos1) {
        return std::make_shared<PerfectGasStiffenedGasDecoder>(dim, perfectGasEos2, stiffenedGasEos1);
    } else if (stiffenedGasEos1 && stiffenedGasEos2) {
        return std::make_shared<StiffenedGasStiffenedGasDecoder>(dim, stiffenedGasEos1, stiffenedGasEos2);
    }
    throw std::invalid_argument("Unknown combination of equation of states for ablate::finiteVolume::processes::TwoPhaseEulerAdvection::TwoPhaseDecoder");
}

/**PerfectGasPerfectGasDecoder**************/

/**PerfectGasStiffenedGasDecoder**************/

#include <signal.h>

// Roots for a*x*x + b*x + c
void SolveQuadratic(const PetscReal a, const PetscReal b, const PetscReal c, PetscReal *x1, PetscReal *x2) {
  if (PetscAbsReal(a) < PETSC_SMALL) {
    *x1 = *x2 = -c/b;
  }
  else if (PetscAbsReal(c) < PETSC_SMALL) {
    *x1 = 0.0;
    *x2 = -b/a;
  }
  else {
    PetscReal disc = PetscSqrtReal(b*b - 4.0*a*c);
    if (b > 0) {
      *x1 = 0.5*(-b - disc)/a;
      *x2 = 2.0*c/(-b - disc);
    }
    else {
      *x1 = 2.0*c/(-b + disc);
      *x2 = 0.5*(-b + disc)/a;
    }


  }
}
PetscReal Heaviside(const PetscReal x, const PetscReal x0, const PetscReal e) {
  if (x < x0) {
    return 0.0;
  }
  else if (x - x0 > e) {
    return 1.0;
  }
  else {
    PetscReal x_x0 = x - x0;
    return x_x0*x_x0*x_x0*(10.0*e*e + 6.0*x_x0*x_x0 - 15.0*e*x_x0)/(e*e*e*e*e);
  }
}
//static PetscInt cnt = 0;


/**StiffenedGasStiffenedGasDecoder**************/
ablate::finiteVolume::processes::TwoPhaseEulerAdvection::StiffenedGasStiffenedGasDecoder::StiffenedGasStiffenedGasDecoder(PetscInt dim, const std::shared_ptr<eos::StiffenedGas> &eosGas,
                                                                                                                          const std::shared_ptr<eos::StiffenedGas> &eosLiquid)
    : eosGas(eosGas), eosLiquid(eosLiquid) {
    // Create the fake euler field
    auto fakeEulerField = ablate::domain::Field{.name = CompressibleFlowFields::EULER_FIELD,
                                                .numberComponents = 2 + dim,
                                                .components = {},
                                                .id = PETSC_DEFAULT,
                                                .subId = PETSC_DEFAULT,
                                                .offset = 0,
                                                .location = ablate::domain::FieldLocation::SOL,
                                                .type = ablate::domain::FieldType::FVM,
                                                .tags = {}};

    // size up the scratch vars
    gasEulerFieldScratch.resize(2 + dim);
    liquidEulerFieldScratch.resize(2 + dim);

    // extract/store compute calls
    gasComputeTemperature = eosGas->GetThermodynamicFunction(eos::ThermodynamicProperty::Temperature, {fakeEulerField});
    gasComputeInternalEnergy = eosGas->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::InternalSensibleEnergy, {fakeEulerField});
    gasComputeSpeedOfSound = eosGas->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SpeedOfSound, {fakeEulerField});
    gasComputePressure = eosGas->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Pressure, {fakeEulerField});

    liquidComputeTemperature = eosLiquid->GetThermodynamicFunction(eos::ThermodynamicProperty::Temperature, {fakeEulerField});
    liquidComputeInternalEnergy = eosLiquid->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::InternalSensibleEnergy, {fakeEulerField});
    liquidComputeSpeedOfSound = eosLiquid->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SpeedOfSound, {fakeEulerField});
    liquidComputePressure = eosLiquid->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Pressure, {fakeEulerField});
}

void ablate::finiteVolume::processes::TwoPhaseEulerAdvection::StiffenedGasStiffenedGasDecoder::DecodeTwoPhaseEulerState(PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues,
                                                                                                                        const PetscReal *normal, PetscReal *density, PetscReal *densityG,
                                                                                                                        PetscReal *densityL, PetscReal *normalVelocity, PetscReal *velocity,
                                                                                                                        PetscReal *internalEnergy, PetscReal *internalEnergyG,
                                                                                                                        PetscReal *internalEnergyL, PetscReal *aG, PetscReal *aL, PetscReal *MG,
                                                                                                                        PetscReal *ML, PetscReal *p, PetscReal *T, PetscReal *alpha) {
    const int EULER_FIELD = 2;
    const int VF_FIELD = 1;

    // decode
    *density = conservedValues[CompressibleFlowFields::RHO + uOff[EULER_FIELD]];
    PetscReal totalEnergy = conservedValues[CompressibleFlowFields::RHOE + uOff[EULER_FIELD]] / (*density);
    PetscReal densityVF = conservedValues[uOff[VF_FIELD]];


    if (*density < PETSC_SMALL) { // This occurs when a cell hasn't been initialized yet. Usually FVM boundary cells
        *densityG = 0.0;
        *densityL = 0.0;
        *internalEnergyG = 0.0;
        *internalEnergyL = 0.0;
        *alpha = 0.0;
        *p = 0.0;
        *aG = 0.0;
        *aL = 0.0;
        *MG = 0.0;
        *ML = 0.0;

        return;
    }


    // Get the velocity in this direction, and kinetic energy
    (*normalVelocity) = 0.0;
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        velocity[d] = conservedValues[CompressibleFlowFields::RHOU + d + uOff[EULER_FIELD]] / (*density);
        (*normalVelocity) += velocity[d] * normal[d];
        ke += PetscSqr(velocity[d]);
    }
    ke *= 0.5;
    (*internalEnergy) = (totalEnergy)-ke;

    PetscReal cp1 = eosGas->GetSpecificHeatCp();
    PetscReal cp2 = eosLiquid->GetSpecificHeatCp();
    PetscReal p01 = eosGas->GetReferencePressure();
    PetscReal p02 = eosLiquid->GetReferencePressure();
    PetscReal gamma1 = eosGas->GetSpecificHeatRatio();
    PetscReal gamma2 = eosLiquid->GetSpecificHeatRatio();

    DecodeDataStructStiff decodeDataStruct{
        .etot = (*internalEnergy),
        .rhotot = (*density),
        .Yg = densityVF / (*density),
        .Yl = ((*density) - densityVF) / (*density),
        .gam1 = gamma1,
        .gam2 = gamma2,
        .cpg = cp1,
        .cpl = cp2,
        .p0g = p01,
        .p0l = p02,
    };

    PetscReal rhoG = NAN, rhoL = NAN, eG = NAN, eL = NAN;

    if (decodeDataStruct.Yg < PETSC_SMALL || decodeDataStruct.Yl < PETSC_SMALL) {
        rhoL = decodeDataStruct.rhotot;
        eL = decodeDataStruct.etot;
        rhoG = decodeDataStruct.rhotot;
        eG = decodeDataStruct.etot;
    }
    else {
      SNES snes;
      Vec x, r;
      Mat J;
      VecCreate(PETSC_COMM_SELF, &x) >> utilities::PetscUtilities::checkError;
      VecSetSizes(x, PETSC_DECIDE, 4) >> utilities::PetscUtilities::checkError;
      VecSetFromOptions(x) >> utilities::PetscUtilities::checkError;

      // Set the initial guess to the conserved energy and the internal energy
      PetscScalar *ax;
      VecGetArray(x, &ax) >> utilities::PetscUtilities::checkError;
      ax[0] = decodeDataStruct.rhotot; // rho 1
      ax[1] = decodeDataStruct.rhotot; // rho 2
      ax[2] = decodeDataStruct.etot;   // e1
      ax[3] = decodeDataStruct.etot;   // e2
      VecRestoreArray(x, &ax) >> utilities::PetscUtilities::checkError;


      VecDuplicate(x, &r) >> utilities::PetscUtilities::checkError;

      MatCreate(PETSC_COMM_SELF, &J) >> utilities::PetscUtilities::checkError;
      MatSetSizes(J, 4, 4, 4, 4) >> utilities::PetscUtilities::checkError;
      MatSetType(J, MATDENSE) >> utilities::PetscUtilities::checkError; // The KSP fails is this isn't a dense matrix
      MatSetFromOptions(J) >> utilities::PetscUtilities::checkError;
      MatSetUp(J) >> utilities::PetscUtilities::checkError;

      SNESCreate(PETSC_COMM_SELF, &snes) >> utilities::PetscUtilities::checkError;
      SNESSetOptionsPrefix(snes, "gasSolver_");
      SNESSetFunction(snes, r, FormFunctionStiff, &decodeDataStruct) >> utilities::PetscUtilities::checkError;
      SNESSetJacobian(snes, J, J, FormJacobianStiff, &decodeDataStruct) >> utilities::PetscUtilities::checkError;
      SNESSetTolerances(snes, 1E-14, 1E-10, 1E-10, 1000, 10000) >> utilities::PetscUtilities::checkError;  // refine relative tolerance for more accurate pressure value
      SNESSetFromOptions(snes) >> utilities::PetscUtilities::checkError;
      SNESSolve(snes, NULL, x) >> utilities::PetscUtilities::checkError;

      SNESConvergedReason reason;
      SNESGetConvergedReason(snes, &reason) >> utilities::PetscUtilities::checkError;

      if (reason < 0 || reason == SNES_CONVERGED_ITS) {
        throw std::runtime_error("SNES for stiffened gas-stiffened gas decode failed.\n");
      }

      VecGetArray(x, &ax) >> utilities::PetscUtilities::checkError;
      rhoG = ax[0];
      rhoL = ax[1];
      eG   = ax[2];
      eL   = ax[3];
      VecRestoreArray(x, &ax) >> utilities::PetscUtilities::checkError;

      SNESDestroy(&snes) >> utilities::PetscUtilities::checkError;
      VecDestroy(&x) >> utilities::PetscUtilities::checkError;
      VecDestroy(&r) >> utilities::PetscUtilities::checkError;
      MatDestroy(&J) >> utilities::PetscUtilities::checkError;
    }

    PetscReal etG = eG + ke;
    PetscReal etL = eL + ke;

    PetscReal pG = 0;
    PetscReal pL = 0;
    PetscReal a1 = 0;
    PetscReal a2 = 0;

    // Fill the scratch array for gas
    liquidEulerFieldScratch[CompressibleFlowFields::RHO] = rhoL;
    liquidEulerFieldScratch[CompressibleFlowFields::RHOE] = rhoL * etL;
    for (PetscInt d = 0; d < dim; d++) {
        liquidEulerFieldScratch[CompressibleFlowFields::RHOU + d] = velocity[d] * rhoL;
    }

    // Fill the scratch array for gas
    gasEulerFieldScratch[CompressibleFlowFields::RHO] = rhoG;
    gasEulerFieldScratch[CompressibleFlowFields::RHOE] = rhoG * etG;
    for (PetscInt d = 0; d < dim; d++) {
        gasEulerFieldScratch[CompressibleFlowFields::RHOU + d] = velocity[d] * rhoG;
    }


    PetscReal TL = 0, TG = 0;
    liquidComputeTemperature.function(liquidEulerFieldScratch.data(), &TL, liquidComputeTemperature.context.get()) >> utilities::PetscUtilities::checkError;
    gasComputeTemperature.function(gasEulerFieldScratch.data(), &TG, gasComputeTemperature.context.get()) >> utilities::PetscUtilities::checkError;
    *T = 0.5*(TL + TG);

    // Decode the gas
    {
//        liquidComputeTemperature.function(liquidEulerFieldScratch.data(), T, liquidComputeTemperature.context.get()) >> utilities::PetscUtilities::checkError;
        liquidComputeInternalEnergy.function(liquidEulerFieldScratch.data(), *T, &eL, liquidComputeInternalEnergy.context.get()) >> utilities::PetscUtilities::checkError;
        liquidComputeSpeedOfSound.function(liquidEulerFieldScratch.data(), *T, &a2, liquidComputeSpeedOfSound.context.get()) >> utilities::PetscUtilities::checkError;
        liquidComputePressure.function(liquidEulerFieldScratch.data(), *T, &pL, liquidComputePressure.context.get()) >> utilities::PetscUtilities::checkError;
    }

    // Decode the gas
    {
//        gasComputeTemperature.function(gasEulerFieldScratch.data(), T, gasComputeTemperature.context.get()) >> utilities::PetscUtilities::checkError;
        gasComputeInternalEnergy.function(gasEulerFieldScratch.data(), *T, &eG, gasComputeInternalEnergy.context.get()) >> utilities::PetscUtilities::checkError;
        gasComputeSpeedOfSound.function(gasEulerFieldScratch.data(), *T, &a1, gasComputeSpeedOfSound.context.get()) >> utilities::PetscUtilities::checkError;
        gasComputePressure.function(gasEulerFieldScratch.data(), *T, &pG, gasComputePressure.context.get()) >> utilities::PetscUtilities::checkError;
    }

// if (PetscAbsReal(pL - pG) > PetscMin(pG, pL)*PETSC_SMALL) printf("%e\t%e\t%e\n", pL, pG, PetscAbsReal(pL-pG));

    // once state defined
    *densityG = rhoG;
    *densityL = rhoL;
    *internalEnergyG = eG;
    *internalEnergyL = eL;
    *alpha = densityVF / (*densityG);
    *p = 0.5*(pL+pG);  // pressure equilibrium, pG = pL
    *aG = a1;
    *aL = a2;
    *MG = (*normalVelocity) / (*aG);
    *ML = (*normalVelocity) / (*aL);

}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::TwoPhaseEulerAdvection, "", ARG(ablate::eos::EOS, "eos", "must be twoPhase"),
         OPT(ablate::parameters::Parameters, "parameters", "the parameters used by advection: cfl(.5)"), ARG(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculatorGasGas", ""),
         ARG(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculatorGasLiquid", ""), ARG(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculatorLiquidGas", ""),
         ARG(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculatorLiquidLiquid", ""));
