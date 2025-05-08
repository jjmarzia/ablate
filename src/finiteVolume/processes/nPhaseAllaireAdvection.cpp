#include "nPhaseAllaireAdvection.hpp"

#include <utility>
#include "eos/stiffenedGas.hpp"
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
    if (this->eosNPhase) {
        auto nPhaseEOS = std::dynamic_pointer_cast<eos::NPhase>(this->eosNPhase);
        // populate component eoses
        if (nPhaseEOS) {
          for (std::size_t k=0; k<eosk.size(); k++) {
            eosk[k] = nPhaseEOS->GetEOSk(k); //might not make sense given the eosk size might not be defined yet
          }
        } else {
            throw std::invalid_argument("invalid EOS. nPhaseAllaireAdvection requires NPhase equation of state.");
        }
    }

    // If there is a flux calculator assumed advection
    if (this->fluxCalculatorNStiff) {
        // cfl
        timeStepData.cfl = parameters->Get<PetscReal>("cfl", 0.5);
    }
}


ablate::finiteVolume::processes::NPhaseAllaireAdvection::~NPhaseAllaireAdvection() {
    // Destructor implementation
}

void ablate::finiteVolume::processes::NPhaseAllaireAdvection::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    // Before each step, compute the alpha
    auto multiphasePreStage = std::bind(&ablate::finiteVolume::processes::NPhaseAllaireAdvection::MultiphaseFlowPreStage, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    flow.RegisterPreStage(multiphasePreStage);

    ablate::domain::SubDomain& subDomain = flow.GetSubDomain();

    // Create the decoder based upon the eoses
    decoder = CreateNPhaseDecoder(subDomain.GetDimensions(), eosk);

    // Currently, no option for species advection
//    flow.RegisterRHSFunction(CompressibleFlowCompleteFlux, this);
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



}
#include <signal.h>
// Update the volume fraction, velocity, temperature, pressure fields, and gas density fields (if they exist).
PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvection::UpdateAuxFieldsNPhase(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[],
                                                                                                   const PetscScalar *conservedValues, const PetscInt aOff[], PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;

    if (!auxField) PetscFunctionReturn(0);

    auto nPhaseAllaireAdvection = (NPhaseAllaireAdvection *)ctx;

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

      nPhaseAllaireAdvection->decoder->DecodeNPhaseAllaireState(
          dim, uOff, conservedValues, norm, &density, &densityk, &normalVelocity, velocity, &internalEnergy, &internalEnergyk, &ak,
          &Mk, &p, &Tk, &alphak);

      // //rho = sumk alphakrhok; why is this needed if density returns in the decoder?
      // density = 0.0;
      // for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
      //   density += conservedValues[NPhaseFlowFields::ALPHAKRHOK + k]; //alphakrhok is not yet enumerated in the way ALLAIRE_FIELD->RHOU is
      // }

      for (PetscInt d = 0; d < dim; d++) velocity[d] = conservedValues[NPhaseFlowFields::RHOU + d] / density;
    }

    auto fields = nPhaseAllaireAdvection->auxUpdateFields.data();

    for (std::size_t f = 0; f < nPhaseAllaireAdvection->auxUpdateFields.size(); ++f) {
      if (fields[f] == CompressibleFlowFields::VELOCITY_FIELD) {
        for (PetscInt d = 0; d < dim; d++) {
          auxField[aOff[f] + d] = velocity[d];
        }
      }
      else if (fields[f] == NPhaseFlowFields::PRESSURE) {
        auxField[aOff[f]] = p;
      }
      //do for all fields
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
      else if (fields[f] == NPhaseFlowFields::ALPHAKRHOK) { //not in aux but just in case
        for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
          auxField[aOff[f] + k] = densityk[k];
        }
      }
      else if (fields[f] == NPhaseFlowFields::ALPHAK) { //not in aux but just in case
        for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
          auxField[aOff[f] + k] = ak[k];
        }
      }
    }

    PetscFunctionReturn(0);
}
#include <iostream>


PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvection::MultiphaseFlowPreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime) {
    PetscFunctionBegin;
    // Get flow field data
    const auto &fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver &>(solver);
    ablate::domain::Range cellRange;
    fvSolver.GetCellRangeWithoutGhost(cellRange);
    PetscInt dim;
    PetscCall(DMGetDimension(fvSolver.GetSubDomain().GetDM(), &dim));

    const auto &allaireOffset = fvSolver.GetSubDomain().GetField(NPhaseFlowFields::ALLAIRE_FIELD).offset;
    const auto &alphakOffset = fvSolver.GetSubDomain().GetField(ALPHAK).offset;
    const auto &alphakRhokOffset = fvSolver.GetSubDomain().GetField(ALPHAKRHOK).offset;

    DM dm = fvSolver.GetSubDomain().GetDM();
    Vec globFlowVec;
    PetscCall(TSGetSolution(flowTs, &globFlowVec));

    PetscScalar *flowArray;
    PetscCall(VecGetArray(globFlowVec, &flowArray));

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
    auto nPhaseAllaireAdvection = (NPhaseAllaireAdvection *)ctx;

    // Compute the norm of cell face
    PetscReal norm[3];
    NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = MagVector(dim, fg->normal);


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

      // a_L = 1/ (sumk alphak_L / ak_L)
      // a_R = 1/ (sumk alphak_R / ak_R)
    PetscReal aL=0.0;
    for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
      aL += alphak_L[k] / ak_L[k];
    }
    aL = 1.0 / aL;
    PetscReal aR=0.0;
    for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
      aR += alphak_R[k] / ak_R[k];
    }
    aR = 1.0 / aR;

    PetscReal massFlux = 0.0, p12 = 0.0;
    fluxCalculator::Direction direction = nPhaseAllaireAdvection->fluxCalculatorNStiff->GetFluxCalculatorFunction()(
      nPhaseAllaireAdvection->fluxCalculatorNStiff->GetFluxCalculatorContext(), normalVelocityL, aL, densityL, pL, normalVelocityR, aR, densityR, pR, &massFlux, &p12);

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

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvection::NPhaseFlowComputeAlphakRhokFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscScalar *fieldL,
                                                                                                      const PetscScalar *fieldR, const PetscInt *aOff, const PetscScalar *auxL, const PetscScalar *auxR,
                                                                                                      PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    auto nPhaseAllaireAdvection = (NPhaseAllaireAdvection *)ctx;

    // Compute the norm
    PetscReal norm[3];
    NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = MagVector(dim, fg->normal);

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

    //flux[0] is flawed, it should be NPHASEFLOWFIELDS::ALPHAKRHOK(k) or variant
    for (size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
        fluxCalculator::Direction directionk = nPhaseAllaireAdvection->fluxCalculatorNStiff->GetFluxCalculatorFunction()(
            nPhaseAllaireAdvection->fluxCalculatorNStiff->GetFluxCalculatorContext(), normalVelocityL, ak_L[k], densityk_L[k], pL, normalVelocityR, ak_R[k], densityk_R[k], pR, &massFlux, &p12);
        if (directionk == fluxCalculator::LEFT) {
            flux[0] = massFlux * areaMag * alphak_L[k];
        } else if (directionk == fluxCalculator::RIGHT) {
            flux[0] = massFlux * areaMag * alphak_R[k];
        } else {
            flux[0] = massFlux * areaMag * 0.5 * (alphak_L[k] + alphak_R[k]);
        }
    }

    PetscFunctionReturn(0);
}

//need a PetscErrorCode NPhaseFlowComputeAlphakFlux 

std::shared_ptr<ablate::finiteVolume::processes::NPhaseAllaireAdvection::NPhaseDecoder> ablate::finiteVolume::processes::NPhaseAllaireAdvection::CreateNPhaseDecoder(
    PetscInt dim, const std::vector<std::shared_ptr<eos::EOS>> &eosk) {

    // return std::make_shared<NStiffDecoder>(dim, eosk);
    std::vector<std::shared_ptr<ablate::eos::StiffenedGas>> stiffGases;
    for (const auto& eos : eosk) {
        stiffGases.push_back(std::dynamic_pointer_cast<ablate::eos::StiffenedGas>(eos));
    }
    return std::make_shared<NStiffDecoder>(dim, stiffGases);
}


#include <signal.h>




/**StiffenedGasStiffenedGasDecoder**************/
ablate::finiteVolume::processes::NPhaseAllaireAdvection::NStiffDecoder::NStiffDecoder(PetscInt dim, const std::vector<std::shared_ptr<eos::StiffenedGas>> &eosk)
    : eosk(eosk) {
    // Create the fake euler field
    auto fakeAllaireField = ablate::domain::Field{.name = NPhaseFlowFields::ALLAIRE_FIELD,
                                                .numberComponents = 1 + dim,
                                                .components = {},
                                                .id = PETSC_DEFAULT,
                                                .subId = PETSC_DEFAULT,
                                                .offset = 0,
                                                .location = ablate::domain::FieldLocation::SOL,
                                                .type = ablate::domain::FieldType::FVM,
                                                .tags = {}};

    // size up the scratch vars
    for (std::size_t k = 0; k < eosk.size(); k++){
      kAllaireFieldScratch[k].resize(1 + dim);

      kComputeTemperature[k] = eosk[k]->GetThermodynamicFunction(eos::ThermodynamicProperty::Temperature, {fakeAllaireField});
      kComputeInternalEnergy[k] = eosk[k]->GetThermodynamicFunction(eos::ThermodynamicProperty::InternalSensibleEnergy, {fakeAllaireField});
      kComputeSpeedOfSound[k] = eosk[k]->GetThermodynamicFunction(eos::ThermodynamicProperty::SpeedOfSound, {fakeAllaireField});
      kComputePressure[k] = eosk[k]->GetThermodynamicFunction(eos::ThermodynamicProperty::Pressure, {fakeAllaireField});
    }
}

void ablate::finiteVolume::processes::NPhaseAllaireAdvection::NStiffDecoder::DecodeNPhaseAllaireState(PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues,
                                                                                                                        const PetscReal *normal, PetscReal *densityOut, std::vector<PetscReal> *densitykOut, PetscReal *normalVelocityOut, PetscReal *velocityOut,
                                                                                                                        PetscReal *internalEnergyOut, std::vector<PetscReal> *internalEnergykOut, std::vector<PetscReal> *akOut, std::vector<PetscReal> *MkOut,
                                                                                                                         PetscReal *pOut, std::vector<PetscReal> *TkOut, std::vector<PetscReal> *alphakOut) {

    

    //these are wrong; change asap
    const int ALLAIRE_FIELD = 2;

    // decode; these are wrong, change asap
    PetscReal density = 1000; //sum alphakrhok
    PetscReal totalEnergy = conservedValues[NPhaseFlowFields::RHOE + uOff[ALLAIRE_FIELD]] / density;

    for (std::size_t k = 0; k < eosk.size(); k++) {
        (*densitykOut)[k] = NAN;
        (*internalEnergykOut)[k] = NAN;
        (*akOut)[k] = NAN;
        (*MkOut)[k] = NAN;
        (*TkOut)[k] = NAN;
        (*alphakOut)[k] = 0.0;
    }
    *pOut = NAN;


    // Get the velocity in this direction, and kinetic energy
    PetscReal normalVelocity = 0.0;
    PetscReal velocity[3] = {0.0, 0.0, 0.0};
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        velocity[d] = conservedValues[NPhaseFlowFields::RHOU + d + uOff[ALLAIRE_FIELD]] / density;
        normalVelocity += velocity[d] * normal[d];
        ke += velocity[d]*velocity[d];
    }
    ke *= 0.5;
    PetscReal internalEnergy = totalEnergy - ke;


    if (density < PETSC_SMALL) { // This occurs when a cell hasn't been initialized yet. Usually FVM boundary cells
        *normalVelocityOut = 0.0;
        for (PetscInt d = 0; d < dim; ++d) velocity[d] = 0.0;
        *densityOut = 0.0;
        *internalEnergyOut = 0.0;
        for (std::size_t k = 0; k < eosk.size(); k++) {
            (*densitykOut)[k] = 0.0;
            (*internalEnergykOut)[k] = 0.0;
            (*akOut)[k] = 0.0;
            (*MkOut)[k] = 0.0;
            (*TkOut)[k] = 0.0;
            (*alphakOut)[k] = 0.0;
        }
        return;
    }

    std::vector<PetscReal> rhok;
    std::vector<PetscReal> ek;
    std::vector<PetscReal> Tk;
    std::vector<PetscReal> pk;
    std::vector<PetscReal> alphak;
    std::vector<PetscReal> Cpk;
    std::vector<PetscReal> gammak;
    std::vector<PetscReal> pik;
    

    for (std::size_t k = 0; k < eosk.size(); k++) {
        rhok[k] = NAN;
        ek[k] = NAN;
        Tk[k] = NAN;
        pk[k] = NAN;
        alphak[k] = 0.0;
        Cpk[k] = eosk[k]->GetSpecificHeatCp();
        gammak[k] = eosk[k]->GetSpecificHeatRatio();
        pik[k] = eosk[k]->GetReferencePressure();
    }

    const PetscReal alphaMin = 1e-10;

    PetscReal p = internalEnergy*density;
    PetscReal pdenom = 0;
      for (size_t k = 0; k < eosk.size(); k++){
        p -= alphak[k]*pik[k]*gammak[k]/(gammak[k] - 1.0); 
        pdenom += alphak[k]/(gammak[k]-1.0);
      }
      p /= pdenom;

    auto isMixture = true;
    for (size_t k = 0; k < eosk.size(); k++) {
        if (alphak[k] > 1-alphaMin) {
            pk[k] = p;
            rhok[k] = density;
            ek[k] = internalEnergy;
            Tk[k] = gammak[k]*(ek[k] - pik[k]/rhok[k])/Cpk[k];
            alphak[k] = 1.0;
            isMixture = false;
        }
    }
    if (isMixture){
      for (size_t k = 0; k < eosk.size(); k++){
        pk[k] = p;
        rhok[k] = rhok[k]*alphak[k];
        ek[k] = (p + gammak[k]*pik[k])/((gammak[k] - 1.0)*rhok[k]);
        Tk[k] = gammak[k]*(ek[k] - pik[k]/rhok[k])/Cpk[k];
      }
    }

    for (std::size_t k = 0; k < eosk.size(); k++) {
      (*akOut)[k] = (PetscReal)PetscSqrtReal((gammak[k]-1)*gammak[k]*Cpk[k]*Tk[k]);
      (*MkOut)[k] = (PetscReal)(normalVelocity / (*akOut)[k]);
      (*alphakOut)[k] = (PetscReal)alphak[k]; //check, is this being fed in properly ?
      (*densitykOut)[k] = (PetscReal)rhok[k];
      *densityOut = (PetscReal)density;
      (*internalEnergykOut)[k] = (PetscReal)ek[k];
      (*TkOut)[k] = (PetscReal)Tk[k];
      *pOut = (PetscReal)p;
    }
    *internalEnergyOut = internalEnergy;
    for (PetscInt d = 0; d < dim; d++) velocityOut[d] = (PetscReal)velocity[d];
    *normalVelocityOut = (PetscReal)normalVelocity;

}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::NPhaseAllaireAdvection, "", ARG(ablate::eos::EOS, "eos", "must be nPhase"),
         OPT(ablate::parameters::Parameters, "parameters", "the parameters used by advection: cfl(.5)"), 
         ARG(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculatorNStiff", ""));
