#include "twoPhaseEulerAdvection.hpp"

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

PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::FormFunctionGas(SNES snes, Vec x, Vec F, void *ctx) {
    auto decodeDataStruct = (DecodeDataStructGas *)ctx;
    const PetscReal *ax;
    PetscReal *aF;
    VecGetArrayRead(x, &ax) >> utilities::PetscUtilities::checkError;
    // ax = [rhog, rhol, T]
    PetscReal rhoG = ax[0];
    PetscReal rhoL = ax[1];
    PetscReal T = ax[2];
    VecRestoreArrayRead(x, &ax) >> utilities::PetscUtilities::checkError;

    PetscReal gammaG = decodeDataStruct->gamG;
    PetscReal gammaL = decodeDataStruct->gamL;
    PetscReal Yg = decodeDataStruct->Yg;
    PetscReal Yl = decodeDataStruct->Yl;
    PetscReal rho = decodeDataStruct->density;
    PetscReal e = decodeDataStruct->internalEnergy;
    PetscReal cvG = decodeDataStruct->cvG;
    PetscReal cpL = decodeDataStruct->cpL;
    PetscReal p0L = decodeDataStruct->p0L;

    VecGetArray(F, &aF) >> utilities::PetscUtilities::checkError;

    aF[0] = (cpL*(-1 + gammaL)*rhoL*T)/gammaL - p0L - cvG*(-1 + gammaG)*rhoG*T;
    aF[1] = -(e*rhoL) + cvG*rhoL*T*Yg + (p0L + (cpL*rhoL*T)/gammaL)*Yl;
    aF[2] = -(rhoG*rhoL) + rho*rhoL*Yg + rho*rhoG*Yl;

    VecRestoreArray(F, &aF) >> utilities::PetscUtilities::checkError;
    return 0;
}

PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::FormJacobianGas(SNES snes, Vec x, Mat J, Mat P, void *ctx) {
    auto decodeDataStruct = (DecodeDataStructGas *)ctx;

    const PetscReal *ax;
    VecGetArrayRead(x, &ax) >> utilities::PetscUtilities::checkError;
    // ax = [rhog, rhol, T]
    const PetscReal rhoG = ax[0];
    const PetscReal rhoL = ax[1];
    const PetscReal T = ax[2];
    VecRestoreArrayRead(x, &ax) >> utilities::PetscUtilities::checkError;

    const PetscReal gammaG = decodeDataStruct->gamG;
    const PetscReal gammaL = decodeDataStruct->gamL;
    const PetscReal Yg = decodeDataStruct->Yg;
    const PetscReal Yl = decodeDataStruct->Yl;
    const PetscReal rho = decodeDataStruct->density;
    const PetscReal e = decodeDataStruct->internalEnergy;
    const PetscReal cvG = decodeDataStruct->cvG;
    const PetscReal cpL = decodeDataStruct->cpL;
//    PetscReal p0L = decodeDataStruct->p0L;

    PetscScalar *v;
    MatDenseGetArray(P, &v) >> utilities::PetscUtilities::checkError;
    v[0] = -(cvG*(-1 + gammaG)*T);
    v[1] = (cpL*(-1 + gammaL)*T)/gammaL;
    v[2] = -(cvG*(-1 + gammaG)*rhoG) + (cpL*(-1 + gammaL)*rhoL)/gammaL;
    v[3] = 0;
    v[4] = -e + cvG*T*Yg + (cpL*T*Yl)/gammaL;
    v[5] = cvG*rhoL*Yg + (cpL*rhoL*Yl)/gammaL;
    v[6] = -rhoL + rho*Yl;
    v[7] = -rhoG + rho*Yg;
    v[8] = 0;
    MatDenseRestoreArray(P, &v) >> utilities::PetscUtilities::checkError;

    if (J != P) {
      MatCopy(P, J, SAME_NONZERO_PATTERN);
    }
    return 0;
}

PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::FormFunctionStiff(SNES snes, Vec x, Vec F, void *ctx) {
    auto decodeDataStruct = (DecodeDataStructStiff *)ctx;
    const PetscReal *ax;
    PetscReal *aF;
    VecGetArrayRead(x, &ax);
    // ax = [rhog, rhol, eg, el]
    PetscReal rhoG = ax[0];
    PetscReal rhoL = ax[1];
    PetscReal eG = ax[2];
    PetscReal eL = ax[3];

    PetscReal gamma1 = decodeDataStruct->gam1;
    PetscReal gamma2 = decodeDataStruct->gam2;
    PetscReal Y1 = decodeDataStruct->Yg;
    PetscReal Y2 = decodeDataStruct->Yl;
    PetscReal rho = decodeDataStruct->rhotot;
    PetscReal e = decodeDataStruct->etot;
    PetscReal cp1 = decodeDataStruct->cpg;
    PetscReal cp2 = decodeDataStruct->cpl;
    PetscReal p01 = decodeDataStruct->p0g;
    PetscReal p02 = decodeDataStruct->p0l;

    VecGetArray(F, &aF);
    aF[0] = (gamma1 - 1) * eG * rhoG - gamma1 * p01 - (gamma2 - 1) * eL * rhoL + gamma2 * p02;  // pG - pL = 0, pressure equilibrium
    aF[1] = gamma1 / cp1 * rhoL * (eG * rhoG - p01) - gamma2 / cp2 * rhoG * (eL * rhoL - p02);  // TG - TL = 0, temperature equilibrium
    aF[2] = Y1 * rho * rhoL + Y2 * rho * rhoG - rhoG * rhoL;
    aF[3] = Y1 * eG + Y2 * eL - e;

//printf("%+e\t%+e\t%+e\t%+e\n", rhoG, rhoL, eG, eL);

    VecRestoreArrayRead(x, &ax);
    VecRestoreArray(F, &aF);
    return 0;
}

PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::FormJacobianStiff(SNES snes, Vec x, Mat J, Mat P, void *ctx) {
    auto decodeDataStruct = (DecodeDataStructStiff *)ctx;
    const PetscReal *ax;
    PetscReal v[16];
    PetscInt row[4] = {0, 1, 2, 3}, col[4] = {0, 1, 2, 3};
    VecGetArrayRead(x, &ax);
    // ax = [rhog, rhol, eg, el]
    PetscReal rhoG = ax[0];
    PetscReal rhoL = ax[1];
    PetscReal eG = ax[2];
    PetscReal eL = ax[3];

    PetscReal gamma1 = decodeDataStruct->gam1;
    PetscReal gamma2 = decodeDataStruct->gam2;
    PetscReal Y1 = decodeDataStruct->Yg;
    PetscReal Y2 = decodeDataStruct->Yl;
    PetscReal rho = decodeDataStruct->rhotot;
    //    PetscReal e = decodeDataStruct->etot;
    PetscReal cp1 = decodeDataStruct->cpg;
    PetscReal cp2 = decodeDataStruct->cpl;
    PetscReal p01 = decodeDataStruct->p0g;
    PetscReal p02 = decodeDataStruct->p0l;

    // need to check Jacobian, not getting correct solution
    v[0] = (gamma1 - 1) * eG;
    v[1] = -(gamma2 - 1) * eL;
    v[2] = (gamma1 - 1) * rhoG;
    v[3] = -(gamma2 - 1) * rhoL;
    v[4] = gamma1 / cp1 * eG * rhoL - gamma2 / cp2 * eL * rhoL + gamma2 / cp2 * p02;
    v[5] = gamma1 / cp1 * eG * rhoG - gamma1 / cp1 * p01 - gamma2 / cp2 * eL * rhoG;
    v[6] = gamma1 / cp1 * rhoG * rhoL;
    v[7] = -gamma2 / cp2 * rhoG * rhoL;
    v[8] = Y2 * rho - rhoL;
    v[9] = Y1 * rho - rhoG;
    v[10] = 0.0;
    v[11] = 0.0;
    v[12] = 0.0;
    v[13] = 0.0;
    v[14] = Y1;
    v[15] = Y2;
    VecRestoreArrayRead(x, &ax);
    MatSetValues(P, 4, row, 4, col, v, INSERT_VALUES);
    MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);
    if (J != P) {
        MatSetValues(J, 4, row, 4, col, v, INSERT_VALUES);
        MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
    }
    return 0;
}

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
    auto intSharpProcess = std::make_shared<ablate::finiteVolume::processes::IntSharp>(1000, 0.01, false);
    intSharpProcess->Initialize(flow);



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
      else if (fields[f] == VOLUME_FRACTION_FIELD) { // In case it's ever moved to the AUX vector
        auxField[aOff[f]] = alpha;
      }
    }

    PetscFunctionReturn(0);
}

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
    
    //compute the term for all cells
    std::shared_ptr<ablate::finiteVolume::processes::IntSharp> intSharpProcess;
    intSharpProcess->ComputeTerm(fvSolver, dm, stagetime, globFlowVec, locFVec, intSharpProcess.get());

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

        //get the intsharp term at this cell
        const auto &fluxGrad = intSharpProcess->fluxGradValues[i - cellRange.start]; //fluxGradValues has size (nCells x dim); we are grabbing the ith row such that fluxGrad size is dim
  
        //keep old alpha
        const PetscScalar oldAlpha = allFields[vfOffset];
  
        //update alpha according to intsharp-calculated flux grad values
        //commenting this out nullifies the entire remainder of the loop
        for (PetscInt d = 0; d < dim; ++d) { 
          if (!std::isnan(fluxGrad[d]) && fluxGrad[d] != 0.0) { //check to make sure intsharp has actually been computed here
            allFields[vfOffset] -= fluxGrad[d]; //this can be thought of as the RHS of the material derivative of alpha in pseudo time. 
          }       
        }
  
        //recompute conserved variables:
        //update rhogAlpha based on new alpha
        allFields[rhoAlphaOffset] = (allFields[vfOffset] / oldAlpha) * allFields[rhoAlphaOffset];

        //update euler field based on new alpha; 
        //here we are assuming rhoG/L old = rhoG/L new, e old = e new 
        allFields[ablate::finiteVolume::CompressibleFlowFields::RHO] = allFields[vfOffset]*densityG + (1-allFields[vfOffset])*densityL;
        allFields[ablate::finiteVolume::CompressibleFlowFields::RHOE] = allFields[ablate::finiteVolume::CompressibleFlowFields::RHO]*internalEnergy;
        for (PetscInt d = 0; d < dim; ++d) { allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU+d] = allFields[ablate::finiteVolume::CompressibleFlowFields::RHO] * velocity[d]; }

        //redo decode with new euler fields (density G/L and alpha will be redundant but since RHOE is updated, eG/eL -->p,T will be changed)
        decoder->DecodeTwoPhaseEulerState(
          dim, uOff, allFields, norm, &density, &densityG, &densityL, &normalVelocity, velocity, &internalEnergy, &internalEnergyG, &internalEnergyL, &aG, &aL, &MG, &ML, &p, &t, &alpha);
    }

    // //update sol vec
    // PetscCall(DMLocalToGlobalBegin(dm, locFVec, ADD_VALUES, globFlowVec));
    // PetscCall(DMLocalToGlobalEnd(dm, locFVec, ADD_VALUES, globFlowVec));

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
ablate::finiteVolume::processes::TwoPhaseEulerAdvection::PerfectGasPerfectGasDecoder::PerfectGasPerfectGasDecoder(PetscInt dim, const std::shared_ptr<eos::PerfectGas> &eosGas,
                                                                                                                  const std::shared_ptr<eos::PerfectGas> &eosLiquid)
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

void ablate::finiteVolume::processes::TwoPhaseEulerAdvection::PerfectGasPerfectGasDecoder::DecodeTwoPhaseEulerState(PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues,
                                                                                                                    const PetscReal *normal, PetscReal *density, PetscReal *densityG,
                                                                                                                    PetscReal *densityL, PetscReal *normalVelocity, PetscReal *velocity,
                                                                                                                    PetscReal *internalEnergy, PetscReal *internalEnergyG, PetscReal *internalEnergyL,
                                                                                                                    PetscReal *aG, PetscReal *aL, PetscReal *MG, PetscReal *ML, PetscReal *p,
                                                                                                                    PetscReal *T, PetscReal *alpha) {


    // (RHO, RHOE, RHOU, RHOV, RHOW)
    const int EULER_FIELD = 2;
    const int VF_FIELD = 1;

    // decode
    *density = conservedValues[CompressibleFlowFields::RHO + uOff[EULER_FIELD]];

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


    PetscReal totalEnergy = conservedValues[CompressibleFlowFields::RHOE + uOff[EULER_FIELD]] / (*density);
    PetscReal densityVF = conservedValues[uOff[VF_FIELD]];

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

    // mass fractions
    PetscReal Yg = densityVF / (*density);
    PetscReal Yl = ((*density) - densityVF) / (*density);

    PetscReal R1 = eosGas->GetGasConstant();
    PetscReal R2 = eosLiquid->GetGasConstant();
    PetscReal gamma1 = eosGas->GetSpecificHeatRatio();
    PetscReal gamma2 = eosLiquid->GetSpecificHeatRatio();
    PetscReal cv1 = R1 / (gamma1 - 1);
    PetscReal cv2 = R2 / (gamma2 - 1);

    PetscReal eG = (*internalEnergy) / (Yg + Yl * cv2 / cv1);
    PetscReal etG = eG + ke;
    PetscReal eL = cv2 / cv1 * eG;

    PetscReal etL = eL + ke;
    PetscReal rhoG = (*density) * (Yg + Yl * eL / eG * (gamma2 - 1) / (gamma1 - 1));
    PetscReal rhoL = rhoG * eG / eL * (gamma1 - 1) / (gamma2 - 1);

    PetscReal pG = 0;
    PetscReal pL;
    PetscReal a1 = 0;
    PetscReal a2 = 0;

    // Fill the scratch array for gas
    liquidEulerFieldScratch[CompressibleFlowFields::RHO] = rhoL;
    liquidEulerFieldScratch[CompressibleFlowFields::RHOE] = rhoL * etL;
    for (PetscInt d = 0; d < dim; d++) {
        liquidEulerFieldScratch[CompressibleFlowFields::RHOU + d] = velocity[d] * rhoL;
    }

    // Decode the gas
    {
        liquidComputeTemperature.function(liquidEulerFieldScratch.data(), T, liquidComputeTemperature.context.get()) >> utilities::PetscUtilities::checkError;
        liquidComputeInternalEnergy.function(liquidEulerFieldScratch.data(), *T, &eL, liquidComputeInternalEnergy.context.get()) >> utilities::PetscUtilities::checkError;
        liquidComputeSpeedOfSound.function(liquidEulerFieldScratch.data(), *T, &a2, liquidComputeSpeedOfSound.context.get()) >> utilities::PetscUtilities::checkError;
        liquidComputePressure.function(liquidEulerFieldScratch.data(), *T, &pL, liquidComputePressure.context.get()) >> utilities::PetscUtilities::checkError;
    }

    // Fill the scratch array for gas
    gasEulerFieldScratch[CompressibleFlowFields::RHO] = rhoG;
    gasEulerFieldScratch[CompressibleFlowFields::RHOE] = rhoG * etG;
    for (PetscInt d = 0; d < dim; d++) {
        gasEulerFieldScratch[CompressibleFlowFields::RHOU + d] = velocity[d] * rhoG;
    }

    // Decode the gas
    {
        gasComputeTemperature.function(gasEulerFieldScratch.data(), T, gasComputeTemperature.context.get()) >> utilities::PetscUtilities::checkError;
        gasComputeInternalEnergy.function(gasEulerFieldScratch.data(), *T, &eG, gasComputeInternalEnergy.context.get()) >> utilities::PetscUtilities::checkError;
        gasComputeSpeedOfSound.function(gasEulerFieldScratch.data(), *T, &a1, gasComputeSpeedOfSound.context.get()) >> utilities::PetscUtilities::checkError;
        gasComputePressure.function(gasEulerFieldScratch.data(), *T, &pG, gasComputePressure.context.get()) >> utilities::PetscUtilities::checkError;
    }

    // once state defined
    *densityG = rhoG;
    *densityL = rhoL;
    *internalEnergyG = eG;
    *internalEnergyL = eL;
    *alpha = densityVF / (*densityG);
    *p = pG;  // pressure equilibrium, pG = pL
    *aG = a1;
    *aL = a2;
    *MG = (*normalVelocity) / (*aG);
    *ML = (*normalVelocity) / (*aL);
}

/**PerfectGasStiffenedGasDecoder**************/
ablate::finiteVolume::processes::TwoPhaseEulerAdvection::PerfectGasStiffenedGasDecoder::PerfectGasStiffenedGasDecoder(PetscInt dim, const std::shared_ptr<eos::PerfectGas> &eosGas,
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

void ablate::finiteVolume::processes::TwoPhaseEulerAdvection::PerfectGasStiffenedGasDecoder::DecodeTwoPhaseEulerState(PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues,
                                                                                                                      const PetscReal *normal, PetscReal *densityOut, PetscReal *densityG,
                                                                                                                      PetscReal *densityL, PetscReal *normalVelocityOut, PetscReal *velocityOut,
                                                                                                                      PetscReal *internalEnergyOut, PetscReal *internalEnergyG, PetscReal *internalEnergyL,
                                                                                                                      PetscReal *aG, PetscReal *aL, PetscReal *MG, PetscReal *ML, PetscReal *p,
                                                                                                                      PetscReal *T, PetscReal *alpha) {

    const int EULER_FIELD = 2;
    const int VF_FIELD = 1;

    // decode
    PetscReal density = conservedValues[CompressibleFlowFields::RHO + uOff[EULER_FIELD]];
    PetscReal totalEnergy = conservedValues[CompressibleFlowFields::RHOE + uOff[EULER_FIELD]] / (density);
    PetscReal densityVF = conservedValues[uOff[VF_FIELD]];

    *densityOut = *densityG = *densityL = NAN;
    *internalEnergyOut = *internalEnergyG = *internalEnergyL = NAN;
    *alpha = NAN;
    *p = NAN;
    *aG = *aL = NAN;
    *MG = *ML = NAN;

    // Get the velocity in this direction, and kinetic energy
    PetscReal normalVelocity = 0.0;
    PetscReal ke = 0.0;
    PetscReal velocity[3] = {0.0, 0.0, 0.0};
    for (PetscInt d = 0; d < dim; d++) {
        velocity[d] = conservedValues[CompressibleFlowFields::RHOU + d + uOff[EULER_FIELD]] / density;
        normalVelocity += velocity[d] * normal[d];
        ke += velocity[d]*velocity[d];
    }
    ke *= 0.5;
    PetscReal internalEnergy = totalEnergy - ke;

    if (density < PETSC_SMALL) { // This occurs when a cell hasn't been initialized yet. Usually FVM boundary cells
        *normalVelocityOut = 0.0;
        for (PetscInt d = 0; d < dim; ++d) velocityOut[d] = 0.0;
        *densityOut = 0.0;
        *densityG = 0.0;
        *densityL = 0.0;
        *internalEnergyOut = 0.0;
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


    // mass fractions
    PetscReal Yg = PetscMin(1.0, PetscMax(0.0, densityVF / density));
    PetscReal Yl = 1.0 - Yg;

    PetscReal rhoL = NAN, rhoG = NAN, eG = NAN, eL = NAN;
    PetscReal TL = NAN, TG = NAN, pG = NAN, pL = NAN, alphaG = NAN;

    const PetscReal RG = eosGas->GetGasConstant();
    const PetscReal cpL = eosLiquid->GetSpecificHeatCp();
    const PetscReal p0L = eosLiquid->GetReferencePressure();
    const PetscReal gammaG = eosGas->GetSpecificHeatRatio();
    const PetscReal gammaL = eosLiquid->GetSpecificHeatRatio();
    const PetscReal cvG = RG / (gammaG - 1.0);
    const PetscReal cvL = cpL/gammaL;

//    liquidEulerFieldScratch[CompressibleFlowFields::RHOE] = NAN;
//    gasEulerFieldScratch[CompressibleFlowFields::RHOE] = NAN;
//    for (PetscInt d = 0; d < dim; d++) {
//        liquidEulerFieldScratch[CompressibleFlowFields::RHOU + d] = NAN;
//        gasEulerFieldScratch[CompressibleFlowFields::RHOU + d] = NAN;
//    }



    const PetscReal alphaMin = 1e-10;
    const PetscReal alphaMax = 1000*alphaMin;

    if (Yg < alphaMin) { // All liquid
      rhoL = density;
      eL = internalEnergy;
      TL = (eL*rhoL - p0L)/(cvL*rhoL);

      PetscReal rho0 = 998.23;

      if (rhoL < rho0) {
        PetscReal p0  = (gammaL - 1.0)*rho0*eL - gammaL*p0L; // What the pressure would be if the density was higher
        PetscReal dp0 = (gammaL - 1.0)*eL;  // The slope of the pressure
        PetscReal fac = p0 - dp0*rho0;
        PetscReal a = dp0*PetscSqr(p0*rho0/fac);
        PetscReal b = -dp0*PetscSqr(rho0)/fac;
        PetscReal c = PetscSqr(p0)/fac;
        pL = a/(b-rhoL) + c;

      }
      else {
        pL = (gammaL - 1.0)*rhoL*eL - gammaL*p0L;
      }

//      PetscReal p0  = (gammaL - 1.0)*rho0*eL - gammaL*p0L; // What the pressure would be if the density was higher
//      PetscReal dp0 = (gammaL - 1.0)*eL;  // The slope of the pressure

//      pL = (gammaL - 1.0)*rhoL*eL - gammaL*p0L;
//      PetscReal H = Heaviside(rhoL, rho0, rho0*1e-3);
//      pL = (1.0 - H)*(p0 + dp0*(rhoL - rho0)*1e-4) + H*( pL );

      TG = TL;
      pG = pL;
      rhoG = pG / (RG*TG);
      eG = cvG*TG;

      alphaG = 0.0;

    }
    else if (Yl < alphaMin) { //All gas
      rhoG = density;
      eG = internalEnergy;
      TG = eG/cvG;
      pG = (gammaG - 1.0)*rhoG*eG;

      TL = TG;
      pL = pG;
      rhoL = (pL + p0L)/((gammaL-1.0)*cvL*TL);
      eL = cvL*TL + p0L/rhoL;

      alphaG = 1.0;
    }
    else {
      PetscReal e = internalEnergy;
      PetscReal rho = density;


      PetscReal A = rho*(cvG*Yg + cvL*Yl)*(cvG*gammaG*Yg + cvL*gammaL*Yl);
      PetscReal B = cvG*(p0L - e*(1.0 + gammaG)*rho)*Yg + cvL*(-(e*rho) + gammaL*(p0L - e*rho))*Yl;
      PetscReal C = e*(e*rho - p0L);

      PetscReal x1, x2;
      SolveQuadratic(A, B, C, &x1, &x2);

      TG = TL = PetscMax(x1, x2);


      // This doesn't seem to make a difference
      if (Yg > Yl) {
        rhoL = gammaL*p0L*Yl/(e*gammaL - cvG*gammaL*TG*Yg - cpL*TL*Yl);
        rhoG = rho*rhoL*Yg/(rhoL - rho*Yl);
      }
      else {
        rhoG = p0L*rho*Yg/(p0L - e*rho + cvG*rho*TG*Yg + cvL*rho*TG*Yl);
        rhoL = rho*rhoG*Yl/(rhoG - rho*Yg);
      }

      pL = (gammaL - 1.0)*cvL*rhoL*TL - p0L;
      pG = (gammaG - 1.0)*cvG*rhoG*TG;



//      A = -(cvG*Yg) - cvL*Yl;
//      B = -(cvG*p0L*Yg) + cvG*e*(-1 + gammaG)*rho*Yg - cvL*gammaL*p0L*Yl + cvL*e*(-1 + gammaL)*rho*Yl;
//      C = cvG*e*(-1 + gammaG)*p0L*rho*Yg;
//      SolveQuadratic(A, B, C, &x1, &x2);
//      pG = pL = PetscMax(x1, x2);

//      if (Yg > Yl) {
//        rhoG = ((-(gammaL*(pG + p0L)) + gammaG*(pG + gammaL*p0L))*rho*Yg)/((-1 + gammaG)*(pG + e*rho + gammaL*(p0L - e*rho)));
//        rhoL = -((rho*rhoG*Yl)/(-rhoG + rho*Yg));
//      }
//      else {
//        rhoL = ((gammaL*(pL + p0L) - gammaG*(pL + gammaL*p0L))*rho*Yl)/((-1 + gammaL)*(pL - e*(-1 + gammaG)*rho));
//        rhoG = -((rho*rhoL*Yg)/(-rhoL + rho*Yl));
//      }

//      TL = -((pL + p0L)/(cvL*rhoL - cvL*gammaL*rhoL));
//      TG = pG/(cvG*(-1 + gammaG)*rhoG);



      eL = cvL*TL + p0L/rhoL;
      eG = cvG*TG;

      alphaG = densityVF / rhoG;

//       Blend the pressure and energy
      if (Yg < alphaMax) {
        PetscReal pL0 = pL, pG0 = pG;
        PetscReal eL0 = eL, eG0 = eG;
        PetscReal xi = (Yg - alphaMin) / (alphaMax - alphaMin);
        PetscReal G = xi*xi*xi*(10 - 15*xi + 6*xi*xi);
        G = -xi*xi*(2.0*xi - 3.0);
        pG = G*pG0 + (1.0 - G)*pL0;
        eG = G*eG0 + (1.0 - G)*eL0;
      }
      else if (Yl < alphaMax) {
        PetscReal pL0 = pL, pG0 = pG;
        PetscReal eL0 = eL, eG0 = eG;
        PetscReal xi = (Yl - alphaMin) / (alphaMax - alphaMin);
        PetscReal G = xi*xi*xi*(10 - 15*xi + 6*xi*xi);
        G = -xi*xi*(2.0*xi - 3.0);
        pL = G*pL0 + (1.0 - G)*pG0;
        eL = G*eL0 + (1.0 - G)*eG0;
      }



      // Check all of the required relationships
      if (PetscAbsReal(Yg/rhoG + Yl/rhoL - 1.0/density) > 1e-10) throw std::runtime_error("Eq (32) is not satisfied.\n");
//      if (PetscAbsReal(Yg*eG + Yl*eL - e) > 1e-10) throw std::runtime_error("Eq (33) is not satisfied.\n");
//      if (PetscAbsReal(pL - pG) > 1e-10) throw std::runtime_error("Pressure equilibrium is not satisfied.\n");
    }



    *aL = (PetscReal)PetscSqrtReal((gammaL - 1.0)*gammaL*cvL*TL);
    *aG = (PetscReal)PetscSqrtReal((gammaG - 1.0)*gammaG*cvG*TG);
//    liquidComputeInternalEnergy.function(liquidEulerFieldScratch.data(), TL, &eL, liquidComputeInternalEnergy.context.get()) >> utilities::PetscUtilities::checkError;
//    gasComputeInternalEnergy.function(gasEulerFieldScratch.data(), TG, &eG, gasComputeInternalEnergy.context.get()) >> utilities::PetscUtilities::checkError;

//    liquidComputeSpeedOfSound.function(liquidEulerFieldScratch.data(), TL, aL, liquidComputeSpeedOfSound.context.get()) >> utilities::PetscUtilities::checkError;
//    gasComputeSpeedOfSound.function(gasEulerFieldScratch.data(), TG, aG, gasComputeSpeedOfSound.context.get()) >> utilities::PetscUtilities::checkError;


    if (TL < PETSC_SMALL || TG < PETSC_SMALL) {
      throw std::runtime_error("Decode is returning negative temperature.\n");
    }

    if (pL < PETSC_SMALL || pG < PETSC_SMALL) {
//      printf("%ld\n", cnt);
      printf("   T: %+e\n", (PetscReal)TG);
      printf("  pR: %+e\n", (PetscReal)pG);
      printf("  pL: %+e\n", (PetscReal)pL);
      printf("  eR: %+e\n", (PetscReal)eG);
      printf("  eL: %+e\n", (PetscReal)eL);
      printf("rhoL: %+e\n", (PetscReal)rhoL);
      printf("rhoG: %+e\n", (PetscReal)rhoG);
      printf("%e\n", (PetscReal)alphaG);
      printf("  Yg: %+e\n", (PetscReal)Yg);
      printf("  Yl: %+e\n", (PetscReal)Yl);
      printf("%s::%s::%d\n", __FUNCTION__, __FILE__, __LINE__);
      printf("\n");
      raise(SIGSEGV);
      throw std::runtime_error("Decode is returning negative pressure.\n");
    }

    if (eL < PETSC_SMALL || eG < PETSC_SMALL) {
//      printf("%ld\n", cnt);
      printf("   T: %+e\n", (PetscReal)TG);
      printf("  pR: %+e\n", (PetscReal)pG);
      printf("  pL: %+e\n", (PetscReal)pL);
      printf("  eR: %+e\n", (PetscReal)eG);
      printf("  eL: %+e\n", (PetscReal)eL);
      printf("rhoL: %+e\n", (PetscReal)rhoL);
      printf("rhoG: %+e\n", (PetscReal)rhoG);
      printf("%e\n", (PetscReal)alphaG);
      printf("  Yg: %+e\n", (PetscReal)Yg);
      printf("  Yl: %+e\n", (PetscReal)Yl);
      printf("%s::%s::%d\n", __FUNCTION__, __FILE__, __LINE__);
      printf("\n");
      raise(SIGSEGV);
      throw std::runtime_error("Decode is returning negative energy.\n");
    }

    if (PetscAbsReal(TL - TG) > 1e-4*PetscMin(TL, TG)) {
      throw std::runtime_error("Decode is not returning temperature equilibrium.\n");
    }

    if (PetscAbsReal(pL - pG) > 1e-4*PetscMin(pL, pG)) {
      throw std::runtime_error("Decode is not returning pressure equilibrium.\n");
    }

    // once state defined
    *alpha = (PetscReal)(PetscMin(1.0, PetscMax(0.0, alphaG)));
//    *alpha = alphaG;
    for (PetscInt d = 0; d < dim; d++) velocityOut[d] = (PetscReal)velocity[d];
    *normalVelocityOut = (PetscReal)normalVelocity;
    *densityOut = (PetscReal)density;
    *internalEnergyOut = (PetscReal)internalEnergy;
    *densityG = (PetscReal)rhoG;
    *densityL = (PetscReal)rhoL;
    *internalEnergyG = (PetscReal)eG;
    *internalEnergyL = (PetscReal)eL;
    *T = (PetscReal)(0.5*(TG + TL));
    *p = PetscMax((PetscReal)pG, (PetscReal)pL);
    *MG = (PetscReal)(normalVelocity / PetscSqrtReal((gammaG - 1.0)*gammaG*cvG*TG));
    *ML = (PetscReal)(normalVelocity / PetscSqrtReal((gammaL - 1.0)*gammaL*cvL*TL));

//    PetscReal a1t = PetscSqrtReal((gammaG - 1) * cvG * (*T));
//    PetscReal a2t = PetscSqrtReal((gammaL - 1) / gammaL * cpL * (*T));
//    PetscReal ainv = Yg / (rhoG * rhoG * a1t * a1t) + Yl / (rhoL * rhoL * a2t * a2t);
//    PetscReal amix = PetscSqrtReal(1 / ainv) / (*density);
//    PetscReal bmodt = (*density) * amix * amix;
//    if (bmodt < 0.0) {
//        throw std::invalid_argument("isothermal bulk modulus of mixture negative");
//    }

}

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

if (PetscAbsReal(pL - pG) > PetscMin(pG, pL)*PETSC_SMALL) printf("%e\t%e\t%e\n", pL, pG, PetscAbsReal(pL-pG));

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
