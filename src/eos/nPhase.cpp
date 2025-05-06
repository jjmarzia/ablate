#include "nPhase.hpp"
#include "finiteVolume/nPhaseFlowFields.hpp"
#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"
#include "eos/stiffenedGas.hpp"

static inline void NStiffDecode(PetscInt dim, ablate::eos::NPhase::DecodeIn *in, ablate::eos::NPhase::DecodeOut *out) {
    
    //rhok = alphakrhok/alphak
    for (std::size_t k = 0; k < in->alphak.size(); ++k) {
        out->rhok[k] = in->alphakrhok[k] / in->alphak[k];
    }

    //rho = sum(alphak rhok)
    out->rho = 0.0;
    for (std::size_t k = 0; k < in->alphak.size(); ++k) {
        out->rho += in->alphakrhok[k];
    }

    //ui = rhoui/rho
    for (PetscInt d = 0; d < dim; d++) {
        out->ui[d] = in->rhoui[d] / out->rho;
    }

    // p = (rhoe - 0.5 uiui - sumk[ alphak gammak pik / (gammak - 1) ] )/ sumk [ alphak/(gammak - 1) ] 
    out->p = in->rhoe;
    for (PetscInt d = 0; d < dim; d++) {
        out->p -= 0.5 * out->ui[d] * in->rhoui[d];
    }
    PetscReal pterm1 = 0.0;
    PetscReal pterm2 = 0.0;
    for (std::size_t k = 0; k < in->alphak.size(); ++k) {
        pterm1 += in->alphak[k] * in->parameters.gammak[k] * in->parameters.pik[k] / (in->parameters.gammak[k] - 1);
        pterm2 += in->alphak[k] / (in->parameters.gammak[k] - 1);
    }
    out->p = (out->p + pterm1) / pterm2;

    //e = rhoe/rho
    out->e = in->rhoe / out->rho;

    //epsk = (p + gammak pik)/((gammak-1)rhok)
    for (std::size_t k = 0; k < in->alphak.size(); ++k) {
        out->epsk[k] = (in->parameters.pik[k] + in->parameters.gammak[k] * in->parameters.pik[k]) / ((in->parameters.gammak[k] - 1) * out->rhok[k]);
    }

    //ek = epsk + uiui/2; might not need this
    // PetscReal uiui = 0.0;
    // for (PetscInt d = 0; d < dim; d++) {
    //     uiui += out->ui[d] * out->ui[d];
    // }
    // for (std::size_t k = 0; k < in->alphak.size(); ++k) {
    //     out->ek[k] = out->epsk[k] + 0.5 * uiui;
    // }

    //Tk = gammak*(epsk - pik/rhok)/Cpk
    for (std::size_t k = 0; k < in->alphak.size(); ++k) {
        out->Tk[k] = in->parameters.gammak[k] * (out->epsk[k] - in->parameters.pik[k] / out->rhok[k]) / in->parameters.Cpk[k];
    }

    //c = sqrt( (sumk alphak/(gammak(p+pik))  )^-1/rho   ); might not need this
    out->c = 0.0;
    for (std::size_t k = 0; k < in->alphak.size(); ++k) {
        out->c += in->alphak[k] / (in->parameters.gammak[k] * (out->p + in->parameters.pik[k]));
    }
    out->c = PetscSqrtReal( (1.0 / out->c) / out->rho);
}

// ablate::eos::NPhase::NPhase(std::shared_ptr<eos::EOS> eos1, std::shared_ptr<eos::EOS> eos2) : EOS("twoPhase"), eos1(std::move(eos1)), eos2(std::move(eos2)) {
//do the above twophase but for nphase
ablate::eos::NPhase::NPhase(std::vector<std::shared_ptr<eos::EOS>> eosk) : EOS("nPhase"), eosk(std::move(eosk)) {
    // check that eos is nPhase
    if (this->eosk.size() < 2) {
        throw std::invalid_argument("you need at least two phases");
    }

    // populate component eoses
    for (const auto &eos : this->eosk) {
        if (!eos) {
            throw std::invalid_argument("invalid eos");
        }
        auto eosPhase = std::dynamic_pointer_cast<eos::StiffenedGas>(eos);
        parameters.gammak.push_back(eosPhase->GetSpecificHeatRatio());
        parameters.pik.push_back(eosPhase->GetReferencePressure());
        parameters.Cpk.push_back(eosPhase->GetSpecificHeatCp());
        // parameters.numberSpeciesk.push_back(eosPhase->GetSpeciesVariables().size());
        // parameters.speciesk.push_back(eosPhase->GetSpeciesVariables());
        // species.insert(species.end(), eosPhase->GetSpeciesVariables().begin(), eosPhase->GetSpeciesVariables().end());
    }
}

void ablate::eos::NPhase::View(std::ostream &stream) const {
    stream << "EOS with " << eosk.size() << " phases:" << std::endl;
    for (std::size_t k = 0; k < eosk.size(); ++k) {
        stream << "  phase " << k + 1 << ": ";
        if (eosk[k]) {
            stream << *eosk[k];
        } else {
            stream << "null EOS";
        }
        stream << std::endl;
    }
}

ablate::eos::ThermodynamicFunction ablate::eos::NPhase::GetThermodynamicFunction(
    ablate::eos::ThermodynamicProperty property, 
    const std::vector<domain::Field> &fields) const {

    auto allaireField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { 
        return field.name == ablate::finiteVolume::NPhaseFlowFields::ALLAIRE_FIELD; 
    });
    if (allaireField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::NPhase requires the ablate::finiteVolume::NPhaseFlowFields::ALLAIRE_FIELD Field");
    }

    auto alphakrhokField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { 
        return field.name == ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK(0);
    });
    auto alphakField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { 
        return field.name == ablate::finiteVolume::NPhaseFlowFields::ALPHAK(0);
    });

    // Look for the euler field
    // auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::NPhaseFlowFields::EULER_FIELD; });
    // auto densityVFField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD; });
    // auto volumeFractionField =
    //     std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD; });
    // // maybe need to throw error for not having densityVF or volumeFraction fields
    // if (eulerField == fields.end()) {
    //     throw std::invalid_argument("The ablate::eos::TwoPhase requires the ablate::finiteVolume::NPhaseFlowFields::EULER_FIELD Field");
    // }

    // Determine the property size
    PetscInt propertySize = 1;

    return ThermodynamicFunction{
        .function = thermodynamicFunctionsNStiff.at(property), //.first was for back when we had a pair
        .context = std::make_shared<FunctionContext>(FunctionContext{.dim = allaireField->numberComponents - 1,
        .allaireOffset = allaireField->offset,
        .alphakrhokOffset = alphakrhokField->offset,
        .alphakOffset = alphakField->offset,
        // .allaireOffset = eulerField->offset,
        // .alphakrhokOffset = ,
        // .alphakOffset = volumeFractionField->offset,
        .parameters = parameters}),
        .propertySize = propertySize};
}

ablate::eos::EOSFunction ablate::eos::NPhase::GetFieldFunctionFunction(
    const std::string &field, 
    ablate::eos::ThermodynamicProperty property1, 
    ablate::eos::ThermodynamicProperty property2,
    std::vector<std::string> otherProperties) const {
    if (otherProperties != std::vector<std::string>{VF}) {  
        throw std::invalid_argument("ablate::eos::TwoPhase expects other properties to include VF (volume fraction) as first entry");
    }

        //assume that ALLAIRE_FIELD was specified and that we have p,tk, alphak
    auto tp = [this](std::vector<PetscReal> tk, std::vector<PetscReal> alphak, PetscReal pressure, PetscInt dim, const PetscReal velocity[], 
            PetscReal conserved[]) {

            std::vector<PetscReal> rhok;
            std::vector<PetscReal> epsk;

            //if tk and alphak are of different size, throw an error
            if (tk.size() != alphak.size()) {
                throw std::invalid_argument("tk and alphak must be the same size");
            }

            PetscInt phases = tk.size();

            //loop over the elements of rhok and epsk for however many phases exist
            for (PetscInt k=0; k<phases; k++){
                rhok.push_back((pressure + parameters.pik[k]) / (parameters.gammak[k] - 1) * parameters.gammak[k] / parameters.Cpk[k] / tk[k]);
                epsk.push_back((tk[k] * parameters.Cpk[k] / parameters.gammak[k] + parameters.pik[k]) / (parameters.gammak[k] - 1));
                
            }
            //density is sum of alphak * rhok
            PetscReal density = 0.0;
            for (PetscInt k=0; k<phases; k++){
                density += alphak[k] * rhok[k];
            }
            //rhoe is 0.5*uiui + sum_k rho_k * alphak * epsk
            PetscReal rhoe = 0.0;
            for (PetscInt k=0; k<phases; k++){
                rhoe += rhok[k] * alphak[k] * epsk[k];
            }
            PetscReal uiui = 0.0;
            for (PetscInt d = 0; d < dim; d++) {
                uiui += velocity[d] * velocity[d];
            }
            rhoe += 0.5 * uiui;

            // convert to total sensibleEnergy
            PetscReal kineticEnergy = 0;
            for (PetscInt d = 0; d < dim; d++) {
                kineticEnergy += PetscSqr(velocity[d]);
            }
            kineticEnergy *= 0.5;

            conserved[ablate::finiteVolume::NPhaseFlowFields::RHOE] = rhoe;

            for (PetscInt d = 0; d < dim; d++) {
                conserved[ablate::finiteVolume::NPhaseFlowFields::RHOU + d] = density * velocity[d];
            }
        };
        return tp;
}

//done

// (const PetscReal *conserved, PetscReal *p, void *ctx)
PetscErrorCode ablate::eos::NPhase::ComputeDecode(const PetscReal *conserved, DecodeIn &decodeIn, DecodeOut &decodeOut, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.parameters = functionContext->parameters;
    PetscReal phases = functionContext->parameters.gammak.size();
    //if any of pik size, gammak size, Cpk size are not the same as phases, throw an error
    if (functionContext->parameters.pik.size() != phases || functionContext->parameters.gammak.size() != phases || functionContext->parameters.Cpk.size() != phases) {
        throw std::invalid_argument("pik, gammak, and Cpk must be the same size");
    }
    decodeIn.alphak.resize(phases);
    decodeIn.alphakrhok.resize(phases);
    for (std::size_t k = 0; k < phases; ++k) {
        decodeIn.alphak[k] = conserved[functionContext->alphakOffset + k];
        decodeIn.alphakrhok[k] = conserved[functionContext->alphakrhokOffset + k];
    }
    //make decodeIn.rhoui a vector of size dim with components RHOU, RHOV, RHOW
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        decodeIn.rhoui[d] = conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOU + d];
    }
    decodeIn.rhoe = conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOE];
    decodeIn.parameters = functionContext->parameters;
    NStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::NPhase::PressureFunctionNStiff(const PetscReal *conserved, PetscReal *p, void *ctx) {
    //call ComputeDecode to get p
    PetscFunctionBeginUser;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    ComputeDecode(conserved, decodeIn, decodeOut, ctx);
    *p = decodeOut.p;
    PetscFunctionReturn(0);
}

//still need work

PetscErrorCode ablate::eos::NPhase::TemperatureFunctionNStiff(const PetscReal *conserved, PetscReal *temperature, void *ctx) {
    //call ComputeDecode to get p
    PetscFunctionBeginUser;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    ComputeDecode(conserved, decodeIn, decodeOut, ctx);
    *temperature = decodeOut.Tk[0];
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::NPhase::InternalSensibleEnergyFunction(const PetscReal *conserved, PetscReal *internalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // get the velocity for kinetic energy
    PetscReal density = conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    *internalEnergy = conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOE] / density - ke;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::NPhase::SensibleEnthalpyFunctionNStiff(const PetscReal *conserved, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    // Total Enthalpy == Sensible Enthalpy = e + p/rho
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHO];

    PetscReal sensibleInternalEnergy;
    InternalSensibleEnergyFunction(conserved, &sensibleInternalEnergy, ctx);

    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->alphakOffset];
    decodeIn.alphaRho1 = conserved[functionContext->alphakrhokOffset];
    decodeIn.rho = density;
    decodeIn.e = sensibleInternalEnergy;
    decodeIn.parameters = functionContext->parameters;

    NStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    PetscReal p = decodeOut.p;  // [rho1, rho2, e1, e2, p, T]

    // compute enthalpy
    *sensibleEnthalpy = sensibleInternalEnergy + p / density;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::NPhase::SpecificHeatConstantVolumeFunctionNStiff(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    // cv_mix
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->alphakOffset];
    decodeIn.alphaRho1 = conserved[functionContext->alphakrhokOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    PetscReal cv1, cv2, at1, at2;  // initialize variables for at_mix

    NStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    cv1 = functionContext->parameters.Cp1 / functionContext->parameters.gamma1;
    cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
    at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) / functionContext->parameters.gamma1 * functionContext->parameters.Cp1 * decodeOut.T);  // stiffened gas eos
    at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) / functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * decodeOut.T);  // stiffened gas eos

    PetscReal rho1 = decodeOut.rho1;
    PetscReal rho2 = decodeOut.rho2;
    PetscReal T = decodeOut.T;  // [rho1, rho2, e1, e2, p, T]
    PetscReal gamma1 = functionContext->parameters.gamma1;
    PetscReal gamma2 = functionContext->parameters.gamma2;
    PetscReal Y1 = conserved[functionContext->alphakrhokOffset] / density;
    PetscReal Y2 = (density - conserved[functionContext->alphakrhokOffset]) / density;

    // mixed specific heat constant volume
    PetscReal w1 = Y1 / PetscSqr(rho1 * at1);
    PetscReal w2 = Y2 / PetscSqr(rho2 * at2);
    (*specificHeat) = Y1 * cv1 + Y2 * cv2 + (w1 * w2) / (w1 + w2) * PetscSqr(cv1 * (gamma1 - 1) * rho1 - cv2 * (gamma2 - 1) * rho2) * T;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::NPhase::SpecificHeatConstantPressureFunctionNStiff(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    PetscReal Y1 = conserved[functionContext->alphakrhokOffset] / conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHO];
    PetscReal Y2 = (conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHO] - conserved[functionContext->alphakrhokOffset]) /
                   conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHO];
    PetscReal cp1, cp2;
    cp1 = parameters.Cp1;
    cp2 = parameters.Cp2;

    // mixed specific heat constant pressure
    (*specificHeat) = Y1 * cp1 + Y2 * cp2;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::NPhase::SpeedOfSoundFunctionNStiff(const PetscReal *conserved, PetscReal *a, void *ctx) {
    PetscFunctionBeginUser;
    // isentropic sound speed a_mix
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->alphakOffset];
    decodeIn.alphaRho1 = conserved[functionContext->alphakrhokOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    PetscReal cv1, cv2, at1, at2;  // initialize variables for at_mix

    NStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    cv1 = functionContext->parameters.Cp1 / functionContext->parameters.gamma1;
    cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
    at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) / functionContext->parameters.gamma1 * functionContext->parameters.Cp1 * decodeOut.T);  // stiffened gas eos
    at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) / functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * decodeOut.T);  // stiffened gas eos

    PetscReal rho1 = decodeOut.rho1;
    PetscReal rho2 = decodeOut.rho2;
    PetscReal T = decodeOut.T;  // [rho1, rho2, e1, e2, p, T]
    PetscReal gamma1 = functionContext->parameters.gamma1;
    PetscReal gamma2 = functionContext->parameters.gamma2;
    PetscReal Y1 = conserved[functionContext->alphakrhokOffset] / density;
    PetscReal Y2 = (density - conserved[functionContext->alphakrhokOffset]) / density;

    // mixed specific heat constant volume
    PetscReal w1 = Y1 / PetscSqr(rho1 * at1);
    PetscReal w2 = Y2 / PetscSqr(rho2 * at2);
    PetscReal cv_mix = Y1 * cv1 + Y2 * cv2 + (w1 * w2) / (w1 + w2) * PetscSqr(cv1 * (gamma1 - 1) * rho1 - cv2 * (gamma2 - 1) * rho2) * T;
    // mixed isothermal sound speed
    PetscReal at_mix = PetscSqrtReal(1 / (w1 + w2)) / density;
    PetscReal Gamma = (w1 * cv1 * (gamma1 - 1) * rho1 + w2 * cv2 * (gamma2 - 1) * rho2) / ((w1 + w2) * cv_mix * density);
    // mixed isentropic sound speed
    *a = PetscSqrtReal(PetscSqr(at_mix) + PetscSqr(Gamma) * cv_mix * T);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::NPhase::SpeciesSensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    for (PetscInt s = 0; s < parameters.numberSpecies1 + parameters.numberSpecies2; s++) {
        hi[s] = 0.0;
    }
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::NPhase::DensityFunction(const PetscReal *conserved, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHO];
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::eos::EOS, ablate::eos::NPhase, "N phase eos", 
    ARG(std::vector<ablate::eos::EOS>, "eosk", "vector of EOSs for each phase"));
    // ARG(ablate::eos::EOS, "eos1", "eos for fluid 1, must be prefect or stiffened gas."),
    //      ARG(ablate::eos::EOS, "eos2", "eos for fluid 2, must be perfect or stiffened gas."));
