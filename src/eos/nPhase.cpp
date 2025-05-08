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
        return field.name == ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK;
    });
    auto alphakField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { 
        return field.name == ablate::finiteVolume::NPhaseFlowFields::ALPHAK;
    });

    // Determine the property size
    PetscInt propertySize = 1;

    return ThermodynamicFunction{
        .function = thermodynamicFunctionsNStiff.at(property).first, //.first if you have a pair
        .context = std::make_shared<FunctionContext>(FunctionContext{.dim = allaireField->numberComponents - 1,
        .allaireOffset = allaireField->offset,
        .alphakrhokOffset = alphakrhokField->offset,
        .alphakOffset = alphakField->offset,
        .parameters = parameters}),
        .propertySize = propertySize};
}

ablate::eos::EOSFunction ablate::eos::NPhase::GetFieldFunctionFunction(
    const std::string &field, 
    ablate::eos::ThermodynamicProperty property1, 
    ablate::eos::ThermodynamicProperty property2,
    std::vector<std::string> otherProperties) const {

    if (otherProperties != std::vector<std::string>{ALPHAKRHOK, ALPHAK}) {  
        throw std::invalid_argument("ablate::eos::NPhase expects other properties to include ALPHAKRHOK (partial density of 1..k..N phasesa) and ALPHAK (volume fraction of 1..k..N phases) as first and second entries");
    }

    // auto tp = [this](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[])
        //assume that ALLAIRE_FIELD was specified and that we have p,tk, (alphak, alphakrhok) = otherk
    auto tp = [this](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal otherk[], 
            PetscReal conserved[]) {

            std::size_t phases = this->parameters.gammak.size();
            if (this->parameters.pik.size() != phases || this->parameters.gammak.size() != phases || this->parameters.Cpk.size() != phases) {
                throw std::invalid_argument("pik, gammak, and Cpk must be the same size");
            }

            //otherk is formerly yi
            //otherk must be a vector of size 2*phases, first half is alphakrhok, second half is alphak
            const PetscReal *alphakrhok = &otherk[0];
            const PetscReal *alphak = &otherk[phases];
            
            //if any of pik size, gammak size, Cpk size are not the same as phases, throw an error
            

            PetscReal rhok[phases];
            PetscReal epsk[phases];

            //loop over the elements of rhok and epsk for however many phases exist

            for (std::size_t k=0; k<phases; k++){
                // rhok[k] = (pressure + parameters.pik[k]) / (parameters.gammak[k] - 1) * parameters.gammak[k] / parameters.Cpk[k] / tk[k]; //check ? this was in twophase
                // epsk[k] = (tk[k] * parameters.Cpk[k] / parameters.gammak[k] + parameters.pik[k]) / (parameters.gammak[k] - 1); //check ? this was in twophase

                rhok[k] = alphakrhok[k] / alphak[k];
                epsk[k] = (pressure + parameters.gammak[k] * parameters.pik[k]) / ((parameters.gammak[k] - 1) * rhok[k]);
                
            }

            //density is sum of alphak * rhok
            PetscReal density = 0.0;
            for (std::size_t k=0; k<phases; k++){
                density += alphakrhok[k];
            }
            //rhoe is 0.5*uiui + sum_k rho_k * alphak * epsk
            PetscReal rhoe = 0.0;
            for (std::size_t k=0; k<phases; k++){
                rhoe += alphakrhok[k] * epsk[k];
            }
            PetscReal uiui = 0.0;
            for (PetscInt d = 0; d < dim; d++) {
                uiui += velocity[d] * velocity[d];
            }
            rhoe += 0.5 * uiui;

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
    // DecodeOut decodeOut;
    // DecodeIn decodeIn;
    decodeIn.parameters = functionContext->parameters;
    PetscReal phases = functionContext->parameters.gammak.size();
    //if any of pik size, gammak size, Cpk size are not the same as phases, throw an error
    if (functionContext->parameters.pik.size() != phases || functionContext->parameters.gammak.size() != phases || functionContext->parameters.Cpk.size() != phases) {
        throw std::invalid_argument("pik, gammak, and Cpk must be the same size");
    }
    decodeIn.alphak.resize(phases);
    decodeIn.alphakrhok.resize(phases);
    for (PetscInt k = 0; k < phases; ++k) {
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

PetscErrorCode ablate::eos::NPhase::TemperatureFunctionNStiff(const PetscReal *conserved, PetscReal *temperature, void *ctx) {
    //call ComputeDecode to get p
    PetscFunctionBeginUser;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    ComputeDecode(conserved, decodeIn, decodeOut, ctx);
    for (std::size_t k = 0; k < decodeIn.alphak.size(); ++k) {
        temperature[k] = decodeOut.Tk[k]; //PetscReal can be an array of temperatures
    }
    PetscFunctionReturn(0);
}


//the map requires a pair including (static function, temperature function), whereas we don't have use for temperature functions,
//so this is its placeholder
PetscErrorCode ablate::eos::NPhase::nullNStiff(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) {
    return 0;
}

const std::vector<std::string>& ablate::eos::NPhase::GetSpeciesVariables() const {
    static const std::vector<std::string> none{};
    return none;
}

const std::vector<std::string>& ablate::eos::NPhase::GetProgressVariables() const {
    static const std::vector<std::string> none{};
    return none;
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::NPhase::GetThermodynamicTemperatureFunction(
    ablate::eos::ThermodynamicProperty property, const std::vector<ablate::domain::Field>& fields) const {
    throw std::invalid_argument("GetThermodynamicTemperatureFunction is not implemented for ablate::eos::NPhase");
}

PetscErrorCode ablate::eos::NPhase::InternalSensibleEnergyFunction(const PetscReal *conserved, PetscReal *internalEnergy, void *ctx) {

    //total eps = sumk epsk

    PetscFunctionBeginUser;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    ComputeDecode(conserved, decodeIn, decodeOut, ctx);
    auto functionContext = (FunctionContext *)ctx;

    // ui = rhoui/(rho = sumk alphakrhok)
    PetscReal uiui = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        uiui += (decodeIn.rhoui[d] / decodeOut.rho) * (decodeIn.rhoui[d] / decodeOut.rho);
    }
    *internalEnergy = decodeIn.rhoe / decodeOut.rho - 0.5 * uiui;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::NPhase::SensibleEnthalpyFunctionNStiff(const PetscReal *conserved, PetscReal *sensibleEnthalpy, void *ctx) {
    
    //internalsensibleenergyfunction + p/rho

    PetscFunctionBeginUser;
    // auto functionContext = (FunctionContext *)ctx;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    ComputeDecode(conserved, decodeIn, decodeOut, ctx);

    PetscReal sensibleInternalEnergy;
    InternalSensibleEnergyFunction(conserved, &sensibleInternalEnergy, ctx);

    PetscReal p = decodeOut.p;  // [rho1, rho2, e1, e2, p, T]

    // compute enthalpy
    *sensibleEnthalpy = sensibleInternalEnergy + p / decodeOut.rho;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::NPhase::SpecificHeatConstantVolumeFunctionNStiff(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {

    //cv = sumk ( alphak Cvk Tk ) / sumk (alphak Tk) 
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    ComputeDecode(conserved, decodeIn, decodeOut, ctx);

    *specificHeat = 0.0;
    for (std::size_t k = 0; k < decodeIn.alphak.size(); ++k) {
        *specificHeat += decodeIn.alphak[k] * (functionContext->parameters.Cpk[k]/functionContext->parameters.gammak[k]) * decodeOut.Tk[k];
    }
    PetscReal denom = 0.0;
    for (std::size_t k = 0; k < decodeIn.alphak.size(); ++k) {
        denom += decodeIn.alphak[k] * decodeOut.Tk[k];
    }
    *specificHeat /= denom;

    PetscFunctionReturn(0);


}

PetscErrorCode ablate::eos::NPhase::SpecificHeatConstantPressureFunctionNStiff(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    ComputeDecode(conserved, decodeIn, decodeOut, ctx);

    PetscInt phases = decodeIn.alphakrhok.size();


    // Cp = sumk (Yk cpk Tk) / sumk  (Yk Tk), Yk = rhokalphak/rho
    *specificHeat = 0.0;
    for (PetscInt k = 0; k < phases; ++k) {
        *specificHeat += (conserved[functionContext->alphakrhokOffset + k] / decodeOut.rho) * functionContext->parameters.Cpk[k] * decodeOut.Tk[k];
    }
    PetscReal denom = 0.0;
    for (PetscInt k = 0; k < phases; ++k) {
        denom += (conserved[functionContext->alphakrhokOffset + k] / decodeOut.rho) * decodeOut.Tk[k];
    }
    *specificHeat /= denom;

    PetscFunctionReturn(0);
}

//still need work

PetscErrorCode ablate::eos::NPhase::SpeedOfSoundFunctionNStiff(const PetscReal *conserved, PetscReal *a, void *ctx) {

    PetscFunctionBeginUser;
    // auto functionContext = (FunctionContext *)ctx;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    ComputeDecode(conserved, decodeIn, decodeOut, ctx);

    *a = decodeOut.c;

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::NPhase::SpeciesSensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;

    //total number species = sumk numberSpeciesk
    PetscReal totalNumberSpecies = 0;
    PetscInt phases = parameters.gammak.size();
    for (PetscInt k = 0; k < phases; ++k) {
        totalNumberSpecies += parameters.numberSpeciesk[k];
    }

    for (PetscInt s = 0; s < totalNumberSpecies; s++) {
        hi[s] = 0.0;
    }
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::NPhase::DensityFunction(const PetscReal *conserved, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    // auto functionContext = (FunctionContext *)ctx;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    ComputeDecode(conserved, decodeIn, decodeOut, ctx);

    *density = decodeOut.rho;

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::eos::EOS, ablate::eos::NPhase, "N phase eos", 
    ARG(std::vector<ablate::eos::EOS>, "eosk", "vector of EOSs for each phase"));
    // ARG(ablate::eos::EOS, "eos1", "eos for fluid 1, must be prefect or stiffened gas."),
    //      ARG(ablate::eos::EOS, "eos2", "eos for fluid 2, must be perfect or stiffened gas."));
