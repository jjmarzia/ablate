#include "kthStiffenedGas.hpp"
#include "finiteVolume/nPhaseFlowFields.hpp"
#include "registrar.hpp"

ablate::eos::KthStiffenedGas::KthStiffenedGas(std::shared_ptr<ablate::parameters::Parameters> parametersIn, std::vector<std::string> species) : EOS("kthStiffenedGas"), species(species) {
    parameters.gamma = parametersIn->GetExpect<PetscReal>("gamma");
    parameters.Cp = parametersIn->GetExpect<PetscReal>("Cp");
    parameters.p0 = parametersIn->GetExpect<PetscReal>("p0");
    parameters.numberSpecies = (PetscInt)species.size();
}

void ablate::eos::KthStiffenedGas::View(std::ostream& stream) const {
    stream << "KthStiffenedGas:"
          << "\n\tgamma: " << parameters.gamma << "\n\tCp: " << parameters.Cp << "\n\tp0: " << parameters.p0;
}

PetscErrorCode ablate::eos::KthStiffenedGas::ComputeTotalDensity(const PetscReal conserved[], const std::vector<domain::Field>& fields, PetscReal* density) {
    PetscFunctionBeginUser;
    
    // Get the alphakrhok field
    auto alphakrhokField = std::find_if(fields.begin(), fields.end(), [](const auto& field) { 
        return field.name == ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK;
    });
    if (alphakrhokField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::KthStiffenedGas requires the ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK Field");
    }

    // Total density is sum of alphakrhok
    *density = 0.0;
    for (PetscInt k = 0; k < alphakrhokField->numberComponents; k++) {
        *density += conserved[alphakrhokField->offset + k];
    }
    PetscFunctionReturn(0);
}

ablate::eos::ThermodynamicFunction ablate::eos::KthStiffenedGas::GetThermodynamicFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field>& fields) const {
    // Look for the allaire field
    auto allaireField = std::find_if(fields.begin(), fields.end(), [](const auto& field) { return field.name == ablate::finiteVolume::NPhaseFlowFields::ALLAIRE_FIELD; });
    if (allaireField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::KthStiffenedGas requires the ablate::finiteVolume::NPhaseFlowFields::ALLAIRE_FIELD Field");
    }

    return ThermodynamicFunction{
        .function = std::get<0>(thermodynamicFunctions.at(property)),
        .context = std::make_shared<FunctionContext>(FunctionContext{.dim = allaireField->numberComponents - 1, .allaireOffset = allaireField->offset, .parameters = parameters, .fields = &fields}),
        .propertySize = std::get<2>(thermodynamicFunctions.at(property)) == SPECIES_SIZE ? (PetscInt)species.size() : PetscInt(std::get<2>(thermodynamicFunctions.at(property)))};
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::KthStiffenedGas::GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty property,
                                                                                                               const std::vector<domain::Field>& fields) const {
    // Look for the allaire field
    auto allaireField = std::find_if(fields.begin(), fields.end(), [](const auto& field) { return field.name == ablate::finiteVolume::NPhaseFlowFields::ALLAIRE_FIELD; });
    if (allaireField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::KthStiffenedGas requires the ablate::finiteVolume::NPhaseFlowFields::ALLAIRE_FIELD Field");
    }

    return ThermodynamicTemperatureFunction{
        .function = std::get<1>(thermodynamicFunctions.at(property)),
        .context = std::make_shared<FunctionContext>(FunctionContext{.dim = allaireField->numberComponents - 1, .allaireOffset = allaireField->offset, .parameters = parameters, .fields = &fields}),
        .propertySize = std::get<2>(thermodynamicFunctions.at(property)) == SPECIES_SIZE ? (PetscInt)species.size() : PetscInt(std::get<2>(thermodynamicFunctions.at(property)))};
}

ablate::eos::EOSFunction ablate::eos::KthStiffenedGas::GetFieldFunctionFunction(const std::string& field, ablate::eos::ThermodynamicProperty property1, ablate::eos::ThermodynamicProperty property2,
                                                                              std::vector<std::string> otherProperties) const {
    if (ablate::finiteVolume::NPhaseFlowFields::ALLAIRE_FIELD == field) {
        if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) ||
            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature)) {
            auto tp = [this](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density
                PetscReal gam = parameters.gamma;
                PetscReal density = ((pressure + parameters.p0) / (gam - 1)) * gam / parameters.Cp / temperature;

                // compute the sensible internal energy
                PetscReal sensibleInternalEnergy = temperature * parameters.Cp / parameters.gamma + parameters.p0 / density;

                // convert to total sensibleEnergy
                PetscReal kineticEnergy = 0;
                for (PetscInt d = 0; d < dim; d++) {
                    kineticEnergy += PetscSqr(velocity[d]);
                }
                kineticEnergy *= 0.5;

                conserved[ablate::finiteVolume::NPhaseFlowFields::RHOE] = density * (kineticEnergy + sensibleInternalEnergy);
                for (PetscInt d = 0; d < dim; d++) {
                    conserved[ablate::finiteVolume::NPhaseFlowFields::RHOU + d] = density * velocity[d];
                }
            };
            if (property1 == ThermodynamicProperty::Temperature) {
                return tp;
            } else {
                return [tp](PetscReal pressure, PetscReal temperature, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                    tp(temperature, pressure, dim, velocity, yi, conserved);
                };
            }
        }

        // pressure and energy
        if ((property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::InternalSensibleEnergy) ||
            (property1 == ThermodynamicProperty::InternalSensibleEnergy && property2 == ThermodynamicProperty::Pressure)) {
            auto iep = [this](PetscReal internalSensibleEnergy, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density
                PetscReal density = (pressure + parameters.gamma * parameters.p0) / ((parameters.gamma - 1.0) * internalSensibleEnergy);

                // convert to total sensibleEnergy
                PetscReal kineticEnergy = 0;
                for (PetscInt d = 0; d < dim; d++) {
                    kineticEnergy += PetscSqr(velocity[d]);
                }
                kineticEnergy *= 0.5;

                conserved[ablate::finiteVolume::NPhaseFlowFields::RHOE] = density * (kineticEnergy + internalSensibleEnergy);
                for (PetscInt d = 0; d < dim; d++) {
                    conserved[ablate::finiteVolume::NPhaseFlowFields::RHOU + d] = density * velocity[d];
                }
            };
            if (property1 == ThermodynamicProperty::InternalSensibleEnergy) {
                return iep;
            } else {
                return [iep](PetscReal pressure, PetscReal internalSensibleEnergy, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                    iep(internalSensibleEnergy, pressure, dim, velocity, yi, conserved);
                };
            }
        }

        throw std::invalid_argument("Unknown property combination(" + std::string(to_string(property1)) + "," + std::string(to_string(property2)) + ") for " + field +
                                  " for ablate::eos::KthStiffenedGas.");
    }
    throw std::invalid_argument("Unknown field type " + field + " for ablate::eos::KthStiffenedGas.");
}

PetscErrorCode ablate::eos::KthStiffenedGas::DensityFunction(const PetscReal conserved[], PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext*)ctx;
    PetscCall(ComputeTotalDensity(conserved, *functionContext->fields, property));
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::KthStiffenedGas::PressureFunction(const PetscReal conserved[], PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext*)ctx;

    // Get total density
    PetscReal density;
    PetscCall(ComputeTotalDensity(conserved, *functionContext->fields, &density));

    // Get kinetic energy
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // Get internal energy from total energy minus kinetic
    PetscReal internalEnergy = conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOE] / density - ke;
    *property = (functionContext->parameters.gamma - 1.0) * density * internalEnergy - functionContext->parameters.gamma * functionContext->parameters.p0;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::KthStiffenedGas::TemperatureFunction(const PetscReal conserved[], PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext*)ctx;

    // Get total density
    PetscReal density;
    PetscCall(ComputeTotalDensity(conserved, *functionContext->fields, &density));

    // Get kinetic energy
    PetscReal speedSquare = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        speedSquare += PetscSqr(conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOU + d] / density);
    }

    // Get internal energy from total energy minus kinetic
    PetscReal internalEnergy = conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOE] / density - 0.5 * speedSquare;
    *property = (internalEnergy - functionContext->parameters.p0 / density) * functionContext->parameters.gamma / functionContext->parameters.Cp;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::KthStiffenedGas::InternalSensibleEnergyFunction(const PetscReal conserved[], PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext*)ctx;

    // Get total density
    PetscReal density;
    PetscCall(ComputeTotalDensity(conserved, *functionContext->fields, &density));

    // Get kinetic energy
    PetscReal speedSquare = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        speedSquare += PetscSqr(conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOU + d] / density);
    }

    // Get internal energy from total energy minus kinetic
    *property = conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOE] / density - 0.5 * speedSquare;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::KthStiffenedGas::SensibleEnthalpyFunction(const PetscReal conserved[], PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext*)ctx;

    // Get total density
    PetscReal density;
    PetscCall(ComputeTotalDensity(conserved, *functionContext->fields, &density));

    PetscReal sensibleInternalEnergy;
    PetscCall(InternalSensibleEnergyFunction(conserved, &sensibleInternalEnergy, ctx));

    PetscReal pressure;
    PetscCall(PressureFunction(conserved, &pressure, ctx));

    // Total Enthalpy == Sensible Enthalpy = e + p/rho
    *property = sensibleInternalEnergy + pressure / density;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::KthStiffenedGas::SpecificHeatConstantVolumeFunction(const PetscReal conserved[], PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext*)ctx;
    *property = functionContext->parameters.Cp / functionContext->parameters.gamma;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::KthStiffenedGas::SpecificHeatConstantPressureFunction(const PetscReal conserved[], PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext*)ctx;
    *property = functionContext->parameters.Cp;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::KthStiffenedGas::SpeedOfSoundFunction(const PetscReal conserved[], PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext*)ctx;

    // Get total density
    PetscReal density;
    PetscCall(ComputeTotalDensity(conserved, *functionContext->fields, &density));

    // Get kinetic energy
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // Get internal energy from total energy minus kinetic
    PetscReal internalEnergy = conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOE] / density - ke;
    PetscReal p = (functionContext->parameters.gamma - 1.0) * density * internalEnergy - functionContext->parameters.gamma * functionContext->parameters.p0;
    *property = PetscSqrtReal(functionContext->parameters.gamma * (p + functionContext->parameters.p0) / density);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::KthStiffenedGas::SpeciesSensibleEnthalpyFunction(const PetscReal conserved[], PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext*)ctx;

    // Get total density
    PetscReal density;
    PetscCall(ComputeTotalDensity(conserved, *functionContext->fields, &density));

    PetscReal sensibleInternalEnergy;
    PetscCall(InternalSensibleEnergyFunction(conserved, &sensibleInternalEnergy, ctx));

    PetscReal pressure;
    PetscCall(PressureFunction(conserved, &pressure, ctx));

    // Total Enthalpy == Sensible Enthalpy = e + p/rho
    *property = sensibleInternalEnergy + pressure / density;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::KthStiffenedGas::DensityTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext*)ctx;
    PetscCall(ComputeTotalDensity(conserved, *functionContext->fields, property));
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::KthStiffenedGas::PressureTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext*)ctx;

    // Get total density
    PetscReal density;
    PetscCall(ComputeTotalDensity(conserved, *functionContext->fields, &density));

    // Get kinetic energy
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // Get internal energy from total energy minus kinetic
    PetscReal internalEnergy = conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOE] / density - ke;
    *property = (functionContext->parameters.gamma - 1.0) * density * internalEnergy - functionContext->parameters.gamma * functionContext->parameters.p0;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::KthStiffenedGas::TemperatureTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    *property = T;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::KthStiffenedGas::InternalSensibleEnergyTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext*)ctx;

    // Get total density
    PetscReal density;
    PetscCall(ComputeTotalDensity(conserved, *functionContext->fields, &density));

    // Get kinetic energy
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // Get internal energy from total energy minus kinetic
    *property = conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOE] / density - ke;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::KthStiffenedGas::SensibleEnthalpyTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext*)ctx;

    // Get total density
    PetscReal density;
    PetscCall(ComputeTotalDensity(conserved, *functionContext->fields, &density));

    PetscReal sensibleInternalEnergy;
    PetscCall(InternalSensibleEnergyFunction(conserved, &sensibleInternalEnergy, ctx));

    PetscReal pressure;
    PetscCall(PressureFunction(conserved, &pressure, ctx));

    // Total Enthalpy == Sensible Enthalpy = e + p/rho
    *property = sensibleInternalEnergy + pressure / density;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::KthStiffenedGas::SpecificHeatConstantVolumeTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext*)ctx;
    *property = functionContext->parameters.Cp / functionContext->parameters.gamma;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::KthStiffenedGas::SpecificHeatConstantPressureTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext*)ctx;
    *property = functionContext->parameters.Cp;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::KthStiffenedGas::SpeedOfSoundTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext*)ctx;

    // Get total density
    PetscReal density;
    PetscCall(ComputeTotalDensity(conserved, *functionContext->fields, &density));

    // Get kinetic energy
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // Get internal energy from total energy minus kinetic
    PetscReal internalEnergy = conserved[functionContext->allaireOffset + ablate::finiteVolume::NPhaseFlowFields::RHOE] / density - ke;
    PetscReal p = (functionContext->parameters.gamma - 1.0) * density * internalEnergy - functionContext->parameters.gamma * functionContext->parameters.p0;
    *property = PetscSqrtReal(functionContext->parameters.gamma * (p + functionContext->parameters.p0) / density);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::KthStiffenedGas::SpeciesSensibleEnthalpyTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext*)ctx;

    // Get total density
    PetscReal density;
    PetscCall(ComputeTotalDensity(conserved, *functionContext->fields, &density));

    PetscReal sensibleInternalEnergy;
    PetscCall(InternalSensibleEnergyFunction(conserved, &sensibleInternalEnergy, ctx));

    PetscReal pressure;
    PetscCall(PressureFunction(conserved, &pressure, ctx));

    // Total Enthalpy == Sensible Enthalpy = e + p/rho
    *property = sensibleInternalEnergy + pressure / density;
    PetscFunctionReturn(0);
}

REGISTER(ablate::eos::EOS, ablate::eos::KthStiffenedGas, "kth stiffened gas eos", 
    ARG(ablate::parameters::Parameters, "parameters", "parameters for the kth stiffened gas eos"),
    OPT(std::vector<std::string>, "species", "species to track.  Note: species mass fractions do not change eos")); 