#ifndef ABLATELIBRARY_TWOPHASE_HPP
#define ABLATELIBRARY_TWOPHASE_HPP

#include <memory>
#include "eos.hpp"
#include "eos/perfectGas.hpp"
#include "eos/stiffenedGas.hpp"
#include "parameters/parameters.hpp"
namespace ablate::eos {
class TwoPhase : public EOS {  // , public std::enabled_shared_from_this<TwoPhase>
   public:
    inline const static std::string VF = "volumeFraction";

   private:
    const std::vector<std::shared_ptr<eos::EOS>> eosk;

    struct Parameters {

        //for n phase we need gammak, pik, Cpk
        std::vector<PetscReal> gammak;
        std::vector<PetscReal> pik;
        std::vector<PetscReal> Cpk;

        //just in case
        std::vector<PetscReal> numberSpeciesk; 
        std::vector<std::vector<std::string>> speciesk; 

    };
    Parameters parameters;
    struct FunctionContext {

        PetscInt dim;

        PetscInt alphakOffset;
        PetscInt alphakrhokOffset;
        PetscInt rhouiOffset;
        PetscInt rhoeOffset;

        // PetscInt eulerOffset;
        // PetscInt densityVFOffset;
        // PetscInt volumeFractionOffset;
        Parameters parameters;
    };

   public:

    struct DecodeIn {
        std::vector<PetscReal> alphak;
        std::vector<PetscReal> alphakrhok;
        std::vector<PetscReal> rhoui;
        PetscReal rhoe;
        Parameters parameters;
    };
    struct DecodeOut {
        PetscReal p;
        std::vector<PetscReal> rhok;
        std::vector<PetscReal> ui;
        PetscReal rho;
        PetscReal e; //total energy eps + uiui/2
        std::vector<PetscReal> epsk; //internal energy of phase k 
        std::vector<PetscReal> Tk;
        // std::vector<PetscReal> ck;
        PetscReal c;
    };

   private:
    // functions for all cases
    static PetscErrorCode DensityFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode InternalSensibleEnergyFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpeciesSensibleEnthalpyFunction(const PetscReal conserved[], PetscReal* property, void* ctx);

    // functions for LiquidLiquid
    static PetscErrorCode PressureFunctionLiquidLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureFunctionLiquidLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SensibleEnthalpyFunctionLiquidLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantVolumeFunctionLiquidLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantPressureFunctionLiquidLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpeedOfSoundFunctionLiquidLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);

    using ThermodynamicStaticFunction = PetscErrorCode (*)(const PetscReal conserved[], PetscReal* property, void* ctx);
// map for LiquidLiquid case
    std::map<ThermodynamicProperty, ThermodynamicStaticFunction> thermodynamicFunctionsLiquidLiquid = {
        {ThermodynamicProperty::Density, DensityFunction},
        {ThermodynamicProperty::Pressure, PressureFunctionLiquidLiquid},
        {ThermodynamicProperty::Temperature, TemperatureFunctionLiquidLiquid},
        {ThermodynamicProperty::InternalSensibleEnergy, InternalSensibleEnergyFunction},
        {ThermodynamicProperty::SensibleEnthalpy, SensibleEnthalpyFunctionLiquidLiquid},
        {ThermodynamicProperty::SpecificHeatConstantVolume, SpecificHeatConstantVolumeFunctionLiquidLiquid},
        {ThermodynamicProperty::SpecificHeatConstantPressure, SpecificHeatConstantPressureFunctionLiquidLiquid},
        {ThermodynamicProperty::SpeedOfSound, SpeedOfSoundFunctionLiquidLiquid},
        {ThermodynamicProperty::SpeciesSensibleEnthalpy, SpeciesSensibleEnthalpyFunction}};

    /**
     * Store a list of properties that are sized by species, everything is assumed to be size one
     */
    const std::set<ThermodynamicProperty> speciesSizedProperties = {ThermodynamicProperty::SpeciesSensibleEnthalpy};

   public:
    explicit NPhase(std::vector<std::shared_ptr<eos::EOS>> eosk) : eosk(std::move(eosk)) {} //need std::move?

    void View(std::ostream& stream) const override;

    const std::shared_ptr<ablate::eos::EOS> GetEOSk(PetscInt k) const { 
        if (k < 0 || k >= eosk.size()) {
            throw std::out_of_range("invalid index");
        }
        return eosk[k]; 
    }

    ThermodynamicFunction GetThermodynamicFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;

    EOSFunction GetFieldFunctionFunction(const std::string& field, ThermodynamicProperty property1, ThermodynamicProperty property2, std::vector<std::string> otherProperties) const override;
    const std::vector<std::string>& GetFieldFunctionProperties() const override { return otherPropertiesList; }  // list of other properties i.e. VF;

    const std::vector<std::string>& GetSpeciesVariables() const override { return species; }  // lists species of eos1 first, then eos2, no distinction for which fluid the species exists in
    [[nodiscard]] virtual const std::vector<std::string>& GetProgressVariables() const override { return ablate::utilities::VectorUtilities::Empty<std::string>; }
};
}  // namespace ablate::eos

#endif  // ABLATELIBRARY_TWOPHASE_HPP
