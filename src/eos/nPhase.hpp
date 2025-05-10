#ifndef ABLATELIBRARY_NPHASE_HPP
#define ABLATELIBRARY_NPHASE_HPP

#include <memory>
#include "eos.hpp"
#include "eos/perfectGas.hpp"
#include "eos/stiffenedGas.hpp"
#include "parameters/parameters.hpp"
namespace ablate::eos {
class NPhase : public EOS {  // , public std::enabled_shared_from_this<NPhase>
   public:
    std::vector<std::string> species;
    inline const static std::string ALPHAKRHOK = "alphakrhok";
    inline const static std::string ALPHAK = "alphak";

   private:
    const std::vector<std::shared_ptr<eos::EOS>> eosk;
    std::vector<std::string> otherPropertiesList = {"ALPHAKRHOK, ALPHAK"};

    struct Parameters {

        //for n phase we need gammak, pik, Cpk
        std::vector<PetscReal> gammak;
        std::vector<PetscReal> pik;
        std::vector<PetscReal> Cpk;

        //just in case?
        std::vector<PetscReal> numberSpeciesk; 
        std::vector<std::vector<std::string>> speciesk; 

    };
    Parameters parameters;
    struct FunctionContext {

        PetscInt dim;

        //Allaire variables
        PetscInt allaireOffset;
        //need equiv of alphakOffset, alphakrhokOffset for n phases
        PetscInt alphakOffset;
        PetscInt alphakrhokOffset;
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

    static PetscErrorCode ComputeDecode(const PetscReal conserved[], DecodeIn &decodeIn, DecodeOut &decodeOut, void* ctx);

    static PetscErrorCode PressureFunctionNStiff(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureFunctionNStiff(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SensibleEnthalpyFunctionNStiff(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantVolumeFunctionNStiff(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantPressureFunctionNStiff(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpeedOfSoundFunctionNStiff(const PetscReal conserved[], PetscReal* property, void* ctx);

    static PetscErrorCode nullNStiff(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);

    using ThermodynamicStaticFunction = PetscErrorCode (*)(const PetscReal conserved[], PetscReal* property, void* ctx);
    using nullfunction = PetscErrorCode (*)(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);

    std::map<ThermodynamicProperty, std::pair<ThermodynamicStaticFunction, nullfunction>> thermodynamicFunctionsNStiff = {
        //p, rhok, ui, rho, e, epsk, Tk, ck
    
        {ThermodynamicProperty::Density, {DensityFunction, nullNStiff}},
        {ThermodynamicProperty::Pressure, {PressureFunctionNStiff, nullNStiff}},
        {ThermodynamicProperty::Temperature, {TemperatureFunctionNStiff, nullNStiff}},
        {ThermodynamicProperty::InternalSensibleEnergy, {InternalSensibleEnergyFunction, nullNStiff}},
        {ThermodynamicProperty::SensibleEnthalpy, {SensibleEnthalpyFunctionNStiff, nullNStiff}},
        {ThermodynamicProperty::SpecificHeatConstantVolume, {SpecificHeatConstantVolumeFunctionNStiff, nullNStiff}},
        {ThermodynamicProperty::SpecificHeatConstantPressure, {SpecificHeatConstantPressureFunctionNStiff, nullNStiff}},
        {ThermodynamicProperty::SpeedOfSound, {SpeedOfSoundFunctionNStiff, nullNStiff}},
        {ThermodynamicProperty::SpeciesSensibleEnthalpy, {SpeciesSensibleEnthalpyFunction, nullNStiff}}};

    /**
     * Store a list of properties that are sized by species, everything is assumed to be size one
     */
    const std::set<ThermodynamicProperty> speciesSizedProperties = {ThermodynamicProperty::SpeciesSensibleEnthalpy};

   public:
    explicit NPhase(std::vector<std::shared_ptr<eos::EOS>> eosk);

    void View(std::ostream& stream) const override;

    const std::shared_ptr<ablate::eos::EOS> GetEOSk(std::size_t k) const { 
        if (k < 0 || k >= eosk.size()) {
            throw std::out_of_range("invalid index");
        }
        return eosk[k]; 
    }

    ThermodynamicFunction GetThermodynamicFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;

    ThermodynamicTemperatureFunction GetThermodynamicTemperatureFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;
    const std::vector<std::string>& GetSpeciesVariables() const override;
    const std::vector<std::string>& GetProgressVariables() const override;


    EOSFunction GetFieldFunctionFunction(const std::string& field, ThermodynamicProperty property1, ThermodynamicProperty property2, std::vector<std::string> otherProperties) const override;
    const std::vector<std::string>& GetFieldFunctionProperties() const override { return otherPropertiesList; }  // list of other properties i.e. VF;

    std::size_t GetNumberOfPhases() const { return eosk.size(); }

};
}  // namespace ablate::eos

#endif  // ABLATELIBRARY_NPHASE_HPP
