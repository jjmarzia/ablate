#include "nPhaseFlowFields.hpp"

#include <utility>
#include "domain/fieldDescription.hpp"
#include "utilities/vectorUtilities.hpp"

ablate::finiteVolume::NPhaseFlowFields::NPhaseFlowFields(std::shared_ptr<eos::EOS> eos, std::shared_ptr<domain::Region> region,
                                                                     std::shared_ptr<parameters::Parameters> conservedFieldParameters)
    : eos(std::move(eos)), region(std::move(region)), conservedFieldOptions(std::move(conservedFieldParameters)) {}

std::vector<std::shared_ptr<ablate::domain::FieldDescription>> ablate::finiteVolume::NPhaseFlowFields::GetFields() {
    std::vector<std::shared_ptr<ablate::domain::FieldDescription>> flowFields{
        std::make_shared<domain::FieldDescription>(
            ALLAIRE_FIELD, ALLAIRE_FIELD,
            std::vector<std::string>{"rhoe", "rhovel" + domain::FieldDescription::DIMENSION},
            domain::FieldLocation::SOL,
            domain::FieldType::FVM,
            region,
            conservedFieldOptions),

        std::make_shared<domain::FieldDescription>(
            VELOCITY_FIELD, VELOCITY_FIELD, 
            std::vector<std::string>{"vel" + domain::FieldDescription::DIMENSION}, 
            domain::FieldLocation::AUX, 
            domain::FieldType::FVM, 
            region, 
            auxFieldOptions)
        };
    return flowFields;
}

#include "registrar.hpp"
REGISTER(ablate::domain::FieldDescriptor, ablate::finiteVolume::NPhaseFlowFields, "fields needed for nPhase flow",
         ARG(ablate::eos::EOS, "eos", "the equation of state to be used for the flow"), 
         OPT(ablate::domain::Region, "region", "the region for the flow (defaults to entire domain)"),
         OPT(ablate::parameters::Parameters, "conservedFieldOptions", "petsc options used for the conserved fields.  Common options would be petscfv_type and petsclimiter_type"));
