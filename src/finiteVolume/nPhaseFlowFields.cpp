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

        //register alphakrhok, alphak
        std::make_shared<domain::FieldDescription>(
            ALPHAKRHOK, ALPHAKRHOK,
            std::vector<std::string>{"alphakrhok"},
            domain::FieldLocation::SOL,
            domain::FieldType::FVM,
            region,
            conservedFieldOptions),

        std::make_shared<domain::FieldDescription>(
            ALPHAK, ALPHAK,
            std::vector<std::string>{"alphak"},
            domain::FieldLocation::SOL,
            domain::FieldType::FVM,
            region,
            conservedFieldOptions),

        //do tk, p, rho, rhok, e, ek
        // std::make_shared<domain::FieldDescription>(
        //     TK, TK, 
        //     std::vector<std::string>{"tk"}, // N phases ?
        //     domain::FieldLocation::AUX, 
        //     domain::FieldType::FVM, 
        //     region, 
        //     conservedFieldOptions),

        std::make_shared<domain::FieldDescription>(
            UI, UI, 
            std::vector<std::string>{"vel" + domain::FieldDescription::DIMENSION}, 
            domain::FieldLocation::AUX, 
            domain::FieldType::FVM, 
            region, 
            auxFieldOptions)
        };
        
        // if (!eos->GetSpeciesVariables().empty()) {
        //     flowFields.emplace_back(std::make_shared<domain::FieldDescription>(
        //         DENSITY_YI_FIELD, DENSITY_YI_FIELD, eos->GetSpeciesVariables(), domain::FieldLocation::SOL, domain::FieldType::FVM, region, conservedFieldOptions, eos->GetFieldTags()));
        //     flowFields.emplace_back(
        //         std::make_shared<domain::FieldDescription>(YI_FIELD, YI_FIELD, eos->GetSpeciesVariables(), domain::FieldLocation::AUX, domain::FieldType::FVM, region, auxFieldOptions));
        // }
    
        // if (!eos->GetProgressVariables().empty()) {
        //     flowFields.emplace_back(std::make_shared<domain::FieldDescription>(DENSITY_PROGRESS_FIELD,
        //                                                                        DENSITY_PROGRESS_FIELD,
        //                                                                        eos->GetProgressVariables(),
        //                                                                        domain::FieldLocation::SOL,
        //                                                                        domain::FieldType::FVM,
        //                                                                        region,
        //                                                                        conservedFieldOptions,
        //                                                                        ablate::utilities::VectorUtilities::Merge(eos->GetFieldTags(), {EV_TAG})));
        //     flowFields.emplace_back(
        //         std::make_shared<domain::FieldDescription>(PROGRESS_FIELD, PROGRESS_FIELD, eos->GetProgressVariables(), domain::FieldLocation::AUX, domain::FieldType::FVM, region, auxFieldOptions));
        // }
    
        // // check the eos/chemModel for any additional required fields
        for (auto& fieldDescriptor : eos->GetAdditionalFields()) {
            for (auto& field : fieldDescriptor->GetFields()) {
                switch (field->location) {
                    case domain::FieldLocation::SOL:
                        flowFields.push_back(field->Specialize(domain::FieldType::FVM, region, conservedFieldOptions));
                        break;
                    case domain::FieldLocation::AUX:
                        flowFields.push_back(field->Specialize(domain::FieldType::FVM, region, auxFieldOptions));
                        break;
                }
            }
        }


    return flowFields;
}

#include "registrar.hpp"
REGISTER(ablate::domain::FieldDescriptor, ablate::finiteVolume::NPhaseFlowFields, "fields needed for nPhase flow",
         ARG(ablate::eos::EOS, "eos", "the equation of state to be used for the flow (use stiffened gas?)"), 
         OPT(ablate::domain::Region, "region", "the region for the flow (defaults to entire domain)"),
         OPT(ablate::parameters::Parameters, "conservedFieldOptions", "petsc options used for the conserved fields.  Common options would be petscfv_type and petsclimiter_type"));
