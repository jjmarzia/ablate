#include "nPhaseFlowFields.hpp"

#include <utility>
#include "domain/fieldDescription.hpp"
#include "utilities/vectorUtilities.hpp"
#include "eos/nPhase.hpp"

ablate::finiteVolume::NPhaseFlowFields::NPhaseFlowFields(std::shared_ptr<eos::EOS> eos, 
    std::shared_ptr<domain::Region> region,
                                                                     
    std::shared_ptr<parameters::Parameters> conservedFieldParameters,
    PetscInt dimensions)
    : eos(std::move(eos)), 
    region(std::move(region)), 
    conservedFieldOptions(std::move(conservedFieldParameters)), 
    dim(dimensions) {  }

std::vector<std::shared_ptr<ablate::domain::FieldDescription>> ablate::finiteVolume::NPhaseFlowFields::GetFields() {
    // Get number of phases from EOS
    auto nPhaseEOS = std::dynamic_pointer_cast<eos::NPhase>(eos);
    if (!nPhaseEOS) {
        throw std::invalid_argument("EOS must be of type NPhase");
    }
    std::size_t phases = nPhaseEOS->GetNumberOfPhases();

    // Create component names for alphakrhok and alphak
    std::vector<std::string> alphakrhokComponents;
    std::vector<std::string> alphakComponents;
    for (std::size_t k = 0; k < phases; k++) {
        alphakrhokComponents.push_back("alphakrhok" + std::to_string(k));
        alphakComponents.push_back("alphak" + std::to_string(k));
    }

    // Create component names for Aij (unique i<j pairs)
    std::vector<std::string> aijComponents;
    if (phases >= 2) {
        for (std::size_t i = 0; i < phases; i++) {
            for (std::size_t j = i + 1; j < phases; j++) {
                aijComponents.push_back("aij_" + std::to_string(i) + "_" + std::to_string(j));
            }
        }
    }

    std::vector<std::shared_ptr<ablate::domain::FieldDescription>> flowFields{
        std::make_shared<domain::FieldDescription>(
            ALLAIRE_FIELD, ALLAIRE_FIELD,
            std::vector<std::string>{"rhoe", "rhovel" + domain::FieldDescription::DIMENSION},
            domain::FieldLocation::SOL,
            domain::FieldType::FVM,
            region,
            ablate::parameters::MapParameters::Create({
        {"petscfv_type", "leastsquares"}, 
        {"petsclimiter_type", "none"},
        {"petscfv_compute_gradients", "false"} //TRUE
    })),

        //register alphak FIRST, then alphakrhok
        std::make_shared<domain::FieldDescription>(
            ALPHAK, ALPHAK,
            alphakComponents,
            domain::FieldLocation::SOL,
            domain::FieldType::FVM,
            region,
            ablate::parameters::MapParameters::Create({
        {"petscfv_type", "leastsquares"}, 
        {"petsclimiter_type", "none"},
        {"petscfv_compute_gradients", "false"} //TRUE
    })),

        std::make_shared<domain::FieldDescription>(
            ALPHAKRHOK, ALPHAKRHOK,
            alphakrhokComponents,
            domain::FieldLocation::SOL,
            domain::FieldType::FVM,
            region,
            ablate::parameters::MapParameters::Create({
        {"petscfv_type", "leastsquares"}, 
        {"petsclimiter_type", "none"},
        {"petscfv_compute_gradients", "false"} //should be FALSE as reconstructed ALPHAKRHOK should come from slope limited ALPHAK
        //i.e. (ALPHAKRHOK)_R = (ALPHAK)_R rhok
    })),

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
            auxFieldOptions),



        // Aij interface indicator per unique pair (i<j)
        std::make_shared<domain::FieldDescription>(
            AIJ, AIJ,
            aijComponents,
            domain::FieldLocation::AUX,
            domain::FieldType::FVM,
            region,
            auxFieldOptions),

        std::make_shared<domain::FieldDescription>(
            "gradAij", "gradAij",
            [&](){
                std::vector<std::string> gradAijComponents;
                for (std::size_t i = 0; i < phases; i++){
                    for (std::size_t j = i + 1; j < phases; j++){
                        for (PetscInt d = 0; d < 2; d++){
                            std::string dimName = (d==0) ? "x" : (d==1) ? "y" : "z";
                            gradAijComponents.push_back("gradAij_" + std::to_string(i) + "_" + std::to_string(j) + "_" + dimName);
                        }
                    }
                }
                return gradAijComponents;
            }(),
            domain::FieldLocation::AUX,
            domain::FieldType::FVM,
            region,
            auxFieldOptions),

        std::make_shared<domain::FieldDescription>(
            "sfmom", "sfmom", 
            [&](){
                std::vector<std::string> sfmomComponents;
                for (PetscInt d = 0; d < 2; d++){
                    sfmomComponents.push_back("sfmom_" + std::to_string(d));
                }
                return sfmomComponents;
            }(),
            domain::FieldLocation::AUX, 
            domain::FieldType::FVM, 
            region, 
            auxFieldOptions),

        std::make_shared<domain::FieldDescription>(
            "nAij", "nAij",
            [&](){
                std::vector<std::string> nAijComponents;
                for (std::size_t i = 0; i < phases; i++){
                    for (std::size_t j = i + 1; j < phases; j++){
                        for (PetscInt d = 0; d < 2; d++){
                            std::string dimName = (d==0) ? "x" : (d==1) ? "y" : "z";
                            nAijComponents.push_back("nAij_" + std::to_string(i) + "_" + std::to_string(j) + "_" + dimName);
                        }
                    }
                }
                return nAijComponents;
            }(),
            domain::FieldLocation::AUX,
            domain::FieldType::FVM,
            region,
            auxFieldOptions),

        std::make_shared<domain::FieldDescription>(
            "kappaij", "kappaij",
            [&](){
                std::vector<std::string> kappaijComponents;
                for (std::size_t i = 0; i < phases; i++){
                    for (std::size_t j = i + 1; j < phases; j++){
                        kappaijComponents.push_back("kappaij_" + std::to_string(i) + "_" + std::to_string(j));
                    }
                }
                return kappaijComponents;
            }(),
            domain::FieldLocation::AUX,
            domain::FieldType::FVM,
            region,
            auxFieldOptions)

        };

    
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
