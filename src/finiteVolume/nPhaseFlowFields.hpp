#ifndef ABLATELIBRARY_NPHASEFLOWFIELDS_HPP
#define ABLATELIBRARY_NPHASEFLOWFIELDS_HPP

#include <domain/region.hpp>
#include <memory>
#include <string>
#include <vector>
#include "domain/fieldDescriptor.hpp"
#include "eos/eos.hpp"
#include "parameters/mapParameters.hpp"

namespace ablate::finiteVolume {

class NPhaseFlowFields : public domain::FieldDescriptor {
   public:
   //unchanging conserved components
    typedef enum {RHOE, RHOU, RHOV, RHOW} AllaireComponents;

    //! the primary field containing the AllaireComponents
    inline const static std::string ALLAIRE_FIELD = "allaire";

    //! The conserved prefix used for fields that have a conserved and non conserved form
    // inline const static std::string CONSERVED = "density";

    // alpha_k rho_k for each phase; this might actually belong in nPhaseEulerAdvection ?
    inline static std::string ALPHAKRHOK(PetscInt phase){
        return "alpharho" + std::to_string(phase);
    } 
    inline static std::string ALPHAK(PetscInt phase){
        return "alpha" + std::to_string(phase);
    }

    inline static std::string RHOK(PetscInt phase){
        return "rho" + std::to_string(phase);
    }

    inline static std::string TK(PetscInt phase){
        return "t" + std::to_string(phase);
    }

    inline static std::string EPSK(PetscInt phase){
        return "eps" + std::to_string(phase);
    }

    //! some common aux fields
    inline const static std::string VELOCITY_FIELD = "ui";
    inline const static std::string PRESSURE_FIELD = "p";
    inline const static std::string DENSITY_FIELD = "rho";
    inline const static std::string ENERGY_FIELD = "e";

   protected:
    const std::shared_ptr<eos::EOS> eos;
    const std::shared_ptr<domain::Region> region;
    const std::shared_ptr<parameters::Parameters> conservedFieldOptions;
    const std::shared_ptr<parameters::Parameters> auxFieldOptions = ablate::parameters::MapParameters::Create({{"petscfv_type", "leastsquares"}, {"petsclimiter_type", "none"}});

   public:
    /**
     * Create a helper class that produces the required compressible flow fields based upon the eos and specifed region
     * @param eos the eos used to determine the species
     * @param region the region for all of the fields
     * @param conservedFieldParameters override the default field parameters for the conserved field
     */
    explicit NPhaseFlowFields(std::shared_ptr<eos::EOS> eos, std::shared_ptr<domain::Region> region = {}, std::shared_ptr<parameters::Parameters> conservedFieldParameters = {});

    /**
     * override and return the compressible flow fields
     * @return
     */
    std::vector<std::shared_ptr<domain::FieldDescription>> GetFields() override;
};

// std::istream& operator>>(std::istream& is, NPhaseFlowFields::ValidRange& v);

}  // namespace ablate::finiteVolume

#endif  // ABLATELIBRARY_NPHASEFLOWFIELDS_HPP
