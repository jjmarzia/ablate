#ifndef ABLATELIBRARY_GHOST_HPP
#define ABLATELIBRARY_GHOST_HPP

#include <domain/subDomain.hpp>
#include "boundaryCondition.hpp"

namespace ablate::finiteVolume::boundaryConditions {

class Ghost : public BoundaryCondition {
    typedef PetscErrorCode (*UpdateFunction)(PetscReal time, const PetscReal* c, const PetscReal* n, const PetscScalar* a_xI, PetscScalar* a_xG, void* ctx);

   private:
    const std::string labelName;
    const std::vector<PetscInt> labelIds;
    const UpdateFunction updateFunction;
    const void* updateContext;

   protected:
    // Store some field information
    PetscInt dim;
    PetscInt fieldSize;
    // the field offset for the a_xI values;
    PetscInt fieldOffset;

   public:
    BoundaryCondition::Type type() const override { return BoundaryCondition::Type::GHOST; }

    Ghost(std::string fieldName, std::string boundaryName, std::vector<int> labelIds, UpdateFunction updateFunction, void* updateContext, std::string labelName = {});

    Ghost(std::string fieldName, std::string boundaryName, int labelId, UpdateFunction updateFunction, void* updateContext, std::string labelName = {});

    virtual ~Ghost() override = default;

    void SetupBoundary(DM dm, PetscDS problem, PetscInt fieldId) override;

    void SetupBoundary(std::shared_ptr<ablate::domain::SubDomain> subDomain, PetscInt fieldId) override { SetupBoundary(subDomain->GetDM(), subDomain->GetDiscreteSystem(), fieldId); }
    void ComputeBoundary(PetscReal time, Vec locX, Vec locX_t, Vec cellGeomVec) override {};

};

}  // namespace ablate::finiteVolume::boundaryConditions

#endif  // ABLATELIBRARY_GHOST_HPP
