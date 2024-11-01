#ifndef ABLATELIBRARY_FINITEVOLUME_BOUNDARYCONDITION_HPP
#define ABLATELIBRARY_FINITEVOLUME_BOUNDARYCONDITION_HPP
#include <memory>
#include <string>
#include "domain/fieldDescription.hpp"
#include "domain/subDomain.hpp"
#include "mathFunctions/mathFunction.hpp"

namespace ablate::finiteVolume::boundaryConditions {
class BoundaryCondition {
   private:
    const std::string boundaryName;
    const std::string fieldName;

   protected:
    BoundaryCondition(const std::string boundaryName, const std::string fieldName) : boundaryName(boundaryName), fieldName(fieldName) {}

   public:

    typedef enum { GHOST, BOUNDARYCELL } Type;

    const std::string& GetBoundaryName() const { return boundaryName; }
    const std::string& GetFieldName() const { return fieldName; }

    virtual ~BoundaryCondition() = default;
    virtual void SetupBoundary(DM dm, PetscDS problem, PetscInt fieldId) = 0;

    virtual void SetupBoundary(std::shared_ptr<ablate::domain::SubDomain> subDomain, PetscInt fieldId) = 0;
    virtual void ComputeBoundary(PetscReal time, Vec locX, Vec locX_t, Vec cellGeomVec) = 0;

    virtual BoundaryCondition::Type type() const = 0;

};
}  // namespace ablate::finiteVolume::boundaryConditions
#endif  // ABLATELIBRARY_BOUNDARYCONDITION_HPP
