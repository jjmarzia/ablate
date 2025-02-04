#ifndef ABLATELIBRARY_BOUNDARYCELL_HPP
#define ABLATELIBRARY_BOUNDARYCELL_HPP

#include <domain/subDomain.hpp>
#include "boundaryCondition.hpp"
#include "domain/field.hpp"

// Boundary conditions for cells marked as boundaryCells

namespace ablate::finiteVolume::boundaryConditions {

class BoundaryCell : public BoundaryCondition {
//  typedef void (*UpdateFunction)(PetscReal time, const PetscReal* x, PetscScalar* vals);

   private:
    std::shared_ptr<ablate::domain::SubDomain> subDomain = nullptr;
    std::vector<std::string> labelIds = {};
    IS pointsIS = nullptr;
    bool constant = false;
    bool calculated = false;


   protected:

    // Store some field information
    PetscInt dim = -1;
    PetscInt fieldSize = -1;

    virtual void updateFunction(PetscReal, const PetscReal*, PetscScalar*) = 0;
    virtual void updateFunction(PetscReal, const PetscReal*, PetscScalar*, PetscInt*, PetscInt*){throw std::logic_error("this BC isn't asking for petscint point or petscint fieldid"); }


   public:

    BoundaryCondition::Type type() const override { return BoundaryCondition::Type::BOUNDARYCELL; }

    BoundaryCell(std::string fieldName, std::string boundaryName, std::vector<std::string> labelIds, bool constant = false);

    virtual ~BoundaryCell() override;

    void SetupBoundary(std::shared_ptr<ablate::domain::SubDomain> subDomain, PetscInt fieldId) override;
    void SetupBoundary(DM dm, PetscDS problem, PetscInt fieldId) override {}

    void ComputeBoundary(PetscReal time, Vec locX, Vec locX_t, Vec cellGeomVec) override;
};

}  // namespace ablate::finiteVolume::boundaryConditions

#endif  // ABLATELIBRARY_BOUNDARYCELL_HPP
