#ifndef ABLATELIBRARY_ESSENTIALBOUNDARY_HPP
#define ABLATELIBRARY_ESSENTIALBOUNDARY_HPP

#include <mathFunctions/fieldFunction.hpp>
#include "boundaryCell.hpp"

namespace ablate::finiteVolume::boundaryConditions {

class EssentialBoundary : public BoundaryCell {
   private:
    void updateFunction(PetscReal time, const PetscReal* x, PetscScalar* vals) override;

    const std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction;

   public:
    EssentialBoundary(std::string boundaryName, std::vector<std::string> labelIds, std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction, bool constant);
};
}  // namespace ablate::finiteVolume::boundaryConditions
#endif  // ABLATELIBRARY_ESSENTIALBOUNDARY_HPP
