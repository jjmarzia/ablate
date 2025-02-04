#ifndef ABLATELIBRARY_ZEROGRADIENTBOUNDARY_HPP
#define ABLATELIBRARY_ZEROGRADIENTBOUNDARY_HPP

#include <mathFunctions/fieldFunction.hpp>
#include "boundaryCell.hpp"

namespace ablate::finiteVolume::boundaryConditions {

class ZeroGradientBoundary : public BoundaryCell {
   private:
    void updateFunction(PetscReal time, const PetscReal* x, PetscScalar* vals) override;
    const std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction;
    bool zeroMomentum;

   public:
    zeroGradientBoundary(std::string boundaryName, std::vector<std::string> labelIds, std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction, bool constant, bool zeroMomentum);
};
}  // namespace ablate::finiteVolume::boundaryConditions
#endif  // ABLATELIBRARY_ZEROGRADIENTBOUNDARY_HPP
