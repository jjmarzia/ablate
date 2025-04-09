#ifndef ABLATELIBRARY_ZERODERBOUNDARY_HPP
#define ABLATELIBRARY_ZERODERBOUNDARY_HPP

#include <mathFunctions/fieldFunction.hpp>
#include "boundaryCell.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "finiteVolume/processes/flowProcess.hpp"
#include "finiteVolume/processes/process.hpp"
#include "solver/solver.hpp"
#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"

namespace ablate::finiteVolume::boundaryConditions {

class ZeroDerBoundary : public BoundaryCell {
   private:
    void updateFunction(PetscReal time, const PetscReal* x, PetscScalar* vals, PetscInt point) override;

    const std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction;

   public:
    ZeroDerBoundary(std::string boundaryName, std::vector<std::string> labelIds, std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction);
    inline const static std::string VOLUME_FRACTION_FIELD = eos::TwoPhase::VF;
    inline const static std::string DENSITY_VF_FIELD = ablate::finiteVolume::CompressibleFlowFields::CONSERVED + VOLUME_FRACTION_FIELD;
};
}  // namespace ablate::finiteVolume::boundaryConditions
#endif  // ABLATELIBRARY_ZERODERBOUNDARY_HPP

