#ifndef ABLATELIBRARY_NEUMANN_HPP
#define ABLATELIBRARY_NEUMANN_HPP

#include <mathFunctions/fieldFunction.hpp>
#include "ghost.hpp"

namespace ablate::finiteVolume::boundaryConditions {

class Neumann : public Ghost {
   private:
    static PetscErrorCode NeumannUpdate(PetscReal time, const PetscReal* c, const PetscReal* n, const PetscScalar* a_xI, PetscScalar* a_xG, void* ctx);

    const std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction;

    /**
     * Uses linear interpolation to force the value at the face
     */
    const bool enforceAtFace;

   public:
    Neumann(std::string boundaryName, std::vector<int> labelId, std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction, std::string labelName = {}, bool enforceAtFace = false);
};
}  // namespace ablate::finiteVolume::boundaryConditions
#endif  // ABLATELIBRARY_ESSENTIALGHOST_HPP
