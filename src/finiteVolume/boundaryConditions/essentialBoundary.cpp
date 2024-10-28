#include "essentialBoundary.hpp"
ablate::finiteVolume::boundaryConditions::EssentialBoundary::EssentialBoundary(std::string boundaryName, std::vector<std::string> labelIds, std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction, bool constant)
    : BoundaryCell(boundaryFunction->GetName(), boundaryName, labelIds, constant), boundaryFunction(boundaryFunction) {}

void ablate::finiteVolume::boundaryConditions::EssentialBoundary::updateFunction(PetscReal time, const PetscReal *x, PetscScalar *vals) {
    boundaryFunction->GetSolutionField().GetPetscFunction()(dim, time, x, fieldSize, vals, boundaryFunction->GetSolutionField().GetContext());
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::boundaryConditions::BoundaryCondition, ablate::finiteVolume::boundaryConditions::EssentialBoundary, "essential (Dirichlet condition) for boundary cells created by adding a layer next to the domain. See boxMeshBoundaryCells for an example.",
         ARG(std::string, "boundaryName", "the name for this boundary condition"),
         ARG(std::vector<std::string>, "labelIds", "labels to apply this BC to"),
         ARG(ablate::mathFunctions::FieldFunction, "boundaryValue", "the field function used to describe the boundary"),
         OPT(bool, "constant", "the boundary condition does not depend on time, default is false"));
