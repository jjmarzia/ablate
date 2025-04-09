#include "essential.hpp"
#include "utilities/petscUtilities.hpp"

PetscErrorCode ablate::finiteElement::boundaryConditions::Essential::BoundaryValueFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    auto boundary = (Essential *)ctx;
    return boundary->boundaryFunction->GetSolutionField().GetPetscFunction()(dim, time, x, Nf, u, boundary->boundaryFunction->GetSolutionField().GetContext());
}
PetscErrorCode ablate::finiteElement::boundaryConditions::Essential::BoundaryTimeDerivativeFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    auto boundary = (Essential *)ctx;
    return boundary->boundaryFunction->GetTimeDerivative().GetPetscFunction()(dim, time, x, Nf, u, boundary->boundaryFunction->GetTimeDerivative().GetContext());
}

ablate::finiteElement::boundaryConditions::Essential::Essential(std::string boundaryName, int labelId, std::shared_ptr<mathFunctions::FieldFunction> boundaryFunctionIn, std::string labelNameIn)
    : Essential(boundaryName, std::vector<int>{labelId}, boundaryFunctionIn, labelNameIn) {}

ablate::finiteElement::boundaryConditions::Essential::Essential(std::string boundaryName, std::vector<int> labelIdsIn, std::shared_ptr<mathFunctions::FieldFunction> boundaryFunctionIn,
                                                                std::string labelNameIn)
    : BoundaryCondition(boundaryName, boundaryFunctionIn->GetName()),
      labelName(labelNameIn.empty() ? "marker" : labelNameIn),
      labelIds(labelIdsIn.begin(), labelIdsIn.end()),
      boundaryFunction(boundaryFunctionIn) {}

ablate::mathFunctions::PetscFunction ablate::finiteElement::boundaryConditions::Essential::GetBoundaryFunction() { return BoundaryValueFunction; }
ablate::mathFunctions::PetscFunction ablate::finiteElement::boundaryConditions::Essential::GetBoundaryTimeDerivativeFunction() { return BoundaryTimeDerivativeFunction; }

void *ablate::finiteElement::boundaryConditions::Essential::GetContext() { return this; }

void ablate::finiteElement::boundaryConditions::Essential::SetupBoundary(DM dm, PetscDS problem, PetscInt fieldId) {
    DMLabel label;
    DMGetLabel(dm, labelName.c_str(), &label) >> utilities::PetscUtilities::checkError;


    PetscErrorCode ierr;
    PetscPrintf(PETSC_COMM_WORLD, "Setting up boundary condition: %s with label: %s\n", GetBoundaryName().c_str(), labelName.c_str());
    ierr = DMGetLabel(dm, labelName.c_str(), &label);
    if (ierr || !label) {
        PetscPrintf(PETSC_COMM_WORLD, "Error: Label %s not found in the mesh.\n", labelName.c_str());
    }
    PetscPrintf(PETSC_COMM_WORLD, "Applying boundary condition to label IDs: ");
    for (auto id : labelIds) {
        PetscPrintf(PETSC_COMM_WORLD, "%d ", id);
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n");

    ierr = PetscDSAddBoundary(problem,
                       DM_BC_ESSENTIAL,
                       GetBoundaryName().c_str(),
                       label,
                       labelIds.size(),
                       &labelIds[0],
                       fieldId,
                       0,
                       NULL,
                       (void (*)(void))GetBoundaryFunction(),
                       (void (*)(void))GetBoundaryTimeDerivativeFunction(),
                       GetContext(),
                       NULL);
    if (ierr) {
        PetscPrintf(PETSC_COMM_WORLD, "Error: Failed to add boundary condition %s.\n", GetBoundaryName().c_str());
    }
}

#include "registrar.hpp"
REGISTER(ablate::finiteElement::boundaryConditions::BoundaryCondition, ablate::finiteElement::boundaryConditions::Essential, "essential (Dirichlet condition) for FE based problems",
         ARG(std::string, "boundaryName", "the name for this boundary condition"), ARG(std::vector<int>, "labelIds", "the ids on the mesh to apply the boundary condition"),
         ARG(ablate::mathFunctions::FieldFunction, "boundaryValue", "the field function used to describe the boundary"),
         OPT(std::string, "labelName", "the mesh label holding the boundary ids (default marker)"));