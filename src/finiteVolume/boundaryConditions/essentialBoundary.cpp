#include "essentialBoundary.hpp"
ablate::finiteVolume::boundaryConditions::EssentialBoundary::EssentialBoundary(std::string boundaryName, std::vector<std::string> labelIds, std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction)
    : BoundaryCell(boundaryFunction->GetName(), boundaryName, labelIds), boundaryFunction(boundaryFunction) {

    if (!boundaryFunction) {
        throw std::invalid_argument("EssentialBoundary must be constructed with a valid boundary function");
    }
    if (boundaryFunction->GetSolutionField().GetPetscFunction() == nullptr) {
        throw std::invalid_argument(
            "EssentialBoundary must be constructed with a valid boundary function that has a solution field (i.e. mathFunction must not be null).");
    }



        PetscPrintf(PETSC_COMM_WORLD, "EssentialBoundary created: %s\n", boundaryName.c_str());
        PetscPrintf(PETSC_COMM_WORLD, "Associated labels: ");
        for (const auto& label : labelIds) {
            PetscPrintf(PETSC_COMM_WORLD, "%s ", label.c_str());
        }
        PetscPrintf(PETSC_COMM_WORLD, "\n");
    }

void ablate::finiteVolume::boundaryConditions::EssentialBoundary::updateFunction(PetscReal time, const PetscReal *x, PetscScalar *vals, PetscInt point) {

    // PetscPrintf(PETSC_COMM_WORLD, "prior to update function");
    boundaryFunction->GetSolutionField().GetPetscFunction()(dim, time, x, fieldSize, vals, boundaryFunction->GetSolutionField().GetContext());

    // PetscPrintf(PETSC_COMM_WORLD, "updateFunction called for boundary");
}

void ablate::finiteVolume::boundaryConditions::EssentialBoundary::ExtraSetup() {
    // PetscPrintf(PETSC_COMM_WORLD, "EssentialBoundary ExtraSetup called\n");
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::boundaryConditions::BoundaryCondition, ablate::finiteVolume::boundaryConditions::EssentialBoundary, "essential (Dirichlet condition) for boundary cells created by adding a layer next to the domain. See boxMeshBoundaryCells for an example.",
         ARG(std::string, "boundaryName", "the name for this boundary condition"),
         ARG(std::vector<std::string>, "labelIds", "labels to apply this BC to"),
         ARG(ablate::mathFunctions::FieldFunction, "boundaryValue", "the field function used to describe the boundary"));
