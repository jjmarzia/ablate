#include "neumann.hpp"

ablate::finiteVolume::boundaryConditions::Neumann::Neumann(std::string boundaryName, std::vector<int> labelIds, std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction,
                                                                         std::string labelName, bool enforceAtFace, bool zeroMomentum)
    : Ghost(boundaryFunction->GetName(), boundaryName, labelIds, NeumannUpdate, this, labelName), boundaryFunction(boundaryFunction), enforceAtFace(enforceAtFace), zeroMomentum(zeroMomentum) {}
PetscErrorCode ablate::finiteVolume::boundaryConditions::Neumann::NeumannUpdate(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    PetscFunctionBeginUser;
    // cast the pointer back to a math function
    ablate::finiteVolume::boundaryConditions::Neumann *neumann = (ablate::finiteVolume::boundaryConditions::Neumann *)ctx;

    // Use the petsc function directly
    PetscCall(neumann->boundaryFunction->GetSolutionField().GetPetscFunction()(
        neumann->dim, time, c, neumann->fieldSize, a_xG, neumann->boundaryFunction->GetSolutionField().GetContext()));

if (true){
//    if (neumann->enforceAtFace) {
        for (PetscInt f = 0; f < neumann->fieldSize; f++) {


//axG is definitely the boundary value specified in the yaml (which in essentialghost is supposed to be fixed);
//axI is definitely the neighboring cell value
//so let's say the neighbor is 1 and the ghost val is 5.
//essentialghost: axG --> 2axG - axI = 2*5 - 1 = 9 (this puts a 9 in the ghost cell and a 1 in the neighbor cell so that the middle face is 5, hence enforceatface)
//neumann: axG --> axI = 1 (this puts a 1 in the ghost cell and a 1 in the neighbor cell, and the face is also 1).

//                std::cout << "f=" << f <<  "  before: axG=" << a_xG[f] << ",   axI=" << a_xI[f] << "\n";
if (neumann->zeroMomentum){
if (f<2){ a_xG[f] = a_xI[neumann->fieldOffset + f]; } //rho, rhoE
else{ a_xG[f] = 0; } //rhou, rhov, rhow
}

else{ a_xG[f] = a_xI[neumann->fieldOffset + f]; } //rho, rhoe, rhou, rhov, rhow
//                std::cout << "f=" << f <<  "  after: axG=" << a_xG[f] << ",   axI=" << a_xI[f] << "\n";

        }
    }
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::boundaryConditions::BoundaryCondition, ablate::finiteVolume::boundaryConditions::Neumann, "essential (Dirichlet condition) for ghost cell based boundaries",
         ARG(std::string, "boundaryName", "the name for this boundary condition"), ARG(std::vector<int>, "labelIds", "the ids on the mesh to apply the boundary condition"),
         ARG(ablate::mathFunctions::FieldFunction, "boundaryValue", "the field function used to describe the boundary"),
         OPT(std::string, "labelName", "the mesh label holding the boundary ids (default Face Sets)"),
         OPT(bool, "enforceAtFace", "optionally update the boundary to enforce the value at the face instead of the cell using linear interpolation (default false)"),
         OPT(bool, "zeroMomentum", "if true, set \rho\vec{u} to zero (default false)"));
