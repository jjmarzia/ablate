#include "zeroGradientBoundary.hpp"
ablate::finiteVolume::boundaryConditions::zeroGradientBoundary::zeroGradientBoundary(std::string boundaryName, std::vector<std::string> labelIds, std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction, bool constant, bool zeroMomentum)
    : BoundaryCell(boundaryFunction->GetName(), boundaryName, labelIds, constant, zeroMomentum), boundaryFunction(boundaryFunction) {}


auto ablate::finiteVolume::boundaryConditions::zeroGradientBoundary::getNeighoringInteriorCell(DM dm, PetscInt boundaryCell, PetscInt *neighborCell){

PetscInt conesize; DMPlexGetConeSize(dm, boundaryCell, &conesize);
const PetscInt *cone; DMPlexGetCone(dm, boundaryCell, &cone); //cone[consize] ?
for (PetscInt i=0; i < conesize; ++i){
PetscInt face = cone[i];
DMLabel label; DMGetLabel(dm, "boundaryFaces", &label); //here's the object corresponding to the boundaryFaces label
PetscInt labelValue; DMLabelGetValue(label, face, &labelValue); //this will return labelValue=-1 if the face doesn't have the label or labelValue=X if it does, where X=whatever integer corresponds to the boundaryFaces label (we don't care about the integer except that it's >-1

if (labelValue >= 0){
PetscInt support[2];
DMPlexGetSupport(dm, face, support);
*neighborCell = (support[0] == boundaryCell) ? support[1] : support[0]; return 0;
}

}
return -1;
}

//auto ablate::finiteVolume::boundaryConditions::zeroGradientBoundary::getCellIDFromCoordinate(DM dm, PetscInt dim, const PetscReal x[], PetscInt *cellID){
//PetscInt cStart, cEnd;
//PetscReal minDist = PETSC_MAX_REAL;
//DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
//for (PetscInt c = cStart; c < cEnd; ++c) {
//        PetscReal centroid[3]; PetscReal volume; DMPlexComputeCellGeometryFVM(dm, c, &volume, centroid, NULL);
//        PetscReal dist = 0.0; for (PetscInt d = 0; d < dim; ++d) {  dist += PetscSqr(x[d] - centroid[d]); }
//        dist = PetscSqrtReal(dist);
//        if (dist < minDist) { minDist = dist; *cellID = c; }
//}
//return 0;
//}


void ablate::finiteVolume::boundaryConditions::zeroGradientBoundary::updateFunction(PetscInt point, PetscInt fieldID, PetscScalar dataArray, DM dataDM, PetscScalar *vals) {

VecGetDM(locX, &dmData) >> utilities::PetscUtilities::checkError;


PetscInt *neighborCell; getNeighoringInteriorCell(dm, point, &neighborCell);
PetscScalar *neighborVals;
xDMPlexPointLocalRef(dataDM, neighborCell, fieldID, dataArray, &neighborVals);
for (PetscInt i=0; i < fieldSize; ++i){ vals[i] = neighborVals[i]; }

//further step needed or no?

}

//previously essentialboundary was doing the following:
//    boundaryFunction->GetSolutionField().GetPetscFunction()(dim, time, x, fieldSize, vals, boundaryFunction->GetSolutionField().GetContext());

//boundaryFunction is the math expression (provided by yaml?) that the field values are set to
// getsolutionfield is the solution field associated with the boundary condition (euler, vf, etc)
// getpetscfunction is the "actual function" that petsc uses to assign the field values at the point in the appropriate solution field based on the boundaryfunction.
// { dim, time, x, fieldSize, vals, boundaryFunction->GetSolutionField().GetContext()) } = {dimension number, current simulation time, spatial coords of the boundary cell, 
//                                                                                          number of components in the field, the array where the compute results are stored, additional context }

//you don't need a boundaryfunction for zeroGradient because the user isn't providing any math expression that determines the values.
//however this poses a problem in terms of the relationship between the structure of boundaryCell and the would-be structure of zeroGrad: you need to put the logic of zero grad in zeroGradient,
//but the boundaryCell is the location where the boundary Cells are being looped over.
//this is conflicting because the logic of zero grad is cell-specific, it's not a generalized math function like essentialBoundary.

//ok...it's actually not conflicting and here's why
//here's the loop over boundary cell points (in boundarycell which is meant to tailor to essentialBoundary)
//    for (PetscInt p = 0; p < nPoints; ++p) {
//      PetscFVCellGeom* cg; DMPlexPointLocalRead(dmCell, points[p], cellGeomArray, &cg) //get the cell geometry corresponding to the cell ID (the cell geometry just contains the centroid and the volume, the first of which we want) 
//      PetscScalar *vals; xDMPlexPointLocalRef(dmData, points[p], field.id, array, &vals); //get the vals given by the yaml
//      updateFunction(time, cg->centroid, vals); //use updateFunction from essentialboundary to imbue this centroid at this time step with the specific vals in the yaml
//    }

//vals in the above context is grabbed from the yaml, but we can override vals WITHIN updateFunction cell-by-cell in zeroGradient so that boundaryCell does what we want.

//how to get the points corresponding to boundary cells over which we need to iterate though? this is how it's done in boundarycell:
//    PetscScalar *array;
//    PetscInt nPoints = 0;
//    const PetscInt *points;
//    DM dmData, dmCell;
//    const PetscScalar* cellGeomArray;
//    ablate::domain::Field field = subDomain->GetField(GetFieldName());
//
//    ISGetIndices(pointsIS, &points) >> utilities::PetscUtilities::checkError;
//    ISGetLocalSize(pointsIS, &nPoints) >> utilities::PetscUtilities::checkError;
//
//    VecGetDM(locX, &dmData) >> utilities::PetscUtilities::checkError;
//
//    VecGetDM(cellGeomVec, &dmCell) >> utilities::PetscUtilities::checkError;
//    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
//
//    VecGetArray(locX, &array) >> utilities::PetscUtilities::checkError;



//basically boundaryCell is telling zeroGrad, "ok I have a time step and a centroid and some vals I inherited from the yaml. Please apply the update rule."
//then zeroGrad says to boundaryCell "sure so I'm gonna go ahead and ignore the vals you just gave me, as well as the time step, and simply infer the update based on the neighbor interior cell."

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::boundaryConditions::BoundaryCondition, ablate::finiteVolume::boundaryConditions::zeroGradientBoundary, "Zero gradient condition for boundary cells created by adding a layer next to the domain. See boxMeshBoundaryCells for an example.",
         ARG(std::string, "boundaryName", "the name for this boundary condition"),
         ARG(std::vector<std::string>, "labelIds", "labels to apply this BC to"),
         ARG(ablate::mathFunctions::FieldFunction, "boundaryValue", "the field function used to describe the boundary (just supply an accurate fieldName, the other parameters will are overridden for now so they can be arbitrary)"),
         OPT(bool, "constant", "the boundary condition does not depend on time, default is false"));
         OPT(bool, "zeroMomentum", "enforce \rho\vec{u}=0 while the other euler variables remain zero derivative (no slip BC), default is false"));
