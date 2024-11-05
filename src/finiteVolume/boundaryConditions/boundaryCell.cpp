#include "boundaryCell.hpp"
#include "solver/solver.hpp"
#include <petsc.h>
#include "utilities/petscUtilities.hpp"
#include "utilities/petscSupport.hpp"

ablate::finiteVolume::boundaryConditions::BoundaryCell::BoundaryCell(std::string fieldName, std::string boundaryName, std::vector<std::string> labelIds,
                                                       bool constant)
    : BoundaryCondition(boundaryName, fieldName),
      labelIds(labelIds),constant(constant) {}

ablate::finiteVolume::boundaryConditions::BoundaryCell::~BoundaryCell() {
  if (pointsIS) ISDestroy(&pointsIS);
}

void ablate::finiteVolume::boundaryConditions::BoundaryCell::SetupBoundary(std::shared_ptr<ablate::domain::SubDomain> subDomainIn, PetscInt fieldId) {

    subDomain = subDomainIn;

    dim = subDomain->GetDimensions();
    fieldSize = subDomain->GetField(GetFieldName()).numberComponents;

    DM dm = subDomain->GetDM();


    // Get a list of all points that will have this boundary condition. For now assume that only cells will be used
    PetscInt pStart, pEnd;
    DMPlexGetHeightStratum(dm, 0, &pStart, &pEnd) >> utilities::PetscUtilities::checkError;

    std::vector<IS> listISs(labelIds.size(), nullptr);

    for (size_t l = 0; l < labelIds.size(); ++l) {

        std::string id = labelIds[l];

        DMLabel label;
        DMGetLabel(dm, id.c_str(), &label) >> utilities::PetscUtilities::checkError;

        // Get the values this label can take
        PetscInt numValues;
        IS valuesIS;
        const PetscInt *values;
        DMLabelGetNumValues(label, &numValues) >> utilities::PetscUtilities::checkError;
        DMLabelGetNonEmptyStratumValuesIS(label, &valuesIS) >> utilities::PetscUtilities::checkError;
        ISGetIndices(valuesIS, &values) >> utilities::PetscUtilities::checkError;

        std::vector<IS> subISs(numValues, nullptr);

        // Pull all of points for each value
        for (PetscInt v = 0; v < numValues; ++v) {
          DMGetStratumIS(dm, id.c_str(), values[v], &subISs[v]) >> utilities::PetscUtilities::checkError;

          // Remove any points not in the range.
          ISGeneralFilter(subISs[v], pStart, pEnd) >> utilities::PetscUtilities::checkError;
        }
        ISRestoreIndices(valuesIS, &values) >> utilities::PetscUtilities::checkError;
        ISDestroy(&valuesIS) >> utilities::PetscUtilities::checkError;

        ISConcatenate(PETSC_COMM_SELF, numValues, &subISs[0], &listISs[l]) >> utilities::PetscUtilities::checkError;
        ISSortRemoveDups(listISs[l]) >> utilities::PetscUtilities::checkError;

        for (PetscInt v = 0; v < numValues; ++v) {
          ISDestroy(&subISs[v]);
        }
    }

    // Now create the points list
    ISConcatenate(PETSC_COMM_SELF, labelIds.size(), &listISs[0], &pointsIS) >> utilities::PetscUtilities::checkError;
    ISSortRemoveDups(pointsIS) >> utilities::PetscUtilities::checkError;

    for (size_t l = 0; l < labelIds.size(); ++l) {
      ISDestroy(&listISs[l]);
    }

}

void ablate::finiteVolume::boundaryConditions::BoundaryCell::ComputeBoundary(PetscReal time, Vec locX, Vec locX_t, Vec cellGeomVec) {
    // time - Current time of the simulation
    // locX - boundary condition
    // locX_t -Time derivative of boundary condition

    if (calculated) return;

    PetscScalar *array;
    PetscInt nPoints = 0;
    const PetscInt *points;
    DM dmData, dmCell;
    const PetscScalar* cellGeomArray;
    ablate::domain::Field field = subDomain->GetField(GetFieldName());

    ISGetIndices(pointsIS, &points) >> utilities::PetscUtilities::checkError;
    ISGetLocalSize(pointsIS, &nPoints) >> utilities::PetscUtilities::checkError;

    VecGetDM(locX, &dmData) >> utilities::PetscUtilities::checkError;

    VecGetDM(cellGeomVec, &dmCell) >> utilities::PetscUtilities::checkError;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;

    VecGetArray(locX, &array) >> utilities::PetscUtilities::checkError;
    for (PetscInt p = 0; p < nPoints; ++p) {

      PetscFVCellGeom* cg;
      DMPlexPointLocalRead(dmCell, points[p], cellGeomArray, &cg) >> utilities::PetscUtilities::checkError;

      PetscScalar *vals;
      xDMPlexPointLocalRef(dmData, points[p], field.id, array, &vals) >> utilities::PetscUtilities::checkError;

      updateFunction(time, cg->centroid, vals);

    }
    VecRestoreArray(locX, &array) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;

    ISRestoreIndices(pointsIS, &points);

    calculated = constant; // If the values do not change over time mark this BC as calculated so it's not done again

}
