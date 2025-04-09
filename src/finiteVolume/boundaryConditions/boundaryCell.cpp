#include "boundaryCell.hpp"
#include "solver/solver.hpp"
#include <petsc.h>
#include "utilities/petscUtilities.hpp"
#include "utilities/petscSupport.hpp"

ablate::finiteVolume::boundaryConditions::BoundaryCell::BoundaryCell(std::string fieldName, std::string boundaryName, std::vector<std::string> labelIds)
    : BoundaryCondition(boundaryName, fieldName),
      labelIds(labelIds) {}

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

        if (!label) {
          throw std::runtime_error("Label " + id + " can not be found when setting up the boundary conditions.");
        }

        // Get the values this label can take
        PetscInt numValues;
        IS valuesIS;
        const PetscInt *values;


        DMLabelGetNonEmptyStratumValuesIS(label, &valuesIS) >> utilities::PetscUtilities::checkError;



        ISGetSize(valuesIS, &numValues) >> utilities::PetscUtilities::checkError;
        ISGetIndices(valuesIS, &values) >> utilities::PetscUtilities::checkError;
        std::vector<IS> subISs(numValues, nullptr);

        //print numvalues
        PetscPrintf(PETSC_COMM_WORLD, "numValues: %d\n", numValues);

        // Pull all of points for each value
        for (PetscInt v = 0; v < numValues; ++v) {
          // PetscPrintf(PETSC_COMM_WORLD, "debug next3\n");
          DMGetStratumIS(dm, id.c_str(), values[v], &subISs[v]) >> utilities::PetscUtilities::checkError;



          // print all of the coordinates in the stratum
          PetscInt nPoints;
          ISGetLocalSize(subISs[v], &nPoints) >> utilities::PetscUtilities::checkError;

          // print the label associated with this region
          PetscPrintf(PETSC_COMM_WORLD, "Label %s with value %d has %d points\n", id.c_str(), values[v], nPoints);

          const PetscInt *points;  
          ISGetIndices(subISs[v], &points) >> utilities::PetscUtilities::checkError;
          for (PetscInt p = 0; p < nPoints; ++p) {
            PetscInt point = points[p];
            PetscReal centroid[3];
            DMPlexComputeCellGeometryFVM(dm, point, nullptr, centroid, nullptr) >> utilities::PetscUtilities::checkError;
            // PetscPrintf(PETSC_COMM_WORLD, "point %d: (%g, %g, %g)\n", point, centroid[0], centroid[1], centroid[2]);
          }
          ISRestoreIndices(subISs[v], &points) >> utilities::PetscUtilities::checkError;


          // PetscPrintf(PETSC_COMM_WORLD, "debug next4\n");
          // Remove any points not in the range.
//check the validity of subISs[v], pstart, pEnd through printing
          // PetscPrintf(PETSC_COMM_WORLD, "subISs[%d]: %p\n", v, (void *)subISs[v]);
          // PetscPrintf(PETSC_COMM_WORLD, "pStart: %d\n", pStart);
          // PetscPrintf(PETSC_COMM_WORLD, "pEnd: %d\n", pEnd);

          // ISGeneralFilter(subISs[v], pStart, pEnd) >> utilities::PetscUtilities::checkError;

          // PetscPrintf(PETSC_COMM_WORLD, "debug next5\n");

        }

        // PetscPrintf(PETSC_COMM_WORLD, "debug next6\n");


        ISRestoreIndices(valuesIS, &values) >> utilities::PetscUtilities::checkError;
        ISDestroy(&valuesIS) >> utilities::PetscUtilities::checkError;

        ISConcatenate(PETSC_COMM_SELF, numValues, &subISs[0], &listISs[l]) >> utilities::PetscUtilities::checkError;
        ISSortRemoveDups(listISs[l]) >> utilities::PetscUtilities::checkError;

        for (PetscInt v = 0; v < numValues; ++v) {
          ISDestroy(&subISs[v]);
        }

        // PetscPrintf(PETSC_COMM_WORLD, "debug last\n");

    }

    // Now create the points list
    ISConcatenate(PETSC_COMM_SELF, labelIds.size(), &listISs[0], &pointsIS) >> utilities::PetscUtilities::checkError;
    ISSortRemoveDups(pointsIS) >> utilities::PetscUtilities::checkError;

    for (size_t l = 0; l < labelIds.size(); ++l) {
      ISDestroy(&listISs[l]);
    }

    ExtraSetup(); // for ZeroDerBoundary

}

void ablate::finiteVolume::boundaryConditions::BoundaryCell::ComputeBoundary(PetscReal time, Vec locX, Vec locX_t, Vec cellGeomVec) {
    // time - Current time of the simulation
    // locX - boundary condition
    // locX_t -Time derivative of boundary condition

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
      updateFunction(time, cg->centroid, vals, points[p]);

    }
    VecRestoreArray(locX, &array) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;

    ISRestoreIndices(pointsIS, &points);

}
