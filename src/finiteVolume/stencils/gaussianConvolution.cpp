#include "gaussianConvolution.hpp"
#include <petsc.h>
#include "utilities/petscSupport.hpp"
#include "utilities/mathUtilities.hpp"
#include "domain/fieldAccessor.hpp"
//#include "levelSetUtilities.hpp"
#include "utilities/constants.hpp"



using namespace ablate::finiteVolume::stencil;

#define xexit(S, ...) {PetscFPrintf(MPI_COMM_WORLD, stderr, \
  "\x1b[1m(%s:%d, %s)\x1b[0m\n  \x1b[1m\x1b[90mexiting:\x1b[0m " S "\n", \
  __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); exit(0);}

// geomDM - Sample DM with the geometry. Other DMs can store the data, but the geometric layout (including all ghost cells, etc) must match this exactly
// nLayers - The number of layers to use. Recommendation is nQuad = 4;
// sigmaFactor - The standard deviation will be sigmaFactor*h. Recommendation is sigmaFactor = 1.0;

GaussianConvolution::GaussianConvolution(DM geomDM, const PetscInt nLayers, const PetscInt sigmaFactor) : nLayers(nLayers), geomDM(geomDM) {

  Vec faceGeomVec;
  DMPlexComputeGeometryFVM(geomDM, &cellGeomVec, &faceGeomVec) >> utilities::PetscUtilities::checkError;
  VecDestroy(&faceGeomVec);

  DMPlexGetHeightStratum(geomDM, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;
  PetscInt nCells = cEnd - cStart;

  PetscMalloc3(nCells, &nCellList, nCells, &cellList, nCells, &cellWeights) >> utilities::PetscUtilities::checkError;

  nCellList -= cStart; // So that it can be index by cell number
  cellList -= cStart;
  cellWeights -= cStart;
  for (PetscInt c = cStart; c < cEnd; ++c) {
    nCellList[c] = -1;
    cellList[c] = nullptr;
    cellWeights[c] = nullptr;
  }

  // The spatial standard deviation to use.
  PetscReal h;
  DMPlexGetMinRadius(geomDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0;
  sigma = sigmaFactor*h;


  // Get the information about periodicity
  const PetscReal *maxCell, *L;
  DMGetPeriodicity(geomDM, &maxCell, NULL, &L) >> ablate::utilities::PetscUtilities::checkError;

  PetscInt dim;
  DMGetDimension(geomDM, &dim);
  for (PetscInt d = 0; d < dim; ++d) {
    if (maxCell[d] > 0.0) {
      maxDist[d] = 2*nLayers*maxCell[d];
      sideLen[d] = L[d];
    }
  }

}

GaussianConvolution::~GaussianConvolution() {

  if (cellGeomVec) VecDestroy(&cellGeomVec) >> utilities::PetscUtilities::checkError;

  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscFree2(cellList[c], cellWeights[c]) >> utilities::PetscUtilities::checkError;
  }
  nCellList += cStart;
  cellList += cStart;
  cellWeights += cStart;
  PetscFree3(nCellList, cellList, cellWeights) >> utilities::PetscUtilities::checkError;

}

// Build the list of cells needed for point p.
void GaussianConvolution::BuildList(const PetscInt p) {

  PetscInt           dim;
  DM                 dmCell;
  const PetscScalar* cellGeomArray;
  PetscFVCellGeom*   cg0;


  DMGetDimension(geomDM, &dim) >> ablate::utilities::PetscUtilities::checkError;
  VecGetDM(cellGeomVec, &dmCell) >> utilities::PetscUtilities::checkError;
  VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;

  DMPlexPointLocalRead(dmCell, p, cellGeomArray, &cg0) >> utilities::PetscUtilities::checkError;

  PetscInt nCells, *localCellList;
  DMPlexGetNeighbors(geomDM, p, nLayers, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &localCellList) >> ablate::utilities::PetscUtilities::checkError;

  PetscMalloc2(nCells, &cellList[p], nCells, &cellWeights[p]);

  if (!cellList[p] || !cellWeights[p]) {
    throw std::runtime_error("Could not allocate memory at a point in GaussianConvolution.");
  }

  PetscInt nnz = 0; // The number of non-zero entries
  for (PetscInt n = 0; n < nCells; ++n) {
    PetscInt neighborCell = localCellList[n];
    PetscFVCellGeom* cg;
    DMPlexPointLocalRead(dmCell, neighborCell, cellGeomArray, &cg) >> utilities::PetscUtilities::checkError;

    // Compute the distance, taking into account periodicity
    PetscReal r = 0.0;
    for (PetscInt d = 0; d < dim; ++d) {
      PetscReal dist = PetscAbsReal(cg0->centroid[d] - cg->centroid[d]);
      dist = (dist > maxDist[d]) ? dist - sideLen[d] : dist;
      r += PetscSqr(dist);
    }

    PetscReal wt = PetscExpReal(0.5*r/PetscSqr(sigma));

    if (wt > PETSC_SMALL) {
      cellList[p][nnz] = neighborCell;
      cellWeights[p][nnz++] = wt;
    }
  }
  nCellList[p] = nnz;

  DMPlexRestoreNeighbors(geomDM, p, nLayers, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &localCellList) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;

}


void GaussianConvolution::Evaluate(DM dm, const PetscInt p, const PetscInt fid, Vec fVec, PetscInt offset, const PetscInt nDof, PetscReal vals[]) {
  const PetscScalar *array;
  VecGetArrayRead(fVec, &array) >> ablate::utilities::PetscUtilities::checkError;
  Evaluate(dm, p, fid, array, offset, nDof, vals);
  VecRestoreArrayRead(fVec, &array) >> ablate::utilities::PetscUtilities::checkError;
}

void GaussianConvolution::FormAllLists() {
  for (PetscInt cell = GaussianConvolution::cStart; cell < GaussianConvolution::cEnd; ++cell){
    BuildList(cell);
  }
}
PetscInt GaussianConvolution::GetCellList(const PetscInt p, const PetscInt **cellListOut) {

  if (!cellList[p]) BuildList(p);  // Build the convolution list

  *(const PetscInt **)cellListOut = cellList[p];
  return nCellList[p];
}


// dm - DM containing the data
// p - Center cell of interest
// fid - field id
// array - Array of the data
// offset - Where the data of interest starts
// nDof - Number of degrees of freedom, i.e. number of components in the vector
// vals - Smoothed values
void GaussianConvolution::Evaluate(DM dm, const PetscInt p, const PetscInt fid, const PetscScalar *array, PetscInt offset, const PetscInt nDof, PetscReal vals[]) {

  if (!cellList[p]) BuildList(p);  // Build the convolution list

  PetscArrayzero(vals, nDof) >> ablate::utilities::PetscUtilities::checkError;

  PetscReal totalWT = 0.0;

  for (PetscInt i = 0; i < nCellList[p]; ++i) {
    PetscInt cell = cellList[p][i];
    const PetscScalar *data;
    xDMPlexPointLocalRead(dm, cell, fid, array, &data);
    for (PetscInt c = 0; c < nDof; ++c) {
      vals[c] += data[offset + c]*cellWeights[p][i];
    }
    totalWT += cellWeights[p][i];
  }

  for (PetscInt c = 0; c < nDof; ++c) vals[c] /= totalWT;

}
