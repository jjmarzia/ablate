#include "intSharp.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"

#define UNUSED(X) {(void)X;}

void ablate::finiteVolume::processes::IntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) { }

ablate::finiteVolume::processes::IntSharp::IntSharp(PetscReal Gamma, PetscReal epsilon) : Gamma(Gamma), epsilon(epsilon) {}

void ablate::finiteVolume::processes::IntSharp::ClearData() {
  if (cellDM) DMDestroy(&cellDM);
  if (fluxDM) DMDestroy(&fluxDM);
  if (vertDM) DMDestroy(&vertDM);
  if (gaussianConv) gaussianConv->~GaussianConvolution();
}

ablate::finiteVolume::processes::IntSharp::~IntSharp() {
  ablate::finiteVolume::processes::IntSharp::ClearData();
}

void ablate::finiteVolume::processes::IntSharp::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {

  DM dm = flow.GetSubDomain().GetDM();
  PetscInt vStart, vEnd, cStart, cEnd;
  PetscInt dim;

  // Clear any previously allocated memory
  ablate::finiteVolume::processes::IntSharp::ClearData();

  DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;
  DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd) >> utilities::PetscUtilities::checkError;
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;

  ablate::utilities::PetscUtilities::CopyDM(dm, cStart, cEnd, 1, &cellDM);        // Cell-based smoothed VOF field
  ablate::utilities::PetscUtilities::CopyDM(dm, vStart, vEnd, 1, &vertDM);  // Vertex-based scalars
  ablate::utilities::PetscUtilities::CopyDM(dm, vStart, vEnd, dim, &fluxDM);  // Vertex-based gradients

  gaussianConv = std::make_shared<ablate::finiteVolume::stencil::GaussianConvolution>(dm, 4, 2.0);

  // If Setup is called twice would this be added twice?
  flow.RegisterRHSFunction(ComputeTerm, this);
}

#include <signal.h>



void SaveCellData(DM dm, const Vec vec, const char fname[255], const PetscInt id, PetscInt Nc, ablate::domain::Range range) {


  const PetscScalar *array;
  PetscInt      dim;
  MPI_Comm      comm = PetscObjectComm((PetscObject)dm);
  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  VecGetArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
  DMGetDimension(dm, &dim);

  PetscInt boundaryCellStart;
  DMPlexGetCellTypeStratum(dm, DM_POLYTOPE_FV_GHOST, &boundaryCellStart, nullptr) >> ablate::utilities::PetscUtilities::checkError;


  for (PetscInt r = 0; r < size; ++r) {
    if ( rank==r ) {

      FILE *f1;
      if ( rank==0 ) f1 = fopen(fname, "w");
      else f1 = fopen(fname, "a");
      if (f1==NULL) throw std::runtime_error("Vertex is marked as next to a cut cell but is not!");

      for (PetscInt c = range.start; c < range.end; ++c) {
        PetscInt cell = range.points ? range.points[c] : c;

        DMPolytopeType ct;
        DMPlexGetCellType(dm, cell, &ct) >> ablate::utilities::PetscUtilities::checkError;

        if (ct < 12) {

          PetscReal x0[3];
          DMPlexComputeCellGeometryFVM(dm, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;
          for (PetscInt d = 0; d < dim; ++d) {
            fprintf(f1, "%+e\t", x0[d]);
          }

          const PetscScalar *val;
          xDMPlexPointLocalRead(dm, cell, id, array, &val) >> ablate::utilities::PetscUtilities::checkError;
          for (PetscInt i = 0; i < Nc; ++i) {
            fprintf(f1, "%+e\t", val[i]);
          }

          fprintf(f1, "\n");
        }
      }
      fclose(f1);
    }

    MPI_Barrier(comm);
  }


  VecRestoreArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
}

// Calculate the value of phi at a vertex as an inverse-distance weighted sum of all cells using this vertex
PetscReal VertexPhi(const PetscInt vert, Vec cellGeomVec, DM phiDM, Vec phiVec) {

  PetscReal vertPhi = 0.0, totalWt = 0.0;
  PetscInt nCells, *cellList;
  DM dmCell;
  const PetscScalar* cellGeomArray;
  PetscScalar *x0;
  PetscInt dim;
  const PetscScalar *phiArray;

  DMGetDimension(phiDM, &dim) >> ablate::utilities::PetscUtilities::checkError;

  VecGetDM(cellGeomVec, &dmCell) >> ablate::utilities::PetscUtilities::checkError;

  DMPlexVertexGetCoordinates(phiDM, 1, &vert, &x0) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArrayRead(cellGeomVec, &cellGeomArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArrayRead(phiVec, &phiArray) >> ablate::utilities::PetscUtilities::checkError;
  DMPlexVertexGetCells(phiDM, vert, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt c = 0; c < nCells; ++c) {
    const PetscInt cell = cellList[c];

    PetscFVCellGeom* cg;
    DMPlexPointLocalRead(dmCell, cell, cellGeomArray, &cg) >> ablate::utilities::PetscUtilities::checkError;
    PetscReal *x = cg->centroid;
    PetscReal r = 0.0;

    for (PetscInt d = 0; d < dim; ++d) r += PetscSqr(x[d] - x0[d]);
    r = PetscSqrtReal(r);

    const PetscScalar *phi;
    DMPlexPointLocalRead(phiDM, cell, phiArray, &phi) >> ablate::utilities::PetscUtilities::checkError;

    vertPhi += (*phi)/r;
    totalWt += 1/r;

  }

  DMPlexVertexRestoreCells(phiDM, vert, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArrayRead(phiVec, &phiArray) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> ablate::utilities::PetscUtilities::checkError;
  DMPlexVertexRestoreCoordinates(phiDM, 1, &vert, &x0) >> ablate::utilities::PetscUtilities::checkError;

  return (vertPhi/totalWt);


}

// Calculate epsilon*grad(phi) - phi*(1-phi)*grad(phi)/||grad(phi)||
void VertexFlux(const PetscReal gamma, const PetscReal epsilon, Vec cellGeomVec, DM phiDM, Vec phiVec, DM maskDM, Vec vertMask, DM fluxDM, Vec fluxVec) {
  PetscInt vStart, vEnd;
  PetscScalar *fluxArray;
  const PetscScalar *maskArray;
  PetscInt dim;

  DMGetDimension(fluxDM, &dim) >> ablate::utilities::PetscUtilities::checkError;
  VecZeroEntries(fluxVec) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(fluxVec, &fluxArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArrayRead(vertMask, &maskArray) >> ablate::utilities::PetscUtilities::checkError;

  DMPlexGetDepthStratum(fluxDM, 0, &vStart, &vEnd) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt v = vStart; v < vEnd; ++v) {
    const PetscScalar *maskVal;
    DMPlexPointLocalRead(maskDM, v, maskArray, &maskVal) >> ablate::utilities::PetscUtilities::checkError;

    if (*maskVal > 0.5) {

      PetscScalar grad[dim];
      DMPlexVertexGradFromCell(phiDM, v, phiVec, -1, 0, grad) >> ablate::utilities::PetscUtilities::checkError;

      const PetscScalar mag = ablate::utilities::MathUtilities::MagVector(dim, grad);
      const PetscReal phi = VertexPhi(v, cellGeomVec, phiDM, phiVec);

      PetscScalar *flux;
      DMPlexPointLocalRef(fluxDM, v, fluxArray, &flux) >> ablate::utilities::PetscUtilities::checkError;

      const PetscReal fac = gamma*(epsilon - phi*(1 - phi)/mag);

      ablate::utilities::MathUtilities::ScaleVector(dim, flux, fac);
    }

  }

  VecRestoreArray(fluxVec, &fluxArray) >> ablate::utilities::PetscUtilities::checkError;

}

PetscErrorCode ablate::finiteVolume::processes::IntSharp::ComputeTerm(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx) {


    PetscFunctionBegin;
int rank;
MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

//PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%s::%s::%d\t%d\n", __FILE__, __FUNCTION__, __LINE__, rank);PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
//exit(0);

    auto *process = (ablate::finiteVolume::processes::IntSharp *)ctx;

    // Everything in the IntSharp process must be declared, otherwise you get an "invalid use of member ... in static member function error
    DM cellDM = process->cellDM;
    DM vertDM = process->vertDM;          UNUSED(vertDM);
    DM fluxDM = process->fluxDM;  UNUSED(fluxDM);
    const PetscScalar *phiRange = process->phiRange; UNUSED(phiRange);
    PetscInt dim = solver.GetSubDomain().GetDimensions(); UNUSED(dim);

    const ablate::domain::Field &phiField = solver.GetSubDomain().GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);

//    PetscReal val;
  std::shared_ptr<ablate::finiteVolume::stencil::GaussianConvolution> gaussianConv = process->gaussianConv;

  ablate::domain::Range cellRange;
  solver.GetCellRangeWithoutGhost(cellRange);

  DM phiDM = dm;
  Vec phiVec = locX;
  if (phiField.location != ablate::domain::FieldLocation::SOL) {
    throw std::runtime_error("The vector containing the VOF field is not SOL");
  }
//  // Get the DM and Vec of the volume-fraction, in case it's not in the SOL vec.
//  DM phiDM = solver.GetSubDomain().GetFieldDM(phiField);
//  Vec phiVec = solver.GetSubDomain().GetVec(phiField);


  // Local and global smooth phi field
  Vec smoothPhiVec[2] = {NULL, NULL};
  DMGetLocalVector(cellDM, &smoothPhiVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMGetGlobalVector(cellDM, &smoothPhiVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

  // Mask indicating the cells of interest
  Vec cellMaskVec[2] = {NULL, NULL};
  DMGetLocalVector(cellDM, &cellMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMGetGlobalVector(cellDM, &cellMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecZeroEntries(cellMaskVec[GLOBAL]);
  VecZeroEntries(cellMaskVec[LOCAL]);

  const PetscScalar *phiArray;
  PetscScalar *smoothPhiArray[2] = {NULL, NULL};
  PetscScalar *cellMaskArray[2] = {NULL, NULL};

  VecGetArrayRead(phiVec, &phiArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(smoothPhiVec[GLOBAL], &smoothPhiArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(cellMaskVec[LOCAL], &cellMaskArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;


  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

    const PetscScalar *phiVal;
    PetscScalar *smoothPhiVal;

    xDMPlexPointLocalRead(phiDM, cell, phiField.id, phiArray, &phiVal) >> ablate::utilities::PetscUtilities::checkError;
    DMPlexPointLocalRef(cellDM, cell, smoothPhiArray[GLOBAL], &smoothPhiVal) >> ablate::utilities::PetscUtilities::checkError;
    *smoothPhiVal = *phiVal;

    if (*phiVal > phiRange[0] && *phiVal < phiRange[1]) {
        const PetscInt *cellList;
        PetscInt nCells = gaussianConv->GetCellList(cell, &cellList);
        (void)nCells;

      for (PetscInt i = 0; i < nCells; ++i) {
        PetscScalar *maskVal;
        DMPlexPointLocalRef(cellDM, cellList[i], cellMaskArray[LOCAL], &maskVal) >> ablate::utilities::PetscUtilities::checkError;
        *maskVal = 1;
      }
    }
  }

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

    const PetscScalar *phiVal;

    xDMPlexPointLocalRead(phiDM, cell, phiField.id, phiArray, &phiVal) >> ablate::utilities::PetscUtilities::checkError;

    if (*phiVal > phiRange[0] && *phiVal < phiRange[1]) {
      PetscScalar *maskVal;
      DMPlexPointLocalRef(cellDM, cell, cellMaskArray[LOCAL], &maskVal) >> ablate::utilities::PetscUtilities::checkError;
      *maskVal = 2;

    }
  }

  VecRestoreArrayRead(phiVec, &phiArray) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArray(smoothPhiVec[GLOBAL], &smoothPhiArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArray(cellMaskVec[LOCAL], &cellMaskArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;

  // Populate the local vector so that overlap cells have values
  DMGlobalToLocal(cellDM, smoothPhiVec[GLOBAL], INSERT_ALL_VALUES, smoothPhiVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMLocalToGlobal(cellDM, cellMaskVec[LOCAL], ADD_ALL_VALUES, cellMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMGlobalToLocal(cellDM, cellMaskVec[GLOBAL], ADD_ALL_VALUES, cellMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;

  // Do one more pass over the cells. Mark any cells which might have a non-zero smoothed phi field.
  VecGetArray(cellMaskVec[GLOBAL], &cellMaskArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(cellMaskVec[LOCAL], &cellMaskArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    const PetscInt cell = cellRange.GetPoint(c);

    PetscScalar *globalMaskVal;
    DMPlexPointLocalRef(cellDM, cell, cellMaskArray[GLOBAL], &globalMaskVal) >> ablate::utilities::PetscUtilities::checkError;

    if (*globalMaskVal > 0.5) {
      const PetscInt *cellList;
      PetscInt nCells = gaussianConv->GetCellList(cell, &cellList);

      for (PetscInt i = 0; i < nCells; ++i) {
        PetscScalar *localMaskVal;
        DMPlexPointLocalRef(cellDM, cellList[i], cellMaskArray[LOCAL], &localMaskVal) >> ablate::utilities::PetscUtilities::checkError;
        *localMaskVal = (*localMaskVal < 0.5) ? 1 : *localMaskVal;
      }
    }
  }
  VecRestoreArray(cellMaskVec[GLOBAL], &cellMaskArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArray(cellMaskVec[LOCAL], &cellMaskArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;

  DMLocalToGlobal(cellDM, cellMaskVec[LOCAL], ADD_ALL_VALUES, cellMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMGlobalToLocal(cellDM, cellMaskVec[GLOBAL], ADD_ALL_VALUES, cellMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;

  SaveCellData(cellDM, cellMaskVec[LOCAL], "maskLocal.txt", -1, 1, cellRange);
  SaveCellData(cellDM, cellMaskVec[GLOBAL], "maskGlobal.txt", -1, 1, cellRange);
  SaveCellData(cellDM, smoothPhiVec[GLOBAL], "phi0.txt", -1, 1, cellRange);


  Vec vertMaskVec[2] = {nullptr, nullptr};
  PetscScalar *vertMaskArray[2] = {nullptr, nullptr};
  DMGetLocalVector(vertDM, &vertMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
//  DMGetGlobalVector(vertDM, &vertMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecZeroEntries(vertMaskVec[LOCAL]);
//  VecZeroEntries(vertMaskVec[GLOBAL]);

  VecGetArray(vertMaskVec[LOCAL], &vertMaskArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;

  VecGetArray(cellMaskVec[GLOBAL], &cellMaskArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(smoothPhiVec[GLOBAL], &smoothPhiArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(smoothPhiVec[LOCAL], &smoothPhiArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    const PetscInt cell = cellRange.GetPoint(c);

    PetscScalar *maskVal;
    DMPlexPointLocalRef(cellDM, cell, cellMaskArray[GLOBAL], &maskVal) >> ablate::utilities::PetscUtilities::checkError;

    if (*maskVal > 0.5) {

        // Mark all of the vertices associated with this cell.
      PetscInt nVert, *vertList;
      DMPlexCellGetVertices(cellDM, cell, &nVert, &vertList) >> ablate::utilities::PetscUtilities::checkError;
      for (PetscInt v = 0; v < nVert; ++v) {
        const PetscInt vert = vertList[v];
        PetscScalar *vertMaskVal;
        DMPlexPointLocalRef(vertDM, vert, vertMaskArray[LOCAL], &vertMaskVal) >> ablate::utilities::PetscUtilities::checkError;
        *vertMaskVal = 1;
      }
      DMPlexCellRestoreVertices(cellDM, cell, &nVert, &vertList) >> ablate::utilities::PetscUtilities::checkError;

      // Calculate the smoothed VOF field
      PetscScalar *phiVal;
      DMPlexPointLocalRef(cellDM, cell, smoothPhiArray[GLOBAL], &phiVal) >> ablate::utilities::PetscUtilities::checkError;

      gaussianConv->Evaluate(cellDM, cell, -1, smoothPhiArray[LOCAL], 0, 1, phiVal);
    }
  }

  VecRestoreArray(vertMaskVec[LOCAL], &vertMaskArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArray(cellMaskVec[GLOBAL], &cellMaskArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArray(smoothPhiVec[GLOBAL], &smoothPhiArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArray(smoothPhiVec[LOCAL], &smoothPhiArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;


  DMGlobalToLocal(cellDM, smoothPhiVec[GLOBAL], INSERT_ALL_VALUES, smoothPhiVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;

  SaveCellData(cellDM, smoothPhiVec[LOCAL], "phi1.txt", -1, 1, cellRange);


  // Flux (interior portion of the divergence term
  Vec fluxVec[2] = {nullptr, nullptr};
  DMGetLocalVector(fluxDM, &fluxVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  Vec cellGeomVec;
  solver.GetGeomVecs(&cellGeomVec, NULL);
  VertexFlux(process->Gamma, process->epsilon, cellGeomVec, cellDM, smoothPhiVec[LOCAL], vertDM, vertMaskVec[LOCAL], fluxDM, fluxVec[LOCAL]);



  // Finally compute the net force at cell centers
  const ablate::domain::Field &eulerField = solver.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);
  const ablate::domain::Field &densityVFField = solver.GetSubDomain().GetField("densityvolumeFraction");

  const PetscScalar *xArray;
  PetscScalar *fArray;

  VecGetArray(smoothPhiVec[GLOBAL], &smoothPhiArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArrayRead(locX, &xArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(locFVec, &fArray) >> ablate::utilities::PetscUtilities::checkError;
FILE *f1 = fopen("force.txt", "w");
  // Net force on the cell-center
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

    const PetscScalar *phiVal;
PetscReal x[3];
DMPlexComputeCellGeometryFVM(dm, cell, NULL, x, NULL);

    xDMPlexPointLocalRead(phiDM, cell, phiField.id, xArray, &phiVal) >> ablate::utilities::PetscUtilities::checkError;

    if (*phiVal > phiRange[0] && *phiVal < phiRange[1]) {
      const PetscScalar *smoothPhiVal, *euler;
      DMPlexPointLocalRead(cellDM, cell, smoothPhiArray[GLOBAL], &smoothPhiVal) >> ablate::utilities::PetscUtilities::checkError;
      xDMPlexPointLocalRead(dm, cell, eulerField.id, xArray, &euler) >> ablate::utilities::PetscUtilities::checkError;
//This is wrong. This is the mixed density, not the density of the gas
      PetscScalar rhoGrad[dim];
      DMPlexCellGradFromCell(dm, cell, locX, eulerField.id, CompressibleFlowFields::RHO, rhoGrad);

      PetscScalar *rhoPhiForce;
      xDMPlexPointLocalRef(dm, cell, densityVFField.id, fArray, &rhoPhiForce) >> ablate::utilities::PetscUtilities::checkError;

      for (PetscInt d = 0; d < dim; ++d) {
        PetscScalar fluxGrad[dim];
        DMPlexCellGradFromVertex(fluxDM, cell, fluxVec[LOCAL], -1, d, fluxGrad);

        *rhoPhiForce += euler[CompressibleFlowFields::RHO]*fluxGrad[d] + (*smoothPhiVal)*euler[CompressibleFlowFields::RHOU + d]*rhoGrad[d];
      }
fprintf(f1, "%+e\t%+e\t%+e\n", x[0], x[1], *rhoPhiForce);
    }
  }
fclose(f1);
  VecRestoreArray(smoothPhiVec[GLOBAL], &smoothPhiArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArrayRead(locX, &xArray) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArray(locFVec, &fArray) >> ablate::utilities::PetscUtilities::checkError;


  // Clear all of the temporary vectors.
  if (smoothPhiVec[LOCAL]) DMRestoreLocalVector(cellDM, &smoothPhiVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  if (cellMaskVec[LOCAL]) DMRestoreLocalVector(cellDM, &cellMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  if (vertMaskVec[LOCAL]) DMRestoreLocalVector(vertDM, &vertMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  if (fluxVec[LOCAL]) DMRestoreLocalVector(fluxDM, &fluxVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;

  if (smoothPhiVec[GLOBAL]) DMRestoreGlobalVector(cellDM, &smoothPhiVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  if (cellMaskVec[GLOBAL]) DMRestoreGlobalVector(cellDM, &cellMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  if (vertMaskVec[GLOBAL]) DMRestoreGlobalVector(vertDM, &vertMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  if (fluxVec[GLOBAL]) DMRestoreGlobalVector(fluxDM, &fluxVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

  solver.RestoreRange(cellRange);

printf("%s::%s::%d\n", __FILE__, __FUNCTION__, __LINE__);
exit(0);
    PetscFunctionReturn(0);

}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::IntSharp, "calculates interface regularization term",
         ARG(PetscReal, "Gamma", "Gamma, velocity scale parameter (approx. umax)"),
         ARG(PetscReal, "epsilon", "epsilon, interface thickness scale parameter (approx. h)")
);
