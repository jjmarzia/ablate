#include "intSharp.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"
#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"
#include <signal.h>
#define UNUSED(X) {(void)X;}

void ablate::finiteVolume::processes::IntSharp::ClearData() {
  if (cellDM) DMDestroy(&cellDM);
  if (fluxDM) DMDestroy(&fluxDM);
  if (vertDM) DMDestroy(&vertDM);
  if (cellGaussianConv) cellGaussianConv->~GaussianConvolution();
  if (vertexGaussianConv) vertexGaussianConv->~GaussianConvolution();
}

ablate::finiteVolume::processes::IntSharp::~IntSharp() {
  ablate::finiteVolume::processes::IntSharp::ClearData();
}

void ablate::finiteVolume::processes::IntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) {

  DM dm = flow.GetSubDomain().GetDM();
  PetscInt vStart, vEnd, cStart, cEnd;
  PetscInt dim;

  // Clear any previously allocated memory
  ablate::finiteVolume::processes::IntSharp::ClearData();

  DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;
  DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd) >> utilities::PetscUtilities::checkError;
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;

  ablate::utilities::PetscUtilities::CopyDM(dm, cStart, cEnd, 1, &cellDM);    // Cell-based smoothed VOF field
  ablate::utilities::PetscUtilities::CopyDM(dm, vStart, vEnd, 1, &vertDM);    // Vertex-based scalars
  ablate::utilities::PetscUtilities::CopyDM(dm, vStart, vEnd, dim, &fluxDM);  // Vertex-based gradients

  cellGaussianConv = std::make_shared<ablate::finiteVolume::stencil::GaussianConvolution>(dm, 1.0, 0, ablate::finiteVolume::stencil::GaussianConvolution::DepthOrHeight::HEIGHT);
  vertexGaussianConv = std::make_shared<ablate::finiteVolume::stencil::GaussianConvolution>(dm, 1.0, 0, ablate::finiteVolume::stencil::GaussianConvolution::DepthOrHeight::DEPTH);


}

ablate::finiteVolume::processes::IntSharp::IntSharp(PetscReal Gamma, PetscReal epsilon, bool addToRHS) : Gamma(Gamma), epsilon(epsilon), addToRHS(addToRHS) {}

//wrapper function to match the expected signature in intsharp::setup
void intSharpPreStageWrapper(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime, ablate::finiteVolume::processes::IntSharp* intSharpProcess) {
  intSharpProcess->PreStage(flowTs, solver, stagetime);
}

void ablate::finiteVolume::processes::IntSharp::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
  // Before each step, sharpen the alpha and propagate changes into conserved values
  auto intSharpPreStage = std::bind(intSharpPreStageWrapper, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, this);
  flow.RegisterPreStage(intSharpPreStage);


  // List of required fields
  std::string fieldList[] = { ablate::finiteVolume::CompressibleFlowFields::GASDENSITY_FIELD,
                              TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD,
                              ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD,
                              "densityvolumeFraction"};

  for (auto field : fieldList) {
    if (!(flow.GetSubDomain().ContainsField(field))) {
      throw std::runtime_error("ablate::finiteVolume::processes::IntSharp expects a "+ field +" field to be defined.");
    }
  }

  flow.RegisterRHSFunction(ComputeTerm, this);
}

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
      if (f1==nullptr) throw std::runtime_error("Vertex is marked as next to a cut cell but is not!");

      for (PetscInt c = range.start; c < range.end; ++c) {
        PetscInt cell = range.points ? range.points[c] : c;

        DMPolytopeType ct;
        DMPlexGetCellType(dm, cell, &ct) >> ablate::utilities::PetscUtilities::checkError;

        if (ct < 12) {

          PetscReal x0[3];
          DMPlexComputeCellGeometryFVM(dm, cell, nullptr, x0, nullptr) >> ablate::utilities::PetscUtilities::checkError;
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

void ablate::finiteVolume::processes::IntSharp::MemoryHelper(DM dm, VecLoc loc, Vec *vec, PetscScalar **array) {

  if (loc==LOCAL) {
    DMGetLocalVector(dm, vec) >> ablate::utilities::PetscUtilities::checkError;
    VecZeroEntries(*vec) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(*vec, array) >> ablate::utilities::PetscUtilities::checkError;
    localVecList.push_back({dm, *vec, *array});
  }
  else {
    DMGetGlobalVector(dm, vec) >> ablate::utilities::PetscUtilities::checkError;
    VecZeroEntries(*vec) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(*vec, array) >> ablate::utilities::PetscUtilities::checkError;
    globalVecList.push_back({dm, *vec, *array});
  }

}

void ablate::finiteVolume::processes::IntSharp::MemoryHelper() {
  for (struct vecData data : localVecList) {
    VecRestoreArray(data.vec, &data.array) >> ablate::utilities::PetscUtilities::checkError;
    DMRestoreLocalVector(data.dm, &data.vec) >> ablate::utilities::PetscUtilities::checkError;
  }
  localVecList.clear();

  for (struct vecData data : globalVecList) {
    VecRestoreArray(data.vec, &data.array) >> ablate::utilities::PetscUtilities::checkError;
    DMRestoreGlobalVector(data.dm, &data.vec) >> ablate::utilities::PetscUtilities::checkError;
  }
  globalVecList.clear();


}

void ablate::finiteVolume::processes::IntSharp::SetMasks(ablate::domain::Range &cellRange, DM phiDM, Vec phiVec, PetscInt phiID, Vec cellMaskVec[2], PetscScalar *cellMaskArray[2], PetscScalar *vertMaskArray) {

  const PetscScalar *phiArray;
  VecGetArrayRead(phiVec, &phiArray) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

    const PetscScalar *phiVal;
    xDMPlexPointLocalRead(phiDM, cell, phiID, phiArray, &phiVal) >> ablate::utilities::PetscUtilities::checkError;

    if (*phiVal > phiRange[0] && *phiVal < phiRange[1]) {
      const PetscInt *cellList;
      PetscInt nCells = cellGaussianConv->GetCellList(cell, &cellList);

      for (PetscInt i = 0; i < nCells; ++i) {
        PetscScalar *maskVal;
        DMPlexPointLocalRef(cellDM, cellList[i], cellMaskArray[LOCAL], &maskVal) >> ablate::utilities::PetscUtilities::checkError;
        *maskVal = 1;
      }
    }
  }

  VecRestoreArrayRead(phiVec, &phiArray) >> ablate::utilities::PetscUtilities::checkError;

  DMLocalToGlobal(cellDM, cellMaskVec[LOCAL], ADD_ALL_VALUES, cellMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMGlobalToLocal(cellDM, cellMaskVec[GLOBAL], ADD_ALL_VALUES, cellMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;


  // Do one more pass over the cells. Mark any cells which might have a non-zero smoothed phi field.
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    const PetscInt cell = cellRange.GetPoint(c);

    PetscScalar *globalMaskVal;
    DMPlexPointLocalRef(cellDM, cell, cellMaskArray[GLOBAL], &globalMaskVal) >> ablate::utilities::PetscUtilities::checkError;

    if (*globalMaskVal > 0.5) {
      const PetscInt *cellList;
      PetscInt nCells = cellGaussianConv->GetCellList(cell, &cellList);

      for (PetscInt i = 0; i < nCells; ++i) {
        PetscScalar *localMaskVal;
        DMPlexPointLocalRef(cellDM, cellList[i], cellMaskArray[LOCAL], &localMaskVal) >> ablate::utilities::PetscUtilities::checkError;
        *localMaskVal = (*localMaskVal < 0.5) ? 1 : *localMaskVal;
      }
    }
  }


  DMLocalToGlobal(cellDM, cellMaskVec[LOCAL], ADD_ALL_VALUES, cellMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMGlobalToLocal(cellDM, cellMaskVec[GLOBAL], ADD_ALL_VALUES, cellMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;


  // Now mark all of the vertices
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
        DMPlexPointLocalRef(vertDM, vert, vertMaskArray, &vertMaskVal) >> ablate::utilities::PetscUtilities::checkError;
        *vertMaskVal = 1;
      }
      DMPlexCellRestoreVertices(cellDM, cell, &nVert, &vertList) >> ablate::utilities::PetscUtilities::checkError;
    }
  }


}


PetscErrorCode ablate::finiteVolume::processes::IntSharp::PreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime) {
  PetscFunctionBegin;
  const auto &fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver &>(solver);
  ablate::domain::Range cellRange; fvSolver.GetCellRangeWithoutGhost(cellRange);
  PetscInt dim; PetscCall(DMGetDimension(fvSolver.GetSubDomain().GetDM(), &dim));
  const auto &eulerOffset = fvSolver.GetSubDomain().GetField(CompressibleFlowFields::EULER_FIELD).offset;
  const auto &vfOffset = fvSolver.GetSubDomain().GetField(VOLUME_FRACTION_FIELD).offset;
  const auto &rhoAlphaOffset = fvSolver.GetSubDomain().GetField(DENSITY_VF_FIELD).offset;
  DM dm = fvSolver.GetSubDomain().GetDM();
  Vec globFlowVec; PetscCall(TSGetSolution(flowTs, &globFlowVec));
  PetscScalar *flowArray; PetscCall(VecGetArray(globFlowVec, &flowArray));
  PetscInt uOff[3]; uOff[0] = vfOffset; uOff[1] = rhoAlphaOffset; uOff[2] = eulerOffset;
  // Get the rhs vector
  Vec locFVec; PetscCall(DMGetLocalVector(dm, &locFVec)); PetscCall(VecZeroEntries(locFVec));


  // compute intsharp term for all cells
  auto intSharpProcess = std::make_shared<ablate::finiteVolume::processes::IntSharp>(0, 0.001, false);
  std::cout << "Debug: intSharpProcess created" << std::endl;
  intSharpProcess->ComputeTerm(fvSolver, dm, stagetime, globFlowVec, locFVec, intSharpProcess.get());
  std::cout << "Debug: intSharpProcess->ComputeTerm called" << std::endl;

  PetscReal norm[3] = {1, 1, 1};

  for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
      const PetscInt cell = cellRange.GetPoint(i);
      PetscScalar *allFields = nullptr; DMPlexPointLocalRef(dm, cell, flowArray, &allFields) >> utilities::PetscUtilities::checkError;
      auto density = allFields[ablate::finiteVolume::CompressibleFlowFields::RHO];
      PetscReal velocity[3]; for (PetscInt d = 0; d < dim; d++) { velocity[d] = allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density; }

      // decode state (we just need densityG and densityL to reconstruct mixture RHO out of new alpha, and internalEnergy to reconstruct RHOE )
      // PetscReal densityG = 1.0, densityL = 1000.0, internalEnergy = 1e5; 
      PetscReal density, densityG, densityL, internalEnergy, normalVelocity, internalEnergyG, internalEnergyL, aG, aL, MG, ML, p, t, alpha;
      intSharpProcess->decoder->DecodeTwoPhaseEulerState(dim, uOff, allFields, norm, &density, &densityG, &densityL, &normalVelocity, velocity, &internalEnergy, &internalEnergyG, &internalEnergyL, &aG, &aL, &MG, &ML, &p, &t, &alpha);

      // update alpha according to intsharp-calculated flux grad values
      const auto &fluxGrad = intSharpProcess->fluxGradValues[i - cellRange.start];
      const PetscScalar oldAlpha = allFields[vfOffset];
      for (PetscInt d = 0; d < dim; ++d) { if (!std::isnan(fluxGrad[d]) && fluxGrad[d] != 0.0) { allFields[vfOffset] -= fluxGrad[d]; } } // this can be thought of as the RHS of the material derivative of alpha in pseudo time
      allFields[vfOffset] = std::max(allFields[vfOffset], 0.0); // enforce alpha >= 0

      // update euler field based on new alpha
      allFields[rhoAlphaOffset] = (allFields[vfOffset] / oldAlpha) * allFields[rhoAlphaOffset];
      allFields[ablate::finiteVolume::CompressibleFlowFields::RHO] = allFields[vfOffset] * densityG + (1 - allFields[vfOffset]) * densityL;
      allFields[ablate::finiteVolume::CompressibleFlowFields::RHOE] = allFields[ablate::finiteVolume::CompressibleFlowFields::RHO] * internalEnergy;
      for (PetscInt d = 0; d < dim; ++d) {
          allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = allFields[ablate::finiteVolume::CompressibleFlowFields::RHO] * velocity[d];
      }

      //now propagate changes made to conserved vars back to the decode (is this necessary? are we storing this?)
      intSharpProcess->decoder->DecodeTwoPhaseEulerState(dim, uOff, allFields, norm, &density, &densityG, &densityL, &normalVelocity, velocity, &internalEnergy, &internalEnergyG, &internalEnergyL, &aG, &aL, &MG, &ML, &p, &t, &alpha);
  }

  // Restore
  PetscCall(DMRestoreLocalVector(dm, &locFVec)); PetscCall(VecRestoreArray(globFlowVec, &flowArray)); fvSolver.RestoreRange(cellRange); PetscFunctionReturn(0);
}

// Note: locX is a local vector, which means it contains all overlap cells
PetscErrorCode ablate::finiteVolume::processes::IntSharp::ComputeTerm(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx) {


    PetscFunctionBegin;
int rank;
MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  ablate::finiteVolume::processes::IntSharp *process = (ablate::finiteVolume::processes::IntSharp *)ctx;

  // Everything in the IntSharp process must be declared, otherwise you get an "invalid use of member ... in static member function error
  DM cellDM = process->cellDM;
  DM vertDM = process->vertDM;          UNUSED(vertDM);
  DM fluxDM = process->fluxDM;  UNUSED(fluxDM);
  const PetscScalar *phiRange = process->phiRange; UNUSED(phiRange);
  PetscInt dim = solver.GetSubDomain().GetDimensions(); UNUSED(dim);

  const ablate::domain::Field &phiField = solver.GetSubDomain().GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
  const ablate::domain::Field &eulerField = solver.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);
  const ablate::domain::Field &gasDensityField = solver.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::GASDENSITY_FIELD);
  const ablate::domain::Field &densityVFField = solver.GetSubDomain().GetField("densityvolumeFraction");


  const auto &ofield = solver.GetSubDomain().GetField("debug");


  std::shared_ptr<ablate::finiteVolume::stencil::GaussianConvolution> cellGaussianConv = process->cellGaussianConv;
  std::shared_ptr<ablate::finiteVolume::stencil::GaussianConvolution> vertexGaussianConv = process->vertexGaussianConv;

  ablate::domain::Range cellRange;
  solver.GetCellRangeWithoutGhost(cellRange);

//  SaveCellData(dm, locX, "phiField.txt", phiField.id, phiField.numberComponents, cellRange);
//  SaveCellData(dm, locX, "eulerField.txt", eulerField.id, eulerField.numberComponents, cellRange);
//  {
//     DM gasDensityDM = solver.GetSubDomain().GetFieldDM(gasDensityField);
//    Vec gasDensityVec = solver.GetSubDomain().GetVec(gasDensityField);
//    SaveCellData(gasDensityDM, gasDensityVec, "gasDensityField.txt", gasDensityField.id, gasDensityField.numberComponents, cellRange);
//  }
//  SaveCellData(dm, locX, "densityVFField.txt", densityVFField.id, densityVFField.numberComponents, cellRange);


  if (phiField.location != ablate::domain::FieldLocation::SOL) {
    throw std::runtime_error("The vector containing the VOF field is not SOL");
  }

  // Mask indicating the cells of interest
  Vec cellMaskVec[2] = {nullptr, nullptr};
  PetscScalar *cellMaskArray[2] = {nullptr, nullptr};
  process->MemoryHelper(cellDM, LOCAL, &cellMaskVec[LOCAL], &cellMaskArray[LOCAL]);
  process->MemoryHelper(cellDM, GLOBAL, &cellMaskVec[GLOBAL], &cellMaskArray[GLOBAL]);

  Vec vertMaskVec = nullptr;
  PetscScalar *vertMaskArray = nullptr;
  process->MemoryHelper(vertDM, LOCAL, &vertMaskVec, &vertMaskArray);

  process->SetMasks(cellRange, dm, locX, phiField.id, cellMaskVec, cellMaskArray, vertMaskArray);


  // Flux (interior portion of the divergence term)
  Vec sharpeningVec[2] = {nullptr, nullptr};
  PetscScalar *sharpeningArray[2] = {nullptr, nullptr};
  process->MemoryHelper(fluxDM, LOCAL, &sharpeningVec[LOCAL], &sharpeningArray[LOCAL]);

  const PetscScalar *xArray;
  VecGetArrayRead(locX, &xArray) >> ablate::utilities::PetscUtilities::checkError;

  PetscInt vStart, vEnd;
  DMPlexGetDepthStratum(vertDM, 0, &vStart, &vEnd) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt v = vStart; v < vEnd; ++v) {

    const PetscScalar *maskVal;
    DMPlexPointLocalRead(vertDM, v, vertMaskArray, &maskVal) >> ablate::utilities::PetscUtilities::checkError;

    if (*maskVal > 0.5) {
      PetscReal smoothPhi;
      vertexGaussianConv->Evaluate(v, nullptr, dm, phiField.id, xArray, 0, 1, &smoothPhi);

      PetscReal phiGrad[dim], norm = 0.0;
      for (PetscInt d = 0; d < dim; ++d) {
        PetscInt dx[3] = {0, 0, 0};
        dx[d] = 1;
        vertexGaussianConv->Evaluate(v, dx, dm, phiField.id, xArray, 0, 1, &phiGrad[d]);
        norm += PetscSqr(phiGrad[d]);
      }
      norm = PetscSqrtReal(norm);

      PetscScalar *sharpeningFlux;
      DMPlexPointLocalRead(fluxDM, v, sharpeningArray[LOCAL], &sharpeningFlux);

      if (norm > PETSC_MACHINE_EPSILON) {
        smoothPhi = process->epsilon - smoothPhi*(1-smoothPhi)/norm;
        for (PetscInt d = 0; d < dim; ++d) {
          sharpeningFlux[d] = phiGrad[d]*smoothPhi;
        }
      }
      else {
        for (PetscInt d = 0; d < dim; ++d) sharpeningFlux[d] = 0.0;
      }
    }
  }

  //  Net force at cell centers
  PetscScalar *fArray;
  VecGetArray(locFVec, &fArray) >> ablate::utilities::PetscUtilities::checkError;

  DM gasDensityDM = solver.GetSubDomain().GetFieldDM(gasDensityField);
  Vec gasDensityVec = solver.GetSubDomain().GetVec(gasDensityField);
  const PetscScalar *gasDensityArray;
  VecGetArrayRead(gasDensityVec, &gasDensityArray);

    Vec auxVec = solver.GetSubDomain().GetAuxVector(); //LOCAL aux vector, not global
    PetscScalar *auxArray; VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;


    // make fluxGradValues (to be accessible by prestage) be a matrix M[nCells][dim]
    process->fluxGradValues.resize(cellRange.end - cellRange.start, std::vector<PetscScalar>(dim, 0.0));

  // Net force on the cell-center
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

    const PetscScalar *phiVal;
    xDMPlexPointLocalRead(dm, cell, phiField.id, xArray, &phiVal) >> ablate::utilities::PetscUtilities::checkError;

    if (*phiVal > phiRange[0] && *phiVal < phiRange[1]) {
      const PetscScalar *euler;
      xDMPlexPointLocalRead(dm, cell, eulerField.id, xArray, &euler) >> ablate::utilities::PetscUtilities::checkError;

      PetscReal smoothRhoG;
      cellGaussianConv->Evaluate(cell, nullptr, gasDensityDM, gasDensityField.id, gasDensityArray, 0, 1, &smoothRhoG);
      smoothRhoG *= process->Gamma;

      PetscReal smoothPhi;
      cellGaussianConv->Evaluate(cell, nullptr, dm, phiField.id, xArray, 0, 1, &smoothPhi);

      PetscScalar *force;
      xDMPlexPointLocalRef(dm, cell, densityVFField.id, fArray, &force) >> ablate::utilities::PetscUtilities::checkError;

PetscScalar *optr;
xDMPlexPointLocalRef(solver.GetSubDomain().GetAuxDM(), cell, ofield.id, auxArray, &optr);


      for (PetscInt d = 0; d < dim; ++d) {
        PetscScalar fluxGrad[dim];
        DMPlexCellGradFromVertex(fluxDM, cell, sharpeningVec[LOCAL], -1, d, fluxGrad);

        PetscReal u = euler[CompressibleFlowFields::RHOU+d]/euler[CompressibleFlowFields::RHO];

        PetscReal dRhoG;
        PetscInt dx[3] = {0, 0, 0};
        dx[d] = 1;
        cellGaussianConv->Evaluate(cell, dx, gasDensityDM, gasDensityField.id, gasDensityArray, 0, 1, &dRhoG);

        if (process->addToRHS) {
          *force -= smoothRhoG*fluxGrad[d] + 0.0*smoothPhi*u*dRhoG;
      }

        *optr -= smoothRhoG*fluxGrad[d];

        // Store the fluxGrad value
        process->fluxGradValues[c - cellRange.start][d] = fluxGrad[d];

      }

    }

  }

  VecRestoreArray(auxVec, &auxArray);

  VecRestoreArrayRead(gasDensityVec, &gasDensityArray);
  VecRestoreArrayRead(locX, &xArray) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArray(locFVec, &fArray) >> ablate::utilities::PetscUtilities::checkError;

  // Clear all of the temporary vectors.
  process->MemoryHelper();
//SaveCellData(dm, locFVec, "force.txt", densityVFField.id, 1, cellRange);
  solver.RestoreRange(cellRange);



//PetscPrintf(PETSC_COMM_WORLD, "%s::%s::%d\n", __FILE__, __FUNCTION__, __LINE__);
//exit(0);
    PetscFunctionReturn(0);

}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::IntSharp, "calculates interface regularization term",
         ARG(PetscReal, "Gamma", "Gamma, velocity scale parameter (approx. umax)"),
         ARG(PetscReal, "epsilon", "epsilon, interface thickness scale parameter (approx. h)"),
         ARG(bool, "addtoRHS", "add to the RHS of the densityVFField equation")
);
