#include "domain/RBF/mq.hpp"
#include "intSharp.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"
#include <fstream>
#include <PetscTime.h>


static void IS_CopyDM(DM oldDM, const PetscInt pStart, const PetscInt pEnd, const PetscInt nDOF, DM *newDM) {
    PetscSection section;
    // Create a sub auxDM
    DM coordDM;
    DMGetCoordinateDM(oldDM, &coordDM) >> ablate::utilities::PetscUtilities::checkError;
    DMClone(oldDM, newDM) >> ablate::utilities::PetscUtilities::checkError;
    // this is a hard coded "dmAux" that petsc looks for
    DMSetCoordinateDM(*newDM, coordDM) >> ablate::utilities::PetscUtilities::checkError;
    PetscSectionCreate(PetscObjectComm((PetscObject)(*newDM)), &section) >> ablate::utilities::PetscUtilities::checkError;
    PetscSectionSetChart(section, pStart, pEnd) >> ablate::utilities::PetscUtilities::checkError;
    for (PetscInt p = pStart; p < pEnd; ++p) PetscSectionSetDof(section, p, nDOF) >> ablate::utilities::PetscUtilities::checkError;
    PetscSectionSetUp(section) >> ablate::utilities::PetscUtilities::checkError;
    DMSetLocalSection(*newDM, section) >> ablate::utilities::PetscUtilities::checkError;
    PetscSectionDestroy(&section) >> ablate::utilities::PetscUtilities::checkError;
    DMSetUp(*newDM) >> ablate::utilities::PetscUtilities::checkError;
    // This builds the global section information based on the local section. It's necessary if we don't create a global vector
    //    right away.
    DMGetGlobalSection(*newDM, &section) >> ablate::utilities::PetscUtilities::checkError;
    /* Calling DMPlexComputeGeometryFVM() generates the value returned by DMPlexGetMinRadius() */
    Vec cellgeom = NULL;
    Vec facegeom = NULL;
    DMPlexComputeGeometryFVM(*newDM, &cellgeom, &facegeom);
    VecDestroy(&cellgeom);
    VecDestroy(&facegeom);
}

void GetCoordinate3D(DM dm, PetscInt dim, PetscInt p, PetscReal *xp, PetscReal *yp, PetscReal *zp){
    //get the coordinates of the point
    PetscReal vol; PetscReal centroid[3];
    DMPlexComputeCellGeometryFVM(dm, p, &vol, centroid, nullptr);
    *xp = centroid[0]; *yp = centroid[1]; *zp = centroid[2];
}
void GetCoordinate1D(DM dm, PetscInt dim, PetscInt p, PetscReal *xp){
    //get the coordinates of the point
    PetscReal vol; PetscReal centroid[dim];
    DMPlexComputeCellGeometryFVM(dm, p, &vol, centroid, nullptr);
    *xp = centroid[0];
}
void PhiNeighborGauss(PetscReal d, PetscReal s, PetscReal *weight){
    PetscReal g0 = PetscExpReal(0/ (2*PetscSqr(s)));
    PetscReal gd = PetscExpReal(-PetscSqr(d)/ (2*PetscSqr(s)));
    *weight = gd/g0;
}
PetscInt phitildepenalty[999999] = { 0 };
void PushGhost(DM dm, Vec LocalVec, Vec GlobalVec, InsertMode ADD_OR_INSERT_VALUES, bool zerovec, bool isphitilde) {
    if ((ADD_OR_INSERT_VALUES == ADD_VALUES) and (zerovec == true)){
        VecZeroEntries(GlobalVec);
    }
    DMLocalToGlobal(dm, LocalVec, ADD_OR_INSERT_VALUES, GlobalVec); //p0 to p1
    DMGlobalToLocal(dm, GlobalVec, INSERT_VALUES, LocalVec); //p1 to p1
    PetscScalar *LocalArray; VecGetArray(LocalVec, &LocalArray);
    PetscInt cStart, cEnd; DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
    if ((ADD_OR_INSERT_VALUES == ADD_VALUES) and (isphitilde)){
        for (PetscInt cell = cStart; cell < cEnd; ++cell){
            phitildepenalty[cell]+=1;
        }
    }
}
void ablate::finiteVolume::processes::IntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
    IntSharp::subDomain = solver.GetSubDomainPtr();
}
ablate::finiteVolume::processes::IntSharp::IntSharp(PetscReal Gamma, PetscReal epsilon, bool flipPhiTilde, bool addtoRHS) : Gamma(Gamma), epsilon(epsilon), flipPhiTilde(flipPhiTilde), addtoRHS(addtoRHS) {}
ablate::finiteVolume::processes::IntSharp::~IntSharp() { DMDestroy(&vertexDM) >> utilities::PetscUtilities::checkError; }

void intSharpPreStageWrapper(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime, ablate::finiteVolume::processes::IntSharp* intSharpProcess) {
  intSharpProcess->PreStage(flowTs, solver, stagetime);
}

PetscErrorCode ablate::finiteVolume::processes::IntSharp::PreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime) {
  PetscFunctionBegin;

  // PetscLogDouble startTime, endTime, // elapsedTime;

  // // // PetscPrintf(PETSC_COMM_WORLD, "PreStage function called at time %g\n", stagetime);
    
  const auto &fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver &>(solver);
  ablate::domain::Range cellRange; 
  fvSolver.GetCellRangeWithoutGhost(cellRange);
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

  // DM auxDM = solver.GetSubDomain().GetAuxDM();
  // Vec auxVec = solver.GetSubDomain().GetAuxVector(); PetscScalar *auxArray = nullptr; PetscCall(VecGetArray(auxVec, &auxArray));

  //insert computeterm logic start

  //init fields
  
  Vec locX = solver.GetSubDomain().GetSolutionVector(); //had to add this whereas previously locX is given in computeterm as input
  ablate::finiteVolume::processes::IntSharp *process = this;

  // // // PetscPrintf(PETSC_COMM_WORLD, "process function called at time %g\n", stagetime);

  std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
  subDomain->UpdateAuxLocalVector();
  // auto dim = solver.GetSubDomain().GetDimensions();
PetscReal xymin[dim], xymax[dim]; DMGetBoundingBox(dm, xymin, xymax);
PetscReal xmin=xymin[0];
PetscReal xmax=xymax[0];
PetscReal ymin=xymin[1];
PetscReal ymax=xymax[1];
PetscReal zmin=xymin[2];
PetscReal zmax=xymax[2];
  const auto &phiField = subDomain->GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);


  const auto &eulerField = solver.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);
  const auto &densityVFField = subDomain->GetField("densityvolumeFraction");
  const auto &ofield = subDomain->GetField("debug");
  const auto &ofield2 = subDomain->GetField("debug2");

  const auto &gasDensityField = subDomain->GetField("gasDensity");
  const auto &liquidDensityField = subDomain->GetField("liquidDensity");
  const auto &mixtureEnergyField = subDomain->GetField("mixtureEnergy");

  auto eulerfID = eulerField.id;
  DM auxDM = subDomain->GetAuxDM();
  Vec auxVec = subDomain->GetAuxVector(); //LOCAL aux vector, not global

  Vec vertexVec; DMGetLocalVector(process->vertexDM, &vertexVec);
  const PetscScalar *solArray; VecGetArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError;
  PetscScalar *auxArray; VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
  PetscScalar *vertexArray; VecGetArray(vertexVec, &vertexArray);
  PetscScalar *fArray; PetscCall(VecGetArray(locFVec, &fArray));
  // ablate::domain::Range cellRange; solver.GetCellRangeWithoutGhost(cellRange);
  PetscInt vStart, vEnd; DMPlexGetDepthStratum(process->vertexDM, 0, &vStart, &vEnd);
  PetscInt cStart, cEnd; DMPlexGetHeightStratum(auxDM, 0, &cStart, &cEnd);

DM sharedVertexDM_1; IS_CopyDM(process->vertexDM, vStart, vEnd, 1, &sharedVertexDM_1);
DM sharedVertexDM_dim; IS_CopyDM(process->vertexDM, vStart, vEnd, dim, &sharedVertexDM_dim);

Vec vxLocalVec, vxGlobalVec, vyLocalVec, vyGlobalVec, vzLocalVec, vzGlobalVec, aLocalVec, aGlobalVec;
PetscScalar *vxLocalArray, *vyLocalArray, *vzLocalArray, *aLocalArray;

DM sharedDM; IS_CopyDM(auxDM, cStart, cEnd, 1, &sharedDM);
Vec divaLocalVec, divaGlobalVec, ismaskLocalVec, ismaskGlobalVec, phitildemaskLocalVec, phitildemaskGlobalVec;
Vec rankLocalVec, rankGlobalVec, phiLocalVec, phiGlobalVec, phitildeLocalVec, phitildeGlobalVec, cellidLocalVec, cellidGlobalVec;
PetscScalar *divaLocalArray, *ismaskLocalArray, *phitildemaskLocalArray;
PetscScalar *rankLocalArray, *phiLocalArray, *phitildeLocalArray, *cellidLocalArray;
Vec xLocalVec, xGlobalVec, yLocalVec, yGlobalVec, zLocalVec, zGlobalVec;
PetscScalar *xLocalArray, *yLocalArray, *zLocalArray;
#define CREATE_VEC_AND_ARRAY(dm, vecLocal, vecGlobal, array) \
  DMCreateLocalVector(dm, &vecLocal); \
  DMCreateGlobalVector(dm, &vecGlobal); \
  VecZeroEntries(vecLocal); \
  VecZeroEntries(vecGlobal); \
  VecGetArray(vecLocal, &array);
CREATE_VEC_AND_ARRAY(sharedVertexDM_1, vxLocalVec, vxGlobalVec, vxLocalArray);
CREATE_VEC_AND_ARRAY(sharedVertexDM_1, vyLocalVec, vyGlobalVec, vyLocalArray);
CREATE_VEC_AND_ARRAY(sharedVertexDM_1, vzLocalVec, vzGlobalVec, vzLocalArray);
CREATE_VEC_AND_ARRAY(sharedVertexDM_dim, aLocalVec, aGlobalVec, aLocalArray);
CREATE_VEC_AND_ARRAY(sharedDM, divaLocalVec, divaGlobalVec, divaLocalArray);
CREATE_VEC_AND_ARRAY(sharedDM, ismaskLocalVec, ismaskGlobalVec, ismaskLocalArray);
CREATE_VEC_AND_ARRAY(sharedDM, phitildemaskLocalVec, phitildemaskGlobalVec, phitildemaskLocalArray);
CREATE_VEC_AND_ARRAY(sharedDM, rankLocalVec, rankGlobalVec, rankLocalArray);
CREATE_VEC_AND_ARRAY(sharedDM, phiLocalVec, phiGlobalVec, phiLocalArray);
CREATE_VEC_AND_ARRAY(sharedDM, phitildeLocalVec, phitildeGlobalVec, phitildeLocalArray);
CREATE_VEC_AND_ARRAY(sharedDM, cellidLocalVec, cellidGlobalVec, cellidLocalArray);
CREATE_VEC_AND_ARRAY(sharedDM, xLocalVec, xGlobalVec, xLocalArray);
CREATE_VEC_AND_ARRAY(sharedDM, yLocalVec, yGlobalVec, yLocalArray);
CREATE_VEC_AND_ARRAY(sharedDM, zLocalVec, zGlobalVec, zLocalArray);
  int rank; MPI_Comm_rank(PETSC_COMM_WORLD, &rank); rank+=1;

  // PetscTime(&startTime);

  //clean up fields
  for (PetscInt cell = cStart; cell < cEnd; ++cell){
          PetscSection globalSection; DMGetGlobalSection(dm, &globalSection);
          PetscInt owned = 1; PetscSectionGetOffset(globalSection, cell, &owned);
          PetscScalar *divaptr; xDMPlexPointLocalRef(sharedDM, cell, -1, divaLocalArray, &divaptr); *divaptr = 0;
          PetscScalar *ismaskptr; xDMPlexPointLocalRef(sharedDM, cell, -1, ismaskLocalArray, &ismaskptr); *ismaskptr = 0;
          PetscScalar *rankptr; xDMPlexPointLocalRef(sharedDM, cell, -1, rankLocalArray, &rankptr); *rankptr = 0;
          PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(sharedDM, cell, -1, phitildemaskLocalArray, &phitildemaskptr); *phitildemaskptr = 0;
          PetscScalar *phitildeptr; xDMPlexPointLocalRef(sharedDM, cell, -1, phitildeLocalArray, &phitildeptr); *phitildeptr = 0;
          PetscScalar *cellidptr; xDMPlexPointLocalRef(sharedDM, cell, -1, cellidLocalArray, &cellidptr); *cellidptr = cell;
          const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
          PetscScalar *phiptr; xDMPlexPointLocalRef(sharedDM, cell, -1, phiLocalArray, &phiptr); if (owned>=0){ *phiptr = *phic; }
          PetscReal xp, yp, zp; GetCoordinate3D(dm, dim, cell, &xp, &yp, &zp);
          PetscScalar *xptr; xDMPlexPointLocalRef(sharedDM, cell, -1, xLocalArray, &xptr);
          PetscScalar *yptr; xDMPlexPointLocalRef(sharedDM, cell, -1, yLocalArray, &yptr);
          PetscScalar *zptr; xDMPlexPointLocalRef(sharedDM, cell, -1, zLocalArray, &zptr);
          *xptr = xp; *yptr = yp; *zptr = zp;
  }
  for (PetscInt cell = cStart; cell < cEnd; ++cell){
      PetscSection globalSection; DMGetGlobalSection(dm, &globalSection);
      PetscInt owned = 1; PetscSectionGetOffset(globalSection, cell, &owned);
      if (owned>=0){ PetscScalar *rankcptr; xDMPlexPointLocalRef(sharedDM, cell, -1, rankLocalArray, &rankcptr); *rankcptr = rank; }
  }
  
  // PetscTime(&endTime);
  // elapsedTime = endTime - startTime;
  // // PetscPrintf(PETSC_COMM_WORLD, "CLEANUP time: %g seconds\n", // elapsedTime);
  // PetscTime(&startTime);

  //init mask field
  for (PetscInt cell = cStart; cell < cEnd; ++cell){
      const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
      if (*phic > 1e-4 and *phic < 1-1e-4) {

              // neighboredit 1
              
              const auto &neighbors = cellNeighbors[cell];
              for (const auto &neighbor : neighbors) {
              // DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
              // for (PetscInt j = 0; j < nNeighbors; ++j) {
                  // PetscInt neighbor = neighbors[j];
                  PetscScalar *ranknptr; xDMPlexPointLocalRef(sharedDM, neighbor, -1, rankLocalArray, &ranknptr);
                  PetscScalar *ismaskptr; xDMPlexPointLocalRef(sharedDM, neighbor, -1, ismaskLocalArray, &ismaskptr);
                  *ismaskptr = *ranknptr;
              }
              // DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
      }
  }
  PushGhost(sharedDM, ismaskLocalVec, ismaskGlobalVec, ADD_VALUES, false, false);

  // PetscTime(&endTime);
  // elapsedTime = endTime - startTime;
  // PetscPrintf(PETSC_COMM_WORLD, "INIT MASK FIELD time: %g seconds\n", // elapsedTime);
  // PetscTime(&startTime);

// phitilde mask
  for (PetscInt cell = cStart; cell < cEnd; ++cell){
      const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
      if (*phic > 1e-4 and *phic < 1-1e-4) { PetscScalar *ismaskptr; xDMPlexPointLocalRef(sharedDM, cell, -1, ismaskLocalArray, &ismaskptr); *ismaskptr = 5; }
  }  

  PetscScalar C=1; PetscScalar N=2.6; PetscScalar layers = ceil(C*N);
  PetscReal rmin; DMPlexGetMinRadius(dm, &rmin); PetscReal h=2*rmin + 0*layers;

  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
      PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(sharedDM, cell, -1, phitildemaskLocalArray, &phitildemaskptr); *phitildemaskptr = 0;
  }
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
      const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic) >> ablate::utilities::PetscUtilities::checkError;
      if (*phic > 0.0001 and *phic < 0.9999) {

// neighboredit 2

const auto &neighbors = cellNeighbors[cell];
for (const auto &neighbor : neighbors) {

              // PetscInt nNeighbors, *neighbors; DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors); //
              // for (PetscInt j = 0; j < nNeighbors; ++j) {
              //     PetscInt neighbor = neighbors[j];
                  PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(sharedDM, neighbor, -1, phitildemaskLocalArray, &phitildemaskptr); *phitildemaskptr = 1;
PetscReal xc, yc, zc; GetCoordinate3D(dm, dim, cell, &xc, &yc, &zc);
PetscReal xn, yn, zn; GetCoordinate3D(dm, dim, neighbor, &xn, &yn, &zn);
              }
              // DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
      }
  }
  PushGhost(sharedDM, phitildemaskLocalVec, phitildemaskGlobalVec, ADD_VALUES, false, false);

  // PetscTime(&endTime);
  // elapsedTime = endTime - startTime;
  // PetscPrintf(PETSC_COMM_WORLD, "PHITILDE MASK FIELD time: %g seconds\n", // elapsedTime);
  // PetscTime(&startTime);

  //phitilde
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
      const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
      PetscReal xc, yc, zc; GetCoordinate3D(dm, dim, cell, &xc, &yc, &zc);
      PetscScalar *phitilde; xDMPlexPointLocalRef(sharedDM, cell, -1, phitildeLocalArray, &phitilde);
      PetscScalar *phitildemask; xDMPlexPointLocalRef(sharedDM, cell, -1, phitildemaskLocalArray, &phitildemask);
      if (*phitildemask < 1e-10){ *phitilde = *phic;
if (process->flipPhiTilde){*phitilde = 1.00- *phitilde;} }
      else{

//neighboredit 3

          // PetscInt nNeighbors, *neighbors; DMPlexGetNeighbors(auxDM, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
          PetscReal weightedphi = 0; PetscReal Tw = 0;

          const auto &neighbors = cellNeighbors[cell];
          for (const auto &neighbor : neighbors) {

          // for (PetscInt j = 0; j < nNeighbors; ++j) {
              // PetscInt neighbor = neighbors[j];
              PetscReal *phin; xDMPlexPointLocalRead(dm, neighbor, phiField.id, solArray, &phin);
              PetscReal xn, yn, zn; GetCoordinate3D(dm, dim, neighbor, &xn, &yn, &zn);
bool periodicfix = true;
if (periodicfix){
PetscReal maxMask = 10*process->epsilon;
if (( PetscAbs(xn-xc) > maxMask) and (xn > xc)){  xn -= (xmax-xmin);  }
if (( PetscAbs(xn-xc) > maxMask) and (xn < xc)){  xn += (xmax-xmin);  }
if (dim>=2){
if (( PetscAbs(yn-yc) > maxMask) and (yn > yc)){  yn -= (ymax-ymin);  }
if (( PetscAbs(yn-yc) > maxMask) and (yn < yc)){  yn += (ymax-ymin);  } }
if (dim==3){
if (( PetscAbs(zn-zc) > maxMask) and (zn > zc)){  zn -= (zmax-zmin);  }
if (( PetscAbs(zn-zc) > maxMask) and (zn < zc)){  zn += (zmax-zmin);  } }
}
              PetscReal d = PetscSqrtReal(PetscSqr(xn - xc) + PetscSqr(yn - yc) + PetscSqr(zn - zc));  // distance
              PetscReal s = C * h;
              PetscReal wn; PhiNeighborGauss(d, s, &wn);
              Tw += wn;
              weightedphi += (*phin * wn);
PetscScalar *rankptr; xDMPlexPointLocalRef(sharedDM, cell, -1, rankLocalArray, &rankptr);
if ((cell==0) and (*rankptr == 5)){ std::cout << "";}
          }
          weightedphi /= Tw;
if (process->flipPhiTilde){weightedphi = 1.000-weightedphi;}
          *phitilde = weightedphi;
          // DMPlexRestoreNeighbors(auxDM, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
      }
  }
  PushGhost(sharedDM, phitildeLocalVec, phitildeGlobalVec, INSERT_VALUES, true, true);

  // PetscTime(&endTime);
  // elapsedTime = endTime - startTime;
  // PetscPrintf(PETSC_COMM_WORLD, "PHITILDE time: %g seconds\n", // elapsedTime);
  // PetscTime(&startTime);

for (PetscInt cell = cStart; cell < cEnd; ++cell) {
PetscScalar *optr2; PetscScalar *phitildeptr;
xDMPlexPointLocalRef(sharedDM, cell, -1, phitildeLocalArray, &phitildeptr);
xDMPlexPointLocalRef(auxDM, cell, ofield2.id, auxArray, &optr2);
*optr2 = *phitildeptr;
PetscScalar *rankptr; xDMPlexPointLocalRef(sharedDM, cell, -1, rankLocalArray, &rankptr);
if ((cell==0) and (*rankptr == 5)){  std::cout << "";   }

}

//init vec a (vertices)
  for (PetscInt vertex = vStart; vertex < vEnd; vertex++) {
      PetscReal vx, vy, vz; GetCoordinate3D(dm, dim, vertex, &vx, &vy, &vz);
      PetscScalar *vxptr; xDMPlexPointLocalRef(sharedVertexDM_1, vertex, -1, vxLocalArray, &vxptr); *vxptr = vx;
      PetscScalar *vyptr; xDMPlexPointLocalRef(sharedVertexDM_1, vertex, -1, vyLocalArray, &vyptr); *vyptr = vy;
      PetscScalar *vzptr; xDMPlexPointLocalRef(sharedVertexDM_1, vertex, -1, vzLocalArray, &vzptr); *vzptr = vz;
      PetscScalar *aptr; xDMPlexPointLocalRef(sharedVertexDM_dim, vertex, -1, aLocalArray, &aptr); *aptr = 0;
  }

  // PetscTime(&endTime);
  // elapsedTime = endTime - startTime;
  // PetscPrintf(PETSC_COMM_WORLD, "INIT VEC A time: %g seconds\n", // elapsedTime);
  // PetscTime(&startTime);

  //build vec a
  for (PetscInt vertex = vStart; vertex < vEnd; vertex++) {
      PetscReal vx, vy, vz; GetCoordinate3D(dm, dim, vertex, &vx, &vy, &vz);
      PetscInt nvn, *vertexneighbors; DMPlexVertexGetCells(dm, vertex, &nvn, &vertexneighbors);
      PetscBool isAdjToMask = PETSC_FALSE;
      for (PetscInt k = 0; k < nvn; k++){
          PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(sharedDM, vertexneighbors[k], -1, phitildemaskLocalArray, &phitildemaskptr);
          if (*phitildemaskptr > 0.5){
              isAdjToMask = PETSC_TRUE;
          }
      }
      PetscScalar gradphiv[dim];
      PetscReal normgradphi = 0.0;
      if (isAdjToMask == PETSC_TRUE){
          if (dim==1){
              PetscScalar *phitildekm1; xDMPlexPointLocalRef(sharedDM, vertexneighbors[0], -1, phitildeLocalArray, &phitildekm1);
              PetscScalar *phitildekp1; xDMPlexPointLocalRef(sharedDM, vertexneighbors[1], -1, phitildeLocalArray, &phitildekp1);
              PetscReal xm1; GetCoordinate1D(dm, dim, vertexneighbors[0], &xm1);
              PetscReal xp1; GetCoordinate1D(dm, dim, vertexneighbors[1], &xp1);
              gradphiv[0]=(*phitildekp1 - *phitildekm1)/(xp1 - xm1);
          }
          else{
DMPlexVertexGradFromCell(sharedDM, vertex, phitildeLocalVec, -1, 0, gradphiv);
}
          for (int k=0; k<dim; ++k){ normgradphi += PetscSqr(gradphiv[k]); }
          normgradphi = PetscSqrtReal(normgradphi);
      }
      else{ for (int k=0; k<dim; ++k){ gradphiv[k] =0; } }

      PetscReal phiv=0;
      if(isAdjToMask == PETSC_TRUE) {
          PetscReal distances[nvn];
          PetscReal shortestdistance = ablate::utilities::Constants::large;
          for (PetscInt k = 0; k < nvn; ++k) {
              PetscInt neighbor = vertexneighbors[k];
              PetscReal nx, ny, nz; GetCoordinate3D(dm, dim, neighbor, &nx, &ny, &nz);


bool periodicfix = true;
if (periodicfix){
PetscReal maxMask = 10*process->epsilon;
if (( PetscAbs(nx-vx) > maxMask) and (nx > vx)){  nx -= (xmax-xmin);  }
if (( PetscAbs(nx-vx) > maxMask) and (nx < vx)){  nx += (xmax-xmin);  }
if (dim>=2){
if (( PetscAbs(ny-vy) > maxMask) and (ny > vy)){  ny -= (ymax-ymin);  }
if (( PetscAbs(ny-vy) > maxMask) and (ny < vy)){  ny += (ymax-ymin);  } }
if (dim==3){
if (( PetscAbs(nz-vz) > maxMask) and (nz > vz)){  nz -= (zmax-zmin);  }
if (( PetscAbs(nz-vz) > maxMask) and (nz < vz)){  nz += (zmax-zmin);  } }
}

              PetscReal distance = PetscSqrtReal(PetscSqr(nx - vx) + PetscSqr(ny - vy) + PetscSqr(nz - vz));
              if (distance < shortestdistance) { shortestdistance = distance; }
              distances[k] = distance;
          }
          PetscReal weights_wrt_short[nvn];
          PetscReal totalweight_wrt_short = 0;
          for (PetscInt k = 0; k < nvn; ++k) {
              PetscReal weight_wrt_short = shortestdistance / distances[k];
              weights_wrt_short[k] = weight_wrt_short;
              totalweight_wrt_short += weight_wrt_short;
          }
          PetscReal weights[nvn];
          for (PetscInt k = 0; k < nvn; ++k) { weights[k] = weights_wrt_short[k] / totalweight_wrt_short; }
          for (PetscInt k = 0; k < nvn; ++k) {
              PetscInt neighbor = vertexneighbors[k];
              PetscReal *phineighbor; xDMPlexPointLocalRef(sharedDM, neighbor, -1, phitildeLocalArray, &phineighbor);
              phiv += (*phineighbor) * (weights[k]);  // unstructured case
          }
      }
      else{ phiv=0; }

      PetscScalar  av[dim];
      PetscReal *avptr; xDMPlexPointLocalRef(sharedVertexDM_dim, vertex, -1, aLocalArray, &avptr); //vertexDM

      for (int k=0; k<dim; ++k){
          if(isAdjToMask == PETSC_TRUE) {
              if (normgradphi > ablate::utilities::Constants::tiny) { av[k] = (process->Gamma * process->epsilon * gradphiv[k]) - (process->Gamma * phiv * (1 - phiv) * (gradphiv[k] / normgradphi)); }
              else { av[k] = (process->Gamma * process->epsilon * gradphiv[k]) - (process->Gamma * phiv * (1 - phiv) * gradphiv[k]); }
              avptr[k] = av[k];
          }
          else{ avptr[k]=0; }
      }
      DMPlexVertexRestoreCells(dm, vertex, &nvn, &vertexneighbors);
  }
  PushGhost(sharedVertexDM_dim, aLocalVec, aGlobalVec, INSERT_VALUES, true, false);

  // PetscTime(&endTime);
  // elapsedTime = endTime - startTime;
  // PetscPrintf(PETSC_COMM_WORLD, "BUILD VEC A time: %g seconds\n", // elapsedTime);
  // PetscTime(&startTime);

  //div a 
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
      PetscScalar *diva; xDMPlexPointLocalRef(sharedDM, cell, -1, divaLocalArray, &diva); *diva=0.0;
      PetscScalar *ismask; xDMPlexPointLocalRef(sharedDM, cell, -1, ismaskLocalArray, &ismask);
      if (*ismask > 0.5){
          if (dim==1){
              PetscInt nVerts, *verts; DMPlexCellGetVertices(dm, cell, &nVerts, &verts);
              PetscScalar *am1; xDMPlexPointLocalRef(sharedVertexDM_dim, verts[0], -1, aLocalArray, &am1);
              PetscScalar *ap1; xDMPlexPointLocalRef(sharedVertexDM_dim, verts[1], -1, aLocalArray, &ap1);
              PetscReal xm1; GetCoordinate1D(dm, dim, verts[0], &xm1);
              PetscReal xp1; GetCoordinate1D(dm, dim, verts[1], &xp1);
              *diva = (*ap1-*am1)/(xp1-xm1);
              DMPlexCellRestoreVertices(dm, cell, &nVerts, &verts);
          }
          else{
              for (PetscInt offset = 0; offset < dim; offset++) {
                  PetscReal nabla_ai[dim];
                  DMPlexCellGradFromVertex(sharedVertexDM_dim, cell, aLocalVec, -1, offset, nabla_ai);
                  *diva += nabla_ai[offset];
              }
          }
      }
      else{ *diva = 0.0; }
  }
  PushGhost(sharedDM, divaLocalVec, divaGlobalVec, INSERT_VALUES, true, false);

  // PetscTime(&endTime);
  // elapsedTime = endTime - startTime;
  // PetscPrintf(PETSC_COMM_WORLD, "DIV A time: %g seconds\n", // elapsedTime);

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      PetscInt cell = cellRange.GetPoint(c);
      const PetscScalar *euler = nullptr; xDMPlexPointLocalRead(dm, cell, eulerfID, solArray, &euler);
      const PetscReal *phik; xDMPlexPointLocalRead(sharedDM, cell, -1, phitildeLocalArray, &phik);
      PetscScalar *eulerSource; xDMPlexPointLocalRef(dm, cell, eulerfID, fArray, &eulerSource);
      PetscScalar *rhophiSource; xDMPlexPointLocalRef(dm, cell, densityVFField.id, fArray, &rhophiSource);
      PetscScalar rhog; const PetscScalar *rhogphig; xDMPlexPointLocalRead(dm, cell, densityVFField.id, solArray, &rhogphig);
      if(*rhogphig > 1e-10){rhog = *rhogphig / *phik;}else{rhog = 0;}
      PetscScalar *diva; xDMPlexPointLocalRef(sharedDM, cell, -1, divaLocalArray, &diva);

      // *diva *= -1; //?

      PetscScalar *optr; xDMPlexPointLocalRef(auxDM, cell, ofield.id, auxArray, &optr);
      *optr = *diva;

if (process->addtoRHS){*rhophiSource += rhog* *diva;}

  }
  subDomain->UpdateAuxLocalVector();


  //insert computeterm logic end

  // // PetscPrintf(PETSC_COMM_WORLD, "diva computed at time %g\n", stagetime);

  // PetscTime(&startTime);

  for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
    const PetscInt cell = cellRange.GetPoint(i);
    PetscScalar *allFields = nullptr; DMPlexPointLocalRef(dm, cell, flowArray, &allFields) >> utilities::PetscUtilities::checkError;
    auto density = allFields[ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal velocity[3]; for (PetscInt d = 0; d < dim; d++) { velocity[d] = allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density; }

    // decode state (we just need densityG and densityL to reconstruct mixture RHO out of new alpha, and internalEnergy to reconstruct RHOE )
    // PetscReal densityG = 1.0, densityL = 1000.0, internalEnergy = 1e5; 
    // PetscReal density, densityG, densityL, internalEnergy, normalVelocity, internalEnergyG, internalEnergyL, aG, aL, MG, ML, p, t, alpha;
    // intSharpProcess->decoder->DecodeTwoPhaseEulerState(dim, uOff, allFields, norm, &density, &densityG, &densityL, &normalVelocity, velocity, &internalEnergy, &internalEnergyG, &internalEnergyL, &aG, &aL, &MG, &ML, &p, &t, &alpha);



// // PetscPrintf(PETSC_COMM_WORLD, "This code has been called %d times\n", callCount);

    PetscReal *densityG, *densityL, *internalEnergy, *diva;
    xDMPlexPointLocalRead(auxDM, cell, gasDensityField.id, auxArray, &densityG) >> utilities::PetscUtilities::checkError;
    xDMPlexPointLocalRead(auxDM, cell, liquidDensityField.id, auxArray, &densityL) >> utilities::PetscUtilities::checkError;
    xDMPlexPointLocalRead(auxDM, cell, mixtureEnergyField.id, auxArray, &internalEnergy) >> utilities::PetscUtilities::checkError;
    xDMPlexPointLocalRef(sharedDM, cell, -1, divaLocalArray, &diva);
    const PetscScalar oldAlpha = allFields[vfOffset];

    

    // update corresponding euler field values based on new alpha
    if (oldAlpha > 1e-3 && oldAlpha < 1-1e-3){

      //inside or outside of oldalpha if statement??
      PetscReal pseudoTime = 1e-4 + 0*oldAlpha;
    allFields[vfOffset] += pseudoTime * *diva;
    if (allFields[vfOffset] < 0.0) { allFields[vfOffset] = 0.0; } 
    else if (allFields[vfOffset] > 1.0) { allFields[vfOffset] = 1.0; }

      allFields[rhoAlphaOffset] = (allFields[vfOffset] / oldAlpha) * allFields[rhoAlphaOffset];

      // densityL = ( allFields[ablate::finiteVolume::CompressibleFlowFields::RHO] - *densityG * oldAlpha )/( 1 - oldAlpha );

    allFields[ablate::finiteVolume::CompressibleFlowFields::RHO] = allFields[vfOffset] * *densityG + (1 - allFields[vfOffset]) * *densityL;
    allFields[ablate::finiteVolume::CompressibleFlowFields::RHOE] = allFields[ablate::finiteVolume::CompressibleFlowFields::RHO] * *internalEnergy;
    for (PetscInt d = 0; d < dim; ++d) {
        allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = allFields[ablate::finiteVolume::CompressibleFlowFields::RHO] * velocity[d];
    }

  }

}

// PetscTime(&endTime);
// elapsedTime = endTime - startTime;
// PetscPrintf(PETSC_COMM_WORLD, "PSEUDO TIMESTEP time: %g seconds\n", // elapsedTime);
// PetscTime(&startTime);

// // PetscPrintf(PETSC_COMM_WORLD, "prestage completed at time %g\n", stagetime);


// Restore
// PetscCall(DMRestoreLocalVector(dm, &locFVec)); 
PetscCall(VecRestoreArray(globFlowVec, &flowArray)); 
// fvSolver.RestoreRange(cellRange); 



  //destroy vecs

  #define RESTORE_VEC_AND_ARRAY(vecLocal, vecGlobal, array) \
  VecRestoreArray(vecLocal, &array); \
  VecDestroy(&vecLocal); \
  VecDestroy(&vecGlobal);
RESTORE_VEC_AND_ARRAY(divaLocalVec, divaGlobalVec, divaLocalArray);
RESTORE_VEC_AND_ARRAY(ismaskLocalVec, ismaskGlobalVec, ismaskLocalArray);
RESTORE_VEC_AND_ARRAY(phitildeLocalVec, phitildeGlobalVec, phitildeLocalArray);
RESTORE_VEC_AND_ARRAY(phitildemaskLocalVec, phitildemaskGlobalVec, phitildemaskLocalArray);
RESTORE_VEC_AND_ARRAY(rankLocalVec, rankGlobalVec, rankLocalArray);
RESTORE_VEC_AND_ARRAY(phiLocalVec, phiGlobalVec, phiLocalArray);
RESTORE_VEC_AND_ARRAY(cellidLocalVec, cellidGlobalVec, cellidLocalArray);
RESTORE_VEC_AND_ARRAY(xLocalVec, xGlobalVec, xLocalArray);
RESTORE_VEC_AND_ARRAY(yLocalVec, yGlobalVec, yLocalArray);
RESTORE_VEC_AND_ARRAY(zLocalVec, zGlobalVec, zLocalArray);
DMDestroy(&sharedDM);

RESTORE_VEC_AND_ARRAY(vxLocalVec, vxGlobalVec, vxLocalArray);
RESTORE_VEC_AND_ARRAY(vyLocalVec, vyGlobalVec, vyLocalArray);
RESTORE_VEC_AND_ARRAY(vzLocalVec, vzGlobalVec, vzLocalArray);
RESTORE_VEC_AND_ARRAY(aLocalVec, aGlobalVec, aLocalArray);
DMDestroy(&sharedVertexDM_1);
DMDestroy(&sharedVertexDM_dim);

  // cleanup
  VecRestoreArrayRead(locX, &solArray);
  VecRestoreArray(auxVec, &auxArray);
  VecRestoreArray(vertexVec, &vertexArray);
  VecRestoreArray(locFVec, &fArray);
  solver.RestoreRange(cellRange);

  DMRestoreLocalVector(process->vertexDM, &vertexVec);
  VecDestroy(&vertexVec); //<--- this is fine

  // // PetscPrintf(PETSC_COMM_WORLD, "destroy vecs at time %g\n", stagetime);



PetscFunctionReturn(0);


}

void ablate::finiteVolume::processes::IntSharp::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    auto dim = flow.GetSubDomain().GetDimensions();
    auto dm = flow.GetSubDomain().GetDM();
    PetscFE fe_coords;
    PetscInt k = 1;
    DMClone(dm, &vertexDM) >> utilities::PetscUtilities::checkError;
    PetscFECreateLagrange(PETSC_COMM_SELF, dim, dim, PETSC_TRUE, k, PETSC_DETERMINE, &fe_coords) >> utilities::PetscUtilities::checkError;
    DMSetField(vertexDM, 0, nullptr, (PetscObject)fe_coords) >> utilities::PetscUtilities::checkError;
    PetscFEDestroy(&fe_coords) >> utilities::PetscUtilities::checkError;
    DMCreateDS(vertexDM) >> utilities::PetscUtilities::checkError;


    // // PetscPrintf(PETSC_COMM_WORLD, "preparing to compute global cell neighbors\n");

    // global cell neighbors
    ablate::domain::Range cellRange; 
    // // PetscPrintf(PETSC_COMM_WORLD, "defined cellrange\n");
    auto fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver*>(&flow);

    if (!fvSolver) {
      // // PetscPrintf(PETSC_COMM_WORLD, "Dynamic cast failed\n");
      return;
    }

    // // PetscPrintf(PETSC_COMM_WORLD, "got dynamic cast \n");
    
    
    // fvSolver->GetCellRangeWithoutGhost(cellRange);
    PetscInt cStart, cEnd; DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
    cellRange.start = cStart; cellRange.end = cEnd;

    // // PetscPrintf(PETSC_COMM_WORLD, "got cellrange: start=%d, end=%d\n", cellRange.start, cellRange.end);
    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
        PetscInt cell = cellRange.GetPoint(i);
        // // PetscPrintf(PETSC_COMM_WORLD, "got cell \n");
        PetscInt nNeighbors, *neighbors;
        PetscReal layers=3;
        // DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);

        DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        // // PetscPrintf(PETSC_COMM_WORLD, "called neighbors \n");
        cellNeighbors[cell] = std::vector<PetscInt>(neighbors, neighbors + nNeighbors);
        // // PetscPrintf(PETSC_COMM_WORLD, "populate cellneighbors array \n");
        DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
    }

    // // PetscPrintf(PETSC_COMM_WORLD, "global cell neighbors computed\n");


    // global vertex neighbors
    PetscInt vStart, vEnd;
    DMPlexGetDepthStratum(vertexDM, 0, &vStart, &vEnd);
    for (PetscInt vertex = vStart; vertex < vEnd; ++vertex) {
        PetscInt nvn, *vertexneighbors;
        DMPlexVertexGetCells(dm, vertex, &nvn, &vertexneighbors);
        vertexNeighbors[vertex] = std::vector<PetscInt>(vertexneighbors, vertexneighbors + nvn);
        DMPlexVertexRestoreCells(dm, vertex, &nvn, &vertexneighbors);
    }

    //prestage
    auto intSharpPreStage = std::bind(intSharpPreStageWrapper, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, this);
    flow.RegisterPreStage(intSharpPreStage);

    //computeterm
    // flow.RegisterRHSFunction(ComputeTerm, this);
}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::IntSharp, "calculates interface regularization term",
         ARG(PetscReal, "Gamma", "Gamma, velocity scale parameter (approx. umax)"),
         ARG(PetscReal, "epsilon", "epsilon, interface thickness scale parameter (approx. h)"),
         ARG(bool, "flipPhiTilde", "if true: phiTilde-->1-phiTilde (set it to true if primary phase is phi=0 or false if phi=1)"),
         ARG(bool, "addtoRHS", "if true: add to the RHS of the equation (default: false)")
);
