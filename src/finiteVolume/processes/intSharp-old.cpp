#include "domain/RBF/mq.hpp"
#include "intSharp.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"
#include <fstream>

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

ablate::finiteVolume::processes::IntSharp::IntSharp(PetscReal Gamma, PetscReal epsilon, bool flipPhiTilde) : Gamma(Gamma), epsilon(epsilon), flipPhiTilde(flipPhiTilde) {}
ablate::finiteVolume::processes::IntSharp::~IntSharp() { DMDestroy(&vertexDM) >> utilities::PetscUtilities::checkError; }

void ablate::finiteVolume::processes::IntSharp::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    auto dim = flow.GetSubDomain().GetDimensions();
    auto dm = flow.GetSubDomain().GetDM();
    // create a domain, vertexDM, to use it in source function for storing any calculated vertex normal. Here the vertex normals will be stored on vertices, therefore k = 1
    PetscFE fe_coords;
    PetscInt k = 1;
    DMClone(dm, &vertexDM) >> utilities::PetscUtilities::checkError;
    PetscFECreateLagrange(PETSC_COMM_SELF, dim, dim, PETSC_TRUE, k, PETSC_DETERMINE, &fe_coords) >> utilities::PetscUtilities::checkError;
    DMSetField(vertexDM, 0, nullptr, (PetscObject)fe_coords) >> utilities::PetscUtilities::checkError;
    PetscFEDestroy(&fe_coords) >> utilities::PetscUtilities::checkError;
    DMCreateDS(vertexDM) >> utilities::PetscUtilities::checkError;
    flow.RegisterRHSFunction(ComputeTerm, this);
}

PetscErrorCode ablate::finiteVolume::processes::IntSharp::ComputeTerm(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx) {
    PetscFunctionBegin;
    auto *process = (ablate::finiteVolume::processes::IntSharp *)ctx;
    std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
    subDomain->UpdateAuxLocalVector();
    //dm = sol DM
    //locX = solvec
    //locFVec = vector of conserved vars / eulerSource fields (rho, rhoe, rhov, ..., rhoet)
    //auxvec = auxvec
    //auxArray = auxArray
    //notions of "process->" refer to the private variables: vertexDM, Gamma, epsilon. (the public variables are a subset: Gamma and epsilon)
    //process->vertexDM = aux DM
    //get fields
    auto dim = solver.GetSubDomain().GetDimensions();

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

    auto eulerfID = eulerField.id;

    // get vecs/arrays
    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector(); //LOCAL aux vector, not global

    Vec vertexVec; DMGetLocalVector(process->vertexDM, &vertexVec);
    const PetscScalar *solArray; VecGetArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError;
    PetscScalar *auxArray; VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
    PetscScalar *vertexArray; VecGetArray(vertexVec, &vertexArray);
    PetscScalar *fArray; PetscCall(VecGetArray(locFVec, &fArray));

    // get ranges
    ablate::domain::Range cellRange; solver.GetCellRangeWithoutGhost(cellRange);
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

    //field ID for non field calls is -1.
    int rank; MPI_Comm_rank(PETSC_COMM_WORLD, &rank); rank+=1;
    //clean up fields
    for (PetscInt cell = cStart; cell < cEnd; ++cell){
            PetscSection globalSection; DMGetGlobalSection(dm, &globalSection);
            PetscInt owned = 1; PetscSectionGetOffset(globalSection, cell, &owned);
//            PetscScalar *divaptr; xDMPlexPointLocalRef(divaDM, cell, -1, divaLocalArray, &divaptr);
            PetscScalar *divaptr; xDMPlexPointLocalRef(sharedDM, cell, -1, divaLocalArray, &divaptr);
            *divaptr = 0;
//            PetscScalar *ismaskptr; xDMPlexPointLocalRef(ismaskDM, cell, -1, ismaskLocalArray, &ismaskptr);
            PetscScalar *ismaskptr; xDMPlexPointLocalRef(sharedDM, cell, -1, ismaskLocalArray, &ismaskptr);
            *ismaskptr = 0;
//            PetscScalar *rankptr; xDMPlexPointLocalRef(rankDM, cell, -1, rankLocalArray, &rankptr);
            PetscScalar *rankptr; xDMPlexPointLocalRef(sharedDM, cell, -1, rankLocalArray, &rankptr);
            *rankptr = 0;
//            PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, cell, -1, phitildemaskLocalArray, &phitildemaskptr);
            PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(sharedDM, cell, -1, phitildemaskLocalArray, &phitildemaskptr);
            *phitildemaskptr = 0;
//            PetscScalar *phitildeptr; xDMPlexPointLocalRef(phitildeDM, cell, -1, phitildeLocalArray, &phitildeptr);
            PetscScalar *phitildeptr; xDMPlexPointLocalRef(sharedDM, cell, -1, phitildeLocalArray, &phitildeptr);
            *phitildeptr = 0;
//            PetscScalar *cellidptr; xDMPlexPointLocalRef(cellidDM, cell, -1, cellidLocalArray, &cellidptr);
            PetscScalar *cellidptr; xDMPlexPointLocalRef(sharedDM, cell, -1, cellidLocalArray, &cellidptr);
            *cellidptr = cell;
            const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
//            PetscScalar *phiptr; xDMPlexPointLocalRef(phiDM, cell, -1, phiLocalArray, &phiptr);
            PetscScalar *phiptr; xDMPlexPointLocalRef(sharedDM, cell, -1, phiLocalArray, &phiptr);
            if (owned>=0){ *phiptr = *phic; }
            PetscReal xp, yp, zp; GetCoordinate3D(dm, dim, cell, &xp, &yp, &zp);
/*            PetscScalar *xptr; xDMPlexPointLocalRef(xDM, cell, -1, xLocalArray, &xptr);
            PetscScalar *yptr; xDMPlexPointLocalRef(yDM, cell, -1, yLocalArray, &yptr);
            PetscScalar *zptr; xDMPlexPointLocalRef(zDM, cell, -1, zLocalArray, &zptr);*/
            PetscScalar *xptr; xDMPlexPointLocalRef(sharedDM, cell, -1, xLocalArray, &xptr);
            PetscScalar *yptr; xDMPlexPointLocalRef(sharedDM, cell, -1, yLocalArray, &yptr);
            PetscScalar *zptr; xDMPlexPointLocalRef(sharedDM, cell, -1, zLocalArray, &zptr);
            *xptr = xp; *yptr = yp; *zptr = zp;
    }
    for (PetscInt cell = cStart; cell < cEnd; ++cell){
        PetscSection globalSection; DMGetGlobalSection(dm, &globalSection);
        PetscInt owned = 1; PetscSectionGetOffset(globalSection, cell, &owned);
        if (owned>=0){
                PetscScalar *rankcptr;
//                xDMPlexPointLocalRef(rankDM, cell, -1, rankLocalArray, &rankcptr);
                xDMPlexPointLocalRef(sharedDM, cell, -1, rankLocalArray, &rankcptr);
                *rankcptr = rank;
        }
    }
    //init mask field
    for (PetscInt cell = cStart; cell < cEnd; ++cell){
        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
        if (*phic > 1e-4 and *phic < 1-1e-4) {
                PetscInt nNeighbors, *neighbors;
                DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
                for (PetscInt j = 0; j < nNeighbors; ++j) {
                    PetscInt neighbor = neighbors[j];
//                    PetscScalar *ranknptr; xDMPlexPointLocalRef(rankDM, neighbor, -1, rankLocalArray, &ranknptr);
//                    PetscScalar *ismaskptr; xDMPlexPointLocalRef(ismaskDM, neighbor, -1, ismaskLocalArray, &ismaskptr);
                    PetscScalar *ranknptr; xDMPlexPointLocalRef(sharedDM, neighbor, -1, rankLocalArray, &ranknptr);
                    PetscScalar *ismaskptr; xDMPlexPointLocalRef(sharedDM, neighbor, -1, ismaskLocalArray, &ismaskptr);
                    *ismaskptr = *ranknptr;
                }
                DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        }
    }
//    PushGhost(ismaskDM, ismaskLocalVec, ismaskGlobalVec, ADD_VALUES, false, false);
    PushGhost(sharedDM, ismaskLocalVec, ismaskGlobalVec, ADD_VALUES, false, false);


    for (PetscInt cell = cStart; cell < cEnd; ++cell){
        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
        if (*phic > 1e-4 and *phic < 1-1e-4) {
//                PetscScalar *ismaskptr; xDMPlexPointLocalRef(ismaskDM, cell, -1, ismaskLocalArray, &ismaskptr);
                PetscScalar *ismaskptr; xDMPlexPointLocalRef(sharedDM, cell, -1, ismaskLocalArray, &ismaskptr);
                *ismaskptr = 5;
        }
    }
    PetscReal rmin; DMPlexGetMinRadius(dm, &rmin); PetscReal h=2*rmin;
    PetscScalar C=1; PetscScalar N=2.6; PetscScalar layers = ceil(C*N);
    // phitilde mask
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
//        PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, cell, -1, phitildemaskLocalArray, &phitildemaskptr);
        PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(sharedDM, cell, -1, phitildemaskLocalArray, &phitildemaskptr);
        *phitildemaskptr = 0;
    }
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic) >> ablate::utilities::PetscUtilities::checkError;
        if (*phic > 0.0001 and *phic < 0.9999) {
                PetscInt nNeighbors, *neighbors; DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors); //
                for (PetscInt j = 0; j < nNeighbors; ++j) {
                    PetscInt neighbor = neighbors[j];
//                    PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, neighbor, -1, phitildemaskLocalArray, &phitildemaskptr);
                    PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(sharedDM, neighbor, -1, phitildemaskLocalArray, &phitildemaskptr);
                    *phitildemaskptr = 1;
PetscReal xc, yc, zc; GetCoordinate3D(dm, dim, cell, &xc, &yc, &zc);
PetscReal xn, yn, zn; GetCoordinate3D(dm, dim, neighbor, &xn, &yn, &zn);
                }
                DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        }
    }
//    PushGhost(phitildemaskDM, phitildemaskLocalVec, phitildemaskGlobalVec, ADD_VALUES, false, false);
    PushGhost(sharedDM, phitildemaskLocalVec, phitildemaskGlobalVec, ADD_VALUES, false, false);


    //phitilde, auxDM COPY
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
        PetscReal xc, yc, zc; GetCoordinate3D(dm, dim, cell, &xc, &yc, &zc);
//        PetscScalar *phitilde; xDMPlexPointLocalRef(phitildeDM, cell, -1, phitildeLocalArray, &phitilde);
//        PetscScalar *phitildemask; xDMPlexPointLocalRef(phitildemaskDM, cell, -1, phitildemaskLocalArray, &phitildemask);
        PetscScalar *phitilde; xDMPlexPointLocalRef(sharedDM, cell, -1, phitildeLocalArray, &phitilde);
        PetscScalar *phitildemask; xDMPlexPointLocalRef(sharedDM, cell, -1, phitildemaskLocalArray, &phitildemask);
        if (*phitildemask < 1e-10){ *phitilde = *phic;
if (process->flipPhiTilde){*phitilde = 1.00- *phitilde;} }
        else{
            PetscInt nNeighbors, *neighbors; DMPlexGetNeighbors(auxDM, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
            PetscReal weightedphi = 0; PetscReal Tw = 0;
            for (PetscInt j = 0; j < nNeighbors; ++j) {
                PetscInt neighbor = neighbors[j];
                PetscReal *phin; xDMPlexPointLocalRead(dm, neighbor, phiField.id, solArray, &phin);
                PetscReal xn, yn, zn; GetCoordinate3D(dm, dim, neighbor, &xn, &yn, &zn);
bool periodicfix = true;
if (periodicfix){
//temporary fix addressing how multiple layers of neighbors for a periodic domain return coordinates on the opposite side
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
//PetscScalar *rankptr; xDMPlexPointLocalRef(rankDM, cell, -1, rankLocalArray, &rankptr);
PetscScalar *rankptr; xDMPlexPointLocalRef(sharedDM, cell, -1, rankLocalArray, &rankptr);
if ((cell==0) and (*rankptr == 5)){ std::cout << "";}
            }
            weightedphi /= Tw;
if (process->flipPhiTilde){weightedphi = 1.000-weightedphi;}
            *phitilde = weightedphi;
            DMPlexRestoreNeighbors(auxDM, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        }
    }
//    PushGhost(phitildeDM, phitildeLocalVec, phitildeGlobalVec, INSERT_VALUES, true, true);
    PushGhost(sharedDM, phitildeLocalVec, phitildeGlobalVec, INSERT_VALUES, true, true);

for (PetscInt cell = cStart; cell < cEnd; ++cell) {
PetscScalar *optr2; PetscScalar *phitildeptr;
//xDMPlexPointLocalRef(phitildeDM, cell, -1, phitildeLocalArray, &phitildeptr);
xDMPlexPointLocalRef(sharedDM, cell, -1, phitildeLocalArray, &phitildeptr);
xDMPlexPointLocalRef(auxDM, cell, ofield2.id, auxArray, &optr2);
*optr2 = *phitildeptr;
//PetscScalar *rankptr; xDMPlexPointLocalRef(rankDM, cell, -1, rankLocalArray, &rankptr);
PetscScalar *rankptr; xDMPlexPointLocalRef(sharedDM, cell, -1, rankLocalArray, &rankptr);
if ((cell==0) and (*rankptr == 5)){  std::cout << "";   }

}

    //clean up vertex based vectors
    for (PetscInt vertex = vStart; vertex < vEnd; vertex++) {
        PetscReal vx, vy, vz; GetCoordinate3D(dm, dim, vertex, &vx, &vy, &vz);
//        PetscScalar *vxptr; xDMPlexPointLocalRef(vxDM, vertex, -1, vxLocalArray, &vxptr);
//        PetscScalar *vyptr; xDMPlexPointLocalRef(vyDM, vertex, -1, vyLocalArray, &vyptr);
//        PetscScalar *vzptr; xDMPlexPointLocalRef(vzDM, vertex, -1, vzLocalArray, &vzptr);
        PetscScalar *vxptr; xDMPlexPointLocalRef(sharedVertexDM_1, vertex, -1, vxLocalArray, &vxptr);
        PetscScalar *vyptr; xDMPlexPointLocalRef(sharedVertexDM_1, vertex, -1, vyLocalArray, &vyptr);
        PetscScalar *vzptr; xDMPlexPointLocalRef(sharedVertexDM_1, vertex, -1, vzLocalArray, &vzptr);
        *vxptr = vx; *vyptr = vy; *vzptr = vz;
//        PetscScalar *aptr; xDMPlexPointLocalRef(aDM, vertex, -1, aLocalArray, &aptr);
        PetscScalar *aptr; xDMPlexPointLocalRef(sharedVertexDM_dim, vertex, -1, aLocalArray, &aptr);

        *aptr = 0;
    }

    //calculate phiv, gradphiv, av (auxDM COPY)
    for (PetscInt vertex = vStart; vertex < vEnd; vertex++) {
        PetscReal vx, vy, vz; GetCoordinate3D(dm, dim, vertex, &vx, &vy, &vz);
        PetscInt nvn, *vertexneighbors; DMPlexVertexGetCells(dm, vertex, &nvn, &vertexneighbors);
        PetscBool isAdjToMask = PETSC_FALSE;
        for (PetscInt k = 0; k < nvn; k++){
//            PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, vertexneighbors[k], -1, phitildemaskLocalArray, &phitildemaskptr);
            PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(sharedDM, vertexneighbors[k], -1, phitildemaskLocalArray, &phitildemaskptr);
            if (*phitildemaskptr > 0.5){
                isAdjToMask = PETSC_TRUE;
            }
        }
        PetscScalar gradphiv[dim];
        PetscReal normgradphi = 0.0;
        if (isAdjToMask == PETSC_TRUE){
            if (dim==1){
                //changed to phitilde
//                PetscScalar *phitildekm1; xDMPlexPointLocalRef(phitildeDM, vertexneighbors[0], -1, phitildeLocalArray, &phitildekm1);
//                PetscScalar *phitildekp1; xDMPlexPointLocalRef(phitildeDM, vertexneighbors[1], -1, phitildeLocalArray, &phitildekp1);
                PetscScalar *phitildekm1; xDMPlexPointLocalRef(sharedDM, vertexneighbors[0], -1, phitildeLocalArray, &phitildekm1);
                PetscScalar *phitildekp1; xDMPlexPointLocalRef(sharedDM, vertexneighbors[1], -1, phitildeLocalArray, &phitildekp1);
                PetscReal xm1; GetCoordinate1D(dm, dim, vertexneighbors[0], &xm1);
                PetscReal xp1; GetCoordinate1D(dm, dim, vertexneighbors[1], &xp1);
                gradphiv[0]=(*phitildekp1 - *phitildekm1)/(xp1 - xm1);
            }
            else{
//DMPlexVertexGradFromCell(phitildeDM, vertex, phitildeLocalVec, -1, 0, gradphiv);
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

//temporary fix addressing how multiple layers of neighbors for a periodic domain return coordinates on the opposite side

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
//                PetscReal *phineighbor; xDMPlexPointLocalRef(phitildeDM, neighbor, -1, phitildeLocalArray, &phineighbor);
                PetscReal *phineighbor; xDMPlexPointLocalRef(sharedDM, neighbor, -1, phitildeLocalArray, &phineighbor);
                phiv += (*phineighbor) * (weights[k]);  // unstructured case
            }
        }
        else{ phiv=0; }

        //get a at vertices (av) (Chiu 2011)
        PetscScalar  av[dim];
//        PetscReal *avptr; xDMPlexPointLocalRef(aDM, vertex, -1, aLocalArray, &avptr); //vertexDM
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
//    PushGhost(aDM, aLocalVec, aGlobalVec, INSERT_VALUES, true, false);
    PushGhost(sharedVertexDM_dim, aLocalVec, aGlobalVec, INSERT_VALUES, true, false);


    //diva (auxDM COPY)
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
//        PetscScalar *diva; xDMPlexPointLocalRef(divaDM, cell, -1, divaLocalArray, &diva); *diva=0.0;
//        PetscScalar *ismask; xDMPlexPointLocalRef(ismaskDM, cell, -1, ismaskLocalArray, &ismask);
        PetscScalar *diva; xDMPlexPointLocalRef(sharedDM, cell, -1, divaLocalArray, &diva); *diva=0.0;
        PetscScalar *ismask; xDMPlexPointLocalRef(sharedDM, cell, -1, ismaskLocalArray, &ismask);
        if (*ismask > 0.5){
            if (dim==1){
                PetscInt nVerts, *verts; DMPlexCellGetVertices(dm, cell, &nVerts, &verts);
//                PetscScalar *am1; xDMPlexPointLocalRef(aDM, verts[0], -1, aLocalArray, &am1);
//                PetscScalar *ap1; xDMPlexPointLocalRef(aDM, verts[1], -1, aLocalArray, &ap1);
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
//                    DMPlexCellGradFromVertex(aDM, cell, aLocalVec, -1, offset, nabla_ai);
                    DMPlexCellGradFromVertex(sharedVertexDM_dim, cell, aLocalVec, -1, offset, nabla_ai);
                    *diva += nabla_ai[offset];
                }
            }
        }
        else{ *diva = 0.0; }
    }

//    PushGhost(divaDM, divaLocalVec, divaGlobalVec, INSERT_VALUES, true, false);
    PushGhost(sharedDM, divaLocalVec, divaGlobalVec, INSERT_VALUES, true, false);

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.GetPoint(c);
        const PetscScalar *euler = nullptr; xDMPlexPointLocalRead(dm, cell, eulerfID, solArray, &euler);
//        const PetscReal *phik; xDMPlexPointLocalRead(phitildeDM, cell, -1, phitildeLocalArray, &phik);
        const PetscReal *phik; xDMPlexPointLocalRead(sharedDM, cell, -1, phitildeLocalArray, &phik);
        PetscScalar *eulerSource; xDMPlexPointLocalRef(dm, cell, eulerfID, fArray, &eulerSource);
        PetscScalar *rhophiSource; xDMPlexPointLocalRef(dm, cell, densityVFField.id, fArray, &rhophiSource);
        PetscScalar rhog; const PetscScalar *rhogphig; xDMPlexPointLocalRead(dm, cell, densityVFField.id, solArray, &rhogphig);
        if(*rhogphig > 1e-10){rhog = *rhogphig / *phik;}else{rhog = 0;}
//        PetscScalar *diva; xDMPlexPointLocalRef(divaDM, cell, -1, divaLocalArray, &diva);
        PetscScalar *diva; xDMPlexPointLocalRef(sharedDM, cell, -1, divaLocalArray, &diva);

//first term
        *rhophiSource += rhog* *diva;

//second term
        auto density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];
        PetscReal ux = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] / density;
        PetscReal uy = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + 1] / density;
        PetscReal uz = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + 2] / density;
        PetscScalar Drhogx = 0.0, Drhogy = 0.0, Drhogz = 0.0;
        PetscScalar gradrhogphi[dim];
        PetscScalar gradphi[dim];
        DMPlexCellGradFromCell(dm, cell, locX, densityVFField.id, 0, gradrhogphi);
//        DMPlexCellGradFromCell(phitildeDM, cell, phitildeLocalVec, -1, 0, gradphi);
        DMPlexCellGradFromCell(sharedDM, cell, phitildeLocalVec, -1, 0, gradphi);
           Drhogx = gradrhogphi[0] - rhog * gradphi[0]; if(*phik > 1e-10){Drhogx = Drhogx / *phik;}else{Drhogx = 0;}
if(dim>1){ Drhogy = gradrhogphi[1] - rhog * gradphi[1]; if(*phik > 1e-10){Drhogy = Drhogy / *phik;}else{Drhogy = 0;} }
if(dim>2){ Drhogz = gradrhogphi[2] - rhog * gradphi[2]; if(*phik > 1e-10){Drhogz = Drhogz / *phik;}else{Drhogz = 0;} }
        *rhophiSource += *phik * (ux * Drhogx + uy * Drhogy + uz * Drhogz)*0;

        PetscScalar *optr; xDMPlexPointLocalRef(auxDM, cell, ofield.id, auxArray, &optr);
        *optr = *diva;

    }
    subDomain->UpdateAuxLocalVector();

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


/*    VecRestoreArray(divaLocalVec, &divaLocalArray);
    DMRestoreLocalVector(divaDM, &divaLocalVec);
    DMRestoreGlobalVector(divaDM, &divaGlobalVec);
    DMDestroy(&divaDM);

    VecRestoreArray(ismaskLocalVec, &ismaskLocalArray);
    DMRestoreLocalVector(ismaskDM, &ismaskLocalVec);
    DMRestoreGlobalVector(ismaskDM, &ismaskGlobalVec);
    DMDestroy(&ismaskDM);

    VecRestoreArray(phitildeLocalVec, &phitildeLocalArray);
    DMRestoreLocalVector(phitildeDM, &phitildeLocalVec);
    DMRestoreGlobalVector(phitildeDM, &phitildeGlobalVec);
    DMDestroy(&phitildeDM);

    VecRestoreArray(phitildemaskLocalVec, &phitildemaskLocalArray);
    DMRestoreLocalVector(phitildemaskDM, &phitildemaskLocalVec);
    DMRestoreGlobalVector(phitildemaskDM, &phitildemaskGlobalVec);
    DMDestroy(&phitildemaskDM);

    VecRestoreArray(rankLocalVec, &rankLocalArray);
    DMRestoreLocalVector(rankDM, &rankLocalVec);
    DMRestoreGlobalVector(rankDM, &rankGlobalVec);
    DMDestroy(&rankDM);

    VecRestoreArray(phiLocalVec, &phiLocalArray);
    DMRestoreLocalVector(phiDM, &phiLocalVec);
    DMRestoreGlobalVector(phiDM, &phiGlobalVec);
    DMDestroy(&phiDM);

    VecRestoreArray(cellidLocalVec, &cellidLocalArray);
    DMRestoreLocalVector(cellidDM, &cellidLocalVec);
    DMRestoreGlobalVector(cellidDM, &cellidGlobalVec);
    DMDestroy(&cellidDM);

    VecRestoreArray(xLocalVec, &xLocalArray);
    DMRestoreLocalVector(xDM, &xLocalVec);
    DMRestoreGlobalVector(xDM, &xGlobalVec);
    DMDestroy(&xDM);

    VecRestoreArray(yLocalVec, &yLocalArray);
    DMRestoreLocalVector(yDM, &yLocalVec);
    DMRestoreGlobalVector(yDM, &yGlobalVec);
    DMDestroy(&yDM);

    VecRestoreArray(zLocalVec, &zLocalArray);
    DMRestoreLocalVector(zDM, &zLocalVec);
    DMRestoreGlobalVector(zDM, &zGlobalVec);
    DMDestroy(&zDM); */

/*    VecRestoreArray(vxLocalVec, &vxLocalArray);
    DMRestoreLocalVector(vxDM, &vxLocalVec);
    DMRestoreGlobalVector(vxDM, &vxGlobalVec);
    DMDestroy(&vxDM);

    VecRestoreArray(vyLocalVec, &vyLocalArray);
    DMRestoreLocalVector(vyDM, &vyLocalVec);
    DMRestoreGlobalVector(vyDM, &vyGlobalVec);
    DMDestroy(&vyDM);

    VecRestoreArray(vzLocalVec, &vzLocalArray);
    DMRestoreLocalVector(vzDM, &vzLocalVec);
    DMRestoreGlobalVector(vzDM, &vzGlobalVec);
    DMDestroy(&vzDM);

    VecRestoreArray(aLocalVec, &aLocalArray);
    DMRestoreLocalVector(aDM, &aLocalVec);
    DMRestoreGlobalVector(aDM, &aGlobalVec);
    DMDestroy(&aDM); */

    // cleanup
    VecRestoreArrayRead(locX, &solArray);
    VecRestoreArray(auxVec, &auxArray);
    VecRestoreArray(vertexVec, &vertexArray);
    VecRestoreArray(locFVec, &fArray);
    solver.RestoreRange(cellRange);

    DMRestoreLocalVector(process->vertexDM, &vertexVec);
    VecDestroy(&vertexVec); //<--- this is fine
//    VecDestroy(&locFVec); //<--- SEGV 11
    PetscFunctionReturn(0);
}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::IntSharp, "calculates interface regularization term",
         ARG(PetscReal, "Gamma", "Gamma, velocity scale parameter (approx. umax)"),
         ARG(PetscReal, "epsilon", "epsilon, interface thickness scale parameter (approx. h)"),
         ARG(bool, "flipPhiTilde", "if true: phiTilde-->1-phiTilde (set it to true if primary phase is phi=0 or false if phi=1)")
);
