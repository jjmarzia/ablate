#include "domain/RBF/mq.hpp"
#include "intSharp.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"

//#include <iostream>
#include <fstream>

//void GetVertexRange(DM dm, const std::shared_ptr<ablate::domain::Region> &region, ablate::domain::Range &vertexRange) {
//    PetscInt depth=0; //zeroth layer of DAG is always that of the vertices
//    ablate::domain::GetRange(dm, region, depth, vertexRange);
//}

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
//    DMLocalToGlobal(dm, LocalVec, INSERT_VALUES, GlobalVec);

    if ((ADD_OR_INSERT_VALUES == ADD_VALUES) and (zerovec == true)){
//        std::cout << "yeah!"<<"\n";
        VecZeroEntries(GlobalVec);
    }
    DMLocalToGlobal(dm, LocalVec, ADD_OR_INSERT_VALUES, GlobalVec); //p0 to p1
    DMGlobalToLocal(dm, GlobalVec, INSERT_VALUES, LocalVec); //p1 to p1

    PetscScalar *LocalArray; VecGetArray(LocalVec, &LocalArray);
    PetscInt cStart, cEnd; DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);

    if ((ADD_OR_INSERT_VALUES == ADD_VALUES) and (isphitilde)){
        for (PetscInt cell = cStart; cell < cEnd; ++cell){
            //        PetscScalar *optr; xDMPlexPointLocalRef(dm, cell, -1, LocalArray, &optr);
            phitildepenalty[cell]+=1;
        }
    }



//    if (phitildemaskDM != PETSC_NULLPTR){
//        for (PetscInt cell = cStart; cell < cEnd; ++cell){
//
//            PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, cell, -1, phitildemaskLocalArray, &phitildemaskptr);
//
//            if( ( *phitildemaskptr >0 ) and (*phitildemaskptr < 17 )){ *optr /= *phitildemaskptr; }
//        }
//    }

}

PetscInt counter=0;
void SaveData(PetscInt rangeStart, PetscInt rangeEnd, DM dm, PetscScalar *array, std::string filename, bool iterateAcrossTime){
        int rank; MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        std::string counterstring;
        if (iterateAcrossTime){
            counter+=1;
            counterstring = std::to_string(counter);
        }
        if (not (iterateAcrossTime)){ counterstring = ""; }
        std::ofstream thefile("/Users/jjmarzia/Desktop/ablate/inputs/parallel/sidi_n2_2ts/"+filename+counterstring+"_rank"+std::to_string(rank)+".txt");
        for (PetscInt cell = rangeStart; cell < rangeEnd; ++cell) {
//            PetscInt cell = range.GetPoint(c);
            PetscScalar *ptr; xDMPlexPointLocalRef(dm, cell, -1, array, &ptr);
            auto s = std::to_string(*ptr);
            if (thefile.is_open()){
                thefile << s; thefile << "\n";
            }
        }
        thefile.close();
}

void ablate::finiteVolume::processes::IntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
    IntSharp::subDomain = solver.GetSubDomainPtr();
}

ablate::finiteVolume::processes::IntSharp::IntSharp(PetscReal Gamma, PetscReal epsilon) : Gamma(Gamma), epsilon(epsilon) {}
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

// std::cout << "computesource boxmeshcreate xymin0 xymin1 xymax0 xymax1: " << xymin[0] << " " << xymin[1] << " " << xymax[0] << " " << xymax[1] << "\n";


    const auto &phiField = subDomain->GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
    const auto &eulerField = solver.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);
    const auto &densityVFField = subDomain->GetField("densityvolumeFraction");
    const auto &ofield = subDomain->GetField("debug");
    const auto &ofield2 = subDomain->GetField("debug2");

//    auto phifID = phiField.id;
    auto eulerfID = eulerField.id;

    // get vecs/arrays
    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector(); //LOCAL aux vector, not global

    Vec vertexVec; DMGetLocalVector(process->vertexDM, &vertexVec); //
    const PetscScalar *solArray; VecGetArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError; //
    PetscScalar *auxArray; VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError; //
    PetscScalar *vertexArray; VecGetArray(vertexVec, &vertexArray); //
    PetscScalar *fArray; PetscCall(VecGetArray(locFVec, &fArray)); //

    // get ranges
    ablate::domain::Range cellRange; solver.GetCellRangeWithoutGhost(cellRange); //
    PetscInt vStart, vEnd; DMPlexGetDepthStratum(process->vertexDM, 0, &vStart, &vEnd); //

    DM vxDM;
    IS_CopyDM(process->vertexDM, vStart, vEnd, 1, &vxDM);
    DM vyDM;
    IS_CopyDM(process->vertexDM, vStart, vEnd, 1, &vyDM);
    Vec vxLocalVec; DMCreateLocalVector(vxDM, &vxLocalVec);
    Vec vxGlobalVec; DMCreateGlobalVector(vxDM, &vxGlobalVec);
    Vec vyLocalVec; DMCreateLocalVector(vyDM, &vyLocalVec);
    Vec vyGlobalVec; DMCreateGlobalVector(vyDM, &vyGlobalVec);
    VecZeroEntries(vxLocalVec);
    VecZeroEntries(vxGlobalVec);
    VecZeroEntries(vyLocalVec);
    VecZeroEntries(vyGlobalVec);
    PetscScalar *vxLocalArray; VecGetArray(vxLocalVec, &vxLocalArray);
    PetscScalar *vyLocalArray; VecGetArray(vyLocalVec, &vyLocalArray);

    DM aDM;
    IS_CopyDM(process->vertexDM, vStart, vEnd, dim, &aDM);
    Vec aLocalVec; DMCreateLocalVector(aDM, &aLocalVec);
    Vec aGlobalVec; DMCreateGlobalVector(aDM, &aGlobalVec);
    VecZeroEntries(aLocalVec);
    VecZeroEntries(aGlobalVec);
    PetscScalar *aLocalArray; VecGetArray(aLocalVec, &aLocalArray);

    PetscInt cStart, cEnd; DMPlexGetHeightStratum(auxDM, 0, &cStart, &cEnd);

    DM divaDM;
    IS_CopyDM(auxDM, cStart, cEnd, 1, &divaDM);
    Vec divaLocalVec; DMCreateLocalVector(divaDM, &divaLocalVec);
    Vec divaGlobalVec; DMCreateGlobalVector(divaDM, &divaGlobalVec);
    VecZeroEntries(divaLocalVec);
    VecZeroEntries(divaGlobalVec);
    PetscScalar *divaLocalArray; VecGetArray(divaLocalVec, &divaLocalArray);

    DM ismaskDM;
    IS_CopyDM(auxDM, cStart, cEnd, 1, &ismaskDM);
    Vec ismaskLocalVec; DMCreateLocalVector(ismaskDM, &ismaskLocalVec);
    Vec ismaskGlobalVec; DMCreateGlobalVector(ismaskDM, &ismaskGlobalVec);
    VecZeroEntries(ismaskLocalVec);
    VecZeroEntries(ismaskGlobalVec);
    PetscScalar *ismaskLocalArray; VecGetArray(ismaskLocalVec, &ismaskLocalArray);

    DM phitildemaskDM;
    IS_CopyDM(auxDM, cStart, cEnd, 1, &phitildemaskDM);
    Vec phitildemaskLocalVec; DMCreateLocalVector(phitildemaskDM, &phitildemaskLocalVec);
    Vec phitildemaskGlobalVec; DMCreateGlobalVector(phitildemaskDM, &phitildemaskGlobalVec);
    VecZeroEntries(phitildemaskLocalVec);
    VecZeroEntries(phitildemaskGlobalVec);
    PetscScalar *phitildemaskLocalArray; VecGetArray(phitildemaskLocalVec, &phitildemaskLocalArray);

    DM rankDM;
    IS_CopyDM(auxDM, cStart, cEnd, 1, &rankDM);
    Vec rankLocalVec; DMCreateLocalVector(rankDM, &rankLocalVec);
    Vec rankGlobalVec; DMCreateGlobalVector(rankDM, &rankGlobalVec);
    VecZeroEntries(rankLocalVec);
    VecZeroEntries(rankGlobalVec);
    PetscScalar *rankLocalArray; VecGetArray(rankLocalVec, &rankLocalArray);

//    PetscViewer viewer;
//    PetscViewerASCIIOpen(MPI_Comm comm, "", PetscViewer *viewer)
//    DMView(rankDM, viewer);

//    DMViewFromOptions(rankDM, NULL, "-dm-view");
//    PetscViewer viewer;
//    PetscViewerFormat informat = PETSC_VIEWER_ASCII_LATEX;
//    PetscViewerOp
//    PetscViewerPushFormat(viewer, informat);
//    DMView(rankDM, viewer);


    DM phiDM;
    IS_CopyDM(auxDM, cStart, cEnd, 1, &phiDM);
    Vec phiLocalVec; DMCreateLocalVector(phiDM, &phiLocalVec);
    Vec phiGlobalVec; DMCreateGlobalVector(phiDM, &phiGlobalVec);
    VecZeroEntries(phiLocalVec);
    VecZeroEntries(phiGlobalVec);
    PetscScalar *phiLocalArray; VecGetArray(phiLocalVec, &phiLocalArray);

    DM phitildeDM;
    IS_CopyDM(auxDM, cStart, cEnd, 1, &phitildeDM);
    Vec phitildeLocalVec; DMCreateLocalVector(phitildeDM, &phitildeLocalVec);
    Vec phitildeGlobalVec; DMCreateGlobalVector(phitildeDM, &phitildeGlobalVec);
    VecZeroEntries(phitildeLocalVec);
    VecZeroEntries(phitildeGlobalVec);
    PetscScalar *phitildeLocalArray; VecGetArray(phitildeLocalVec, &phitildeLocalArray);

    DM cellidDM;
    IS_CopyDM(auxDM, cStart, cEnd, 1, &cellidDM);
    Vec cellidLocalVec; DMCreateLocalVector(cellidDM, &cellidLocalVec);
    Vec cellidGlobalVec; DMCreateGlobalVector(cellidDM, &cellidGlobalVec);
    VecZeroEntries(cellidLocalVec);
    VecZeroEntries(cellidGlobalVec);
    PetscScalar *cellidLocalArray; VecGetArray(cellidLocalVec, &cellidLocalArray);

    DM xDM;
    IS_CopyDM(auxDM, cStart, cEnd, 1, &xDM);
    DM yDM;
    IS_CopyDM(auxDM, cStart, cEnd, 1, &yDM);
    Vec xLocalVec; DMCreateLocalVector(xDM, &xLocalVec);
    Vec xGlobalVec; DMCreateGlobalVector(xDM, &xGlobalVec);
    Vec yLocalVec; DMCreateLocalVector(yDM, &yLocalVec);
    Vec yGlobalVec; DMCreateGlobalVector(yDM, &yGlobalVec);
    VecZeroEntries(xLocalVec);
    VecZeroEntries(xGlobalVec);
    VecZeroEntries(yLocalVec);
    VecZeroEntries(yGlobalVec);
    PetscScalar *xLocalArray; VecGetArray(xLocalVec, &xLocalArray);
    PetscScalar *yLocalArray; VecGetArray(yLocalVec, &yLocalArray);
    //field ID for non field calls is -1.

    int rank; MPI_Comm_rank(PETSC_COMM_WORLD, &rank); rank+=1;

    //print? (if long )
    bool verbose=false;

    //clean up fields

    for (PetscInt cell = cStart; cell < cEnd; ++cell){

            PetscSection globalSection; DMGetGlobalSection(dm, &globalSection);
            PetscInt owned = 1; PetscSectionGetOffset(globalSection, cell, &owned);

            PetscScalar *divaptr; xDMPlexPointLocalRef(divaDM, cell, -1, divaLocalArray, &divaptr);
            *divaptr = 0;
            PetscScalar *ismaskptr; xDMPlexPointLocalRef(ismaskDM, cell, -1, ismaskLocalArray, &ismaskptr);
            *ismaskptr = 0;
            PetscScalar *rankptr; xDMPlexPointLocalRef(rankDM, cell, -1, rankLocalArray, &rankptr);
            *rankptr = 0;
            PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, cell, -1, phitildemaskLocalArray, &phitildemaskptr);
            *phitildemaskptr = 0;
            PetscScalar *phitildeptr; xDMPlexPointLocalRef(phitildeDM, cell, -1, phitildeLocalArray, &phitildeptr);
            *phitildeptr = 0;
            PetscScalar *cellidptr; xDMPlexPointLocalRef(cellidDM, cell, -1, cellidLocalArray, &cellidptr);
            *cellidptr = cell;

            const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
            PetscScalar *phiptr; xDMPlexPointLocalRef(phiDM, cell, -1, phiLocalArray, &phiptr);
            if (owned>=0){ *phiptr = *phic; }

            PetscReal xp, yp, zp; GetCoordinate3D(dm, dim, cell, &xp, &yp, &zp);
            PetscScalar *xptr; xDMPlexPointLocalRef(xDM, cell, -1, xLocalArray, &xptr);
            PetscScalar *yptr; xDMPlexPointLocalRef(yDM, cell, -1, yLocalArray, &yptr);
            *xptr = xp; *yptr = yp;
    }

    if (verbose){SaveData(cellRange.start, cellRange.end, xDM, xLocalArray, "x", false);}
    if (verbose){SaveData(cellRange.start, cellRange.end, yDM, yLocalArray, "y", false);}
    if (verbose){SaveData(cellRange.start, cellRange.end, cellidDM, cellidLocalArray, "xcellid", false);}
    if (verbose){SaveData(cellRange.start, cellRange.end, phiDM, phiLocalArray, "phi", true);}

//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//        PetscInt cell = cellRange.GetPoint(c); //
//        PetscScalar *Mask; xDMPlexPointLocalRef(auxDM, cell, ISMaskField.id, auxArray, &Mask);
//        *Mask = 0;
////        PetscScalar *rank; xDMPlexPointLocalRef(auxDM, cell, rankField.id, auxArray, &rank);
////        *rank = 0;
//    }

    //Initialize rank field
    //rank field reveals how the domain is divided in terms of processors


    for (PetscInt cell = cStart; cell < cEnd; ++cell){
        PetscSection globalSection; DMGetGlobalSection(dm, &globalSection);
        PetscInt owned = 1; PetscSectionGetOffset(globalSection, cell, &owned);
        if (owned>=0){
                PetscScalar *rankcptr;
                xDMPlexPointLocalRef(rankDM, cell, -1, rankLocalArray, &rankcptr);
                *rankcptr = rank;
        }
    }

//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//        PetscInt cell = cellRange.GetPoint(c);
//        //check whether a cell is owned by a processor
//        PetscSection globalSection;
//        DMGetGlobalSection(dm, &globalSection);
//        PetscInt owned = 1;
//        PetscSectionGetOffset(globalSection, cell, &owned);
//        if (owned>=0){
//                PetscScalar *r;
//                xDMPlexPointLocalRef(auxDM, cell, rankField.id, auxArray, &r);
//                *r = rank;
//        }
//    }

    //check the above
//    for (PetscInt cell = cStart; cell < cEnd; ++cell){
//        PetscScalar *rankcptr; xDMPlexPointLocalRef(rankDM, cell, -1, rankLocalArray, &rankcptr);
//        PetscScalar *cellidptr; xDMPlexPointLocalRef(cellidDM, cell, -1, cellidLocalArray, &cellidptr);
//        if (PetscAbs(*rankcptr - *r) > 1e-2){std::cout << "error rank2\n";} //this doesn't print anything; good
//        if (PetscAbs(*cellidptr - cell) > 1e-2){std::cout << "error cellid2\n";} //this doesn't print anything; good
//    }

    //init mask field (auxDM copy)

    for (PetscInt cell = cStart; cell < cEnd; ++cell){
//        const PetscScalar *phic; xDMPlexPointLocalRead(phiDM, cell, -1, phiLocalArray, &phic);
        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
        if (*phic > 1e-4 and *phic < 1-1e-4) {
                PetscInt nNeighbors, *neighbors;
                DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors); //
                for (PetscInt j = 0; j < nNeighbors; ++j) {
                    PetscInt neighbor = neighbors[j];
                    PetscScalar *ranknptr; xDMPlexPointLocalRef(rankDM, neighbor, -1, rankLocalArray, &ranknptr);
                    PetscScalar *ismaskptr; xDMPlexPointLocalRef(ismaskDM, neighbor, -1, ismaskLocalArray, &ismaskptr);
                    *ismaskptr = *ranknptr;
                }
                DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        }
    }
//    PetscReal ismaskpenalty[cEnd];
    PushGhost(ismaskDM, ismaskLocalVec, ismaskGlobalVec, ADD_VALUES, false, false);
    if (verbose){SaveData(cellRange.start, cellRange.end, rankDM, rankLocalArray, "rank", false);}

    for (PetscInt cell = cStart; cell < cEnd; ++cell){
        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
        if (*phic > 1e-4 and *phic < 1-1e-4) {
                PetscScalar *ismaskptr; xDMPlexPointLocalRef(ismaskDM, cell, -1, ismaskLocalArray, &ismaskptr); //after vec surgery
                *ismaskptr = 5;
        }
    }
    if (verbose){SaveData(cellRange.start, cellRange.end, ismaskDM, ismaskLocalArray, "ismask", true);}

    //init mask field (auxDM) (delete asap)
//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//        PetscInt cell = cellRange.GetPoint(c);
//        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic) >> ablate::utilities::PetscUtilities::checkError;
//
//        if (*phic > 1e-4 and *phic < 1-1e-4) {
//                PetscInt nNeighbors, *neighbors;
//                DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors); //
//                for (PetscInt j = 0; j < nNeighbors; ++j) {
//                    PetscInt neighbor = neighbors[j];
//                    PetscScalar *Mask; xDMPlexPointLocalRef(auxDM, neighbor, ISMaskField.id, auxArray, &Mask);
//                    *Mask = 1;
//                }
//                DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
//        }
//    }
//    subDomain->UpdateAuxLocalVector();

    //Initialize phiTildeMask
    // phiTildeMask determines which cells receive a nonzero phiTilde value
    // this is smoothed to a greater extent than the intsharp field. (this might need to change)

    PetscReal rmin; DMPlexGetMinRadius(dm, &rmin); PetscReal h=2*rmin;
    PetscScalar C=1; PetscScalar N=2.6; PetscScalar layers = ceil(C*N);
//    layers = 4; //temporary; C=1.5, N=2.6

    //do phitildemask for auxDM COPY
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, cell, -1, phitildemaskLocalArray, &phitildemaskptr);
        *phitildemaskptr = 0;
    }
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic) >> ablate::utilities::PetscUtilities::checkError;
        if (*phic > 0.0001 and *phic < 0.9999) {
                PetscInt nNeighbors, *neighbors; DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors); //
                for (PetscInt j = 0; j < nNeighbors; ++j) {
                    PetscInt neighbor = neighbors[j];
                    PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, neighbor, -1, phitildemaskLocalArray, &phitildemaskptr);
                    *phitildemaskptr = 1;

PetscReal xc, yc, zc; GetCoordinate3D(dm, dim, cell, &xc, &yc, &zc);
PetscReal xn, yn, zn; GetCoordinate3D(dm, dim, neighbor, &xn, &yn, &zn);
//if (xn < 0){ std::cout << "id="<<cell<< "  cellx " << xc +yc*yn*zc*zn*0 << "   neighborx " << xn << "\n";   }

                }
                DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        }
    }



//    PetscReal phitildemaskpenalty[cEnd];
    PushGhost(phitildemaskDM, phitildemaskLocalVec, phitildemaskGlobalVec, ADD_VALUES, false, false);
    if (verbose){SaveData(cellRange.start, cellRange.end, phitildemaskDM, phitildemaskLocalArray, "phitildemask", true);}

    //do phitildemask for auxDM (delete asap)
//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//        PetscInt cell = cellRange.GetPoint(c);
//        PetscScalar *phiTildeMask; xDMPlexPointLocalRef(auxDM, cell, phiTildeMaskField.id, auxArray, &phiTildeMask);
//        *phiTildeMask = 0;
//    }
//    subDomain->UpdateAuxLocalVector();
//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//        PetscInt cell = cellRange.GetPoint(c);
//        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic) >> ablate::utilities::PetscUtilities::checkError;
//        if (*phic > 0.0001 and *phic < 0.9999) {
//            PetscInt nNeighbors, *neighbors; DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors); //
//            for (PetscInt j = 0; j < nNeighbors; ++j) {
//                PetscInt neighbor = neighbors[j];
//                PetscScalar *phiTildeMask; xDMPlexPointLocalRef(auxDM, neighbor, phiTildeMaskField.id, auxArray, &phiTildeMask);
////                const PetscScalar *phin; xDMPlexPointLocalRead(dm, neighbor, phiField.id, solArray, &phic) >> ablate::utilities::PetscUtilities::checkError;
//                *phiTildeMask = 1;
//            }
//            DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
//        }
//    }
//    subDomain->UpdateAuxLocalVector();

    //phitilde, auxDM COPY

    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
        PetscReal xc, yc, zc; GetCoordinate3D(dm, dim, cell, &xc, &yc, &zc);
        PetscScalar *phitilde; xDMPlexPointLocalRef(phitildeDM, cell, -1, phitildeLocalArray, &phitilde);
        PetscScalar *phitildemask; xDMPlexPointLocalRef(phitildemaskDM, cell, -1, phitildemaskLocalArray, &phitildemask);
        if (*phitildemask < 1e-10){ *phitilde = *phic; }
        else{
            PetscInt nNeighbors, *neighbors; DMPlexGetNeighbors(auxDM, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
            PetscReal weightedphi = 0; PetscReal Tw = 0;
            for (PetscInt j = 0; j < nNeighbors; ++j) {
                PetscInt neighbor = neighbors[j];
                PetscReal *phin; xDMPlexPointLocalRead(dm, neighbor, phiField.id, solArray, &phin);
                PetscReal xn, yn, zn; GetCoordinate3D(dm, dim, neighbor, &xn, &yn, &zn);

bool periodicfix = false;

if (periodicfix){

//temporary fix addressing how multiple layers of neighbors for a periodic domain return coordinates on the opposite side

PetscReal maxMask = 10*(xmax-xmin)/160; // [10(xmax-xmin)/ Nx] <--> corresponds to 10 cells.
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
                PetscReal s = C * h; //6*h
                PetscReal wn; PhiNeighborGauss(d, s, &wn);
                Tw += wn;
                weightedphi += (*phin * wn);

PetscScalar *rankptr; xDMPlexPointLocalRef(rankDM, cell, -1, rankLocalArray, &rankptr);
if ((cell==0) and (*rankptr == 5)){  std::cout << "particular is phitilde " << *phitilde << "  " << "nneighbors " << nNeighbors << "  " << neighbor << "  " << weightedphi << "  " << Tw << "\n";   }

            }
            weightedphi /= Tw;
            *phitilde = weightedphi;


            DMPlexRestoreNeighbors(auxDM, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        }
    }
    PushGhost(phitildeDM, phitildeLocalVec, phitildeGlobalVec, INSERT_VALUES, true, true);
    if (verbose){SaveData(cellRange.start, cellRange.end, phitildeDM, phitildeLocalArray, "phitilde", true);}

for (PetscInt cell = cStart; cell < cEnd; ++cell) {
PetscScalar *optr2; PetscScalar *phitildeptr; 
xDMPlexPointLocalRef(phitildeDM, cell, -1, phitildeLocalArray, &phitildeptr); 
xDMPlexPointLocalRef(auxDM, cell, ofield2.id, auxArray, &optr2);
*optr2 = *phitildeptr;
PetscScalar *rankptr; xDMPlexPointLocalRef(rankDM, cell, -1, rankLocalArray, &rankptr);

if ((cell==0) and (*rankptr == 5)){  std::cout << "ofield2 is phitilde " << *phitildeptr << "\n";   }

}

    //phitilde, auxDM (delete asap)

//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//        PetscInt cell = cellRange.GetPoint(c);
//        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic) >> ablate::utilities::PetscUtilities::checkError;
//        PetscReal xc, yc, zc; GetCoordinate3D(dm, dim, cell, &xc, &yc, &zc);
//
//        //now build phitilde
//        //number of smoothing layers
//
//        PetscScalar *phiTilde; xDMPlexPointLocalRef(auxDM, cell, phiTildeField.id, auxArray, &phiTilde) >> ablate::utilities::PetscUtilities::checkError;
//        PetscScalar *phiTildeMask; xDMPlexPointLocalRef(auxDM, cell, phiTildeMaskField.id, auxArray, &phiTildeMask) >> ablate::utilities::PetscUtilities::checkError;
//        if (*phiTildeMask == 0){
//            *phiTilde=*phic;
//        }
//        else{
//            PetscInt nNeighbors, *neighbors; DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
//            PetscReal weightedphi = 0; PetscReal Tw = 0;
//
//            for (PetscInt j = 0; j < nNeighbors; ++j) {
//                PetscInt neighbor = neighbors[j];
//
//                PetscReal *phin; xDMPlexPointLocalRead(dm, neighbor, phiField.id, solArray, &phin);
//
//                PetscReal xn, yn, zn; GetCoordinate3D(dm, dim, neighbor, &xn, &yn, &zn);
//                PetscReal d = PetscSqrtReal(PetscSqr(xn - xc) + PetscSqr(yn - yc) + PetscSqr(zn - zc));  // distance
//                PetscReal s = C * h; //6*h
//
//                PetscReal wn; PhiNeighborGauss(d, s, &wn);
//                Tw += wn;
//                weightedphi += (*phin * wn);
//            }
//            weightedphi /= Tw;
//            *phiTilde = weightedphi;
//            DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
//        }
//    }
//    subDomain->UpdateAuxLocalVector();

    //clean up vertex based vectors
    for (PetscInt vertex = vStart; vertex < vEnd; vertex++) {
        PetscReal vx, vy, vz; GetCoordinate3D(dm, dim, vertex, &vx, &vy, &vz);
        PetscScalar *vxptr; xDMPlexPointLocalRef(vxDM, vertex, -1, vxLocalArray, &vxptr);
        PetscScalar *vyptr; xDMPlexPointLocalRef(vyDM, vertex, -1, vyLocalArray, &vyptr);
        *vxptr = vx; *vyptr = vy;
        PetscScalar *aptr; xDMPlexPointLocalRef(aDM, vertex, -1, aLocalArray, &aptr);
        *aptr = 0;
    }
    if (verbose){SaveData(vStart, vEnd, vxDM, vxLocalArray, "vx", false);}
    if (verbose){SaveData(vStart, vEnd, vyDM, vyLocalArray, "vy", false);}

    //calculate phiv, gradphiv, av (auxDM COPY)
    for (PetscInt vertex = vStart; vertex < vEnd; vertex++) {
        PetscReal vx, vy, vz; GetCoordinate3D(dm, dim, vertex, &vx, &vy, &vz);
        PetscInt nvn, *vertexneighbors; DMPlexVertexGetCells(dm, vertex, &nvn, &vertexneighbors);
        PetscBool isAdjToMask = PETSC_FALSE;
        for (PetscInt k = 0; k < nvn; k++){
            PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, vertexneighbors[k], -1, phitildemaskLocalArray, &phitildemaskptr);// >> ablate::utilities::PetscUtilities::checkError;//            PetscScalar *Mask; xDMPlexPointLocalRef(auxDM, vertexneighbors[k], ISMaskField.id, auxArray, &Mask);// >> ablate::utilities::PetscUtilities::checkError;
            if (*phitildemaskptr > 0.5){
                isAdjToMask = PETSC_TRUE;
            }
        }
        PetscScalar gradphiv[dim];
        PetscReal normgradphi = 0.0;
        if (isAdjToMask == PETSC_TRUE){
            if (dim==1){
                //changed to phitilde
                PetscScalar *phitildekm1; xDMPlexPointLocalRef(phitildeDM, vertexneighbors[0], -1, phitildeLocalArray, &phitildekm1);
                PetscScalar *phitildekp1; xDMPlexPointLocalRef(phitildeDM, vertexneighbors[1], -1, phitildeLocalArray, &phitildekp1);
                PetscReal xm1; GetCoordinate1D(dm, dim, vertexneighbors[0], &xm1);
                PetscReal xp1; GetCoordinate1D(dm, dim, vertexneighbors[1], &xp1);
//                gradphiv[0]=(*phikp1 - *phikm1)/(1*process->epsilon);
                gradphiv[0]=(*phitildekp1 - *phitildekm1)/(xp1 - xm1);
            }
            else{ 
DMPlexVertexGradFromCell(phitildeDM, vertex, phitildeLocalVec, -1, 0, gradphiv); 

//PetscInt nCells, *thecells; DMPlexVertexGetCells(dm, vertex, &nCells, &thecells);
//for (int j=0; j<nCells; ++j){
//if (thecells[j]==3572){ 
//PetscReal cx, cy, cz; GetCoordinate3D(dm, dim, thecells[j], &cx, &cy, &cz);
//std::cout << "cellxy vertex vx vy gradphiv xy " << cx << "  " << cy << "  " << vertex << "  " << vx << "  " << vy << "  " << gradphiv[0] << "  " << gradphiv[1] <<"\n";  
//}
//}
//DMPlexVertexRestoreCells(dm, vertex, &nCells, &thecells);



}
            for (int k=0; k<dim; ++k){ normgradphi += PetscSqr(gradphiv[k]); }
            normgradphi = PetscSqrtReal(normgradphi);
        }
        else{ for (int k=0; k<dim; ++k){ gradphiv[k] =0; } }

        PetscReal phiv=0;
//        PetscReal Uv[dim];
        if(isAdjToMask == PETSC_TRUE) {
            PetscReal distances[nvn];
            PetscReal shortestdistance = ablate::utilities::Constants::large;
            for (PetscInt k = 0; k < nvn; ++k) {
//                for (int j=0; j<dim; ++j){ Uv[j] = 0; }
                PetscInt neighbor = vertexneighbors[k];
                PetscReal nx, ny, nz; GetCoordinate3D(dm, dim, neighbor, &nx, &ny, &nz);


//temporary fix addressing how multiple layers of neighbors for a periodic domain return coordinates on the opposite side

bool periodicfix = false;

if (periodicfix){

PetscReal maxMask = 5*process->epsilon;
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
                PetscReal *phineighbor; xDMPlexPointLocalRef(phitildeDM, neighbor, -1, phitildeLocalArray, &phineighbor);
//                PetscReal *eulerneighbor; xDMPlexPointLocalRead(dm, neighbor, eulerField.id, solArray, &eulerneighbor);
//                PetscReal Uneighbor[3] = {eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHOU]/eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHO],
//                                          eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHOV]/eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHO],
//                                          eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHOW]/eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHO]};
//                for (int j=0; j<dim; ++j){ Uv[j] += (Uneighbor[j]) * (weights[k]); }
                phiv += (*phineighbor) * (weights[k]);  // unstructured case
            }
        }
        else{ phiv=0; }

        //get a at vertices (av) (Chiu 2011)
        //based on Eq. 1 of:   Jain SS. Accurate conservative phase-field method for simulation of two-phase flows. Journal of Computational Physics. 2022 Nov 15;469:111529.
        PetscScalar  av[dim]; //PetscScalar  Uv[dim];
        PetscReal *avptr; xDMPlexPointLocalRef(aDM, vertex, -1, aLocalArray, &avptr); //vertexDM
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
    PushGhost(aDM, aLocalVec, aGlobalVec, INSERT_VALUES, true, false);
    if (verbose){SaveData(vStart, vEnd, aDM, aLocalArray, "a", true);}


    //compute gradphiv, phiv, av (auxDM; delete asap)

    //march over vertices
//    for (PetscInt vertex = vStart; vertex < vEnd; vertex++) {
//
//        //if all of the vertex's cell neighbors are not in Mask, don't bother with calculation
//        PetscReal vx, vy, vz; GetCoordinat(dm, dim, vertex, &vx, &vy, &vz);
//        PetscInt nvn, *vertexneighbors; DMPlexVertexGetCells(dm, vertex, &nvn, &vertexneighbors); //
//        PetscBool isAdjToMask = PETSC_FALSE;
//        for (PetscInt k = 0; k < nvn; k++){
//            //changed to phitilde mask
//            PetscScalar *Mask; xDMPlexPointLocalRef(auxDM, vertexneighbors[k], phiTildeMaskField.id, auxArray, &Mask);// >> ablate::utilities::PetscUtilities::checkError;
////            PetscScalar *Mask; xDMPlexPointLocalRef(auxDM, vertexneighbors[k], ISMaskField.id, auxArray, &Mask);// >> ablate::utilities::PetscUtilities::checkError;
//            if (*Mask > 0.5){
//                isAdjToMask = PETSC_TRUE;
//            }
//        }
//
//        PetscScalar gradphiv[dim];
//        PetscReal normgradphi = 0.0;
//        if (isAdjToMask == PETSC_TRUE){
//
//            if (dim==1){
//                //changed to phitilde
//                PetscScalar *phikm1; xDMPlexPointLocalRef(auxDM, vertexneighbors[0], phiTildeField.id, auxArray, &phikm1);
//                PetscScalar *phikp1; xDMPlexPointLocalRef(auxDM, vertexneighbors[1], phiTildeField.id, auxArray, &phikp1);
//                PetscReal xm1; GetCoordinate1D(dm, dim, vertexneighbors[0], &xm1);
//                PetscReal xp1; GetCoordinate1D(dm, dim, vertexneighbors[1], &xp1);
//                gradphiv[0]=(*phikp1 - *phikm1)/(1*process->epsilon);
//            }
//            else{
//                DMPlexVertexGradFromCell(auxDM, vertex, auxVec, phiTildeField.id, 0, gradphiv);
//            }
//
//            //get gradphi at vertices (gradphiv) based on cell centered phis
//            for (int k=0; k<dim; ++k){
//                normgradphi += PetscSqr(gradphiv[k]);
//            }
//            normgradphi = PetscSqrtReal(normgradphi);
//        }
//        else{
//            for (int k=0; k<dim; ++k){
//                gradphiv[k] =0;
//            }
//        }
//
//        PetscReal phiv=0;
//        PetscReal Uv[dim];
//
//        if(isAdjToMask == PETSC_TRUE) {
//            PetscReal distances[nvn];
//            PetscReal shortestdistance = ablate::utilities::Constants::large;
//            for (PetscInt k = 0; k < nvn; ++k) {
//
//                for (int j=0; j<dim; ++j){
//                    Uv[j] = 0;
//                }
//
//                PetscInt neighbor = vertexneighbors[k];
//                PetscReal nx, ny, nz;
//                GetCoordinat(dm, dim, neighbor, &nx, &ny, &nz);
//
//                PetscReal distance = PetscSqrtReal(PetscSqr(nx - vx) + PetscSqr(ny - vy) + PetscSqr(nz - vz));
//                if (distance < shortestdistance) {
//                    shortestdistance = distance;
//                }
//                distances[k] = distance;
//            }
//
//            PetscReal weights_wrt_short[nvn];
//            PetscReal totalweight_wrt_short = 0;
//
//            for (PetscInt k = 0; k < nvn; ++k) {
//                PetscReal weight_wrt_short = shortestdistance / distances[k];
//                weights_wrt_short[k] = weight_wrt_short;
//                totalweight_wrt_short += weight_wrt_short;
//            }
//
//            PetscReal weights[nvn];
//            for (PetscInt k = 0; k < nvn; ++k) {
//                weights[k] = weights_wrt_short[k] / totalweight_wrt_short;
//            }
//
//            for (PetscInt k = 0; k < nvn; ++k) {
//                PetscInt neighbor = vertexneighbors[k];
//                //changed to phitilde
//                PetscReal *phineighbor; xDMPlexPointLocalRef(auxDM, neighbor, phiTildeField.id, auxArray, &phineighbor);
//                PetscReal *eulerneighbor; xDMPlexPointLocalRead(dm, neighbor, eulerField.id, solArray, &eulerneighbor);
//                PetscReal Uneighbor[3] = {eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHOU]/eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHO],
//                                          eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHOV]/eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHO],
//                                          eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHOW]/eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHO]};
//                for (int j=0; j<dim; ++j){
//                    Uv[j] += (Uneighbor[j]) * (weights[k]);
//                }
//                phiv += (*phineighbor) * (weights[k]);  // unstructured case
//            }
//        }
//        else{
//            phiv=0;
//        }
//
//        //get a at vertices (av) (Chiu 2011)
//        //based on Eq. 1 of:   Jain SS. Accurate conservative phase-field method for simulation of two-phase flows. Journal of Computational Physics. 2022 Nov 15;469:111529.
//        PetscScalar  av[dim]; //PetscScalar  Uv[dim];
//        PetscReal *avptr;
//        xDMPlexPointLocalRef(process->vertexDM, vertex, 0, vertexArray, &avptr); //vertexDM
//
//        for (int k=0; k<dim; ++k){
//            if(isAdjToMask == PETSC_TRUE) {
//                if (normgradphi > ablate::utilities::Constants::tiny) {
//                    av[k] = (process->Gamma * process->epsilon * gradphiv[k]) - (process->Gamma * phiv * (1 - phiv) * (gradphiv[k] / normgradphi));
//                } else {
//                    av[k] = (process->Gamma * process->epsilon * gradphiv[k]) - (process->Gamma * phiv * (1 - phiv) * gradphiv[k]);
//                }
//                avptr[k] = av[k];
//            }
//            else{
//                avptr[k]=0;
//            }
//        }
//        DMPlexVertexRestoreCells(dm, vertex, &nvn, &vertexneighbors);
//    }
//    subDomain->UpdateAuxLocalVector();

    //diva (auxDM COPY)
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        PetscScalar *diva; xDMPlexPointLocalRef(divaDM, cell, -1, divaLocalArray, &diva); *diva=0.0;
        PetscScalar *ismask; xDMPlexPointLocalRef(ismaskDM, cell, -1, ismaskLocalArray, &ismask);
        if (*ismask > 0.5){
            if (dim==1){
                PetscInt nVerts, *verts; DMPlexCellGetVertices(dm, cell, &nVerts, &verts); //
                PetscScalar *am1; xDMPlexPointLocalRef(aDM, verts[0], -1, aLocalArray, &am1);
                PetscScalar *ap1; xDMPlexPointLocalRef(aDM, verts[1], -1, aLocalArray, &ap1);
                PetscReal xm1; GetCoordinate1D(dm, dim, verts[0], &xm1);
                PetscReal xp1; GetCoordinate1D(dm, dim, verts[1], &xp1);
                *diva = (*ap1-*am1)/(xp1-xm1);
                DMPlexCellRestoreVertices(dm, cell, &nVerts, &verts);
            }
            else{
                for (PetscInt offset = 0; offset < dim; offset++) {
                    PetscReal nabla_ai[dim];
                    DMPlexCellGradFromVertex(aDM, cell, aLocalVec, -1, offset, nabla_ai);
                    *diva += nabla_ai[offset];

//if (cell==3572){ std::cout << offset << "   " << *diva <<"\n";}

                }
            }
        }
        else{ *diva = 0.0; }
//        *rhophiSource += 0*(ux+uy+uz);
    }

//    PetscReal divapenalty[cEnd];
    PushGhost(divaDM, divaLocalVec, divaGlobalVec, INSERT_VALUES, true, false);
    if (verbose){SaveData(cellRange.start, cellRange.end, divaDM, divaLocalArray, "diva", true);}

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.GetPoint(c);
//    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        const PetscScalar *euler = nullptr; xDMPlexPointLocalRead(dm, cell, eulerfID, solArray, &euler);
        const PetscReal *phik; xDMPlexPointLocalRead(phitildeDM, cell, -1, phitildeLocalArray, &phik);
        PetscScalar *eulerSource; xDMPlexPointLocalRef(dm, cell, eulerfID, fArray, &eulerSource);
        PetscScalar *rhophiSource; xDMPlexPointLocalRef(dm, cell, densityVFField.id, fArray, &rhophiSource);// std::cout << "cell   " << cell << "  phi   " << *phik << "   rhs of rhophi equation   " << *rhophiSource << "\n";// *rhophi += 0; std::cout << "  xnew  " << *rhophi << "\n";
        PetscScalar rhog; const PetscScalar *rhogphig; xDMPlexPointLocalRead(dm, cell, densityVFField.id, solArray, &rhogphig);
        if(*rhogphig > 1e-10){rhog = *rhogphig / *phik;}else{rhog = 0;}
        PetscScalar *diva; xDMPlexPointLocalRef(divaDM, cell, -1, divaLocalArray, &diva);


        *rhophiSource += rhog* *diva;

        PetscScalar *optr; xDMPlexPointLocalRef(auxDM, cell, ofield.id, auxArray, &optr);
        *optr = *diva;

    }

//
//
//    // march over cells (auxDM, delete asap)
//    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
//        const PetscInt cell = cellRange.GetPoint(i);
//        PetscReal div=0.0;
////        //changed to phitilde
//////        PetscScalar *Mask; xDMPlexPointLocalRef(auxDM, cell, phiTildeMaskField.id, auxArray, &Mask);
////        PetscScalar *Mask; xDMPlexPointLocalRef(auxDM, cell, ISMaskField.id, auxArray, &Mask);// >> ablate::utilities::PetscUtilities::checkError;
////        if (*Mask > 0.5){
////            if (dim==1){
////                PetscInt nVerts, *verts; DMPlexCellGetVertices(dm, cell, &nVerts, &verts); //
////                PetscScalar *am1; xDMPlexPointLocalRef(process->vertexDM, verts[0], 0, vertexArray, &am1);
////                PetscScalar *ap1; xDMPlexPointLocalRef(process->vertexDM, verts[1], 0, vertexArray, &ap1);
//////                PetscReal xm1, ym1, zm1; GetCoordinat(dm, dim, verts[0], &xm1, &ym1, &zm1);
//////                PetscReal xp1, yp1, zp1; GetCoordinat(dm, dim, verts[1], &xp1, &yp1, &zp1);
////                div = (*ap1-*am1)/(1*process->epsilon);
//////                std::cout << "  cell " << cell << "  neighbor vert  " << verts[0] << "  a  " << *am1 << "   div   " << div << "\n";
////                DMPlexCellRestoreVertices(dm, cell, &nVerts, &verts);
////            }
////            else {
////                for (PetscInt offset = 0; offset < dim; offset++) {
////                    PetscReal nabla_ai[dim];
////                    DMPlexCellGradFromVertex(process->vertexDM, cell, vertexVec, 0, offset, nabla_ai);
////                    div += nabla_ai[offset];
//////                                    div=0;
////                }
////            }
////        }
////        else{ div=0; }
//        PetscReal *divaptr; xDMPlexPointLocalRef(auxDM, cell, divaField.id, auxArray, &divaptr); *divaptr = div;
//        const PetscScalar *euler = nullptr; xDMPlexPointLocalRead(dm, cell, eulerfID, solArray, &euler);
//        //Fdiff = mdiff v_m + \rho v_n v_{m,n}
//        //2d: Fdiffx = mdiff vx + \rho (vx vx,x + vy vx,y)
//        PetscScalar density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];
//        PetscReal ux = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU] / density;
//        PetscReal uy = euler[ablate::finiteVolume::CompressibleFlowFields::RHOV] / density;
//        PetscReal uz = euler[ablate::finiteVolume::CompressibleFlowFields::RHOW] / density;
//
//
//        PetscReal U[3] = {ux, uy, uz};
////        std::cout << "cell " << cell << "   ux  " << ux << "\n";
//        const PetscReal *phik; xDMPlexPointLocalRead(auxDM, cell, phiTildeField.id, auxArray, &phik);
//        PetscReal mdiff=0; PetscReal Fdiffx=0; PetscReal Fdiffy=0; PetscReal Fdiffz=0;
//        PetscReal *phiTildeMask; xDMPlexPointLocalRef(auxDM, cell, phiTildeMaskField.id, auxArray, &phiTildeMask);
//        if (*phiTildeMask>0.5) {
//            // add mdiff, etc. to momentum, energy equations; see local AlphaEq.pdf, SharpeningForce.pdf
//            PetscScalar gradrho[dim];//  for (int k=0; k<dim; ++k){gradrho[k]=0;}
//            PetscScalar gradrhou[dim];//  for (int k=0; k<dim; ++k){gradrhou[k]=0;}
//            PetscScalar gradrhov[dim];//  for (int k=0; k<dim; ++k){gradrhou[k]=0;}
//            PetscScalar gradrhow[dim];//  for (int k=0; k<dim; ++k){gradrhou[k]=0;}
//            PetscScalar gradphi[dim];// for (int k=0; k<dim; ++k){gradphi[k]=0;}
//            PetscScalar gradux[dim];// for (int k=0; k<dim; ++k){gradux[k]=0;}
//            PetscScalar graduy[dim];// for (int k=0; k<dim; ++k){graduy[k]=0;}
//            PetscScalar graduz[dim];// for (int k=0; k<dim; ++k){graduz[k]=0;}
//            //
//            if (dim == 1) {
//                PetscInt nNeighbors, *neighbors;
//                DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
////                const PetscReal *phikm1;xDMPlexPointLocalRead(dm, neighbors[0], phiField.id, solArray, &phikm1);
////                const PetscReal *phikp1;xDMPlexPointLocalRead(dm, neighbors[2], phiField.id, solArray, &phikp1);
//                const PetscReal *phikm1;xDMPlexPointLocalRead(auxDM, neighbors[0], phiTildeField.id, auxArray, &phikm1);
//                const PetscReal *phikp1;xDMPlexPointLocalRead(auxDM, neighbors[2], phiTildeField.id, auxArray, &phikp1);
////                std::cout << "cell    " << cell <<  "   phikpm1  " << *phikm1 << "   " << *phikp1 << "\n";
//                const PetscScalar *eulerkm1; xDMPlexPointLocalRead(dm, neighbors[0], eulerfID, solArray, &eulerkm1);
//                const PetscScalar *eulerkp1; xDMPlexPointLocalRead(dm, neighbors[2], eulerfID, solArray, &eulerkp1);
//                gradphi[0] = (*phikp1 - *phikm1) / (2 * process->epsilon);// gradphi[1]=gradphi[2]=0;
//                gradrho[0] = (eulerkp1[ablate::finiteVolume::CompressibleFlowFields::RHO] - eulerkm1[ablate::finiteVolume::CompressibleFlowFields::RHO]) / (2 * process->epsilon);// gradrho[1]=gradrho[2]=0; //eulerkm1[0] = rhokm1
//                gradrhou[0] = (eulerkp1[ablate::finiteVolume::CompressibleFlowFields::RHOU] - eulerkm1[ablate::finiteVolume::CompressibleFlowFields::RHOU]) / (2 * process->epsilon);// gradrhou[1]=gradrhou[2]=0; //eulerkm1[1] = rhoukm1
//                gradux[0] = (gradrhou[0] - ux*gradrho[0]) / density; // gradux[1]=gradux[2]=0;
//                graduy[0]=graduz[0]=0;
////                std::cout << gradux[0] << "\n";
//                DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
//            } else {
////                DMPlexCellGradFromCell(dm, cell, locX, densityVFField.id, 0, gradrhophi);
//                //changed to phitilde
//                DMPlexCellGradFromCell(auxDM, cell, auxVec, phiTildeField.id, 0, gradphi);
//                DMPlexCellGradFromCell(dm, cell, locX, eulerField.id, 0, gradrho); //offset=0 implies the first element of the euler vector, which is rho
//                DMPlexCellGradFromVertex(dm, cell, locX, eulerField.id, 1, gradrhou);
//                DMPlexCellGradFromVertex(dm, cell, locX, eulerField.id, 2, gradrhov);
//                DMPlexCellGradFromVertex(dm, cell, locX, eulerField.id, 3, gradrhow);
//
//                for (int k = 0; k < dim; ++k) {
//                    gradux[k] = (gradrhou[k] - ux*gradrho[k]) / density;
//                    graduy[k] = (gradrhov[k] - uy*gradrho[k]) / density;
//                    graduz[k] = (gradrhow[k] - uz*gradrho[k]) / density;
//                }
//            }
//            mdiff = density * div;
////            for (int k = 0; k < dim; ++k) {
////                mdiff += (*phik*U[k]*gradrho[k] - density*U[k]*gradphi[k]); //NOT density uk gradrhok, as shown in AlphaEq1
////            }
////            if ((*phik > 1e-4) and (*phik < 1-1e-4)) {mdiff /= *phik;}else{ mdiff = 0;}
//            for (int k = 0; k < dim; ++k) {
//                mdiff += ((*phik-0.5)*U[k]*gradrho[k] - density*U[k]*gradphi[k]); //NOT density uk gradrhok, as shown in AlphaEq1
//            }
//            if (PetscAbs(*phik - 0.5) > 1e-2){mdiff /= (*phik-0.5); } else{ mdiff = 0;}
//            PetscReal *mdiffptr; xDMPlexPointLocalRef(auxDM, cell, mdiffField.id, auxArray, &mdiffptr); *mdiffptr = mdiff;
//            PetscReal *gradrhoptr; xDMPlexPointLocalRef(auxDM, cell, gradrhoField.id, auxArray, &gradrhoptr); *gradrhoptr = gradrho[0];
//            PetscReal *gradphiptr; xDMPlexPointLocalRef(auxDM, cell, gradphiField.id, auxArray, &gradphiptr); *gradphiptr = gradphi[0];
//            Fdiffx += mdiff*ux; Fdiffy += mdiff*uy; Fdiffz += mdiff*uz;
////            for (int j = 0; j < dim; ++j) {
////                std::cout << cell << "   ux,xi  " << gradux[j] << "   " << U[j] << "\n";
////                std::cout << cell << "   uy,xi  " << graduy[j] << "\n";
////                std::cout << cell << "   uz,xi  " << graduz[j] << "\n";
////            }
//            for (int j = 0; j < dim; ++j) {
//                Fdiffx += density*(U[j]*gradux[j]);
//                Fdiffy += density*(U[j]*graduy[j]);
//                Fdiffz += density*(U[j]*graduz[j]);
//            }
////            std::cout << "--->" << Fdiffx << "    " << mdiff*ux << "     " << density*(U[0]*gradux[0]) << "\n";
//        }
////        if (*Mask > 0.5){std::cout << "cell   " << cell << "  ux   " << ux << "\n";}
//
//        PetscReal *Fdiffxptr; xDMPlexPointLocalRef(auxDM, cell, FdiffxField.id, auxArray, &Fdiffxptr); *Fdiffxptr = Fdiffx;
//
//        PetscScalar *eulerSource; xDMPlexPointLocalRef(dm, cell, eulerfID, fArray, &eulerSource);
//        PetscScalar *rhophiSource; xDMPlexPointLocalRef(dm, cell, densityVFField.id, fArray, &rhophiSource);// std::cout << "cell   " << cell << "  phi   " << *phik << "   rhs of rhophi equation   " << *rhophiSource << "\n";// *rhophi += 0; std::cout << "  xnew  " << *rhophi << "\n";
//        PetscScalar rhog; const PetscScalar *rhogphig; xDMPlexPointLocalRead(dm, cell, densityVFField.id, solArray, &rhogphig);
//        if(*rhogphig > 1e-10){rhog = *rhogphig / *phik;}else{rhog = 0;}
//
//        *rhophiSource += rhog*div + (0*Fdiffy*Fdiffz);
////        *rhophiSource += density*div;
//
//        PetscScalar gradrhogphi[dim];
//        PetscScalar gradphi[dim];
//        PetscScalar gradrhog[dim];
//        if (dim == 1) {
//            PetscInt nNeighbors, *neighbors;
//            DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);//
//            const PetscReal *rhogphikm1; xDMPlexPointLocalRead(dm, neighbors[0], densityVFField.id, solArray, &rhogphikm1);
//            const PetscReal *rhogphikp1; xDMPlexPointLocalRead(dm, neighbors[2], densityVFField.id, solArray, &rhogphikp1);
//            const PetscReal *phikm1; xDMPlexPointLocalRead(auxDM, neighbors[0], phiTildeField.id, auxArray, &phikm1);
//            const PetscReal *phikp1; xDMPlexPointLocalRead(auxDM, neighbors[2], phiTildeField.id, auxArray, &phikp1);
//            gradrhogphi[0] = (*rhogphikp1 - *rhogphikm1) / (2 * process->epsilon);
//            gradphi[0] = (*phikp1 - *phikm1) / (2 * process->epsilon);
//            if(*phik>1e-2){
//                gradrhog[0] = 0*(gradrhogphi[0] - rhog*gradphi[0])/(*phik);
//            }
//            else{
//                gradrhog[0]=0;
//            }
//            DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
//        }
//        else{
//            DMPlexCellGradFromCell(dm, cell, locX, densityVFField.id, 0, gradrhog);
//            //this will be gradrhophi, not gradrho; need to change it
//        }
//        *rhophiSource += 0*gradrhog[0]*ux;
//        if (dim>1){*rhophiSource += 0*gradrhog[1]*uy;}
//        if (dim>2){*rhophiSource += 0*gradrhog[2]*uz;}
//
////        PetscScalar *phiSource; xDMPlexPointLocalRef(dm, cell, phiField.id, fArray, &phiSource); std::cout << "cell   " << cell << "   phiSource  " << *phiSource;
////        *phiSource += div; std::cout << "  xnew  " << *phiSource << "\n";
//
////        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU] += div;
////        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += div*ux;
//
////        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU] += Fdiffx;
////        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += Fdiffx*ux;
//
////        if (dim>1){
////            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOV] += Fdiffy;
////            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += Fdiffy*uy;}
////        if (dim>2){
////            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOW] += Fdiffz;
////            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += Fdiffz*uz;
////        }
//
////        std::cout << cell << "  " << *phik << "    " << density << "   " << ux << "   " << div << "    " << mdiff << "\n";
//
//    }
    subDomain->UpdateAuxLocalVector();

    //destroy vecs

    VecRestoreArray(divaLocalVec, &divaLocalArray);
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

    VecRestoreArray(vxLocalVec, &vxLocalArray);
    DMRestoreLocalVector(vxDM, &vxLocalVec);
    DMRestoreGlobalVector(vxDM, &vxGlobalVec);
    DMDestroy(&vxDM);

    VecRestoreArray(vyLocalVec, &vyLocalArray);
    DMRestoreLocalVector(vyDM, &vyLocalVec);
    DMRestoreGlobalVector(vyDM, &vyGlobalVec);
    DMDestroy(&vyDM);

    VecRestoreArray(aLocalVec, &aLocalArray);
    DMRestoreLocalVector(aDM, &aLocalVec);
    DMRestoreGlobalVector(aDM, &aGlobalVec);
    DMDestroy(&aDM);

    //clean up. necessary?
//    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
//        const PetscInt cell = cellRange.GetPoint(i);
//        PetscReal *divaptr; xDMPlexPointLocalRef(auxDM, cell, divaField.id, auxArray, &divaptr);
//        PetscReal *Mask; xDMPlexPointLocalRef(auxDM, cell, ISMaskField.id, auxArray, &Mask);
//        if (*Mask < 0.99){*Mask = 0;}
//    }

    // cleanup
    VecRestoreArrayRead(locX, &solArray);
    VecRestoreArray(auxVec, &auxArray);
    VecRestoreArray(vertexVec, &vertexArray);
    VecRestoreArray(locFVec, &fArray);
    solver.RestoreRange(cellRange);

    DMRestoreLocalVector(process->vertexDM, &vertexVec);

//    VecDestroy(&auxVec);
//    VecDestroy(&locX);
    VecDestroy(&vertexVec); //<--- this is fine
//    VecDestroy(&locFVec); //<--- SEGV 11


    PetscFunctionReturn(0);

}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::IntSharp, "calculates interface regularization term",
         ARG(PetscReal, "Gamma", "Gamma, velocity scale parameter (approx. umax)"),
         ARG(PetscReal, "epsilon", "epsilon, interface thickness scale parameter (approx. h)")
);
