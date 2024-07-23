#include "domain/RBF/mq.hpp"
#include "intSharp.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"

//void GetVertexRange(DM dm, const std::shared_ptr<ablate::domain::Region> &region, ablate::domain::Range &vertexRange) {
//    PetscInt depth=0; //zeroth layer of DAG is always that of the vertices
//    ablate::domain::GetRange(dm, region, depth, vertexRange);
//}

void GetCoordinate(DM dm, PetscInt dim, PetscInt p, PetscReal *xp, PetscReal *yp, PetscReal *zp){
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
void ablate::finiteVolume::processes::IntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
    IntSharp::subDomain = solver.GetSubDomainPtr();
}

//void ablate::domain::SubDomain::UpdateAuxLocalVector() {
//    if (auxDM) {
//        DMLocalToGlobal(auxDM, auxLocalVec, INSERT_VALUES, auxGlobalVec) >> utilities::PetscUtilities::checkError;
//        DMGlobalToLocal(auxDM, auxGlobalVec, INSERT_VALUES, auxLocalVec) >> utilities::PetscUtilities::checkError;
//    }
//}

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
    const auto &phiField = subDomain->GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
    const auto &eulerField = solver.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);
    const auto &ISMaskField = subDomain->GetField("intsharpMask");
    const auto &divaField = subDomain->GetField("diva");
    const auto &mdiffField = subDomain->GetField("mdiff");
    const auto &FdiffxField = subDomain->GetField("Fdiffx");
    const auto &phiTildeField = subDomain->GetField("ISphiTilde");
    const auto &phiTildeMaskField = subDomain->GetField("ISphiTildeMask");
    const auto &gradphiField = subDomain->GetField("gradphi");
    const auto &gradrhoField = subDomain->GetField("gradrho");
    const auto &densityVFField = subDomain->GetField("densityvolumeFraction");

//    auto phifID = phiField.id;
    auto eulerfID = eulerField.id;

    // get vecs/arrays

    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector(); //these do not have a corresponding restore call

    Vec vertexVec; DMGetLocalVector(process->vertexDM, &vertexVec); //
    const PetscScalar *solArray; VecGetArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError; //
    PetscScalar *auxArray; VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError; //
    PetscScalar *vertexArray; VecGetArray(vertexVec, &vertexArray); //
    PetscScalar *fArray; PetscCall(VecGetArray(locFVec, &fArray)); //

    // get ranges
    ablate::domain::Range cellRange; solver.GetCellRangeWithoutGhost(cellRange); //
    PetscInt vStart, vEnd; DMPlexGetDepthStratum(process->vertexDM, 0, &vStart, &vEnd); //

    //Initialize Mask field
    // ISMask determines which cells receive a nonzero interface regularization term value
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.GetPoint(c); //
        PetscScalar *Mask; xDMPlexPointLocalRef(auxDM, cell, ISMaskField.id, auxArray, &Mask); *Mask = 0;
    }
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.GetPoint(c); //
        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic) >> ablate::utilities::PetscUtilities::checkError;
        if (*phic > 0.0001 and *phic < 0.9999) {
            PetscInt nNeighbors, *neighbors; DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors); //
            for (PetscInt j = 0; j < nNeighbors; ++j) {
                PetscInt neighbor = neighbors[j];
                PetscScalar *Mask; xDMPlexPointLocalRef(auxDM, neighbor, ISMaskField.id, auxArray, &Mask); *Mask = 1;
            }
            DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        }
    }

    //Initialize phiTildeMask
    // phiTildeMask determines which cells receive a nonzero phiTilde value
    // this is smoothed to a greater extent than the intsharp field. (this might need to change)

    PetscReal rmin; DMPlexGetMinRadius(dm, &rmin); PetscReal h=2*rmin;
    PetscScalar C=2; PetscScalar N=2.6; PetscInt layers = ceil(C*N);
    layers = 4; //temporary
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscScalar *phiTildeMask; xDMPlexPointLocalRef(auxDM, cell, phiTildeMaskField.id, auxArray, &phiTildeMask); *phiTildeMask = 0;
    }
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.GetPoint(c);
        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic) >> ablate::utilities::PetscUtilities::checkError;
        if (*phic > 0.0001 and *phic < 0.9999) {
            PetscInt nNeighbors, *neighbors; DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors); //
            for (PetscInt j = 0; j < nNeighbors; ++j) {
                PetscInt neighbor = neighbors[j];
                PetscScalar *phiTildeMask; xDMPlexPointLocalRef(auxDM, neighbor, phiTildeMaskField.id, auxArray, &phiTildeMask); *phiTildeMask = 1;
            }
            DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        }
    }

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.GetPoint(c);
        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic) >> ablate::utilities::PetscUtilities::checkError;
        PetscReal xc, yc, zc; GetCoordinate(dm, dim, cell, &xc, &yc, &zc);

        //now build phitilde
        //number of smoothing layers

        PetscScalar *phiTilde; xDMPlexPointLocalRef(auxDM, cell, phiTildeField.id, auxArray, &phiTilde) >> ablate::utilities::PetscUtilities::checkError;
        PetscScalar *phiTildeMask; xDMPlexPointLocalRef(auxDM, cell, phiTildeMaskField.id, auxArray, &phiTildeMask) >> ablate::utilities::PetscUtilities::checkError;
        if (*phiTildeMask == 0){
            *phiTilde=*phic;
        }
        else{
            PetscInt nNeighbors, *neighbors; DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
            PetscReal weightedphi = 0; PetscReal Tw = 0;

            for (PetscInt j = 0; j < nNeighbors; ++j) {
                PetscInt neighbor = neighbors[j];

                PetscReal *phin; xDMPlexPointLocalRead(dm, neighbor, phiField.id, solArray, &phin);

                PetscReal xn, yn, zn; GetCoordinate(dm, dim, neighbor, &xn, &yn, &zn);
                PetscReal d = PetscSqrtReal(PetscSqr(xn - xc) + PetscSqr(yn - yc) + PetscSqr(zn - zc));  // distance
                PetscReal s = C * h; //6*h

                PetscReal wn; PhiNeighborGauss(d, s, &wn);
                Tw += wn;
                weightedphi += (*phin * wn);
            }
            weightedphi /= Tw;
            *phiTilde = weightedphi;
            DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        }
    }

    //march over vertices
    for (PetscInt vertex = vStart; vertex < vEnd; vertex++) {

        //if all of the vertex's cell neighbors are not in Mask, don't bother with calculation
        PetscReal vx, vy, vz; GetCoordinate(dm, dim, vertex, &vx, &vy, &vz);
        PetscInt nvn, *vertexneighbors; DMPlexVertexGetCells(dm, vertex, &nvn, &vertexneighbors); //
        PetscBool isAdjToMask = PETSC_FALSE;
        for (PetscInt k = 0; k < nvn; k++){
            //changed to phitilde mask
            PetscScalar *Mask; xDMPlexPointLocalRef(auxDM, vertexneighbors[k], phiTildeMaskField.id, auxArray, &Mask);// >> ablate::utilities::PetscUtilities::checkError;
//            PetscScalar *Mask; xDMPlexPointLocalRef(auxDM, vertexneighbors[k], ISMaskField.id, auxArray, &Mask);// >> ablate::utilities::PetscUtilities::checkError;
            if (*Mask > 0.5){
                isAdjToMask = PETSC_TRUE;
            }
        }

        PetscScalar gradphiv[dim];
        PetscReal normgradphi = 0.0;
        if (isAdjToMask == PETSC_TRUE){

            if (dim==1){
                //changed to phitilde
                PetscScalar *phikm1; xDMPlexPointLocalRef(auxDM, vertexneighbors[0], phiTildeField.id, auxArray, &phikm1);
                PetscScalar *phikp1; xDMPlexPointLocalRef(auxDM, vertexneighbors[1], phiTildeField.id, auxArray, &phikp1);
                PetscReal xm1; GetCoordinate1D(dm, dim, vertexneighbors[0], &xm1);
                PetscReal xp1; GetCoordinate1D(dm, dim, vertexneighbors[1], &xp1);
                gradphiv[0]=(*phikp1 - *phikm1)/(1*process->epsilon);
            }
            else{
                DMPlexVertexGradFromCell(auxDM, vertex, auxVec, phiTildeField.id, 0, gradphiv);
            }

            //get gradphi at vertices (gradphiv) based on cell centered phis
            for (int k=0; k<dim; ++k){
                normgradphi += PetscSqr(gradphiv[k]);
            }
            normgradphi = PetscSqrtReal(normgradphi);
        }
        else{
            for (int k=0; k<dim; ++k){
                gradphiv[k] =0;
            }
        }

        PetscReal phiv=0;
        PetscReal Uv[dim];

        if(isAdjToMask == PETSC_TRUE) {
            PetscReal distances[nvn];
            PetscReal shortestdistance = ablate::utilities::Constants::large;
            for (PetscInt k = 0; k < nvn; ++k) {

                for (int j=0; j<dim; ++j){
                    Uv[j] = 0;
                }

                PetscInt neighbor = vertexneighbors[k];
                PetscReal nx, ny, nz;
                GetCoordinate(dm, dim, neighbor, &nx, &ny, &nz);

                PetscReal distance = PetscSqrtReal(PetscSqr(nx - vx) + PetscSqr(ny - vy) + PetscSqr(nz - vz));
                if (distance < shortestdistance) {
                    shortestdistance = distance;
                }
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
            for (PetscInt k = 0; k < nvn; ++k) {
                weights[k] = weights_wrt_short[k] / totalweight_wrt_short;
            }

            for (PetscInt k = 0; k < nvn; ++k) {
                PetscInt neighbor = vertexneighbors[k];
                //changed to phitilde
                PetscReal *phineighbor; xDMPlexPointLocalRef(auxDM, neighbor, phiTildeField.id, auxArray, &phineighbor);
                PetscReal *eulerneighbor; xDMPlexPointLocalRead(dm, neighbor, eulerField.id, solArray, &eulerneighbor);
                PetscReal Uneighbor[3] = {eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHOU]/eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHO],
                                          eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHOV]/eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHO],
                                          eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHOW]/eulerneighbor[ablate::finiteVolume::CompressibleFlowFields::RHO]};
                for (int j=0; j<dim; ++j){
                    Uv[j] += (Uneighbor[j]) * (weights[k]);
                }
                phiv += (*phineighbor) * (weights[k]);  // unstructured case
            }
        }
        else{
            phiv=0;
        }

        //get a at vertices (av) (Chiu 2011)
        //based on Eq. 1 of:   Jain SS. Accurate conservative phase-field method for simulation of two-phase flows. Journal of Computational Physics. 2022 Nov 15;469:111529.
        PetscScalar  av[dim]; //PetscScalar  Uv[dim];
        PetscReal *avptr;
        xDMPlexPointLocalRef(process->vertexDM, vertex, 0, vertexArray, &avptr); //vertexDM

        for (int k=0; k<dim; ++k){
            if(isAdjToMask == PETSC_TRUE) {
                if (normgradphi > ablate::utilities::Constants::tiny) {
                    av[k] = (process->Gamma * process->epsilon * gradphiv[k]) - (process->Gamma * phiv * (1 - phiv) * (gradphiv[k] / normgradphi));
                } else {
                    av[k] = (process->Gamma * process->epsilon * gradphiv[k]) - (process->Gamma * phiv * (1 - phiv) * gradphiv[k]);
                }
                avptr[k] = av[k];
            }
            else{
                avptr[k]=0;
            }
        }
        DMPlexVertexRestoreCells(dm, vertex, &nvn, &vertexneighbors);
    }

    // march over cells
    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {

        const PetscInt cell = cellRange.GetPoint(i);

        PetscReal div=0.0;
        //changed to phitilde
//        PetscScalar *Mask; xDMPlexPointLocalRef(auxDM, cell, phiTildeMaskField.id, auxArray, &Mask);
        PetscScalar *Mask; xDMPlexPointLocalRef(auxDM, cell, ISMaskField.id, auxArray, &Mask);// >> ablate::utilities::PetscUtilities::checkError;
        if (*Mask > 0.5){
            if (dim==1){
                PetscInt nVerts, *verts; DMPlexCellGetVertices(dm, cell, &nVerts, &verts); //
                PetscScalar *am1; xDMPlexPointLocalRef(process->vertexDM, verts[0], 0, vertexArray, &am1);
                PetscScalar *ap1; xDMPlexPointLocalRef(process->vertexDM, verts[1], 0, vertexArray, &ap1);
//                PetscReal xm1, ym1, zm1; GetCoordinate(dm, dim, verts[0], &xm1, &ym1, &zm1);
//                PetscReal xp1, yp1, zp1; GetCoordinate(dm, dim, verts[1], &xp1, &yp1, &zp1);
                div = (*ap1-*am1)/(1*process->epsilon);
//                std::cout << "  cell " << cell << "  neighbor vert  " << verts[0] << "  a  " << *am1 << "   div   " << div << "\n";
                DMPlexCellRestoreVertices(dm, cell, &nVerts, &verts);
            }
            else {
                for (PetscInt offset = 0; offset < dim; offset++) {
                    PetscReal nabla_ai[dim];
                    DMPlexCellGradFromVertex(process->vertexDM, cell, vertexVec, 0, offset, nabla_ai);
                    div += nabla_ai[offset];
//                                    div=0;
                }
            }
        }
        else{
            div=0;
        }

        PetscReal *divaptr; xDMPlexPointLocalRef(auxDM, cell, divaField.id, auxArray, &divaptr); *divaptr = div;

        const PetscScalar *euler = nullptr; xDMPlexPointLocalRead(dm, cell, eulerfID, solArray, &euler);


        //Fdiff = mdiff v_m + \rho v_n v_{m,n}
        //2d: Fdiffx = mdiff vx + \rho (vx vx,x + vy vx,y)
        PetscScalar density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];
        PetscReal ux = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU] / density;
        PetscReal uy = euler[ablate::finiteVolume::CompressibleFlowFields::RHOV] / density;
        PetscReal uz = euler[ablate::finiteVolume::CompressibleFlowFields::RHOW] / density;
        PetscReal U[3] = {ux, uy, uz};

//        std::cout << "cell " << cell << "   ux  " << ux << "\n";

        const PetscReal *phik; xDMPlexPointLocalRead(auxDM, cell, phiTildeField.id, auxArray, &phik);
        PetscReal mdiff=0; PetscReal Fdiffx=0; PetscReal Fdiffy=0; PetscReal Fdiffz=0;

        PetscReal *phiTildeMask; xDMPlexPointLocalRef(auxDM, cell, phiTildeMaskField.id, auxArray, &phiTildeMask);
        if (*phiTildeMask>0.5) {
            // add mdiff, etc. to momentum, energy equations; see local AlphaEq.pdf, SharpeningForce.pdf

            PetscScalar gradrho[dim];//  for (int k=0; k<dim; ++k){gradrho[k]=0;}
            PetscScalar gradrhou[dim];//  for (int k=0; k<dim; ++k){gradrhou[k]=0;}
            PetscScalar gradrhov[dim];//  for (int k=0; k<dim; ++k){gradrhou[k]=0;}
            PetscScalar gradrhow[dim];//  for (int k=0; k<dim; ++k){gradrhou[k]=0;}
            PetscScalar gradphi[dim];// for (int k=0; k<dim; ++k){gradphi[k]=0;}
            PetscScalar gradux[dim];// for (int k=0; k<dim; ++k){gradux[k]=0;}
            PetscScalar graduy[dim];// for (int k=0; k<dim; ++k){graduy[k]=0;}
            PetscScalar graduz[dim];// for (int k=0; k<dim; ++k){graduz[k]=0;}
            //

            if (dim == 1) {

                PetscInt nNeighbors, *neighbors;
                DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
//                const PetscReal *phikm1;xDMPlexPointLocalRead(dm, neighbors[0], phiField.id, solArray, &phikm1);
//                const PetscReal *phikp1;xDMPlexPointLocalRead(dm, neighbors[2], phiField.id, solArray, &phikp1);
                const PetscReal *phikm1;xDMPlexPointLocalRead(auxDM, neighbors[0], phiTildeField.id, auxArray, &phikm1);
                const PetscReal *phikp1;xDMPlexPointLocalRead(auxDM, neighbors[2], phiTildeField.id, auxArray, &phikp1);

//                std::cout << "cell    " << cell <<  "   phikpm1  " << *phikm1 << "   " << *phikp1 << "\n";

                const PetscScalar *eulerkm1; xDMPlexPointLocalRead(dm, neighbors[0], eulerfID, solArray, &eulerkm1);
                const PetscScalar *eulerkp1; xDMPlexPointLocalRead(dm, neighbors[2], eulerfID, solArray, &eulerkp1);


                gradphi[0] = (*phikp1 - *phikm1) / (2 * process->epsilon);// gradphi[1]=gradphi[2]=0;
                gradrho[0] = (eulerkp1[ablate::finiteVolume::CompressibleFlowFields::RHO] - eulerkm1[ablate::finiteVolume::CompressibleFlowFields::RHO]) / (2 * process->epsilon);// gradrho[1]=gradrho[2]=0; //eulerkm1[0] = rhokm1
                gradrhou[0] = (eulerkp1[ablate::finiteVolume::CompressibleFlowFields::RHOU] - eulerkm1[ablate::finiteVolume::CompressibleFlowFields::RHOU]) / (2 * process->epsilon);// gradrhou[1]=gradrhou[2]=0; //eulerkm1[1] = rhoukm1
                gradux[0] = (gradrhou[0] - ux*gradrho[0]) / density; // gradux[1]=gradux[2]=0;
                graduy[0]=graduz[0]=0;

//                std::cout << gradux[0] << "\n";
                DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);

            } else {
//                DMPlexCellGradFromCell(dm, cell, locX, densityVFField.id, 0, gradrhophi);
                //changed to phitilde
                DMPlexCellGradFromCell(auxDM, cell, auxVec, phiTildeField.id, 0, gradphi);
                DMPlexCellGradFromCell(dm, cell, locX, eulerField.id, 0, gradrho); //offset=0 implies the first element of the euler vector, which is rho
                DMPlexCellGradFromVertex(dm, cell, locX, eulerField.id, 1, gradrhou);
                DMPlexCellGradFromVertex(dm, cell, locX, eulerField.id, 2, gradrhov);
                DMPlexCellGradFromVertex(dm, cell, locX, eulerField.id, 3, gradrhow);

                for (int k = 0; k < dim; ++k) {
                    gradux[k] = (gradrhou[k] - ux*gradrho[k]) / density;
                    graduy[k] = (gradrhov[k] - uy*gradrho[k]) / density;
                    graduz[k] = (gradrhow[k] - uz*gradrho[k]) / density;
                }
            }

            mdiff = density * div;

//            for (int k = 0; k < dim; ++k) {
//                mdiff += (*phik*U[k]*gradrho[k] - density*U[k]*gradphi[k]); //NOT density uk gradrhok, as shown in AlphaEq1
//            }
//            if ((*phik > 1e-4) and (*phik < 1-1e-4)) {mdiff /= *phik;}else{ mdiff = 0;}

            for (int k = 0; k < dim; ++k) {
                mdiff += ((*phik-0.5)*U[k]*gradrho[k] - density*U[k]*gradphi[k]); //NOT density uk gradrhok, as shown in AlphaEq1
            }
            if (PetscAbs(*phik - 0.5) > 1e-2){mdiff /= (*phik-0.5); } else{ mdiff = 0;}

            PetscReal *mdiffptr; xDMPlexPointLocalRef(auxDM, cell, mdiffField.id, auxArray, &mdiffptr); *mdiffptr = mdiff;
            PetscReal *gradrhoptr; xDMPlexPointLocalRef(auxDM, cell, gradrhoField.id, auxArray, &gradrhoptr); *gradrhoptr = gradrho[0];
            PetscReal *gradphiptr; xDMPlexPointLocalRef(auxDM, cell, gradphiField.id, auxArray, &gradphiptr); *gradphiptr = gradphi[0];

            Fdiffx += mdiff*ux; Fdiffy += mdiff*uy; Fdiffz += mdiff*uz;

//            for (int j = 0; j < dim; ++j) {
//                std::cout << cell << "   ux,xi  " << gradux[j] << "   " << U[j] << "\n";
//                std::cout << cell << "   uy,xi  " << graduy[j] << "\n";
//                std::cout << cell << "   uz,xi  " << graduz[j] << "\n";
//            }

            for (int j = 0; j < dim; ++j) {
                Fdiffx += density*(U[j]*gradux[j]);
                Fdiffy += density*(U[j]*graduy[j]);
                Fdiffz += density*(U[j]*graduz[j]);
            }

//            std::cout << "--->" << Fdiffx << "    " << mdiff*ux << "     " << density*(U[0]*gradux[0]) << "\n";

        }


//        if (*Mask > 0.5){std::cout << "cell   " << cell << "  ux   " << ux << "\n";}

        PetscReal *Fdiffxptr; xDMPlexPointLocalRef(auxDM, cell, FdiffxField.id, auxArray, &Fdiffxptr); *Fdiffxptr = Fdiffx;

        PetscScalar *eulerSource; xDMPlexPointLocalRef(dm, cell, eulerfID, fArray, &eulerSource);

        PetscScalar *rhophiSource; xDMPlexPointLocalRef(dm, cell, densityVFField.id, fArray, &rhophiSource);// std::cout << "cell   " << cell << "  phi   " << *phik << "   rhs of rhophi equation   " << *rhophiSource << "\n";// *rhophi += 0; std::cout << "  xnew  " << *rhophi << "\n";

        PetscScalar rhog; const PetscScalar *rhogphig; xDMPlexPointLocalRead(dm, cell, densityVFField.id, solArray, &rhogphig);
        if(*rhogphig > 1e-10){rhog = *rhogphig / *phik;}else{rhog = 0;}

        *rhophiSource += rhog*div + (0*Fdiffy*Fdiffz);
//        *rhophiSource += density*div;

        PetscScalar gradrhogphi[dim];
        PetscScalar gradphi[dim];
        PetscScalar gradrhog[dim];
        if (dim == 1) {
            PetscInt nNeighbors, *neighbors;
            DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);//
            const PetscReal *rhogphikm1;xDMPlexPointLocalRead(dm, neighbors[0], densityVFField.id, solArray, &rhogphikm1);
            const PetscReal *rhogphikp1;xDMPlexPointLocalRead(dm, neighbors[2], densityVFField.id, solArray, &rhogphikp1);
            const PetscReal *phikm1;xDMPlexPointLocalRead(auxDM, neighbors[0], phiTildeField.id, auxArray, &phikm1);
            const PetscReal *phikp1;xDMPlexPointLocalRead(auxDM, neighbors[2], phiTildeField.id, auxArray, &phikp1);
            gradrhogphi[0] = (*rhogphikp1 - *rhogphikm1) / (2 * process->epsilon);
            gradphi[0] = (*phikp1 - *phikm1) / (2 * process->epsilon);
            if(*phik>1e-2){
                gradrhog[0] = 0*(gradrhogphi[0] - rhog*gradphi[0])/(*phik);
            }
            else{
                gradrhog[0]=0;
            }
            DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        }
        else{
            DMPlexCellGradFromCell(dm, cell, locX, densityVFField.id, 0, gradrhog);
            //this will be gradrhophi, not gradrho; need to change it
        }
        *rhophiSource += 0*gradrhog[0]*ux;
        if (dim>1){*rhophiSource += 0*gradrhog[1]*uy;}
        if (dim>2){*rhophiSource += 0*gradrhog[2]*uz;}

//        PetscScalar *phiSource; xDMPlexPointLocalRef(dm, cell, phiField.id, fArray, &phiSource); std::cout << "cell   " << cell << "   phiSource  " << *phiSource;
//        *phiSource += div; std::cout << "  xnew  " << *phiSource << "\n";

//        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU] += div;
//        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += div*ux;

//        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU] += Fdiffx;
//        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += Fdiffx*ux;

//        if (dim>1){
//            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOV] += Fdiffy;
//            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += Fdiffy*uy;}
//        if (dim>2){
//            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOW] += Fdiffz;
//            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += Fdiffz*uz;
//        }

//        std::cout << cell << "  " << *phik << "    " << density << "   " << ux << "   " << div << "    " << mdiff << "\n";

    }
    //clean up. necessary?
    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
        const PetscInt cell = cellRange.GetPoint(i);
        PetscReal *divaptr; xDMPlexPointLocalRef(auxDM, cell, divaField.id, auxArray, &divaptr);
        PetscReal *Mask; xDMPlexPointLocalRef(auxDM, cell, ISMaskField.id, auxArray, &Mask);
        if (*Mask < 0.99){*Mask = 0;}

    }

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
