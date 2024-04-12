#include "surfaceForce.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"

ablate::finiteVolume::processes::SurfaceForce::SurfaceForce(PetscReal sigma) : sigma(sigma) {}

ablate::finiteVolume::processes::SurfaceForce::~SurfaceForce() { DMDestroy(&vertexDM) >> utilities::PetscUtilities::checkError; }

void ablate::finiteVolume::processes::SurfaceForce::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
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

    flow.RegisterRHSFunction(ComputeSource, this);
}

void PhiNeighborGaussWeight(PetscReal d, PetscReal s, PetscReal *weight){
    PetscReal pi = 3.14159265358979323846264338327950288419716939937510;
    
    PetscReal Coeff = 1/(PetscSqrtReal(2*pi)*s);

    PetscReal g0 = Coeff*PetscExpReal(0/ (2*PetscSqr(s)));
    PetscReal gd = Coeff*PetscExpReal(-PetscSqr(d)/ (2*PetscSqr(s)));
    *weight = gd/g0;
}

void Get3DCoordinate(DM dm, PetscInt p, PetscReal *xp, PetscReal *yp, PetscReal *zp){
    //get the coordinates of the point
    PetscReal vol;
    PetscReal centroid[3];
    DMPlexComputeCellGeometryFVM(dm, p, &vol, centroid, nullptr);
    *xp = centroid[0];
    *yp = centroid[1];
    *zp = centroid[2];
}

void GetVertexRange(DM dm, const std::shared_ptr<ablate::domain::Region> &region, ablate::domain::Range &vertexRange) {
    PetscInt depth=0; //zeroth layer of DAG is always that of the vertices
    ablate::domain::GetRange(dm, region, depth, vertexRange);
}

PetscErrorCode ablate::finiteVolume::processes::SurfaceForce::ComputeSource(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx) {
    PetscFunctionBegin;
    // create space for normal vertex
    Vec normalVertex;
    auto process = (ablate::finiteVolume::processes::SurfaceForce *)ctx;

    std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;

    PetscCall(DMGetLocalVector(process->vertexDM, &normalVertex));
    PetscScalar *array;
    PetscCall(VecGetArray(normalVertex, &array));
    // march over all vertex for normal calculation with cell data and store on vertex
    PetscInt vStart, vEnd;
    // extract vertices of domain
    PetscCall(DMPlexGetDepthStratum(process->vertexDM, 0, &vStart, &vEnd));
    auto dim = solver.GetSubDomain().GetDimensions();

    const auto &phiField = subDomain->GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
    const auto &phiTildeField = subDomain->GetField("phiTilde");
    const auto &phiTildeStructuredField = subDomain->GetField("phiTildeStructured");
    const auto &kappaField = subDomain->GetField("kappa");
    const auto &n0Field = subDomain->GetField("n0");
    const auto &n1Field = subDomain->GetField("n1");
    const auto &n2Field = subDomain->GetField("n2");

    const auto &eulerField = solver.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);

    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector();
    const PetscScalar *solArray;
    PetscScalar *auxArray;

    // march over each cell
    ablate::domain::Range cellRange; ablate::domain::Range vertexRange;
    solver.GetCellRangeWithoutGhost(cellRange);
    PetscScalar *fArray;

    GetVertexRange(dm, ablate::domain::Region::ENTIREDOMAIN, vertexRange);

    PetscCall(VecGetArray(locFVec, &fArray));
    PetscCall(VecGetArrayRead(locX, &solArray));
    VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;

    static PetscReal artificialsubdomain = 1.25;


    PetscReal h; DMPlexGetMinRadius(dm, &h); //h*=3;
    
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.GetPoint(c);
        const PetscScalar *phic;
        xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic) >> ablate::utilities::PetscUtilities::checkError;
        PetscInt nNeighbors, *neighbors;
        DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        
        PetscReal xc, yc, zc;
        Get3DCoordinate(dm, cell, &xc, &yc, &zc);
        //        std::cout << "\n --------- cell " << cell << " -------start--";
        //        std::cout << "\n coordinate=  ("<<xc<<", "<<yc<<")";

        PetscReal weightedphi=0;
        PetscReal avgphi=0;
        PetscReal Tw=0;

        PetscScalar *phiTilde;
        xDMPlexPointLocalRef(auxDM, cell, phiTildeField.id, auxArray, &phiTilde) >> ablate::utilities::PetscUtilities::checkError;

        PetscScalar *phiTildeStructured;
        xDMPlexPointLocalRef(auxDM, cell, phiTildeStructuredField.id, auxArray, &phiTildeStructured) >> ablate::utilities::PetscUtilities::checkError;

        if ( abs(xc) >= artificialsubdomain or abs(yc) >= artificialsubdomain or abs(zc) >= artificialsubdomain){
            *phiTilde=0;
        }
        else {
            for (PetscInt j = 0; j < nNeighbors; ++j) {
                PetscInt neighbor = neighbors[j];
                //                if (cell != neighbor) {
                PetscReal *phin;
                xDMPlexPointLocalRead(dm, neighbor, phiField.id, solArray, &phin);

                PetscReal xn, yn, zn;
                Get3DCoordinate(dm, neighbor, &xn, &yn, &zn);

                PetscReal d = PetscSqrtReal(PetscSqr(xn-xc) + PetscSqr(yn-yc) + PetscSqr(zn-zc)); //distance

                PetscReal s = 1.698643600577*h; //stdev of our phitilde smoothing, NOT our convolution smoothing
                //                    PetscReal s = 3*h; //stdev of our phitilde smoothing, NOT our convolution smoothing

                PetscReal wn; PhiNeighborGaussWeight(d, s, &wn);
                Tw += wn;

                weightedphi += (*phin*wn);

                if (cell == neighbor){
                    avgphi += 4**phin;
                }
                if (abs(d - 2*h) < PETSC_SMALL){ //cardinal neightbor
                    avgphi += 2**phin;
                }
                if (abs(d - 2* sqrt(2)*h) < PETSC_SMALL){ //diagonal neighbor
                    avgphi += *phin;
                }

                //                }
            }
            weightedphi /= Tw;
            avgphi /= 16;
            *phiTilde = weightedphi; //*phiTilde = (0.5 * (*phic)) + (0.5 * (avgphi));
            *phiTildeStructured = avgphi;

        }
    }

//    //march over vertices
//    for (PetscInt j = vStart; j<vEnd; ++j){
//        const PetscInt vertex = vertexRange.points ? vertexRange.points[j] : j;
//
//        //get gradphi at vertices
//        PetscScalar gradphi_v[dim];
//        DMPlexVertexGradFromCell(dm, vertex, locX, phiTildeField.id, 0, gradphi_v);
//
//        double epsmach = 1e-52;
//        PetscReal normgradphi_v =0.0;
//
//        for (int k=0; k<dim; k++){
//            normgradphi_v += pow(gradphi_v[k], 2);
//        }
//        normgradphi_v = pow(normgradphi_v, 0.5);
//
//        //get n at vertices and write
//        PetscScalar norm_v[dim];
//        PetscReal *nptr;
//
//        for (int k=0; k<dim; k++){
//            norm_v[k] = (gradphi_v[k]+epsmach)/(normgradphi_v+epsmach);
//            xDMPlexPointLocalRef(auxDM, vertex, nField.id, auxArray, &nptr);
//            nptr[k] = norm_v[k];
//        }
//    }
//
//    for (PetscInt i = cellRange.start; i<cellRange.end; ++i) {
//        const PetscInt cell = cellRange.points ? cellRange.points[i] : i;  // cell ID
//
//        const PetscReal *phicenter;
//        xDMPlexPointLocalRead(dm, cell, phiTildeField.id, solArray, &phicenter);
//        auto phi_c = *phicenter;
//
//        PetscScalar gradphi_c[dim];
//        PetscScalar norm_c[dim];
//        // get gradphi_c
//        DMPlexCellGradFromCell(dm, cell, locX, phiTildeField.id, 0, gradphi_c);
//
//        // get n_c (local), kappa; only evaluate at cut cells (0.1<phi<0.9
//        PetscReal normgradphi_c = 0.0;
//        PetscReal kappa = 0.0;
//
//        if (phi_c > 0.1 and phi_c < 0.9) {  // cut cell
//            for (int k = 0; k < dim; k++) {
//                normgradphi_c += pow(gradphi_c[k], 2);
//            }
//            normgradphi_c = pow(normgradphi_c, 0.5);
//            double epsmach = 1e-52;
//
//            PetscReal Nx = (gradphi_c[0] + epsmach) / (normgradphi_c + epsmach);
//            PetscReal Ny = (gradphi_c[1] + epsmach) / (normgradphi_c + epsmach);
//            PetscReal Nz = (gradphi_c[2] + epsmach) / (normgradphi_c + epsmach);
//
//            //            for (int k=0; k<dim; k++){
//            //                norm_c[k] = (gradphi_c[k]+epsmach)/(normgradphi_c+epsmach);
//            //            }
//
//            for (PetscInt offset = 0; offset < dim; offset++) {
//                PetscReal nabla_n[dim];
//                DMPlexCellGradFromVertex(process->vertexDM, cell, normalVertex, 0, offset, nabla_n);
//                kappa += nabla_n[offset];
//            }
//
//            PetscScalar *kappaptr, *n0ptr, *n1ptr, *n2ptr;
//            xDMPlexPointLocalRef(auxDM, cell, kappaField.id, auxArray, &kappaptr) >> ablate::utilities::PetscUtilities::checkError;
//            xDMPlexPointLocalRef(auxDM, cell, n0Field.id, auxArray, &n0ptr) >> ablate::utilities::PetscUtilities::checkError;
//            xDMPlexPointLocalRef(auxDM, cell, n1Field.id, auxArray, &n1ptr) >> ablate::utilities::PetscUtilities::checkError;
//            xDMPlexPointLocalRef(auxDM, cell, n2Field.id, auxArray, &n2ptr) >> ablate::utilities::PetscUtilities::checkError;
//
//            *kappaptr = kappa;
//            *n0ptr = Nx;
//            *n1ptr = Ny;
//            *n2ptr = Nz;
//        }
//        if (not(phi_c > 0.1 and phi_c < 0.9)) {  // not cut cell
//            for (int k = 0; k < dim; k++) {
//                norm_c[k] = 0;
//            }
//            kappa = 0;
//        }
//    }

    
    for (PetscInt v = vStart; v < vEnd; v++) {
        PetscScalar *gradPhi_v;
        PetscCall(DMPlexPointLocalFieldRef(process->vertexDM, v, 0, array, &gradPhi_v));
        DMPlexVertexGradFromCell(auxDM, v, auxVec, phiTildeField.id, 0, gradPhi_v);
        if (utilities::MathUtilities::MagVector(dim, gradPhi_v) > 1e-10) {
            utilities::MathUtilities::NormVector(dim, gradPhi_v);
        }
    }


    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
        // compute divergence of normal with normal at vertex to get curvature
        const PetscInt cell = cellRange.GetPoint(i);
        PetscReal kappa = 0.0;
        PetscReal nabla_n[dim];
        for (PetscInt offset = 0; offset < dim; offset++) {
            DMPlexCellGradFromVertex(process->vertexDM, cell, normalVertex, 0, offset, nabla_n);
            kappa += nabla_n[offset];
        }

        // compute normal at cell center with phi in cell center
        PetscScalar gradPhi_c[dim];
        DMPlexCellGradFromCell(auxDM, cell, auxVec, phiTildeField.id, 0, gradPhi_c);
        if (utilities::MathUtilities::MagVector(dim, gradPhi_c) > 1e-10) {
            utilities::MathUtilities::NormVector(dim, gradPhi_c);
        }

        PetscReal Nx = gradPhi_c[0];
        PetscReal Ny = gradPhi_c[1];
        PetscReal Nz = gradPhi_c[2];

        PetscScalar *kappaptr, *n0ptr, *n1ptr, *n2ptr;
        xDMPlexPointLocalRef(auxDM, cell, kappaField.id, auxArray, &kappaptr) >> ablate::utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRef(auxDM, cell, n0Field.id, auxArray, &n0ptr) >> ablate::utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRef(auxDM, cell, n1Field.id, auxArray, &n1ptr) >> ablate::utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRef(auxDM, cell, n2Field.id, auxArray, &n2ptr) >> ablate::utilities::PetscUtilities::checkError;

        *kappaptr = kappa;
        *n0ptr = Nx;
        *n1ptr = Ny;
        *n2ptr = Nz;



        // compute surface force sigma * cur * normal and add to local F vector

        const PetscScalar *euler = nullptr;
        PetscScalar *eulerSource = nullptr;
        PetscCall(DMPlexPointLocalFieldRef(dm, cell, eulerField.id, fArray, &eulerSource));
        PetscCall(DMPlexPointLocalFieldRead(dm, cell, eulerField.id, solArray, &euler));
        auto density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];

        for (PetscInt k = 0; k < dim; ++k) {
            // calculate surface force and energy
            PetscReal surfaceForce = process->sigma * kappa * gradPhi_c[k];
            PetscReal vel = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + k] / density;
            // add in the contributions
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + k] += surfaceForce;
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += surfaceForce * vel;
        }
    }

    // cleanup
    PetscCall(DMRestoreLocalVector(process->vertexDM, &normalVertex));
    PetscCall(VecRestoreArray(locFVec, &fArray));
    PetscCall(VecRestoreArrayRead(locX, &solArray));
    solver.RestoreRange(cellRange);

    PetscFunctionReturn(0);
}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::SurfaceForce, "calculates surface tension force and adds source terms",
         ARG(PetscReal, "sigma", "sigma, surface tension coefficient"));
