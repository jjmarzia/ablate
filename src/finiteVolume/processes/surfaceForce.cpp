#include "domain/RBF/mq.hpp"
#include "surfaceForce.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "levelSet/levelSetUtilities.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"

ablate::finiteVolume::processes::SurfaceForce::SurfaceForce(PetscReal sigma) : sigma(sigma) {}
ablate::finiteVolume::processes::SurfaceForce::~SurfaceForce() { DMDestroy(&vertexDM) >> utilities::PetscUtilities::checkError; }

PetscReal GaussianDerivativeFactor(const PetscReal *x, const PetscReal s,  const PetscInt dx, const PetscInt dy, const PetscInt dz) {

    const PetscReal s2 = PetscSqr(s);

    const PetscInt derHash = 100*dx + 10*dy + dz;

    if (derHash > 0 && PetscAbsReal(s)<PETSC_SMALL) return (0.0);

    switch (derHash) {
        case   0: // Value
            return (1.0);
        case 100: // x
            return (x[0]/s2);
        case  10: // y
            return (x[1]/s2);
        case   1: // z
            return (x[2]/s2);
        case 200: // xx
            return ((x[0]*x[0] - s2)/PetscSqr(s2));
        case  20: // yy
            return ((x[1]*x[1] - s2)/PetscSqr(s2));
        case   2: // zz
            return ((x[2]*x[2] - s2)/PetscSqr(s2));
        case 110: // xy
            return (x[0]*x[1]/PetscSqr(s2));
        case 101: // xz
            return (x[0]*x[2]/PetscSqr(s2));
        case  11: // yz
            return (x[1]*x[2]/PetscSqr(s2));
        default:
            throw std::runtime_error("Unknown derivative request");
    }

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

// Calculate the curvature from a vertex-based level set field using Gaussian convolution.
// Right now this is just 2D for testing purposes.

static PetscInt FindCell(DM dm, const PetscReal x0[], const PetscInt nCells, const PetscInt cells[], PetscReal *distOut) {
    // Return the cell with the cell-center that is the closest to a given point

    PetscReal dist = PETSC_MAX_REAL;
    PetscInt closestCell = -1;
    PetscInt dim;
    DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

    for (PetscInt c = 0; c < nCells; ++c) {
        PetscReal x[dim];
        DMPlexComputeCellGeometryFVM(dm, cells[c], nullptr, x, nullptr) >> ablate::utilities::PetscUtilities::checkError;

        ablate::utilities::MathUtilities::Subtract(dim, x, x0, x);
        PetscReal cellDist = ablate::utilities::MathUtilities::MagVector(dim, x);
        if (cellDist < dist) {
            closestCell = cells[c];
            dist = cellDist;
        }
    }
    if (distOut) *distOut = dist;
    return (closestCell);
}
static PetscInt *interpCellList = nullptr;

//   Hermite-Gauss quadrature points
const PetscInt nQuad = 4; // Size of the 1D quadrature

//   The quadrature is actually sqrt(2) times the quadrature points. This is as we are integrating
//      against the normal distribution, not exp(-x^2)
const PetscReal quad[4] = {-0.74196378430272585764851359672636022482952014750891895361147387899499975465000530,
                           0.74196378430272585764851359672636022482952014750891895361147387899499975465000530,
                           -2.3344142183389772393175122672103621944890707102161406718291603341725665622712306,
                           2.3344142183389772393175122672103621944890707102161406718291603341725665622712306};

// The weights are the true weights divided by sqrt(pi)
const PetscReal weights[4] = {0.45412414523193150818310700622549094933049562338805584403605771393758003145477625,
                              0.45412414523193150818310700622549094933049562338805584403605771393758003145477625,
                              0.045875854768068491816892993774509050669504376611944155963942286062419968545223748,
                              0.045875854768068491816892993774509050669504376611944155963942286062419968545223748};


static PetscReal sigmaFactor = 6;
static PetscReal artificialsubdomain = 1.25;

void BuildInterpCellList(DM dm, const ablate::domain::Range cellRange) {
    PetscReal h;
    DMPlexGetMinRadius(dm, &h) >> ablate::utilities::PetscUtilities::checkError;
    const PetscReal sigma = sigmaFactor*2*h; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size

    PetscInt dim;
    DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

    PetscMalloc1(16*(cellRange.end - cellRange.start), &interpCellList);

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.GetPoint(c);

        PetscReal x0[dim];
        DMPlexComputeCellGeometryFVM(dm, cell, nullptr, x0, nullptr) >> ablate::utilities::PetscUtilities::checkError;

        PetscInt nCells, *cellList;
        DMPlexGetNeighbors(dm, cell, 2, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

        for (PetscInt i = 0; i < nQuad; ++i) {
            for (PetscInt j = 0; j < nQuad; ++j) {
                for (PetscInt k =0; k < nQuad; ++k) {
                    PetscReal x[3] = {x0[0] + sigma * quad[i], x0[1] + sigma * quad[j], x0[2] + sigma * quad[k]};

                    const PetscInt interpCell = FindCell(dm, x, nCells, cellList, nullptr);

                    if (interpCell < 0) {
                        int rank;
                        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
                        throw std::runtime_error("BuildInterpCellList could not determine the location of (" + std::to_string(x[0]) + ", " + std::to_string(x[1]) + ") on rank " +
                                                 std::to_string(rank) + ".");
                    }

                    interpCellList[(c - cellRange.start) * 16 + i * 4 + j] = interpCell;
                }
            }
        }

        DMPlexRestoreNeighbors(dm, cell, 2, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

    }
}

void CurvatureViaGaussian(DM dm, const PetscInt cell, Vec vec, const ablate::domain::Field *lsField, ablate::domain::rbf::MQ *rbf, const double *h, double *H, double *Nx, double *Ny, double *Nz){

    PetscInt dim;
    DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;



    //    PetscReal h;
    //    DMPlexGetMinRadius(dm, &h) >> ablate::utilities::PetscUtilities::checkError;
    //    h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size

    //  const PetscInt nQuad = 3; // Size of the 1D quadrature
    //  const PetscReal quad[] = {0.0, PetscSqrtReal(3.0), -PetscSqrtReal(3.0)};
    //  const PetscReal weights[] = {2.0/3.0, 1.0/6.0, 1.0/6.0};

    PetscReal x0[dim], vol;
    DMPlexComputeCellGeometryFVM(dm, cell, &vol, x0, nullptr) >> ablate::utilities::PetscUtilities::checkError;

    const PetscReal sigma = sigmaFactor*(2*(*h)); //1e-6

    PetscReal cx = 0.0, cy = 0.0, cxx = 0.0, cyy = 0.0, cxy = 0.0;
    PetscReal cz = 0.0, cxz = 0.0, cyz = 0.0, czz = 0.0;


    for (PetscInt i = 0; i < nQuad; ++i) {
        for (PetscInt j = 0; j < nQuad; ++j) {
            for (PetscInt k = 0; k < nQuad; ++k) {
                const PetscReal dist[3] = {sigma*quad[i], sigma*quad[j], sigma*quad[k]};
                PetscReal x[3] = {x0[0] + dist[0], x0[1] + dist[1], x0[2] + dist[2]};

                const PetscReal lsVal = rbf->Interpolate(lsField, vec, x);

                const PetscReal wt = weights[i]*weights[j]*weights[k];

                cx  += wt*GaussianDerivativeFactor(dist, sigma, 1, 0, 0)*lsVal;
                cy  += wt*GaussianDerivativeFactor(dist, sigma, 0, 1, 0)*lsVal;
                cz  += wt*GaussianDerivativeFactor(dist, sigma, 0, 0, 1)*lsVal;
                cxx += wt*GaussianDerivativeFactor(dist, sigma, 2, 0, 0)*lsVal;
                cyy += wt*GaussianDerivativeFactor(dist, sigma, 0, 2, 0)*lsVal;
                czz += wt*GaussianDerivativeFactor(dist, sigma, 0, 0, 2)*lsVal;
                cxy += wt*GaussianDerivativeFactor(dist, sigma, 1, 1, 0)*lsVal;
                cxz += wt*GaussianDerivativeFactor(dist, sigma, 1, 0, 1)*lsVal;
                cyz += wt*GaussianDerivativeFactor(dist, sigma, 0, 1, 1)*lsVal;
            }
        }
    }

    if (PetscPowReal(cx*cx + cy*cy + cz*cz, 0.5) < ablate::utilities::Constants::small){
        *Nx = cx;
        *Ny = cy;
        *Nz = cz;
    }
    else{
        *Nx = (cx)/ PetscPowReal(cx*cx + cy*cy + cz*cz, 0.5);
        *Ny = (cy)/ PetscPowReal(cx*cx + cy*cy + cz*cz, 0.5);
        *Nz = (cz)/ PetscPowReal(cx*cx + cy*cy + cz*cz, 0.5);
    }

    *Nx *= -1; //the gradient is pointing in the direction of max increase; so if the
    *Ny *= -1; //field inside the interface is phi=1 and outside is phi=0 then gradphi will
    *Nz *= -1; //point towards the interior; therefore the outward normal vec is the negative

    if (PetscPowReal(cx*cx + cy*cy + cz*cz, 1.5) < ablate::utilities::Constants::small){
        *H = (cxx*(cy*cy + cz*cz) + cyy*(cx*cx + cz*cz) + czz*(cx*cx + cy*cy) - 2*cxy*cx*cy - 2*cxz*cx*cz - 2*cyz*cy*cz); //3d
        //        *H = (cxx*cy*cy + cyy*cx*cx - 2*cxy*cx*cy); //2d
    }
    else{
        *H = (cxx*(cy*cy + cz*cz) + cyy*(cx*cx + cz*cz) + czz*(cx*cx + cy*cy) - 2*cxy*cx*cy - 2*cxz*cx*cz - 2*cyz*cy*cz)/PetscPowReal(cx*cx + cy*cy + cz*cz, 1.5);
        //        *H = (cxx*cy*cy + cyy*cx*cx - 2*cxy*cx*cy)/PetscPowReal(cx*cx + cy*cy, 1.5); //2d
    }

    *H *= -1; //likewise, curvature is the NEGATIVE divergence of gradphi
}

void PhiNeighborGaussWeight(PetscReal d, PetscReal s, PetscReal *weight){
    PetscReal pi = 3.14159265358979323846264338327950288419716939937510;

    PetscReal Coeff = 1/(PetscSqrtReal(2*pi)*s);

    PetscReal g0 = Coeff*PetscExpReal(0/ (2*PetscSqr(s)));
    PetscReal gd = Coeff*PetscExpReal(-PetscSqr(d)/ (2*PetscSqr(s)));
    *weight = gd/g0;
}


void ablate::finiteVolume::processes::SurfaceForce::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
    SurfaceForce::subDomain = solver.GetSubDomainPtr();
}

void ablate::domain::SubDomain::UpdateAuxLocalVector() {
    if (auxDM) {
        DMLocalToGlobal(auxDM, auxLocalVec, INSERT_VALUES, auxGlobalVec) >> utilities::PetscUtilities::checkError;
        DMGlobalToLocal(auxDM, auxGlobalVec, INSERT_VALUES, auxLocalVec) >> utilities::PetscUtilities::checkError;
    }
}

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

PetscErrorCode ablate::finiteVolume::processes::SurfaceForce::ComputeSource(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx) {
    PetscFunctionBegin;

    //    ablate::finiteVolume::processes::SurfaceForce *process = (ablate::finiteVolume::processes::SurfaceForce *)ctx;
    auto *process = (ablate::finiteVolume::processes::SurfaceForce *)ctx;
    std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
    subDomain->UpdateAuxLocalVector();

    const auto &phiField = subDomain->GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
    const auto &phiTildeField = subDomain->GetField("phiTilde");
    const auto &phiTildeStructuredField = subDomain->GetField("phiTildeStructured");
    const auto &kappaField = subDomain->GetField("kappa");
    const auto &kappaTildeField = subDomain->GetField("kappaTilde");
//    const auto &kappaTildeTildeField = subDomain->GetField("kappaTildeTilde");
    const auto &n0Field = subDomain->GetField("n0");
    const auto &n1Field = subDomain->GetField("n1");
    const auto &n2Field = subDomain->GetField("n2");
    const auto &CSF0Field = subDomain->GetField("SF0");
    const auto &CSF1Field = subDomain->GetField("SF1");
    const auto &CSF2Field = subDomain->GetField("SF2");
    const auto &CSF0TildeField = subDomain->GetField("SF0Tilde");
    const auto &CSF1TildeField = subDomain->GetField("SF1Tilde");
    const auto &CSF2TildeField = subDomain->GetField("SF2Tilde");
    auto dim = solver.GetSubDomain().GetDimensions();
    const auto &eulerField = solver.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);

    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector();
    Vec vertexVec;
    DMGetLocalVector(process->vertexDM, &vertexVec);
    const PetscScalar *solArray;
    PetscScalar *auxArray;
    PetscScalar *vertexArray;
    PetscScalar *fArray;

    VecGetArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(vertexVec, &vertexArray);
    PetscCall(VecGetArray(locFVec, &fArray));

    ablate::domain::Range cellRange;
    solver.GetCellRangeWithoutGhost(cellRange);
    PetscInt vStart, vEnd;
    DMPlexGetDepthStratum(process->vertexDM, 0, &vStart, &vEnd);

    ablate::domain::Range vertexRange;
    GetVertexRange(dm, ablate::domain::Region::ENTIREDOMAIN, vertexRange);

    if (interpCellList == nullptr) {
        BuildInterpCellList(auxDM, cellRange);
    }

    PetscInt polyAug = 2;
    bool doesNotHaveDerivatives = false;
    bool doesNotHaveInterpolation = false;
    PetscReal h;
    DMPlexGetMinRadius(dm, &h);  // h*=3;
    ablate::domain::rbf::MQ cellRBF(polyAug, h, doesNotHaveDerivatives, doesNotHaveInterpolation);
    cellRBF.Setup(subDomain);
    cellRBF.Initialize();

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.GetPoint(c);
        const PetscScalar *phic;
        xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic) >> ablate::utilities::PetscUtilities::checkError;

        PetscReal xc, yc, zc;
        Get3DCoordinate(dm, cell, &xc, &yc, &zc);
        //        std::cout << "\n --------- cell " << cell << " -------start--";
        //        std::cout << "\n coordinate=  ("<<xc<<", "<<yc<<")";

        //here goes phistructured
        //then phistructured ~ 0.5 defines the interface
        PetscScalar *phiTildeStructured;
        xDMPlexPointLocalRef(auxDM, cell, phiTildeStructuredField.id, auxArray, &phiTildeStructured) >> ablate::utilities::PetscUtilities::checkError;

        //the absence of this line was why ph1=/=phistructured
//        *phiTildeStructured=0;
//
//        PetscInt nNeighbors_1layer, *neighbors_1layer;
//        DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors_1layer, &neighbors_1layer);
//
//        if (abs(xc) >= artificialsubdomain or abs(yc) >= artificialsubdomain or abs(zc) >= artificialsubdomain) {
//            *phiTildeStructured = 0;
//        } else {
//            for (PetscInt j = 0; j < nNeighbors_1layer; ++j) {
//                PetscInt neighbor = neighbors_1layer[j];
//                PetscReal *phin;
//                xDMPlexPointLocalRead(dm, neighbor, phiField.id, solArray, &phin);
//
//                *phiTildeStructured += *phin / (2*nNeighbors_1layer);
//
////                if (cell == 5182){
////                    std::cout << "phiStruct " << *phiTildeStructured << "\n";
////                }
//
//
////                std::cout << "phiStruct cell" << cell <<  "  " << *phic << "   neighbor  " << neighbor << "   " << *phin << "\n";
//            }
//        }
////
//        *phiTildeStructured += *phic/2;
//
//        if (cell == 5503){
//            std::cout << "phiStruct " << *phiTildeStructured << "\n";
//        }


//        std::cout << "phiStruct eeeeeeee " << *phiTildeStructured << "\n";

        //now build phitilde
        //number of smoothing layers

        PetscInt nNeighbors, *neighbors;
        DMPlexGetNeighbors(dm, cell, 3, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);

        PetscReal weightedphi = 0;
//        PetscReal avgphi = 0;
        PetscReal Tw = 0;

        PetscScalar *phiTilde;
        xDMPlexPointLocalRef(auxDM, cell, phiTildeField.id, auxArray, &phiTilde) >> ablate::utilities::PetscUtilities::checkError;


        if (abs(xc) >= artificialsubdomain or abs(yc) >= artificialsubdomain or abs(zc) >= artificialsubdomain) {
            *phiTilde = 0;
        } else {
            for (PetscInt j = 0; j < nNeighbors; ++j) {
                PetscInt neighbor = neighbors[j];

                //                if (cell != neighbor) {
                PetscReal *phin;
                xDMPlexPointLocalRead(dm, neighbor, phiField.id, solArray, &phin);

                PetscReal xn, yn, zn;
                Get3DCoordinate(dm, neighbor, &xn, &yn, &zn);

                PetscReal d = PetscSqrtReal(PetscSqr(xn - xc) + PetscSqr(yn - yc) + PetscSqr(zn - zc));  // distance
                //                PetscReal s = 1.698643600577*h; //stdev of our phitilde smoothing, NOT our convolution smoothing
                PetscReal s = 6 * h;

                PetscReal wn;
                PhiNeighborGaussWeight(d, s, &wn);
                Tw += wn;

                weightedphi += (*phin * wn);
            }

            weightedphi /= Tw;
//            avgphi /= 16;
            *phiTilde = weightedphi;  //*phiTilde = (0.5 * (*phic)) + (0.5 * (avgphi));
//            *phiTildeStructured = avgphi;
        }
    }
//    subDomain->UpdateAuxLocalVector();

    // VERTEX BASED

//    PetscReal kappabar = 0, Nkappa=0;
        for (PetscInt j = vertexRange.start; j < vertexRange.end; j++){

            const PetscInt vertex = vertexRange.GetPoint(j);

            PetscScalar *gradPhi_v;

            DMPlexPointLocalFieldRef(process->vertexDM, vertex, 0, vertexArray, &gradPhi_v);
    //        DMPlexVertexGradFromCell(auxDM, vertex, auxVec, phiTildeStructuredField.id, 0, gradPhi_v);
            DMPlexVertexGradFromCell(dm, vertex, locX, phiField.id, 0, gradPhi_v);
            DMPlexVertexGradFromCell(auxDM, vertex, auxVec, phiTildeField.id, 0, gradPhi_v);

            if (utilities::MathUtilities::MagVector(dim, gradPhi_v) > 1e-10) {
                utilities::MathUtilities::NormVector(dim, gradPhi_v);
            }
        }
        subDomain->UpdateAuxLocalVector();
        for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {

            const PetscInt cell = cellRange.GetPoint(i);

            PetscReal xc, yc, zc;
            Get3DCoordinate(dm, cell, &xc, &yc, &zc);
            PetscReal kappa=0, Nx, Ny, Nz;
            const PetscReal *phiTilde; xDMPlexPointLocalRead(auxDM, cell, phiTildeField.id, auxArray, &phiTilde);
            const PetscReal *phiTildeStructured; xDMPlexPointLocalRead(auxDM, cell, phiTildeStructuredField.id, auxArray, &phiTildeStructured);
            const PetscReal *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);

            //cutcell criterion

            PetscReal phi1=0;
            PetscInt nNeighbors_1layer, *neighbors_1layer;
            DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors_1layer, &neighbors_1layer);
            PetscReal M=0;

            if (abs(xc) >= artificialsubdomain or abs(yc) >= artificialsubdomain or abs(zc) >= artificialsubdomain) {
                phi1 = 0;
            } else {
                for (PetscInt j = 0; j < nNeighbors_1layer; ++j) {
                PetscInt neighbor = neighbors_1layer[j];
                if (neighbor!=cell){
                    PetscReal *phin;
                    xDMPlexPointLocalRead(dm, neighbor, phiField.id, solArray, &phin);
//                    phi1 += *phin / (2*nNeighbors_1layer);
                    phi1 += *phin;
                    M+=1;
//                    if (cell==44659){
//                        std::cout <<"neighbor  " << neighbor << "  " << *phin<<"\n";
//                    }

                }
                }
            }
            phi1/= M;
//            phi1 += *phic;

//            if (cell==44659){
//                std::cout <<"cell  " << cell << "  " << phi1<<"\n";
//            }

//            if (cell==4607){
//                std::cout <<"cell  " << cell << "  " << phi1<<"\n";
//            }

//            if (*phiTilde > 0.25 and *phiTilde < 0.75) {
//            if (*phic > 0.0001 and *phic < 0.9999) {
//            if (*phiTildeStructured > 1e-4 and *phiTildeStructured < 1-1e-4) {
//            if (phi1 > 1e-4 and phi1 < 1-1e-4) {
//            if (phi1 > (1e-4)/4 and phi1 < (1-1e-4)/4) {
//            if (phi1 > 0.4 and phi1 < 0.6) {
            if ((phi1 > 0.25 and phi1 < 0.75) and (*phic > 0.001 and *phic < 0.999)) {
//            if (*phiTilde > 0.35 and *phiTilde < 0.65) {
                if (abs(xc) >= artificialsubdomain or abs(yc) >= artificialsubdomain or abs(zc) >= artificialsubdomain) {
                    kappa = Nx = Ny = Nz = 0;//  +phi1*0;
                } else {
                    PetscReal nabla_n[dim];
                    for (PetscInt offset = 0; offset < dim; offset++) {
                        DMPlexCellGradFromVertex(process->vertexDM, cell, vertexVec, 0, offset, nabla_n);
                        kappa += nabla_n[offset];
                    }

                    PetscScalar gradPhi_c[dim];
                    DMPlexCellGradFromCell(auxDM, cell, auxVec, phiTildeField.id, 0, gradPhi_c);
                    //            DMPlexCellGradFromCell(auxDM, cell, auxVec, phiTildeStructuredField.id, 0, gradPhi_c);
                    //            DMPlexCellGradFromCell(dm, cell, locX, phiField.id, 0, gradPhi_c);
                    if (utilities::MathUtilities::MagVector(dim, gradPhi_c) > 1e-10) {
                        utilities::MathUtilities::NormVector(dim, gradPhi_c);
                    }

                    Nx = gradPhi_c[0];
                    Ny = gradPhi_c[1];
                    //            Nz = gradPhi_c[2];
                    Nz = 0;
                }
            }
            else {
                kappa=Nx=Ny=Nz=0;
            }
            kappa *= -1;
            Nx *= -1;
            Ny *= -1;
            Nz *= -1;

            PetscScalar *kappaptr, *n0ptr, *n1ptr, *n2ptr;
            xDMPlexPointLocalRef(auxDM, cell, kappaField.id, auxArray, &kappaptr) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, n0Field.id, auxArray, &n0ptr) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, n1Field.id, auxArray, &n1ptr) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, n2Field.id, auxArray, &n2ptr) >> ablate::utilities::PetscUtilities::checkError;

            *kappaptr = kappa;
            *n0ptr = Nx;
            *n1ptr = Ny;
            *n2ptr = Nz;
        }
//        subDomain->UpdateAuxLocalVector();
        for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
            const PetscInt cell = cellRange.GetPoint(i);

            PetscReal xc, yc, zc;
            Get3DCoordinate(dm, cell, &xc, &yc, &zc);
            PetscReal *kappac=0;
            PetscReal *kappaTilde;
            xDMPlexPointLocalRef(auxDM, cell, kappaField.id, auxArray, &kappac) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, kappaTildeField.id, auxArray, &kappaTilde) >> ablate::utilities::PetscUtilities::checkError;
            *kappaTilde=0;
            PetscInt nNeighbors, *neighbors;
            DMPlexGetNeighbors(dm, cell, 8, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
            //4 layers
            PetscReal Tw = 0;
            if (PetscAbs(*kappac) > 1e-4) {
                for (PetscInt j = 0; j < nNeighbors; ++j) {
                    PetscInt neighbor = neighbors[j];
                        PetscReal *kappan;
                        xDMPlexPointLocalRef(auxDM, neighbor, kappaField.id, auxArray, &kappan) >> ablate::utilities::PetscUtilities::checkError;
                        if (PetscAbs(*kappan) > 1e-4) {
                            PetscReal xn, yn, zn;
                            Get3DCoordinate(dm, neighbor, &xn, &yn, &zn);
                            PetscReal d = PetscSqrtReal(PetscSqr(xn - xc) + PetscSqr(yn - yc) + PetscSqr(zn - zc));  // distance
                            PetscReal s = 18 * h; //6*h
                            PetscReal wn;
                            PhiNeighborGaussWeight(d, s, &wn);
                            Tw += wn;
                            *kappaTilde += (*kappan * wn);
                        }
                }
                *kappaTilde /= Tw;
            }
        }

        for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
            const PetscInt cell = cellRange.GetPoint(i);
            PetscScalar *kappaTildeptr, *n0ptr, *n1ptr, *n2ptr, *CSF0ptr, *CSF1ptr, *CSF2ptr;
            xDMPlexPointLocalRef(auxDM, cell, kappaTildeField.id, auxArray, &kappaTildeptr) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, n0Field.id, auxArray, &n0ptr) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, n1Field.id, auxArray, &n1ptr) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, n2Field.id, auxArray, &n2ptr) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, CSF0Field.id, auxArray, &CSF0ptr) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, CSF1Field.id, auxArray, &CSF1ptr) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, CSF2Field.id, auxArray, &CSF2ptr) >> ablate::utilities::PetscUtilities::checkError;
            *CSF0ptr = process->sigma * *kappaTildeptr * -*n0ptr;
            *CSF1ptr = process->sigma * *kappaTildeptr * -*n1ptr;
            *CSF2ptr = process->sigma * *kappaTildeptr * -*n2ptr;


            if (PetscAbs(*kappaTildeptr) > 1e-4) {
                // temporary analytical sol
                PetscReal xc, yc, zc;
                Get3DCoordinate(dm, cell, &xc, &yc, &zc);
                *CSF0ptr = -(process->sigma * xc) / PetscSqrtReal(xc * xc + yc * yc);
                *CSF1ptr = -(process->sigma * yc) / PetscSqrtReal(xc * xc + yc * yc);
                *CSF2ptr = 0;
            }

        }

        for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
            const PetscInt cell = cellRange.GetPoint(i);
            PetscScalar *CSF0ptr, *CSF1ptr, *CSF2ptr, *kappac;
            xDMPlexPointLocalRef(auxDM, cell, kappaField.id, auxArray, &kappac) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, CSF0Field.id, auxArray, &CSF0ptr) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, CSF1Field.id, auxArray, &CSF1ptr) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, CSF2Field.id, auxArray, &CSF2ptr) >> ablate::utilities::PetscUtilities::checkError;
            PetscReal xc, yc, zc; Get3DCoordinate(dm, cell, &xc, &yc, &zc);
            PetscReal *CSF0Tilde, *CSF1Tilde, *CSF2Tilde;
            xDMPlexPointLocalRef(auxDM, cell, CSF0TildeField.id, auxArray, &CSF0Tilde) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, CSF1TildeField.id, auxArray, &CSF1Tilde) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, CSF2TildeField.id, auxArray, &CSF2Tilde) >> ablate::utilities::PetscUtilities::checkError;

            *CSF0Tilde=0; *CSF1Tilde=0; *CSF2Tilde=0;

            PetscInt nNeighbors, *neighbors;
            DMPlexGetNeighbors(dm, cell, 4, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
            PetscReal Tw = 0;
            for (PetscInt j = 0; j < nNeighbors; ++j) {
                PetscInt neighbor = neighbors[j];
                PetscReal *CSF0n, *CSF1n, *CSF2n, *kappan;
                xDMPlexPointLocalRef(auxDM, neighbor, kappaField.id, auxArray, &kappan) >> ablate::utilities::PetscUtilities::checkError;
                xDMPlexPointLocalRef(auxDM, neighbor, CSF0Field.id, auxArray, &CSF0n) >> ablate::utilities::PetscUtilities::checkError;
                xDMPlexPointLocalRef(auxDM, neighbor, CSF1Field.id, auxArray, &CSF1n) >> ablate::utilities::PetscUtilities::checkError;
                xDMPlexPointLocalRef(auxDM, neighbor, CSF2Field.id, auxArray, &CSF2n) >> ablate::utilities::PetscUtilities::checkError;

//                if (neighbor == cell){
//                        *CSF0Tilde += *CSF0n/2;
//                        *CSF1Tilde += *CSF1n/2;
//                        *CSF2Tilde += *CSF2n/2;
//                }
//                if (neighbor != cell){
//                        *CSF0Tilde += *CSF0n/(2*(nNeighbors-1));
//                        *CSF1Tilde += *CSF1n/(2*(nNeighbors-1));
//                        *CSF2Tilde += *CSF2n/(2*(nNeighbors-1));
//                }
                PetscReal xn, yn, zn;
                Get3DCoordinate(dm, neighbor, &xn, &yn, &zn);

                PetscReal d = PetscSqrtReal(PetscSqr(xn - xc) + PetscSqr(yn - yc) + PetscSqr(zn - zc));  // distance
                PetscReal s = 6 * h;
                PetscReal wn;
                PhiNeighborGaussWeight(d, s, &wn);
                Tw += wn;
                *CSF0Tilde += (*CSF0n * wn);
                *CSF1Tilde += (*CSF1n * wn);
                *CSF2Tilde += (*CSF2n * wn);
            }
            *CSF0Tilde /= Tw;
            *CSF1Tilde /= Tw;
            *CSF2Tilde /= Tw;
        }


        for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
            const PetscInt cell = cellRange.GetPoint(i);
            const PetscScalar *phiptr;

            PetscScalar *CSF0ptr, *CSF1ptr, *CSF2ptr;
            xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phiptr) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, CSF0TildeField.id, auxArray, &CSF0ptr) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, CSF1TildeField.id, auxArray, &CSF1ptr) >> ablate::utilities::PetscUtilities::checkError;
            xDMPlexPointLocalRef(auxDM, cell, CSF2TildeField.id, auxArray, &CSF2ptr) >> ablate::utilities::PetscUtilities::checkError;

//            xDMPlexPointLocalRef(auxDM, cell, CSF0TildeField.id, auxArray, &CSF0Tildeptr) >> ablate::utilities::PetscUtilities::checkError;
//            xDMPlexPointLocalRef(auxDM, cell, CSF1TildeField.id, auxArray, &CSF1Tildeptr) >> ablate::utilities::PetscUtilities::checkError;
//            xDMPlexPointLocalRef(auxDM, cell, CSF2TildeField.id, auxArray, &CSF2Tildeptr) >> ablate::utilities::PetscUtilities::checkError;

            const PetscScalar *euler = nullptr;
            PetscScalar *eulerSource = nullptr;
            PetscCall(DMPlexPointLocalFieldRef(dm, cell, eulerField.id, fArray, &eulerSource));
            PetscCall(DMPlexPointLocalFieldRead(dm, cell, eulerField.id, solArray, &euler));
            auto density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];



            PetscReal ux = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] / density;
            PetscReal uy = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + 1] / density;
            PetscReal uz = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + 2] / density;
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] += *CSF0ptr;
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + 1] += *CSF1ptr;
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + 2] += *CSF2ptr;
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += *CSF0ptr * ux;
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += *CSF1ptr * uy;
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += *CSF2ptr * uz;

        }

        //END OF VERTEX BASED

        //START OF GDF

//    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
//
//        const PetscInt cell = cellRange.GetPoint(i);
//        const PetscReal *phiTilde; xDMPlexPointLocalRead(auxDM, cell, phiTildeField.id, auxArray, &phiTilde);
//        PetscReal kappa, Nx, Ny, Nz;
//
//        //        PetscInt sq_cre=sqrt(cellRange.end);
//        //        bool cond1 = (cell < sq_cre);
//        //        bool cond2 = (cell % sq_cre == 0);
//        //        bool cond3 = ((cell - 1) % sq_cre == 0);
//        //        bool cond4 = (cell >= cellRange.end - sq_cre);
//        //        bool isBoundary = (cond1 or cond2 or cond3 or cond4);
//
//        PetscReal xc, yc, zc;
//        Get3DCoordinate(dm, cell, &xc, &yc, &zc);
//
//        if ( abs(xc) < artificialsubdomain and abs(yc) < artificialsubdomain and abs(zc) < artificialsubdomain){
//            //        if (not (isBoundary)) {
//
//
//            //cutcell criterion
//
//            PetscReal phi1=0;
//            PetscInt nNeighbors_1layer, *neighbors_1layer;
//            DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors_1layer, &neighbors_1layer);
//            PetscReal M=0;
//
//            if (abs(xc) >= artificialsubdomain or abs(yc) >= artificialsubdomain or abs(zc) >= artificialsubdomain) {
//                phi1 = 0;
//            } else {
//                for (PetscInt j = 0; j < nNeighbors_1layer; ++j) {
//                    PetscInt neighbor = neighbors_1layer[j];
//                    if (neighbor!=cell){
//                        PetscReal *phin;
//                        xDMPlexPointLocalRead(dm, neighbor, phiField.id, solArray, &phin);
//            //                    phi1 += *phin / (2*nNeighbors_1layer);
//                        phi1 += *phin;
//                        M+=1;
//            //                    if (cell==44659){
//            //                        std::cout <<"neighbor  " << neighbor << "  " << *phin<<"\n";
//            //                    }
//
//                    }
//                }
//            }
//            phi1/= M;
////            if (phi1 > 0.25 and phi1 < 0.75) {
//            if (*phiTilde > 0.25 and *phiTilde < 0.75) {
//                //                std::cout << "\n CUT CELL, cell  " << cell << "   ("<<xc<<", "<<yc<<", "<<zc<<")";
//                double H, Nx_ptr, Ny_ptr, Nz_ptr;
//                CurvatureViaGaussian(auxDM, cell, auxVec, &phiTildeField, &cellRBF, &h, &H, &Nx_ptr, &Ny_ptr, &Nz_ptr);
//                Nx = Nx_ptr;
//                Ny = Ny_ptr;
//                Nz = Nz_ptr;
//                //                Nz = 0;
//                kappa = H + phi1*0;
//
//
//                //                std::cout << "\n cell " << cell << " ("<<xc<<", "<<yc<<")  do gauss-hermite";
//            }
//            else{
//                kappa = Nx = Ny = Nz = 0;
//            }
//        }
//        else {
//            //            std::cout << "\n NOT CUT CELL, cell  " << cell << "   ("<<xc<<", "<<yc<<", "<<zc<<")";
//            kappa = Nx = Ny = Nz = 0;
//        }
//        PetscReal N[3] = {Nx, Ny, Nz};
//        PetscScalar *kappaptr, *n0ptr, *n1ptr, *n2ptr;
//        xDMPlexPointLocalRef(auxDM, cell, kappaField.id, auxArray, &kappaptr) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, n0Field.id, auxArray, &n0ptr) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, n1Field.id, auxArray, &n1ptr) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, n2Field.id, auxArray, &n2ptr) >> ablate::utilities::PetscUtilities::checkError;
//
//        *kappaptr = kappa;
//        *n0ptr = Nx;
//        *n1ptr = Ny;
//        *n2ptr = Nz;
//
//        // compute surface force sigma * cur * normal and add to local F vector
//        const PetscScalar *euler = nullptr;
//        PetscScalar *eulerSource = nullptr;
//        PetscCall(DMPlexPointLocalFieldRef(dm, cell, eulerField.id, fArray, &eulerSource));
//        PetscCall(DMPlexPointLocalFieldRead(dm, cell, eulerField.id, solArray, &euler));
//        auto density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];
//
//        for (PetscInt k = 0; k < dim; ++k) {
//            // calculate surface force and energy
//            PetscReal surfaceForce = process->sigma * kappa * N[k] + (0*time);
//            PetscReal vel = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + k] / density;
//            // add in the contributions
//
//            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + k] += surfaceForce;
//            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += surfaceForce * vel;
//        }
//    }

        //GDF

    std::cout << "done\n";

    VecRestoreArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
    //    exit(0);
    PetscFunctionReturn(0);
}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::SurfaceForce, "calculates surface tension force and adds source terms",
         ARG(PetscReal, "sigma", "sigma, surface tension coefficient"));