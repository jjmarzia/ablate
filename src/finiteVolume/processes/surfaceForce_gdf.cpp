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
    const auto &n0Field = subDomain->GetField("n0");
    const auto &n1Field = subDomain->GetField("n1");
    const auto &n2Field = subDomain->GetField("n2");
    auto dim = solver.GetSubDomain().GetDimensions();
    const auto &eulerField = solver.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);

    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector();
    const PetscScalar *solArray;
    PetscScalar *auxArray;

    VecGetArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;

    PetscScalar *fArray;
    PetscCall(VecGetArray(locFVec, &fArray));

    ablate::domain::Range cellRange;
    solver.GetCellRangeWithoutGhost(cellRange);

    if ( interpCellList==nullptr ) {
        BuildInterpCellList(auxDM, cellRange);
    }

    PetscInt polyAug = 2;
    bool doesNotHaveDerivatives = false;
    bool doesNotHaveInterpolation = false;
    PetscReal h; DMPlexGetMinRadius(dm, &h); //h*=3;
    ablate::domain::rbf::MQ cellRBF(polyAug, h, doesNotHaveDerivatives, doesNotHaveInterpolation);
    cellRBF.Setup(subDomain);
    cellRBF.Initialize();


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

    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {

        const PetscInt cell = cellRange.GetPoint(i);
        const PetscReal *phiTilde; xDMPlexPointLocalRead(auxDM, cell, phiTildeField.id, auxArray, &phiTilde);
        PetscReal kappa, Nx, Ny, Nz;

        //        PetscInt sq_cre=sqrt(cellRange.end);
        //        bool cond1 = (cell < sq_cre);
        //        bool cond2 = (cell % sq_cre == 0);
        //        bool cond3 = ((cell - 1) % sq_cre == 0);
        //        bool cond4 = (cell >= cellRange.end - sq_cre);
        //        bool isBoundary = (cond1 or cond2 or cond3 or cond4);

        PetscReal xc, yc, zc;
        Get3DCoordinate(dm, cell, &xc, &yc, &zc);

        if ( abs(xc) < artificialsubdomain and abs(yc) < artificialsubdomain and abs(zc) < artificialsubdomain){
            //        if (not (isBoundary)) {
            if (*phiTilde > 0.1 and *phiTilde < 0.9) {
                //                std::cout << "\n CUT CELL, cell  " << cell << "   ("<<xc<<", "<<yc<<", "<<zc<<")";
                double H, Nx_ptr, Ny_ptr, Nz_ptr;
                CurvatureViaGaussian(auxDM, cell, auxVec, &phiTildeStructuredField, &cellRBF, &h, &H, &Nx_ptr, &Ny_ptr, &Nz_ptr);
                Nx = Nx_ptr;
                Ny = Ny_ptr;
                Nz = Nz_ptr;
                //                Nz = 0;
                kappa = H;


                //                std::cout << "\n cell " << cell << " ("<<xc<<", "<<yc<<")  do gauss-hermite";
            }
            else{
                kappa = Nx = Ny = Nz = 0;
            }
        }
        else {
            //            std::cout << "\n NOT CUT CELL, cell  " << cell << "   ("<<xc<<", "<<yc<<", "<<zc<<")";
            kappa = Nx = Ny = Nz = 0;
        }
        PetscReal N[3] = {Nx, Ny, Nz};
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
            PetscReal surfaceForce = process->sigma * kappa * N[k] + (0*time);
            PetscReal vel = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + k] / density;
            // add in the contributions

            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + k] += surfaceForce;
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += surfaceForce * vel;


        }
    }

    VecRestoreArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
    //    exit(0);
    PetscFunctionReturn(0);
}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::SurfaceForce, "calculates surface tension force and adds source terms",
         ARG(PetscReal, "sigma", "sigma, surface tension coefficient"));