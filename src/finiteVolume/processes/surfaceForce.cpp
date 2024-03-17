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

void Get2DCoordinate(DM dm, PetscInt p, PetscReal *xp, PetscReal *yp){
    //get the coordinates of the point
    PetscReal vol;
    PetscReal centroid[2];
    DMPlexComputeCellGeometryFVM(dm, p, &vol, centroid, nullptr);
    *xp = centroid[0];
    *yp = centroid[1];
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
        DMPlexComputeCellGeometryFVM(dm, cells[c], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

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


static PetscReal sigmaFactor = 3.0;

void BuildInterpCellList(DM dm, const ablate::domain::Range cellRange) {


    PetscReal h;
    DMPlexGetMinRadius(dm, &h) >> ablate::utilities::PetscUtilities::checkError;
    h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size
    const PetscReal sigma = sigmaFactor*h;

    PetscInt dim;
    DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

    PetscMalloc1(16*(cellRange.end - cellRange.start), &interpCellList);

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.GetPoint(c);

        PetscReal x0[dim];
        DMPlexComputeCellGeometryFVM(dm, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

        PetscInt nCells, *cellList;
        DMPlexGetNeighbors(dm, cell, 2, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

        for (PetscInt i = 0; i < nQuad; ++i) {
            for (PetscInt j = 0; j < nQuad; ++j) {

                PetscReal x[2] = {x0[0] + sigma*quad[i], x0[1] + sigma*quad[j]};

                const PetscInt interpCell = FindCell(dm, x, nCells, cellList, NULL);

                if (interpCell < 0) {
                    int rank;
                    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
                    throw std::runtime_error("BuildInterpCellList could not determine the location of (" + std::to_string(x[0]) + ", " + std::to_string(x[1]) + ") on rank " + std::to_string(rank) + ".");
                }

                interpCellList[(c - cellRange.start)*16 + i*4 + j] = interpCell;
            }
        }

        DMPlexRestoreNeighbors(dm, cell, 2, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

    }
}

void CurvatureViaGaussian(DM dm, const PetscInt cell, Vec vec, const ablate::domain::Field *lsField, ablate::domain::rbf::MQ *rbf, const double *h, double *H, double *Nx, double *Ny){

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


    const PetscReal a = sigmaFactor*(*h); //1e-6

    PetscReal cx = 0.0, cy = 0.0, cxx = 0.0, cyy = 0.0, cxy = 0.0;


    for (PetscInt i = 0; i < nQuad; ++i) {
        for (PetscInt j = 0; j < nQuad; ++j) {

            const PetscReal dist[2] = {a*quad[i], a*quad[j]};
            PetscReal x[2] = {x0[0] + dist[0], x0[1] + dist[1]};

            const PetscReal lsVal = rbf->Interpolate(lsField, vec, x);

            const PetscReal wt = weights[i]*weights[j];

            cx  += wt*GaussianDerivativeFactor(dist, a, 1, 0, 0)*lsVal;
            cy  += wt*GaussianDerivativeFactor(dist, a, 0, 1, 0)*lsVal;
            cxx += wt*GaussianDerivativeFactor(dist, a, 2, 0, 0)*lsVal;
            cyy += wt*GaussianDerivativeFactor(dist, a, 0, 2, 0)*lsVal;
            cxy += wt*GaussianDerivativeFactor(dist, a, 1, 1, 0)*lsVal;
        }
    }

    if (PetscPowReal(cx*cx + cy*cy, 0.5) < ablate::utilities::Constants::small){
        *Nx = cx;
        *Ny = cy;
    }
    else{
        *Nx = (cx)/ PetscPowReal(cx*cx + cy*cy, 0.5);
        *Ny = cy/ PetscPowReal(cx*cx + cy*cy, 0.5);
    }

    if (PetscPowReal(cx*cx + cy*cy, 1.5) < ablate::utilities::Constants::small){
        *H = (cxx*cy*cy + cyy*cx*cx - 2.0*cxy*cx*cy);
    }
    else{
        *H = (cxx*cy*cy + cyy*cx*cx - 2.0*cxy*cx*cy)/PetscPowReal(cx*cx + cy*cy, 1.5);
    }
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

    ablate::finiteVolume::processes::SurfaceForce *process = (ablate::finiteVolume::processes::SurfaceForce *)ctx;
    std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
    subDomain->UpdateAuxLocalVector();

    const auto &phiField = subDomain->GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
    const auto &phiTildeField = subDomain->GetField("phiTilde");
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

        PetscReal xc, yc;
        Get2DCoordinate(dm, cell, &xc, &yc);
//        std::cout << "\n --------- cell " << cell << " -------start--";
//        std::cout << "\n coordinate=  ("<<xc<<", "<<yc<<")";

        PetscReal M=0;
        PetscReal avgphi=0;
//        PetscReal totaldistance=0;

        PetscScalar *phiTilde;
        xDMPlexPointLocalRef(auxDM, cell, phiTildeField.id, auxArray, &phiTilde) >> ablate::utilities::PetscUtilities::checkError;

//        PetscInt sq_cre=sqrt(cellRange.end);
//        bool cond1 = (cell < sq_cre);
//        bool cond2 = (cell % sq_cre == 0);
//        bool cond3 = ((cell - 1) % sq_cre == 0);
//        bool cond4 = (cell >= cellRange.end - sq_cre);
//        bool isBoundary = (cond1 or cond2 or cond3 or cond4);

        if ( abs(xc) >= 1.5 or abs(yc) >= 1.5){
//        if (isBoundary){
//            *phiTilde = *phic;
            *phiTilde=0;
//            std::cout << "\nisboundary";
        }
        else {
            for (PetscInt j = 0; j < nNeighbors; ++j) {
                PetscInt neighbor = neighbors[j];
                if (cell != neighbor) {
                    M += 1;
                    PetscReal *phin;
                    xDMPlexPointLocalRead(dm, neighbor, phiField.id, solArray, &phin);

                    PetscReal xn, yn;
                    Get2DCoordinate(dm, neighbor, &xn, &yn);

                    //                PetscReal distance = pow(   pow(xc-xn,2)+pow(yc-yn,2)     ,0.5);
                    avgphi += (*phin);
                    //                totaldistance += distance;

//                    std::cout << "\n   neighbor= " << neighbor << "  phin=  " << *phin;
//                    std::cout << "\n      coordinate=  (" << xn << ", " << yn << ", "
//                              << ")";

//                    std::cout << "\ndo smoothing";
                }
            }
            avgphi /= M;
            *phiTilde = (0.5 * (*phic)) + (0.5 * (avgphi));
        }

//        std::cout << "\n" << cell << "  phiavg "<<avgphi;
//        std::cout << "\n" << cell << "  phitilde "<<*phiTilde + 0*process->sigma;
    }
//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//        PetscInt cell = cellRange.GetPoint(c);
//        const PetscScalar *phi, *phiTilde;
//        xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phi) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRead(auxDM, cell, phiTildeField.id, auxArray, &phiTilde) >> ablate::utilities::PetscUtilities::checkError;
//        if (PetscAbsScalar(*phi - *phiTilde) > PETSC_SMALL) {
//            printf("%+f\t%+f\t%+e\n", *phi, *phiTilde, PetscAbsScalar(*phi - *phiTilde));
//        }
//    }

    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {

        const PetscInt cell = cellRange.GetPoint(i);
        const PetscReal *phiTilde; xDMPlexPointLocalRead(auxDM, cell, phiTildeField.id, auxArray, &phiTilde);
        PetscReal kappa=0, Nx=0, Ny=0, Nz=0;

//        PetscInt sq_cre=sqrt(cellRange.end);
//        bool cond1 = (cell < sq_cre);
//        bool cond2 = (cell % sq_cre == 0);
//        bool cond3 = ((cell - 1) % sq_cre == 0);
//        bool cond4 = (cell >= cellRange.end - sq_cre);
//        bool isBoundary = (cond1 or cond2 or cond3 or cond4);

        PetscReal xc, yc;
        Get2DCoordinate(dm, cell, &xc, &yc);

        if ( abs(xc) < 1.5 and abs(yc) < 1.5){
//        if (not (isBoundary)) {
            if (*phiTilde > 0.1 and *phiTilde < 0.9) {
                //                std::cout << "\n CUT CELL, cell  " << cell << "   ("<<xc<<", "<<yc<<", "<<zc<<")";
                double H, Nx_ptr, Ny_ptr;
                CurvatureViaGaussian(auxDM, cell, auxVec, &phiTildeField, &cellRBF, &h, &H, &Nx_ptr, &Ny_ptr);
                Nx = Nx_ptr;
                Ny = Ny_ptr;
                Nz = 0;
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
    printf("All Done!\n");
//    exit(0);
    PetscFunctionReturn(0);
}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::SurfaceForce, "calculates surface tension force and adds source terms",
         ARG(PetscReal, "sigma", "sigma, surface tension coefficient"));