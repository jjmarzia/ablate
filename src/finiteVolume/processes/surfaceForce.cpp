#include "domain/RBF/mq.hpp"
#include "surfaceForce.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "levelSet/levelSetUtilities.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"

#include <fstream>

//ablate::finiteVolume::processes::IntSharp::IntSharp(PetscReal Gamma, PetscReal epsilon) : Gamma(Gamma), epsilon(epsilon) {}

ablate::finiteVolume::processes::SurfaceForce::SurfaceForce(PetscReal sigma, PetscReal C, PetscReal N, bool flipPhiTilde) : sigma(sigma), C(C), N(N), flipPhiTilde(flipPhiTilde) {}
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

void Get1DCoordinate(DM dm, PetscInt dim, PetscInt p, PetscReal *xp){
    //get the coordinates of the point
    PetscReal vol;
    PetscReal centroid[dim];
    DMPlexComputeCellGeometryFVM(dm, p, &vol, centroid, nullptr);
    *xp = centroid[0];
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
//static PetscReal artificialsubdomain = 1.25;

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
    PetscInt dim; DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;
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
//        DMLocalToGlobal(auxDM, auxLocalVec, ADD_VALUES, auxGlobalVec) >> utilities::PetscUtilities::checkError;
        DMGlobalToLocal(auxDM, auxGlobalVec, INSERT_VALUES, auxLocalVec) >> utilities::PetscUtilities::checkError;
    }
}

void PushToGhost(DM dm, Vec LocalVec, Vec GlobalVec, InsertMode ADD_OR_INSERT_VALUES) {
    //    DMLocalToGlobal(dm, LocalVec, INSERT_VALUES, GlobalVec);
    DMLocalToGlobal(dm, LocalVec, ADD_OR_INSERT_VALUES, GlobalVec); //p0 to p1
    DMGlobalToLocal(dm, GlobalVec, INSERT_VALUES, LocalVec); //p1 to p1
}

PetscInt count=0;
void SaveDataToFile(PetscInt rangeStart, PetscInt rangeEnd, DM dm, PetscScalar *array, std::string filename, bool iterateAcrossTime){
    int rank; MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    std::string counterstring;
    if (iterateAcrossTime){
        count+=1;
        counterstring = std::to_string(count);
    }
    if (not (iterateAcrossTime)){ counterstring = ""; }
    std::ofstream thefile("/Users/jjmarzia/Desktop/ablate/inputs/parallel/"+filename+counterstring+"_rank"+std::to_string(rank)+".txt");
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

static void SF_CopyDM(DM oldDM, const PetscInt pStart, const PetscInt pEnd, const PetscInt nDOF, DM *newDM) {

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
        PetscInt nNeighbors, *neighbors, nNeighbors1, *neighbors1;
        PetscReal layers=3;
        // DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);

        DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        cellNeighbors[cell] = std::vector<PetscInt>(neighbors, neighbors + nNeighbors);
        DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);

        DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors1, &neighbors1);
        cellNeighbors1[cell] = std::vector<PetscInt>(neighbors1, neighbors1 + nNeighbors1);
        DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors1, &neighbors1);
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

    flow.RegisterRHSFunction(ComputeSource, this);
}

PetscErrorCode ablate::finiteVolume::processes::SurfaceForce::ComputeSource(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx) {
    PetscFunctionBegin;

    //    ablate::finiteVolume::processes::SurfaceForce *process = (ablate::finiteVolume::processes::SurfaceForce *)ctx;
    auto *process = (ablate::finiteVolume::processes::SurfaceForce *)ctx;
    std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
    subDomain->UpdateAuxLocalVector();

    const auto &phiField = subDomain->GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
    auto dim = solver.GetSubDomain().GetDimensions();

//PetscReal xymin[dim], xymax[dim]; DMGetBoundingBox(dm, xymin, xymax);
//PetscReal xmin=xymin[0];
//PetscReal xmax=xymax[0];
//PetscReal ymin=xymin[1];
//PetscReal ymax=xymax[1];
//PetscReal zmin=xymin[2];
//PetscReal zmax=xymax[2];

PetscReal xmin = -0.05; PetscReal xmax = 0.05; PetscReal ymin = -0.05; PetscReal ymax = 0.05; PetscReal zmin = 0; PetscReal zmax = 0.2;

    const auto &ofield3 = subDomain->GetField("phitilde_surfaceforce");
    const auto &ofield4 = subDomain->GetField("sf_magnitude");

    const auto &ofield5 = subDomain->GetField("rank");
    const auto &ofield6 = subDomain->GetField("kappa");

    const auto &eulerField = solver.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);

    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector();
    Vec vertexVec; DMGetLocalVector(process->vertexDM, &vertexVec);
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


    ablate::domain::Range vertexRange;
    GetVertexRange(dm, ablate::domain::Region::ENTIREDOMAIN, vertexRange);

    PetscInt vStart, vEnd; DMPlexGetDepthStratum(process->vertexDM, 0, &vStart, &vEnd); //this might replace the above later

    DM nvDM; //vertex based normal calc that is used for curvature calculation
    SF_CopyDM(process->vertexDM, vStart, vEnd, dim, &nvDM);
    Vec nvLocalVec; DMCreateLocalVector(nvDM, &nvLocalVec);
    Vec nvGlobalVec; DMCreateGlobalVector(nvDM, &nvGlobalVec);
    VecZeroEntries(nvLocalVec);
    VecZeroEntries(nvGlobalVec);
    PetscScalar *nvLocalArray; VecGetArray(nvLocalVec, &nvLocalArray);

    PetscInt cStart, cEnd; DMPlexGetHeightStratum(auxDM, 0, &cStart, &cEnd);

    DM nDM;
    SF_CopyDM(auxDM, cStart, cEnd, dim, &nDM); //////////////////////// do this? or do componentwise vecs? come back to this
    Vec nLocalVec; DMCreateLocalVector(nDM, &nLocalVec);
    Vec nGlobalVec; DMCreateGlobalVector(nDM, &nGlobalVec);
    VecZeroEntries(nLocalVec);
    VecZeroEntries(nGlobalVec);
    PetscScalar *nLocalArray; VecGetArray(nLocalVec, &nLocalArray);

    DM sfxDM;
    SF_CopyDM(auxDM, cStart, cEnd, 1, &sfxDM);
    Vec sfxLocalVec; DMCreateLocalVector(sfxDM, &sfxLocalVec);
    Vec sfxGlobalVec; DMCreateGlobalVector(sfxDM, &sfxGlobalVec);
    VecZeroEntries(sfxLocalVec);
    VecZeroEntries(sfxGlobalVec);
    PetscScalar *sfxLocalArray; VecGetArray(sfxLocalVec, &sfxLocalArray);

    DM sfyDM;
    SF_CopyDM(auxDM, cStart, cEnd, 1, &sfyDM);
    Vec sfyLocalVec; DMCreateLocalVector(sfyDM, &sfyLocalVec);
    Vec sfyGlobalVec; DMCreateGlobalVector(sfyDM, &sfyGlobalVec);
    VecZeroEntries(sfyLocalVec);
    VecZeroEntries(sfyGlobalVec);
    PetscScalar *sfyLocalArray; VecGetArray(sfyLocalVec, &sfyLocalArray);

    DM sfzDM;
    SF_CopyDM(auxDM, cStart, cEnd, 1, &sfzDM);
    Vec sfzLocalVec; DMCreateLocalVector(sfzDM, &sfzLocalVec);
    Vec sfzGlobalVec; DMCreateGlobalVector(sfzDM, &sfzGlobalVec);
    VecZeroEntries(sfzLocalVec);
    VecZeroEntries(sfzGlobalVec);
    PetscScalar *sfzLocalArray; VecGetArray(sfzLocalVec, &sfzLocalArray);

    DM rankDM;
    SF_CopyDM(auxDM, cStart, cEnd, 1, &rankDM);
    Vec rankLocalVec; DMCreateLocalVector(rankDM, &rankLocalVec);
    Vec rankGlobalVec; DMCreateGlobalVector(rankDM, &rankGlobalVec);
    VecZeroEntries(rankLocalVec);
    VecZeroEntries(rankGlobalVec);
    PetscScalar *rankLocalArray; VecGetArray(rankLocalVec, &rankLocalArray);

    DM kappaDM;
    SF_CopyDM(auxDM, cStart, cEnd, 1, &kappaDM);
    Vec kappaLocalVec; DMCreateLocalVector(kappaDM, &kappaLocalVec);
    Vec kappaGlobalVec; DMCreateGlobalVector(kappaDM, &kappaGlobalVec);
    VecZeroEntries(kappaLocalVec);
    VecZeroEntries(kappaGlobalVec);
    PetscScalar *kappaLocalArray; VecGetArray(kappaLocalVec, &kappaLocalArray);

    DM sfmaskDM;
    SF_CopyDM(auxDM, cStart, cEnd, 1, &sfmaskDM);
    Vec sfmaskLocalVec; DMCreateLocalVector(sfmaskDM, &sfmaskLocalVec);
    Vec sfmaskGlobalVec; DMCreateGlobalVector(sfmaskDM, &sfmaskGlobalVec);
    VecZeroEntries(sfmaskLocalVec);
    VecZeroEntries(sfmaskGlobalVec);
    PetscScalar *sfmaskLocalArray; VecGetArray(sfmaskLocalVec, &sfmaskLocalArray);

    DM phitildemaskDM;
    SF_CopyDM(auxDM, cStart, cEnd, 1, &phitildemaskDM);
    Vec phitildemaskLocalVec; DMCreateLocalVector(phitildemaskDM, &phitildemaskLocalVec);
    Vec phitildemaskGlobalVec; DMCreateGlobalVector(phitildemaskDM, &phitildemaskGlobalVec);
    VecZeroEntries(phitildemaskLocalVec);
    VecZeroEntries(phitildemaskGlobalVec);
    PetscScalar *phitildemaskLocalArray; VecGetArray(phitildemaskLocalVec, &phitildemaskLocalArray);

    DM phiDM;
    SF_CopyDM(auxDM, cStart, cEnd, 1, &phiDM);
    Vec phiLocalVec; DMCreateLocalVector(phiDM, &phiLocalVec);
    Vec phiGlobalVec; DMCreateGlobalVector(phiDM, &phiGlobalVec);
    VecZeroEntries(phiLocalVec);
    VecZeroEntries(phiGlobalVec);
    PetscScalar *phiLocalArray; VecGetArray(phiLocalVec, &phiLocalArray);

    DM phitildeDM;
    SF_CopyDM(auxDM, cStart, cEnd, 1, &phitildeDM);
    Vec phitildeLocalVec; DMCreateLocalVector(phitildeDM, &phitildeLocalVec);
    Vec phitildeGlobalVec; DMCreateGlobalVector(phitildeDM, &phitildeGlobalVec);
    VecZeroEntries(phitildeLocalVec);
    VecZeroEntries(phitildeGlobalVec);
    PetscScalar *phitildeLocalArray; VecGetArray(phitildeLocalVec, &phitildeLocalArray);

    DM cellidDM;
    SF_CopyDM(auxDM, cStart, cEnd, 1, &cellidDM);
    Vec cellidLocalVec; DMCreateLocalVector(cellidDM, &cellidLocalVec);
    Vec cellidGlobalVec; DMCreateGlobalVector(cellidDM, &cellidGlobalVec);
    VecZeroEntries(cellidLocalVec);
    VecZeroEntries(cellidGlobalVec);
    PetscScalar *cellidLocalArray; VecGetArray(cellidLocalVec, &cellidLocalArray);

    DM xDM;
    SF_CopyDM(auxDM, cStart, cEnd, 1, &xDM);
    DM yDM;
    SF_CopyDM(auxDM, cStart, cEnd, 1, &yDM);
    DM zDM;
    SF_CopyDM(auxDM, cStart, cEnd, 1, &zDM);
    Vec xLocalVec; DMCreateLocalVector(xDM, &xLocalVec);
    Vec xGlobalVec; DMCreateGlobalVector(xDM, &xGlobalVec);
    Vec yLocalVec; DMCreateLocalVector(yDM, &yLocalVec);
    Vec yGlobalVec; DMCreateGlobalVector(yDM, &yGlobalVec);
    Vec zLocalVec; DMCreateLocalVector(zDM, &zLocalVec);
    Vec zGlobalVec; DMCreateGlobalVector(zDM, &zGlobalVec);
    VecZeroEntries(xLocalVec);
    VecZeroEntries(xGlobalVec);
    VecZeroEntries(yLocalVec);
    VecZeroEntries(yGlobalVec);
    VecZeroEntries(zLocalVec);
    VecZeroEntries(zGlobalVec);
    PetscScalar *xLocalArray; VecGetArray(xLocalVec, &xLocalArray);
    PetscScalar *yLocalArray; VecGetArray(yLocalVec, &yLocalArray);
    PetscScalar *zLocalArray; VecGetArray(zLocalVec, &zLocalArray);

    bool verbose=false;

    int rank; MPI_Comm_rank(PETSC_COMM_WORLD, &rank); rank+=1;

    for (PetscInt cell = cStart; cell < cEnd; ++cell){
        PetscScalar *kappaptr; xDMPlexPointLocalRef(kappaDM, cell, -1, kappaLocalArray, &kappaptr);
        *kappaptr = 0;
        PetscScalar *sfmaskptr; xDMPlexPointLocalRef(sfmaskDM, cell, -1, sfmaskLocalArray, &sfmaskptr);
        *sfmaskptr = 0;
        PetscScalar *nptr; xDMPlexPointLocalRef(nDM, cell, -1, nLocalArray, &nptr);
        *nptr = 0;
        PetscScalar *sfxptr; xDMPlexPointLocalRef(sfxDM, cell, -1, sfxLocalArray, &sfxptr);
        *sfxptr = 0;
        PetscScalar *sfyptr; xDMPlexPointLocalRef(sfyDM, cell, -1, sfyLocalArray, &sfyptr);
        *sfyptr = 0;
        PetscScalar *sfzptr; xDMPlexPointLocalRef(sfzDM, cell, -1, sfzLocalArray, &sfzptr);
        *sfzptr = 0;

        PetscSection globalSection; DMGetGlobalSection(dm, &globalSection);
        PetscInt owned = 1; PetscSectionGetOffset(globalSection, cell, &owned);
        if (owned>=0){
            PetscScalar *rankcptr; xDMPlexPointLocalRef(rankDM, cell, -1, rankLocalArray, &rankcptr);
            *rankcptr = rank;
        }
        PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, cell, -1, phitildemaskLocalArray, &phitildemaskptr);
        *phitildemaskptr = 0;
        PetscScalar *phitildeptr; xDMPlexPointLocalRef(phitildeDM, cell, -1, phitildeLocalArray, &phitildeptr);
        *phitildeptr = 0;
        PetscScalar *cellidptr; xDMPlexPointLocalRef(cellidDM, cell, -1, cellidLocalArray, &cellidptr);
        *cellidptr = cell;
        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
        PetscScalar *phiptr; xDMPlexPointLocalRef(phiDM, cell, -1, phiLocalArray, &phiptr);
        *phiptr = *phic;
        PetscReal xp, yp, zp; Get3DCoordinate(dm, cell, &xp, &yp, &zp);
        PetscScalar *xptr; xDMPlexPointLocalRef(xDM, cell, -1, xLocalArray, &xptr);
        PetscScalar *yptr; xDMPlexPointLocalRef(yDM, cell, -1, yLocalArray, &yptr);
        PetscScalar *zptr; xDMPlexPointLocalRef(zDM, cell, -1, zLocalArray, &zptr);
        *xptr = xp; *yptr = yp; *zptr = zp;
    }

    //auxDM COPY
    for (PetscInt cell = cStart; cell < cEnd; ++cell){
//        const PetscScalar *phic; xDMPlexPointLocalRead(phiDM, cell, -1, phiLocalArray, &phic);
        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
        if (*phic > 1e-4 and *phic < 1-1e-4) {

            // neighboredit 1

            auto &neighbors = process->cellNeighbors1[cell];
            for (const auto &neighbor : neighbors) {

            // PetscInt nNeighbors, *neighbors;
            // DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
            // for (PetscInt j = 0; j < nNeighbors; ++j) {
                // PetscInt neighbor = neighbors[j];
                PetscScalar *ranknptr; xDMPlexPointLocalRef(rankDM, neighbor, -1, rankLocalArray, &ranknptr);
                PetscScalar *sfmaskptr; xDMPlexPointLocalRef(sfmaskDM, neighbor, -1, sfmaskLocalArray, &sfmaskptr);
                *sfmaskptr = *ranknptr;
            }
            // DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        }
    }
    PushToGhost(sfmaskDM, sfmaskLocalVec, sfmaskGlobalVec, ADD_VALUES);


//    PetscScalar C=2; PetscScalar N=2.6; PetscScalar layers = ceil(C*N);
//    PetscScalar C=1; PetscScalar N=2.6; 
PetscScalar layers = ceil(process->C*process->N);
PetscReal rmin; DMPlexGetMinRadius(dm, &rmin); PetscReal h=2*rmin + 0*layers;

//    layers = 2; //temporary; current limit for parallel
//    layers = 4; //temporary

    //auxDM copy
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, cell, -1, phitildemaskLocalArray, &phitildemaskptr);
        *phitildemaskptr = 0;
    }
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic) >> ablate::utilities::PetscUtilities::checkError;
        if (*phic > 0.0001 and *phic < 0.9999) {

//neighboredit 2

const auto &neighbors = process->cellNeighbors[cell];
for (const auto &neighbor : neighbors) {

            // PetscInt nNeighbors, *neighbors; DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors); //
            // for (PetscInt j = 0; j < nNeighbors; ++j) {
            //     PetscInt neighbor = neighbors[j];
                PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, neighbor, -1, phitildemaskLocalArray, &phitildemaskptr);
                *phitildemaskptr = 1;
            }
            // DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        }
    }
    PushToGhost(phitildemaskDM, phitildemaskLocalVec, phitildemaskGlobalVec, ADD_VALUES);
    if (verbose){SaveDataToFile(cellRange.start, cellRange.end, phitildemaskDM, phitildemaskLocalArray, "phitildemask", true);}

    //phitilde auxDM copy
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
        PetscReal xc, yc, zc; Get3DCoordinate(dm, cell, &xc, &yc, &zc);
        PetscScalar *phitilde; xDMPlexPointLocalRef(phitildeDM, cell, -1, phitildeLocalArray, &phitilde);
        PetscScalar *phitildemask; xDMPlexPointLocalRef(phitildemaskDM, cell, -1, phitildemaskLocalArray, &phitildemask);
        if (*phitildemask < 1e-10){ *phitilde = *phic;
if (process->flipPhiTilde){*phitilde = 1.00- *phitilde;} }
        else{

            // neighboredit 3


            // PetscInt nNeighbors, *neighbors; DMPlexGetNeighbors(auxDM, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
            PetscReal weightedphi = 0; PetscReal Tw = 0;

            auto &neighbors = process->cellNeighbors[cell];
            for (const auto &neighbor : neighbors) {
            // for (PetscInt j = 0; j < nNeighbors; ++j) {
                // PetscInt neighbor = neighbors[j];
                PetscReal *phin; xDMPlexPointLocalRead(dm, neighbor, phiField.id, solArray, &phin);
                PetscReal xn, yn, zn; Get3DCoordinate(dm, neighbor, &xn, &yn, &zn);


bool periodicfix = true;

if (periodicfix){

//temporary fix addressing how multiple layers of neighbors for a periodic domain return coordinates on the opposite side
PetscReal maxMask = 0.5*(ymax-ymin);
if (( PetscAbs(xn-xc) > maxMask) and (xn > xc)){  
    
    // PetscPrintf(PETSC_COMM_WORLD, "C- N+ BEFORE xn: %f, xc: %f, xmax: %f, xmin: %f, maxMask: %f\n", xn, xc, xmax, xmin, maxMask);
    xn -= (xmax-xmin);  
    // PetscPrintf(PETSC_COMM_WORLD, "C- N+ AFTER xn: %f, xc: %f, xmax: %f, xmin: %f, maxMask: %f\n", xn, xc, xmax, xmin, maxMask);
}
if (( PetscAbs(xn-xc) > maxMask) and (xn < xc)){  

    // PetscPrintf(PETSC_COMM_WORLD, "C+ N- BEFORE xn: %f, xc: %f, xmax: %f, xmin: %f, maxMask: %f\n", xn, xc, xmax, xmin, maxMask);    
    xn += (xmax-xmin);
    // PetscPrintf(PETSC_COMM_WORLD, "C+ N- AFTER xn: %f, xc: %f, xmax: %f, xmin: %f, maxMask: %f\n", xn, xc, xmax, xmin, maxMask);

}
if (dim>=2){
if (( PetscAbs(yn-yc) > maxMask) and (yn > yc)){  yn -= (ymax-ymin);  }
if (( PetscAbs(yn-yc) > maxMask) and (yn < yc)){  yn += (ymax-ymin);  } }
if (dim==3){
if (( PetscAbs(zn-zc) > maxMask) and (zn > zc)){  zn -= (zmax-zmin);  }
if (( PetscAbs(zn-zc) > maxMask) and (zn < zc)){  zn += (zmax-zmin);  } }

}


                PetscReal d = PetscSqrtReal(PetscSqr(xn - xc) + PetscSqr(yn - yc) + PetscSqr(zn - zc));  // distance
                PetscReal s = process->C * h; //6*h
                PetscReal wn; PhiNeighborGaussWeight(d, s, &wn);
                Tw += wn;
                weightedphi += (*phin * wn);

PetscScalar *rankptr; xDMPlexPointLocalRef(rankDM, cell, -1, rankLocalArray, &rankptr);
//this loop is where phitilde is being actually calculated.
//Whether this next line is either included or commented out changes the result in the next loop
if ((cell!=-1) and (*rankptr != -1)){ std::cout << ""; }//std::cout << ""; }//  std::cout << "weightedphi and Tw (surfaceForce)" << weightedphi << "  " << Tw << "\n";}

            }
            weightedphi /= Tw;

if (process->flipPhiTilde){weightedphi = 1.000-weightedphi;}
            *phitilde = weightedphi;


            // DMPlexRestoreNeighbors(auxDM, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        }
    }
    PushToGhost(phitildeDM, phitildeLocalVec, phitildeGlobalVec, INSERT_VALUES);
    if (verbose){SaveDataToFile(cellRange.start, cellRange.end, phitildeDM, phitildeLocalArray, "phitilde", true);}

//MPI_Barrier(PETSC_COMM_WORLD);

//this is a check to see if phitilde is being recalled correctly.
for (PetscInt cell = cStart; cell < cEnd; ++cell) {
PetscScalar *optr3; PetscScalar *phitildeptr; 
xDMPlexPointLocalRef(phitildeDM, cell, -1, phitildeLocalArray, &phitildeptr); 
xDMPlexPointLocalRef(auxDM, cell, ofield3.id, auxArray, &optr3);
*optr3 = *phitildeptr;
PetscScalar *rankptr; xDMPlexPointLocalRef(rankDM, cell, -1, rankLocalArray, &rankptr);
if ((cell==0) and (*rankptr == 5)){  std::cout << "";   }
}


    //auxDM copy
    for (PetscInt vertex = vStart; vertex < vEnd; vertex++) {
        PetscScalar *nvptr; xDMPlexPointLocalRef(nvDM, vertex, -1, nvLocalArray, &nvptr);
        *nvptr = 0;
    }
    for (PetscInt vertex = vStart; vertex < vEnd; vertex++) {
        PetscReal vx, vy, vz; Get3DCoordinate(dm, vertex, &vx, &vy, &vz);
        PetscInt nCells, *cells; DMPlexVertexGetCells(dm, vertex, &nCells, &cells);
        PetscBool isAdjToMask = PETSC_FALSE;
        for (PetscInt k = 0; k < nCells; k++){
            PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, cells[k], -1, phitildemaskLocalArray, &phitildemaskptr) >> ablate::utilities::PetscUtilities::checkError;
            if (*phitildemaskptr > 0.5){
                isAdjToMask = PETSC_TRUE;
            }
        }
        PetscScalar *nv; xDMPlexPointLocalRef(nvDM, vertex, -1, nvLocalArray, &nv);
        if (isAdjToMask == PETSC_TRUE){
            DMPlexVertexGradFromCell(phitildeDM, vertex, phitildeLocalVec, -1, 0, nv);
            //surface area force DOES normalize;
            //surface volume force DOES NOT normalize
            if (utilities::MathUtilities::MagVector(dim, nv) > 1e-10) { utilities::MathUtilities::NormVector(dim, nv); }
        }
        else{ *nv=0; }
        DMPlexVertexRestoreCells(dm, vertex, &nCells, &cells);
    }
    PushToGhost(nvDM, nvLocalVec, nvGlobalVec, INSERT_VALUES);


    //kappa auxDM copy
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        PetscReal xc, yc, zc; Get3DCoordinate(dm, cell, &xc, &yc, &zc);
        PetscReal kappa=0, Nx, Ny, Nz;
        const PetscReal *phitilde; xDMPlexPointLocalRead(phitildeDM, cell, -1, phitildeLocalArray, &phitilde);
        const PetscReal *phic; xDMPlexPointLocalRead(phiDM, cell, -1, phiLocalArray, &phic);
        PetscScalar *sfmask; xDMPlexPointLocalRef(sfmaskDM, cell, -1, sfmaskLocalArray, &sfmask);
        if (*sfmask > 0.5){
            PetscReal nabla_n[dim];
            for (PetscInt offset = 0; offset < dim; offset++) {
                DMPlexCellGradFromVertex(nvDM, cell, nvLocalVec, -1, offset, nabla_n);
                kappa += nabla_n[offset];
            }
            PetscScalar gradphic[dim];
            if (dim==1){
                PetscInt gNeighbors, *neighbors; DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &gNeighbors, &neighbors);
                const PetscScalar *phitildekm1; xDMPlexPointLocalRead(phitildeDM, neighbors[0], -1, phitildeLocalArray, &phitildekm1) >> ablate::utilities::PetscUtilities::checkError;
                const PetscScalar *phitildekp1; xDMPlexPointLocalRead(phitildeDM, neighbors[2], -1, phitildeLocalArray, &phitildekp1) >> ablate::utilities::PetscUtilities::checkError;
                PetscReal xm1, ym1, zm1; Get3DCoordinate(dm, neighbors[0], &xm1, &ym1, &zm1);
                PetscReal xp1, yp1, zp1; Get3DCoordinate(dm, neighbors[2], &xp1, &yp1, &zp1);
                gradphic[0] = (*phitildekp1-*phitildekm1)/(xp1-xm1);
                gradphic[1] = 0;
                DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &gNeighbors, &neighbors);
            }
            else{ DMPlexCellGradFromCell(phitildeDM, cell, phitildeLocalVec, -1, 0, gradphic);}

            Nx = gradphic[0];
            Ny = gradphic[1];
            Nz = gradphic[2];
            if (dim==1){ PetscReal xp; Get1DCoordinate(dm, dim, cell, &xp); kappa=-1; }
        }
        else { kappa=Nx=Ny=Nz=0; }
        kappa *= -1; Nx *= -1; Ny *= -1; Nz *= -1;
        PetscScalar *kappaptr; xDMPlexPointLocalRef(kappaDM, cell, -1, kappaLocalArray, &kappaptr) >> ablate::utilities::PetscUtilities::checkError;
        PetscScalar *nptr; xDMPlexPointLocalRef(nDM, cell, -1, nLocalArray, &nptr) >> ablate::utilities::PetscUtilities::checkError;
        *kappaptr = kappa;
        nptr[0]=Nx; nptr[1]=Ny; nptr[2]=Nz;
    }
    PushToGhost(kappaDM, kappaLocalVec, kappaGlobalVec, INSERT_VALUES);

    if (verbose){SaveDataToFile(cellRange.start, cellRange.end, xDM, xLocalArray, "x", false);}
    if (verbose){SaveDataToFile(cellRange.start, cellRange.end, yDM, yLocalArray, "y", false);}
    if (verbose){SaveDataToFile(cellRange.start, cellRange.end, zDM, zLocalArray, "z", false);}
    if (verbose){SaveDataToFile(cellRange.start, cellRange.end, kappaDM, kappaLocalArray, "kappa", true);}

    for (PetscInt cell = cStart; cell < cEnd; ++cell){
        PetscReal *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, cell, -1, phitildemaskLocalArray, &phitildemaskptr);
        PetscScalar *kappaptr; xDMPlexPointLocalRef(kappaDM, cell, -1, kappaLocalArray, &kappaptr);
        PetscScalar *nptr; xDMPlexPointLocalRef(nDM, cell, -1, nLocalArray, &nptr);
        PetscScalar *sfxptr; xDMPlexPointLocalRef(sfxDM, cell, -1, sfxLocalArray, &sfxptr);
        PetscScalar *sfyptr; xDMPlexPointLocalRef(sfyDM, cell, -1, sfyLocalArray, &sfyptr);
        PetscScalar *sfzptr; xDMPlexPointLocalRef(sfzDM, cell, -1, sfzLocalArray, &sfzptr);
        if(*phitildemaskptr > 0.5){
            *sfxptr = process->sigma * *kappaptr * -nptr[0];
            *sfyptr = process->sigma * *kappaptr * -nptr[1];
            *sfzptr = process->sigma * *kappaptr * -nptr[2];
        }
        else{ *sfxptr = 0; *sfyptr = 0; *sfzptr = 0; } //sfptr[2]=0; }
    }
    PushToGhost(sfxDM, sfxLocalVec, sfxGlobalVec, INSERT_VALUES);
    PushToGhost(sfyDM, sfyLocalVec, sfyGlobalVec, INSERT_VALUES);
    PushToGhost(sfzDM, sfzLocalVec, sfzGlobalVec, INSERT_VALUES);

    //add to rhs (auxdm copy)
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.GetPoint(c);
        //get the phi ptr
        PetscScalar *phiptr; xDMPlexPointLocalRef(phiDM, cell, -1, phiLocalArray, &phiptr);

        PetscReal *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, cell, -1, phitildemaskLocalArray, &phitildemaskptr);
//        PetscScalar *kappaptr; xDMPlexPointLocalRef(kappaDM, cell, -1, kappaLocalArray, &kappaptr);
//        PetscScalar *nptr; xDMPlexPointLocalRef(nDM, cell, -1, nLocalArray, &nptr);
        PetscScalar *sfxptr; xDMPlexPointLocalRef(sfxDM, cell, -1, sfxLocalArray, &sfxptr);
        PetscScalar *sfyptr; xDMPlexPointLocalRef(sfyDM, cell, -1, sfyLocalArray, &sfyptr);
        PetscScalar *sfzptr; xDMPlexPointLocalRef(sfzDM, cell, -1, sfzLocalArray, &sfzptr);

        const PetscScalar *euler = nullptr;
        PetscScalar *eulerSource = nullptr;
        PetscCall(DMPlexPointLocalFieldRef(dm, cell, eulerField.id, fArray, &eulerSource));
        PetscCall(DMPlexPointLocalFieldRead(dm, cell, eulerField.id, solArray, &euler));
        auto density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];
        PetscReal ux = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] / density;
        PetscReal uy = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + 1] / density;
        PetscReal uz = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + 2] / density;


PetscScalar *xc; xDMPlexPointLocalRef(xDM, cell, -1, xLocalArray, &xc);
PetscScalar *yc; xDMPlexPointLocalRef(yDM, cell, -1, yLocalArray, &yc);

//just focus on the pinchoff
if (*xc < 0.027 || *xc > 0.048){*sfxptr = 0; *sfyptr = 0; *sfzptr = 0;}

        if (PetscAbs(*sfxptr) > 1e-10){eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU] += *sfxptr;}
        if (PetscAbs(*sfyptr) > 1e-10){eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOV] += *sfyptr;}
        if (dim==3){if (PetscAbs(*sfzptr) > 1e-10){eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOW] += *sfzptr;}}
        if (PetscAbs(*sfxptr) > 1e-10){eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += *sfxptr * ux;}
        if (PetscAbs(*sfyptr) > 1e-10){eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += *sfyptr * uy;}
        if (dim==3){if (PetscAbs(*sfzptr) > 1e-10){eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += *sfzptr * uz;}}

        //pin liquid parent
        if (*yc < 0.01042 && *phiptr < 0.9999) {
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU] = -1e-10;
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOV] = -1e-10;
            if (dim==3){eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOW] = -1e-10;}
        }


        PetscScalar *optr3; xDMPlexPointLocalRef(auxDM, cell, ofield3.id, auxArray, &optr3);
        PetscScalar *optr4; xDMPlexPointLocalRef(auxDM, cell, ofield4.id, auxArray, &optr4);

PetscScalar *optr5; xDMPlexPointLocalRef(auxDM, cell, ofield5.id, auxArray, &optr5);
PetscScalar *optr6; xDMPlexPointLocalRef(auxDM, cell, ofield6.id, auxArray, &optr6);

//        PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, cell, -1, phitildemaskLocalArray, &phitildemaskptr);

        PetscScalar *phitildeptr; xDMPlexPointLocalRef(phitildeDM, cell, -1, phitildeLocalArray, &phitildeptr);
        *optr3 = *phitildeptr;
        if (dim==3){*optr4 = PetscSqrtReal(PetscSqr(*sfxptr) + PetscSqr(*sfyptr) + PetscSqr(*sfzptr));}
        if (dim<3){*optr4 = PetscSqrtReal(PetscSqr(*sfxptr) + PetscSqr(*sfyptr));}



PetscScalar *rankptr; xDMPlexPointLocalRef(rankDM, cell, -1, rankLocalArray, &rankptr);
PetscScalar *kappaptr; xDMPlexPointLocalRef(kappaDM, cell, -1, kappaLocalArray, &kappaptr);

*optr5 = *rankptr;
*optr6 = *kappaptr; // *phitildemaskptr;


    }
    if (verbose){
        SaveDataToFile(cellRange.start, cellRange.end, sfxDM, sfxLocalArray, "sfx", true);
        SaveDataToFile(cellRange.start, cellRange.end, sfyDM, sfyLocalArray, "sfy", true);
        SaveDataToFile(cellRange.start, cellRange.end, sfzDM, sfzLocalArray, "sfz", true);
    }



//    std::cout << "surfaceForce is done\n";

    VecRestoreArray(nvLocalVec, &nvLocalArray);
    DMRestoreLocalVector(nvDM, &nvLocalVec);
    DMRestoreGlobalVector(nvDM, &nvGlobalVec);
    DMDestroy(&nvDM);

    VecRestoreArray(nLocalVec, &nLocalArray);
    DMRestoreLocalVector(nDM, &nLocalVec);
    DMRestoreGlobalVector(nDM, &nGlobalVec);
    DMDestroy(&nDM);

    VecRestoreArray(sfxLocalVec, &sfxLocalArray);
    DMRestoreLocalVector(sfxDM, &sfxLocalVec);
    DMRestoreGlobalVector(sfxDM, &sfxGlobalVec);
    DMDestroy(&sfxDM);

    VecRestoreArray(sfyLocalVec, &sfyLocalArray);
    DMRestoreLocalVector(sfyDM, &sfyLocalVec);
    DMRestoreGlobalVector(sfyDM, &sfyGlobalVec);
    DMDestroy(&sfyDM);

    VecRestoreArray(sfzLocalVec, &sfzLocalArray);
    DMRestoreLocalVector(sfzDM, &sfzLocalVec);
    DMRestoreGlobalVector(sfzDM, &sfzGlobalVec);
    DMDestroy(&sfzDM);
    
    VecRestoreArray(kappaLocalVec, &kappaLocalArray);
    DMRestoreLocalVector(kappaDM, &kappaLocalVec);
    DMRestoreGlobalVector(kappaDM, &kappaGlobalVec);
    DMDestroy(&kappaDM);
    
    VecRestoreArray(sfmaskLocalVec, &sfmaskLocalArray);
    DMRestoreLocalVector(sfmaskDM, &sfmaskLocalVec);
    DMRestoreGlobalVector(sfmaskDM, &sfmaskGlobalVec);
    DMDestroy(&sfmaskDM);
    
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
    DMDestroy(&zDM);

    VecRestoreArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(vertexVec, &vertexArray);
    VecRestoreArray(locFVec, &fArray);
    DMRestoreLocalVector(process->vertexDM, &vertexVec);
    VecDestroy(&vertexVec);
    solver.RestoreRange(cellRange);
    solver.RestoreRange(vertexRange);


    PetscFunctionReturn(0);
}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::SurfaceForce, "calculates surface tension force and adds source terms",
         ARG(PetscReal, "sigma", "sigma, surface tension coefficient"),
         ARG(PetscReal, "C", "stdev length with respect to grid spacing magnitude (default 1)"),
         ARG(PetscReal, "N", "number of stdevs that the convolution integral captures (default 2.6 for 99 pct accuracy if C=1)"),
         ARG(bool, "flipPhiTilde", "if true: phiTilde-->1-phiTilde (set it to true if primary phase is phi=0 or false if phi=1)")
);