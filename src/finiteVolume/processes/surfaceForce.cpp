#include "domain/RBF/mq.hpp"
#include "surfaceForce.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "levelSet/levelSetUtilities.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"

#include <fstream>

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
    PetscInt rank; MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
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

    flow.RegisterRHSFunction(ComputeSource, this);
}

PetscErrorCode ablate::finiteVolume::processes::SurfaceForce::ComputeSource(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx) {
    PetscFunctionBegin;

    //    ablate::finiteVolume::processes::SurfaceForce *process = (ablate::finiteVolume::processes::SurfaceForce *)ctx;
    auto *process = (ablate::finiteVolume::processes::SurfaceForce *)ctx;
    std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
    subDomain->UpdateAuxLocalVector();

    const auto &phiField = subDomain->GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
//    const auto &phiTildeField = subDomain->GetField("SFphiTilde");
//    const auto &kappaField = subDomain->GetField("kappa");
////    const auto &kappaTildeField = subDomain->GetField("kappaTilde");
////    const auto &xField = subDomain->GetField("x");
////    const auto &yField = subDomain->GetField("y");
////    const auto &zField = subDomain->GetField("z");
//    const auto &n0Field = subDomain->GetField("n0");
//    const auto &n1Field = subDomain->GetField("n1");
//    const auto &n2Field = subDomain->GetField("n2");
//    const auto &CSF0Field = subDomain->GetField("SF0");
//    const auto &CSF1Field = subDomain->GetField("SF1");
//    const auto &CSF2Field = subDomain->GetField("SF2");
//    const auto &SFMaskField = subDomain->GetField("SFMask");
//    const auto &phiTildeMaskField = subDomain->GetField("SFphiTildeMask");
//    const auto &CSF0TildeField = subDomain->GetField("SF0Tilde");
//    const auto &CSF1TildeField = subDomain->GetField("SF1Tilde");
//    const auto &CSF2TildeField = subDomain->GetField("SF2Tilde");
    auto dim = solver.GetSubDomain().GetDimensions();
    const auto &ofield = subDomain->GetField("debug");
    const auto &ofield2 = subDomain->GetField("debug2");
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

//    if (interpCellList == nullptr) {
//        BuildInterpCellList(auxDM, cellRange);
//    }

//    PetscInt polyAug = 2;
//    bool doesNotHaveDerivatives = false;
//    bool doesNotHaveInterpolation = false;

//    ablate::domain::rbf::MQ cellRBF(polyAug, rmin, doesNotHaveDerivatives, doesNotHaveInterpolation);
//    cellRBF.Setup(subDomain);
//    cellRBF.Initialize();

    bool verbose=false;

    PetscInt rank; MPI_Comm_rank(PETSC_COMM_WORLD, &rank); rank+=1;

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
        *xptr = xp; *yptr = yp;
    }
//    //Mask field determines which cells will be operated on at all. auxDM (delete asap_
//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//        PetscInt cell = cellRange.GetPoint(c);
//        PetscScalar *Mask; xDMPlexPointLocalRef(auxDM, cell, SFMaskField.id, auxArray, &Mask);// >> ablate::utilities::PetscUtilities::checkError;
//        *Mask = 0;
//    }

    //auxDM COPY
    for (PetscInt cell = cStart; cell < cEnd; ++cell){
//        const PetscScalar *phic; xDMPlexPointLocalRead(phiDM, cell, -1, phiLocalArray, &phic);
        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
        if (*phic > 1e-4 and *phic < 1-1e-4) {
            PetscInt nNeighbors, *neighbors;
            DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
            for (PetscInt j = 0; j < nNeighbors; ++j) {
                PetscInt neighbor = neighbors[j];
                PetscScalar *ranknptr; xDMPlexPointLocalRef(rankDM, neighbor, -1, rankLocalArray, &ranknptr);
                PetscScalar *sfmaskptr; xDMPlexPointLocalRef(sfmaskDM, neighbor, -1, sfmaskLocalArray, &sfmaskptr);
                *sfmaskptr = *ranknptr;
            }
            DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        }
    }
    PushToGhost(sfmaskDM, sfmaskLocalVec, sfmaskGlobalVec, ADD_VALUES);

//    for (PetscInt cell = cStart; cell < cEnd; ++cell){
//        const PetscScalar *phic; xDMPlexPointLocalRead(phiDM, cell, -1, phiLocalArray, &phic);
//        if (*phic > 1e-4 and *phic < 1-1e-4) {
//            PetscScalar *sfmaskptr; xDMPlexPointLocalRef(sfmaskDM, cell, -1, sfmaskLocalArray, &sfmaskptr); //after vec surgery
//            *sfmaskptr = 5;
//        }
//    }

//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//        PetscInt cell = cellRange.GetPoint(c);
//        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic) >> ablate::utilities::PetscUtilities::checkError;
//        if (*phic > 0.0001 and *phic < 0.9999) {
//            PetscInt nNeighbors, *neighbors; DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
//            for (PetscInt j = 0; j < nNeighbors; ++j) {
//                PetscInt neighbor = neighbors[j];
//                PetscScalar *Mask; xDMPlexPointLocalRef(auxDM, neighbor, SFMaskField.id, auxArray, &Mask) >> ablate::utilities::PetscUtilities::checkError;
//                *Mask = 1;
//            }
//            DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
//        }
//    }

    PetscReal rmin; DMPlexGetMinRadius(dm, &rmin); PetscReal h=2*rmin;
    PetscScalar C=2; PetscScalar N=2.6; PetscScalar layers = ceil(C*N);
//    layers = 2; //temporary; current limit for parallel
    layers = 4; //temporary

    //auxDM copy
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
            }
            DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        }
    }
    PushToGhost(phitildemaskDM, phitildemaskLocalVec, phitildemaskGlobalVec, ADD_VALUES);
    if (verbose){SaveDataToFile(cellRange.start, cellRange.end, phitildemaskDM, phitildemaskLocalArray, "phitildemask", true);}

    //auxDM (delete)
//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//        PetscInt cell = cellRange.GetPoint(c);
//        PetscScalar *phiTildeMask; xDMPlexPointLocalRef(auxDM, cell, phiTildeMaskField.id, auxArray, &phiTildeMask);// >> ablate::utilities::PetscUtilities::checkError;
//        *phiTildeMask = 0;
//    }
//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//        PetscInt cell = cellRange.GetPoint(c);
//        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic) >> ablate::utilities::PetscUtilities::checkError;
//        if (*phic > 0.0001 and *phic < 0.9999) {
//            PetscInt nNeighbors, *neighbors; DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
//            for (PetscInt j = 0; j < nNeighbors; ++j) {
//                PetscInt neighbor = neighbors[j];
//                PetscScalar *phiTildeMask; xDMPlexPointLocalRef(auxDM, neighbor, phiTildeMaskField.id, auxArray, &phiTildeMask) >> ablate::utilities::PetscUtilities::checkError;
//                *phiTildeMask = 1;
//            }
//            DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
//        }
//    }

    //phitilde auxDM copy
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
        PetscReal xc, yc, zc; Get3DCoordinate(dm, cell, &xc, &yc, &zc);
        PetscScalar *phitilde; xDMPlexPointLocalRef(phitildeDM, cell, -1, phitildeLocalArray, &phitilde);
        PetscScalar *phitildemask; xDMPlexPointLocalRef(phitildemaskDM, cell, -1, phitildemaskLocalArray, &phitildemask);
        if (*phitildemask < 1e-10){ *phitilde = *phic; }
        else{
            PetscInt nNeighbors, *neighbors; DMPlexGetNeighbors(auxDM, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
            PetscReal weightedphi = 0; PetscReal Tw = 0;
            for (PetscInt j = 0; j < nNeighbors; ++j) {
                PetscInt neighbor = neighbors[j];
                PetscReal *phin; xDMPlexPointLocalRead(dm, neighbor, phiField.id, solArray, &phin);
                PetscReal xn, yn, zn; Get3DCoordinate(dm, neighbor, &xn, &yn, &zn);
                PetscReal d = PetscSqrtReal(PetscSqr(xn - xc) + PetscSqr(yn - yc) + PetscSqr(zn - zc));  // distance
                PetscReal s = C * h; //6*h
                PetscReal wn; PhiNeighborGaussWeight(d, s, &wn);
                Tw += wn;
                weightedphi += (*phin * wn);
            }
            weightedphi /= Tw;
            *phitilde = weightedphi;
            DMPlexRestoreNeighbors(auxDM, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        }
    }
    PushToGhost(phitildeDM, phitildeLocalVec, phitildeGlobalVec, INSERT_VALUES);
    if (verbose){SaveDataToFile(cellRange.start, cellRange.end, phitildeDM, phitildeLocalArray, "phitilde", true);}

    //good up to this point

    //auxdm (delete asap)
//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//        PetscInt cell = cellRange.GetPoint(c);
//        const PetscScalar *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic) >> ablate::utilities::PetscUtilities::checkError;
//        PetscReal xc, yc, zc; Get3DCoordinate(dm, cell, &xc, &yc, &zc);
//        PetscScalar *phiTilde; xDMPlexPointLocalRef(auxDM, cell, phiTildeField.id, auxArray, &phiTilde) >> ablate::utilities::PetscUtilities::checkError;
//        PetscScalar *phiTildeMask; xDMPlexPointLocalRef(auxDM, cell, phiTildeMaskField.id, auxArray, &phiTildeMask) >> ablate::utilities::PetscUtilities::checkError;
//        if (*phiTildeMask < 0.5){ *phiTilde=*phic; }
//        else{
//            PetscInt nNeighbors, *neighbors; DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
//            PetscReal weightedphi = 0;
//            PetscReal Tw = 0;
//            for (PetscInt j = 0; j < nNeighbors; ++j) {
//                PetscInt neighbor = neighbors[j];
//                PetscReal *phin;
//                xDMPlexPointLocalRead(dm, neighbor, phiField.id, solArray, &phin);
//                PetscReal xn, yn, zn;
//                Get3DCoordinate(dm, neighbor, &xn, &yn, &zn);
//                PetscReal d = PetscSqrtReal(PetscSqr(xn - xc) + PetscSqr(yn - yc) + PetscSqr(zn - zc));  // distance
//                PetscReal s = C * h; //6*h
//                PetscReal wn; PhiNeighborGaussWeight(d, s, &wn);
//                Tw += wn;
//                weightedphi += (*phin * wn);
//            }
//            weightedphi /= Tw;
//            *phiTilde = weightedphi;
//            DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
//        }
//    }
    // VERTEX BASED

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



    //auxDM (delete asap)
    //vertex based normals<-cell based phi
//    for (PetscInt j = vertexRange.start; j < vertexRange.end; j++){
//        const PetscInt vertex = vertexRange.GetPoint(j);
//
//        //if ALL of the vertex's cell neighbors are not in Mask, don't bother with calculation.
//        PetscInt nCells, *cells; DMPlexVertexGetCells(dm, vertex, &nCells, &cells);
//        PetscBool isAdjToMask = PETSC_FALSE;
//        for (PetscInt k = 0; k < nCells; k++){
//            PetscScalar *Mask; xDMPlexPointLocalRef(auxDM, cells[k], phiTildeMaskField.id, auxArray, &Mask) >> ablate::utilities::PetscUtilities::checkError;
//            if (*Mask > 0.5){
//                isAdjToMask = PETSC_TRUE;
//            }
//        }
////        std::cout << "    vertex " << vertex  << "  Mask? " << isAdjToMask << "\n";
//        PetscScalar *gradPhi_v; DMPlexPointLocalFieldRef(process->vertexDM, vertex, 0, vertexArray, &gradPhi_v);
//        if (isAdjToMask == PETSC_TRUE){
//            DMPlexVertexGradFromCell(auxDM, vertex, auxVec, phiTildeField.id, 0, gradPhi_v);
//            //surface area force DOES normalize;
//            //surface volume force DOES NOT normalize
//            if (utilities::MathUtilities::MagVector(dim, gradPhi_v) > 1e-10) { utilities::MathUtilities::NormVector(dim, gradPhi_v);}
//        }
//        else{ *gradPhi_v = 0;}
//        DMPlexVertexRestoreCells(dm, vertex, &nCells, &cells);
//    }
    //        subDomain->UpdateAuxLocalVector();


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
    if (verbose){SaveDataToFile(cellRange.start, cellRange.end, kappaDM, kappaLocalArray, "kappa", true);}

    //auxDM (delete asap)
    //cell based kappa<-vertex based normals, cell based normals<-cell based phi
//    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
//
//        const PetscInt cell = cellRange.GetPoint(i);
//        PetscReal xc, yc, zc; Get3DCoordinate(dm, cell, &xc, &yc, &zc);
//        PetscReal kappa=0, Nx, Ny, Nz;
//        const PetscReal *phiTilde; xDMPlexPointLocalRead(auxDM, cell, phiTildeField.id, auxArray, &phiTilde);
//        const PetscReal *phic; xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
//        PetscScalar *Mask; xDMPlexPointLocalRef(auxDM, cell, SFMaskField.id, auxArray, &Mask);
//            if (*Mask > 0.5){
//
//                PetscReal nabla_n[dim];
//                for (PetscInt offset = 0; offset < dim; offset++) {
//
////                std::cout << "cell   " << cell << "   (" << xc << ",    "<<yc<<")"<<"\n";
//
//                    DMPlexCellGradFromVertex(process->vertexDM, cell, vertexVec, 0, offset, nabla_n);
//                    kappa += nabla_n[offset];
//                }
//
//                PetscScalar gradPhi_c[dim];
//
//                //if 1D, just do centered differencing to calculate gradphi at centers. otherwise do Morgan/Waltz grad calc.
//                if (dim==1){
//
//                    PetscInt gNeighbors, *neighbors; DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &gNeighbors, &neighbors);
//                    const PetscScalar *phiTildekm1; xDMPlexPointLocalRead(auxDM, neighbors[0], phiTildeField.id, auxArray, &phiTildekm1) >> ablate::utilities::PetscUtilities::checkError;
////                    const PetscScalar *phik; xDMPlexPointLocalRead(dm, neighbors[1], phiField.id, solArray, &phik) >> ablate::utilities::PetscUtilities::checkError;
//                    const PetscScalar *phiTildekp1; xDMPlexPointLocalRead(auxDM, neighbors[2], phiTildeField.id, auxArray, &phiTildekp1) >> ablate::utilities::PetscUtilities::checkError;
//                    PetscReal xm1, ym1, zm1; Get3DCoordinate(dm, neighbors[0], &xm1, &ym1, &zm1);
//                    PetscReal xp1, yp1, zp1; Get3DCoordinate(dm, neighbors[2], &xp1, &yp1, &zp1);
//                    gradPhi_c[0] = (*phiTildekp1-*phiTildekm1)/(xp1-xm1);
//                    gradPhi_c[1] = 0;
//
//                    DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &gNeighbors, &neighbors);
//                }
//                else{
//                    DMPlexCellGradFromCell(auxDM, cell, auxVec, phiTildeField.id, 0, gradPhi_c);
//                }
//                //surface area force DOES normalize;
//                //surface volume force DOES NOT normalize
////                if (utilities::MathUtilities::MagVector(dim, gradPhi_c) > 1e-10) {
////                    utilities::MathUtilities::NormVector(dim, gradPhi_c);
////                }
//
//                Nx = gradPhi_c[0];
//                Ny = gradPhi_c[1];
//                //Nz = gradPhi_c[2];
//                Nz = 0;
//                if (dim==1){
//                    PetscReal xp; Get1DCoordinate(dm, dim, cell, &xp);
//                    kappa=-1;//artificial curvature!!!!!!! 1
////                    kappa = -1/(PetscAbs(xp-0.495));//artificial curvature!!!!!!! 1/r
//                }
//        }
//        else {
//            kappa=Nx=Ny=Nz=0;
//        }
//        kappa *= -1;
//        Nx *= -1;
//        Ny *= -1;
//        Nz *= -1;
//
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
//    }
    //        subDomain->UpdateAuxLocalVector();

    //kappaTilde
//    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
//        const PetscInt cell = cellRange.GetPoint(i);
//
//        PetscReal xc, yc, zc;
//        Get3DCoordinate(dm, cell, &xc, &yc, &zc);
//        PetscReal *kappac=0;
//        PetscReal *kappaTilde;
//        xDMPlexPointLocalRef(auxDM, cell, kappaField.id, auxArray, &kappac) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, kappaTildeField.id, auxArray, &kappaTilde) >> ablate::utilities::PetscUtilities::checkError;
//        *kappaTilde=0;
//        PetscInt nNeighbors, *neighbors;
//        DMPlexGetNeighbors(dm, cell, 8, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
//        //4 layers
//        PetscReal Tw = 0;
//        if (PetscAbs(*kappac) > 1e-4) {
//            for (PetscInt j = 0; j < nNeighbors; ++j) {
//                PetscInt neighbor = neighbors[j];
//                PetscReal *kappan;
//                xDMPlexPointLocalRef(auxDM, neighbor, kappaField.id, auxArray, &kappan) >> ablate::utilities::PetscUtilities::checkError;
//                if (PetscAbs(*kappan) > 1e-4) {
//                    PetscReal xn, yn, zn;
//                    Get3DCoordinate(dm, neighbor, &xn, &yn, &zn);
//                    PetscReal d = PetscSqrtReal(PetscSqr(xn - xc) + PetscSqr(yn - yc) + PetscSqr(zn - zc));  // distance
//                    PetscReal s = 18 * rmin; //6*h
//                    PetscReal wn;
//                    PhiNeighborGaussWeight(d, s, &wn);
//                    Tw += wn;
//                    *kappaTilde += (*kappan * wn);
//                }
//            }
//            *kappaTilde /= Tw;
//        }
//    }

    for (PetscInt cell = cStart; cell < cEnd; ++cell){
        PetscReal *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, cell, -1, phitildemaskLocalArray, &phitildemaskptr);
        PetscScalar *kappaptr; xDMPlexPointLocalRef(kappaDM, cell, -1, kappaLocalArray, &kappaptr);
        PetscScalar *nptr; xDMPlexPointLocalRef(nDM, cell, -1, nLocalArray, &nptr);
        PetscScalar *sfxptr; xDMPlexPointLocalRef(sfxDM, cell, -1, sfxLocalArray, &sfxptr);
        PetscScalar *sfyptr; xDMPlexPointLocalRef(sfyDM, cell, -1, sfyLocalArray, &sfyptr);
        if(*phitildemaskptr > 0.5){
            *sfxptr = process->sigma * *kappaptr * -nptr[0];
            *sfyptr = process->sigma * *kappaptr * -nptr[1];
            //            sfptr[2] = process->sigma * *kappaptr * -nptr[2];
        }
        else{ *sfxptr = 0; *sfyptr = 0; } //sfptr[2]=0; }
    }
    PushToGhost(sfxDM, sfxLocalVec, sfxGlobalVec, INSERT_VALUES);
    PushToGhost(sfyDM, sfyLocalVec, sfyGlobalVec, INSERT_VALUES);

    //add to rhs (auxdm copy)
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.GetPoint(c);

        PetscReal *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, cell, -1, phitildemaskLocalArray, &phitildemaskptr);
//        PetscScalar *kappaptr; xDMPlexPointLocalRef(kappaDM, cell, -1, kappaLocalArray, &kappaptr);
//        PetscScalar *nptr; xDMPlexPointLocalRef(nDM, cell, -1, nLocalArray, &nptr);
        PetscScalar *sfxptr; xDMPlexPointLocalRef(sfxDM, cell, -1, sfxLocalArray, &sfxptr);
        PetscScalar *sfyptr; xDMPlexPointLocalRef(sfyDM, cell, -1, sfyLocalArray, &sfyptr);

        const PetscScalar *euler = nullptr;
        PetscScalar *eulerSource = nullptr;
        PetscCall(DMPlexPointLocalFieldRef(dm, cell, eulerField.id, fArray, &eulerSource));
        PetscCall(DMPlexPointLocalFieldRead(dm, cell, eulerField.id, solArray, &euler));
        auto density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];
        PetscReal ux = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] / density;
        PetscReal uy = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + 1] / density;
//        PetscReal uz = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + 2] / density;
        if (PetscAbs(*sfxptr) > 1e-10){eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU] += *sfxptr;}
        if (PetscAbs(*sfyptr) > 1e-10){eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOV] += *sfyptr;}
//        if (PetscAbs(sfptr[2]) > 1e-10){eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOW] += sfptr[2];}
        if (PetscAbs(*sfxptr) > 1e-10){eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += *sfxptr * ux;}
        if (PetscAbs(*sfyptr) > 1e-10){eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += *sfyptr * uy;}
//        if (PetscAbs(sfptr[2]) > 1e-10){eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += sfptr[2]*uz;}

        PetscScalar *optr; xDMPlexPointLocalRef(auxDM, cell, ofield.id, auxArray, &optr);
        PetscScalar *optr2; xDMPlexPointLocalRef(auxDM, cell, ofield2.id, auxArray, &optr2);
//        PetscScalar *phitildemaskptr; xDMPlexPointLocalRef(phitildemaskDM, cell, -1, phitildemaskLocalArray, &phitildemaskptr);
        *optr = *phitildemaskptr;
        *optr2 = *sfxptr;

    }
    if (verbose){
        SaveDataToFile(cellRange.start, cellRange.end, sfxDM, sfxLocalArray, "sfx", true);
        SaveDataToFile(cellRange.start, cellRange.end, sfyDM, sfyLocalArray, "sfy", true);
    }


    //CSF auxDM delete asap
//    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
//
//        const PetscInt cell = cellRange.GetPoint(i);
////        PetscReal vol, centroid[3]; DMPlexComputeCellGeometryFVM(dm, cell, &vol, centroid, nullptr);
////        std::cout << "cell   " << cell << "   ("<<centroid[0]<<",  "<<centroid[1]<<")\n";
//
////        PetscScalar *kappaTildeptr;
//        PetscScalar *kappaptr, *n0ptr, *n1ptr, *n2ptr, *CSF0ptr, *CSF1ptr, *CSF2ptr;
//        PetscReal *phiTildeMaskptr; xDMPlexPointLocalRef(auxDM, cell, phiTildeMaskField.id, auxArray, &phiTildeMaskptr);
////        xDMPlexPointLocalRef(auxDM, cell, kappaTildeField.id, auxArray, &kappaTildeptr) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, kappaField.id, auxArray, &kappaptr) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, n0Field.id, auxArray, &n0ptr) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, n1Field.id, auxArray, &n1ptr) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, n2Field.id, auxArray, &n2ptr) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, CSF0Field.id, auxArray, &CSF0ptr) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, CSF1Field.id, auxArray, &CSF1ptr) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, CSF2Field.id, auxArray, &CSF2ptr) >> ablate::utilities::PetscUtilities::checkError;
////        *CSF0ptr = process->sigma * *kappaTildeptr * -*n0ptr;
////        *CSF1ptr = process->sigma * *kappaTildeptr * -*n1ptr;
////        *CSF2ptr = process->sigma * *kappaTildeptr * -*n2ptr;
//
//
//        if(*phiTildeMaskptr > 0.5){
//        *CSF0ptr = process->sigma * *kappaptr * -*n0ptr;
//        *CSF1ptr = process->sigma * *kappaptr * -*n1ptr;
//        *CSF2ptr = process->sigma * *kappaptr * -*n2ptr;
//        }
//        else{
//        *CSF0ptr = 0;
//        *CSF1ptr = 0;
//        *CSF2ptr = 0;
//        }
//
//    }

    //CSFTilde
//    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
//        const PetscInt cell = cellRange.GetPoint(i);
//        PetscScalar *CSF0ptr, *CSF1ptr, *CSF2ptr, *kappac;
//        xDMPlexPointLocalRef(auxDM, cell, kappaField.id, auxArray, &kappac) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, CSF0Field.id, auxArray, &CSF0ptr) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, CSF1Field.id, auxArray, &CSF1ptr) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, CSF2Field.id, auxArray, &CSF2ptr) >> ablate::utilities::PetscUtilities::checkError;
//        PetscReal xc, yc, zc; Get3DCoordinate(dm, cell, &xc, &yc, &zc);
//        PetscReal *CSF0Tilde, *CSF1Tilde, *CSF2Tilde;
//        xDMPlexPointLocalRef(auxDM, cell, CSF0TildeField.id, auxArray, &CSF0Tilde) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, CSF1TildeField.id, auxArray, &CSF1Tilde) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, CSF2TildeField.id, auxArray, &CSF2Tilde) >> ablate::utilities::PetscUtilities::checkError;
//
//        *CSF0Tilde=0; *CSF1Tilde=0; *CSF2Tilde=0;
//
//        PetscInt nNeighbors, *neighbors;
//        DMPlexGetNeighbors(dm, cell, 4, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
//        PetscReal Tw = 0;
//        for (PetscInt j = 0; j < nNeighbors; ++j) {
//            PetscInt neighbor = neighbors[j];
//            PetscReal *CSF0n, *CSF1n, *CSF2n, *kappan;
//            xDMPlexPointLocalRef(auxDM, neighbor, kappaField.id, auxArray, &kappan) >> ablate::utilities::PetscUtilities::checkError;
//            xDMPlexPointLocalRef(auxDM, neighbor, CSF0Field.id, auxArray, &CSF0n) >> ablate::utilities::PetscUtilities::checkError;
//            xDMPlexPointLocalRef(auxDM, neighbor, CSF1Field.id, auxArray, &CSF1n) >> ablate::utilities::PetscUtilities::checkError;
//            xDMPlexPointLocalRef(auxDM, neighbor, CSF2Field.id, auxArray, &CSF2n) >> ablate::utilities::PetscUtilities::checkError;
//
//            PetscReal xn, yn, zn;
//            Get3DCoordinate(dm, neighbor, &xn, &yn, &zn);
//
//            PetscReal d = PetscSqrtReal(PetscSqr(xn - xc) + PetscSqr(yn - yc) + PetscSqr(zn - zc));  // distance
//            PetscReal s = 6 * h;
//            PetscReal wn;
//            PhiNeighborGaussWeight(d, s, &wn);
//            Tw += wn;
//            *CSF0Tilde += (*CSF0n * wn);
//            *CSF1Tilde += (*CSF1n * wn);
//            *CSF2Tilde += (*CSF2n * wn);
//        }
//        *CSF0Tilde /= Tw;
//        *CSF1Tilde /= Tw;
//        *CSF2Tilde /= Tw;
//    }


//    //Advect
//    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
//        const PetscInt cell = cellRange.GetPoint(i);
//        const PetscScalar *phiptr;
//
//        PetscScalar *CSF0ptr, *CSF1ptr, *CSF2ptr;
//        xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phiptr) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, CSF0Field.id, auxArray, &CSF0ptr) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, CSF1Field.id, auxArray, &CSF1ptr) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRef(auxDM, cell, CSF2Field.id, auxArray, &CSF2ptr) >> ablate::utilities::PetscUtilities::checkError;
//
////        if ((isnan(*CSF0ptr)) or (isnan(*CSF0ptr)) or (isnan(*CSF0ptr))){
////            std::cout << cell << "   csf   " << *CSF0ptr << "   " << *CSF1ptr << "   " << *CSF2ptr<<"\n";
////        }
//
////        PetscReal x, y, z; Get3DCoordinate(dm, cell, &x, &y, &z);
////        PetscScalar *xptr=0, *yptr=0, *zptr=0;
////        xDMPlexPointLocalRef(auxDM, cell, xField.id, auxArray, &xptr) >> ablate::utilities::PetscUtilities::checkError;
////        xDMPlexPointLocalRef(auxDM, cell, yField.id, auxArray, &yptr) >> ablate::utilities::PetscUtilities::checkError;
////        xDMPlexPointLocalRef(auxDM, cell, zField.id, auxArray, &zptr) >> ablate::utilities::PetscUtilities::checkError;
////        *xptr = x; *yptr =y; *zptr =z;
//
//        //temp delete
//        const PetscScalar *euler = nullptr;
//        PetscScalar *eulerSource = nullptr;
//        PetscCall(DMPlexPointLocalFieldRef(dm, cell, eulerField.id, fArray, &eulerSource));
//        PetscCall(DMPlexPointLocalFieldRead(dm, cell, eulerField.id, solArray, &euler));
//        auto density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];
//        PetscReal ux = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] / density;
//        PetscReal uy = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + 1] / density;
//        PetscReal uz = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + 2] / density;
//        if (PetscAbs(*CSF0ptr) > 1e-10){eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU] += *CSF0ptr;}
//        if (PetscAbs(*CSF1ptr) > 1e-10){ eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + 1] += *CSF1ptr;}
//        if (PetscAbs(*CSF2ptr) > 1e-10){ eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + 2] += *CSF2ptr;}
//        if (PetscAbs(*CSF0ptr) > 1e-10){ eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += *CSF0ptr * ux;}
//        if (PetscAbs(*CSF1ptr) > 1e-10){eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += *CSF1ptr * uy;}
//        if (PetscAbs(*CSF2ptr) > 1e-10){eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += *CSF2ptr * uz;}
//
//    }

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

    VecRestoreArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(vertexVec, &vertexArray);
    VecRestoreArray(locFVec, &fArray);
    DMRestoreLocalVector(process->vertexDM, &vertexVec);
    VecDestroy(&vertexVec);
    solver.RestoreRange(cellRange);
    solver.RestoreRange(vertexRange);

    //    VecDestroy(&auxVec);
    //    VecDestroy(&locX);


    PetscFunctionReturn(0);
}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::SurfaceForce, "calculates surface tension force and adds source terms",
         ARG(PetscReal, "sigma", "sigma, surface tension coefficient"));