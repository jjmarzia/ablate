#include "intSharp.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"

void GetVertexRange(DM dm, const std::shared_ptr<ablate::domain::Region> &region, ablate::domain::Range &vertexRange) {
    PetscInt depth=0; //zeroth layer of DAG is always that of the vertices
    ablate::domain::GetRange(dm, region, depth, vertexRange);
}

void GetCoordinate(DM dm, PetscInt dim, PetscInt p, PetscReal *xp, PetscReal *yp, PetscReal *zp){
    //get the coordinates of the point
    PetscReal vol;
    PetscReal centroid[dim];
    DMPlexComputeCellGeometryFVM(dm, p, &vol, centroid, nullptr);
    *xp = centroid[0];
    *yp = centroid[1];
    *zp = centroid[2];
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
    auto process = (ablate::finiteVolume::processes::IntSharp *)ctx;

    //dm = sol DM
    //locX = solvec
    //locFVec = vector of conserved vars / eulerSource fields (rho, rhoe, rhov, ..., rhoet)
    //auxvec = auxvec
    //auxArray = auxArray
    //notions of "process->" refer to the private variables: vertexDM, Gamma, epsilon. (the public variables are a subset: Gamma and epsilon)
    //process->vertexDM = aux DM

    //get fields
    auto dim = solver.GetSubDomain().GetDimensions();
    const auto &phiField = solver.GetSubDomain().GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
    const auto &eulerField = solver.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);
    auto phifID = phiField.id;
    auto eulerfID = eulerField.id;

    // get vecs/arrays
    Vec auxvec; DMGetLocalVector(process->vertexDM, &auxvec);
    PetscScalar *auxArray; VecGetArray(auxvec, &auxArray);
    PetscScalar *fArray; VecGetArray(locFVec, &fArray);
    PetscScalar *solArray; VecGetArray(locX, &solArray);

    // get ranges
    ablate::domain::Range cellRange; solver.GetCellRangeWithoutGhost(cellRange);
    PetscInt vStart, vEnd; DMPlexGetDepthStratum(process->vertexDM, 0, &vStart, &vEnd);

    //march over vertices
    for (PetscInt vertex = vStart; vertex < vEnd; vertex++) {

        const double epsmach = 1e-52;
        PetscReal vx, vy, vz; GetCoordinate(dm, dim, vertex, &vx, &vy, &vz);
        PetscInt nvn, *vertexneighbors;
        DMPlexVertexGetCells(dm, vertex, &nvn, &vertexneighbors);
        PetscReal distances[nvn];
        PetscReal shortestdistance=999999;
        for (PetscInt k =0; k< nvn; ++k){
            PetscInt neighbor = vertexneighbors[k];
            PetscReal nx, ny, nz; GetCoordinate(dm, dim, neighbor, &nx, &ny, &nz);
            PetscReal distance = PetscSqrtReal(PetscSqr(nx-vx) + PetscSqr(ny-vy) + PetscSqr(nz-vz));
            if (distance < shortestdistance){shortestdistance=distance;}
            distances[k]=distance;
        }
        PetscReal weights_wrt_short[nvn];
        PetscReal totalweight_wrt_short=0;
        for (PetscInt k =0; k< nvn; ++k){
            PetscReal weight_wrt_short = shortestdistance/distances[k];
            weights_wrt_short[k] =  weight_wrt_short;
            totalweight_wrt_short += weight_wrt_short;
        }
        PetscReal weights[nvn];
        for (PetscInt k =0; k< nvn; ++k){weights[k] = weights_wrt_short[k]/totalweight_wrt_short;}

        PetscReal phiv=0;
        for (PetscInt k =0; k< nvn; ++k){
            PetscInt neighbor = vertexneighbors[k];
            const PetscReal *phineighbor;
            xDMPlexPointLocalRef(dm, neighbor, phifID, solArray, &phineighbor);
            //                    phiv += (*phineighbor/nvn); //would only work for structured
            phiv += (*phineighbor)*(weights[k]); //unstructured case
        }
        
        //get gradphi at vertices (gradphiv) based on cell centered phis
        PetscScalar gradphiv[dim];
        DMPlexVertexGradFromCell(dm, vertex, locX, phifID, 0, gradphiv);
        PetscReal normgradphi = 0.0;
        for (int k=0; k<dim; ++k){
            normgradphi += pow(gradphiv[k],2);
        }
        normgradphi = pow(normgradphi, 0.5);
        
        //get a at vertices (av) (Chiu 2011)
        //based on Eq. 1 of:   Jain SS. Accurate conservative phase-field method for simulation of two-phase flows. Journal of Computational Physics. 2022 Nov 15;469:111529.
        PetscScalar  av[dim];
        PetscReal *avptr;
        xDMPlexPointLocalRef(process->vertexDM, vertex, 0, auxArray, &avptr); //vertexDM
        for (int k=0; k<dim; ++k){
            av[k] = (process->Gamma * process->epsilon * gradphiv[k]) - (process->Gamma * phiv * (1-phiv) * ((gradphiv[k] + epsmach)/(normgradphi + epsmach)));
            avptr[k]=av[k];
        }
    }

    // march over cells
    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {

        const PetscInt cell = cellRange.GetPoint(i);

        PetscReal div=0.0;
        for (PetscInt offset=0; offset<dim; offset++){
            PetscReal nabla_ai[dim];
            DMPlexCellGradFromVertex(process->vertexDM, cell, auxvec, 0, offset, nabla_ai);
            div +=nabla_ai[offset];
        }
        PetscReal *divaptr;
        xDMPlexPointLocalRef(dm, cell, 0, solArray, &divaptr);
        *divaptr = div;


        // add diva to advection equation RHS
        const PetscScalar *euler = nullptr;
        PetscScalar *eulerSource = nullptr;
        xDMPlexPointLocalRef(dm, cell, eulerfID, fArray, &eulerSource);
        xDMPlexPointLocalRef(dm, cell, eulerfID, solArray, &euler);

        eulerSource[0] += div; //first euler equation for multiphase corresponds to vol frac (might need to be density * vf ?)

    }

    // cleanup
    DMRestoreLocalVector(process->vertexDM, &auxvec);
    VecRestoreArray(locFVec, &fArray);
    VecRestoreArray(locX, &solArray);
//    VecRestoreArray(auxvec, &auxArray);
    solver.RestoreRange(cellRange);
    PetscFunctionReturn(0);

}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::IntSharp, "calculates interface regularization term",
         ARG(PetscReal, "Gamma", "Gamma, velocity scale parameter (approx. umax)"),
         ARG(PetscReal, "epsilon", "epsilon, interface thickness scale parameter (approx. h)")
);
