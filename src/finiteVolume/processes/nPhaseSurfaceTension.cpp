#include "nPhaseSurfaceTension.hpp"
#include "eos/kthStiffenedGas.hpp"
#include "eos/nPhase.hpp"
#include "finiteVolume/nPhaseFlowFields.hpp"
#include "nPhaseAllaireAdvection.hpp"
#include "utilities/petscUtilities.hpp"
#include <petsc/private/dmpleximpl.h>

namespace ablate::finiteVolume::processes {

ablate::finiteVolume::processes::NPhaseSurfaceTension::NPhaseSurfaceTension(const std::vector<PetscReal> &sigmaij) : sigmaij(sigmaij) {}

void ablate::finiteVolume::processes::NPhaseSurfaceTension::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
    NPhaseSurfaceTension::subDomain = solver.GetSubDomainPtr();
}

void NPhaseSurfaceTension::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
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
    auto fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver*>(&flow);

    if (!fvSolver) {
      return;
    }

    // PetscPrintf(MPI_COMM_WORLD, "fvSolver = %p\n", fvSolver);

    // fvSolver->GetCellRangeWithoutGhost(cellRange);
    PetscInt cStart, cEnd; DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
    cellRange.start = cStart; cellRange.end = cEnd;

    // PetscPrintf(MPI_COMM_WORLD, "cellRange = %d, %d\n", cellRange.start, cellRange.end);

    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
        PetscInt cell = cellRange.GetPoint(i);
        PetscInt nNeighbors, *neighbors, nNeighbors1, *neighbors1;
        PetscReal layers=3;

        DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        cellNeighbors[cell] = std::vector<PetscInt>(neighbors, neighbors + nNeighbors);
        DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);

        DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors1, &neighbors1);
        cellNeighbors1[cell] = std::vector<PetscInt>(neighbors1, neighbors1 + nNeighbors1);
        DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors1, &neighbors1);

        // PetscPrintf(MPI_COMM_WORLD, "cell = %d\n", cell);
        // PetscPrintf(MPI_COMM_WORLD, "nNeighbors = %d\n", nNeighbors);
        // PetscPrintf(MPI_COMM_WORLD, "nNeighbors1 = %d\n", nNeighbors1);

        // PetscPrintf(MPI_COMM_WORLD, "neighbors = %p\n", neighbors);
        // PetscPrintf(MPI_COMM_WORLD, "neighbors1 = %p\n", neighbors1);

    }

    // global vertex neighbors
    PetscInt vStart, vEnd;
    DMPlexGetDepthStratum(vertexDM, 0, &vStart, &vEnd);
    for (PetscInt vertex = vStart; vertex < vEnd; ++vertex) {
        PetscInt nvn, *vertexneighbors;
        DMPlexVertexGetCells(vertexDM, vertex, &nvn, &vertexneighbors);
        vertexNeighbors[vertex] = std::vector<PetscInt>(vertexneighbors, vertexneighbors + nvn);
        DMPlexVertexRestoreCells(vertexDM, vertex, &nvn, &vertexneighbors);

        // PetscPrintf(MPI_COMM_WORLD, "vertex = %d\n", vertex);
        // PetscPrintf(MPI_COMM_WORLD, "nvn = %d\n", nvn);
        // PetscPrintf(MPI_COMM_WORLD, "vertexneighbors = %p\n", vertexneighbors);
    }

    flow.RegisterRHSFunction(ComputeSource, this);
}

PetscErrorCode ablate::finiteVolume::processes::NPhaseSurfaceTension::ComputeSource(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx) {
    PetscFunctionBegin;

    auto *process = (ablate::finiteVolume::processes::NPhaseSurfaceTension *)ctx;
    std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
    
    const auto &aijField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::AIJ);
    auto dim = solver.GetSubDomain().GetDimensions();

    PetscPrintf(MPI_COMM_WORLD, "dim = %d\n", dim);

    const auto &allaireField = solver.GetSubDomain().GetField(ablate::finiteVolume::NPhaseFlowFields::ALLAIRE_FIELD);

    const auto &gradAijField = subDomain->GetField("gradAij");
    const auto &nAijField = subDomain->GetField("nAij");
    const auto &kappaijField = subDomain->GetField("kappaij");
    const auto &sfmomField = subDomain->GetField("sfmom");


    PetscPrintf(MPI_COMM_WORLD, "allaireField.numberComponents = %d\n", allaireField.numberComponents);


    DM auxDM = subDomain->GetAuxDM();

    PetscPrintf(MPI_COMM_WORLD, "auxdm\n");
    Vec auxVec = subDomain->GetAuxVector();
    PetscInt cStart, cEnd; DMPlexGetHeightStratum(auxDM, 0, &cStart, &cEnd);

    // PetscPrintf(MPI_COMM_WORLD, "auxVec = %p\n", auxVec);
    Vec vertexVec; 
    DMGetLocalVector(process->vertexDM, &vertexVec);
    // PetscPrintf(MPI_COMM_WORLD, "vertexVec = %p\n", vertexVec);
    const PetscScalar *solArray;
    // PetscPrintf(MPI_COMM_WORLD, "solArray = %p\n", solArray);
    PetscScalar *auxArray;
    // PetscPrintf(MPI_COMM_WORLD, "auxArray = %p\n", auxArray);
    PetscScalar *vertexArray;
    // PetscPrintf(MPI_COMM_WORLD, "vertexArray = %p\n", vertexArray);
    PetscScalar *fArray;

    // PetscPrintf(MPI_COMM_WORLD, "locX = %p\n", locX);
    // PetscPrintf(MPI_COMM_WORLD, "locFVec = %p\n", locFVec);
    // PetscPrintf(MPI_COMM_WORLD, "ctx = %p\n", ctx);

    VecGetArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(vertexVec, &vertexArray);
    VecGetArray(locFVec, &fArray);

    // PetscPrintf(MPI_COMM_WORLD, "solArray = %p\n", solArray);
    // PetscPrintf(MPI_COMM_WORLD, "auxVec = %p\n", auxVec);
    // PetscPrintf(MPI_COMM_WORLD, "vertexVec = %p\n", vertexVec);

    ablate::domain::Range cellRange;
    solver.GetCellRangeWithoutGhost(cellRange);

    // PetscPrintf(MPI_COMM_WORLD, "cellRange = %d, %d\n", cellRange.start, cellRange.end);

    //get size of eos
    // auto nPhaseEOS = std::dynamic_pointer_cast<eos::NPhase>(process->eosNPhase);
    // std::size_t phases = nPhaseEOS->GetNumberOfPhases();


    //get size of alphak field
    const auto &alphakField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
    std::size_t phases = alphakField.numberComponents;

    // PetscPrintf(MPI_COMM_WORLD, "alphakField = %p\n", alphakField);

    PetscInt nPairs = (phases*phases - phases) / 2;

    

    PetscPrintf(MPI_COMM_WORLD, "phases = %lu, pairs = %d\n", phases, nPairs);
    PetscReal h;
    DMPlexGetMinRadius(auxDM, &h);
    h *= 4.0;

    // if (process->sigmaij.size() != nPairs) {
    //     PetscPrintf(MPI_COMM_WORLD, "ERROR: Expected %d surface tension coefficients, got %zu\n", 
    //                 nPairs, process->sigmaij.size());
    //     return 1;
    // }

    //populate gradAij
    for (PetscInt cell = cStart; cell < cEnd; ++cell){
        
        const PetscScalar *Aij;
        xDMPlexPointLocalRead(auxDM, cell, aijField.id, auxArray, &Aij);

        PetscScalar *gradAijArray;
        xDMPlexPointLocalRef(auxDM, cell, gradAijField.id, auxArray, &gradAijArray);
        

            for (PetscInt ij = 0; ij < nPairs; ++ij){
                PetscScalar gradAij[dim];
                DMPlexCellGradFromCell(auxDM, cell, auxVec, aijField.id, ij, gradAij);
                for (PetscInt d = 0; d < dim; ++d){

                    // PetscPrintf(MPI_COMM_WORLD, "1/h = %f\n", 1/h - PETSC_SMALL);
                    if (PetscAbsReal(gradAij[d]) < 1/h - PETSC_SMALL){
                        gradAijArray[ij*dim + d] = gradAij[d];
                    } else {
                        gradAijArray[ij*dim + d] = 0.0;
                    }
                }
            }
    }

    //populate nAij
    for (PetscInt cell = cStart; cell < cEnd; ++cell){
        PetscScalar *nAijArray;
        xDMPlexPointLocalRef(auxDM, cell, nAijField.id, auxArray, &nAijArray);
        PetscScalar *gradAijArray;
        xDMPlexPointLocalRef(auxDM, cell, gradAijField.id, auxArray, &gradAijArray);

        for (PetscInt ij = 0; ij < nPairs; ++ij){
            PetscScalar normGradAij = 0.0;
            for (PetscInt d = 0; d < dim; ++d){
                normGradAij += gradAijArray[ij*dim + d] * gradAijArray[ij*dim + d];
            }
            normGradAij = PetscSqrtReal(normGradAij);
            if (normGradAij > PETSC_SMALL) {
                for (PetscInt d = 0; d < dim; ++d){
                    nAijArray[ij*dim + d] = gradAijArray[ij*dim + d] / normGradAij;
            }
        }
    }
}

    //populate kappaij
    for (PetscInt cell = cStart; cell < cEnd; ++cell){
        PetscScalar *kappaijArray;
        xDMPlexPointLocalRef(auxDM, cell, kappaijField.id, auxArray, &kappaijArray);
        for (PetscInt ij = 0; ij < nPairs; ++ij){
            PetscReal kappaij = 0.0;
            for (PetscInt d=0; d<dim; ++d){
                PetscScalar gradComponent[dim];
                DMPlexCellGradFromCell(auxDM, cell, auxVec, nAijField.id, ij*dim + d, gradComponent);
                kappaij += gradComponent[d];
            }
            if (PetscAbsReal(kappaij) < 1/(2*h) - PETSC_SMALL){
                kappaijArray[ij] = kappaij;
            } else {
                kappaijArray[ij] = 0.0;
            }
        }
    }
    
    for (PetscInt cell = cStart; cell < cEnd; ++cell){
        const PetscScalar *allaire = nullptr;
        PetscScalar *allaireSource = nullptr;
        const auto &alphakrhokField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK);
        DMPlexPointLocalFieldRef(dm, cell, allaireField.id, fArray, &allaireSource);
        DMPlexPointLocalFieldRead(dm, cell, allaireField.id, solArray, &allaire);
        PetscScalar *nAijArray;
        xDMPlexPointLocalRef(auxDM, cell, nAijField.id, auxArray, &nAijArray);
        PetscScalar *AijArray;
        xDMPlexPointLocalRef(auxDM, cell, aijField.id, auxArray, &AijArray);
        PetscScalar *kappaijArray;
        xDMPlexPointLocalRef(auxDM, cell, kappaijField.id, auxArray, &kappaijArray);
        PetscScalar *sfmomArray;
        xDMPlexPointLocalRef(auxDM, cell, sfmomField.id, auxArray, &sfmomArray);
        for (PetscInt d = 0; d < dim; ++d){
            sfmomArray[d] = 0.0;
        }
        for (PetscInt ij = 0; ij < nPairs; ++ij){
            for (PetscInt d = 0; d < dim; ++d){

                PetscReal Sfmom = (AijArray[ij] > 0.1 && AijArray[ij] < 0.9) ? -kappaijArray[ij] * process->sigmaij[ij] * nAijArray[ij*dim + d] : 0.0;

                allaireSource[ablate::finiteVolume::NPhaseFlowFields::RHOU + d] += Sfmom;
                sfmomArray[d] += Sfmom;
                if (Sfmom != 0.0){
                    PetscPrintf(MPI_COMM_WORLD, "Sfmom = %f\n", Sfmom);
                }

                PetscReal u = 0;
                PetscReal rho = 0;
                for (size_t k=0; k<phases; ++k){
                    rho += allaire[alphakrhokField.offset + k];
                }
                // PetscPrintf(MPI_COMM_WORLD, "rho = %f\n", rho);
                u = allaire[ablate::finiteVolume::NPhaseFlowFields::RHOU + d] / rho;
                allaireSource[ablate::finiteVolume::NPhaseFlowFields::RHOE] += Sfmom * u;
            }
        }
    }

    //surfacetension momentum_p = -sum_ij(kappaij * sigma_ij * gradAij_p)
    //here we need to get the sigma_ij from the input

    //surfacetension energy = momentum_p \cdot u_p
    //here we need to get the u_p

    // for (PetscInt cell = cStart; cell < cEnd; ++cell){
    //     //
    // }

    VecRestoreArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(vertexVec, &vertexArray);
    VecRestoreArray(locFVec, &fArray);

    DMRestoreLocalVector(process->vertexDM, &vertexVec) >> ablate::utilities::PetscUtilities::checkError;
    VecDestroy(&vertexVec) >> ablate::utilities::PetscUtilities::checkError;
    solver.RestoreRange(cellRange);




    return 0;
}

}  // namespace ablate::finiteVolume::processes

// #include "registrar.hpp"
// REGISTER_WITHOUT_ARGUMENTS(ablate::finiteVolume::processes::Process, 
//     ablate::finiteVolume::processes::NPhaseSurfaceTension, 
//     "N-phase surface tension (connectivity setup only)");

    #include "registrar.hpp"
    REGISTER(ablate::finiteVolume::processes::Process, 
        ablate::finiteVolume::processes::NPhaseSurfaceTension, 
        "N-phase surface tension with user-defined coefficients",
        ARG(std::vector<PetscReal>, "surfaceTensionCoeffs", "Surface tension coefficients for each phase pair (must match number of phase pairs)"));
