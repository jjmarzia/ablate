#include "domain/RBF/mq.hpp"
#include "surfaceForce.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "levelSet/levelSetUtilities.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "mathFunctions/functionFactory.hpp"

// Called every time the mesh changes
void ablate::finiteVolume::processes::SurfaceForce::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
    SurfaceForce::subDomain = solver.GetSubDomainPtr();
}
PetscErrorCode ablate::finiteVolume::processes::SurfaceForce::ComputeSource(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx) {
    PetscFunctionBegin;

    ablate::finiteVolume::processes::SurfaceForce *process = (ablate::finiteVolume::processes::SurfaceForce *)ctx;
    std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
    const auto &phiField = subDomain->GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
    const auto &phiTildeField = subDomain->GetField("phiTilde");
    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector();
    const PetscScalar *solArray;
    PetscScalar *auxArray;

    VecGetArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;

    ablate::domain::Range cellRange;
    solver.GetCellRangeWithoutGhost(cellRange);
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.GetPoint(c);
        const PetscScalar *phi;
        xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phi) >> ablate::utilities::PetscUtilities::checkError;

        PetscScalar *phiTilde;
        xDMPlexPointLocalRef(auxDM, cell, phiTildeField.id, auxArray, &phiTilde) >> ablate::utilities::PetscUtilities::checkError;
        *phiTilde = *phi;
    }
    subDomain->UpdateAuxLocalVector();
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.GetPoint(c);
        const PetscScalar *phi, *phiTilde;
        xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phi) >> ablate::utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(auxDM, cell, phiTildeField.id, auxArray, &phiTilde) >> ablate::utilities::PetscUtilities::checkError;
        if (PetscAbsScalar(*phi - *phiTilde) > PETSC_SMALL) {
            printf("%+f\t%+f\t%+e\n", *phi, *phiTilde, PetscAbsScalar(*phi - *phiTilde));
        }
    }
    VecRestoreArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
    printf("All Done!\n");
//    exit(0);
    PetscFunctionReturn(0);
}

