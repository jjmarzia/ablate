#include "locations.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
ablate::finiteVolume::processes::locations::locations() {}
void ablate::finiteVolume::processes::locations::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) { flow.RegisterRHSFunction(ComputeSource, this); }
// Called every time the mesh changes
void ablate::finiteVolume::processes::locations::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
  locations::subDomain = solver.GetSubDomainPtr();
}
PetscErrorCode ablate::finiteVolume::processes::locations::ComputeSource(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx) {
    PetscFunctionBegin;
    auto process = (ablate::finiteVolume::processes::locations *)ctx;
    std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;

    const ablate::domain::Field *cellLocs = &(subDomain->GetField("cellLocations"));
    const ablate::domain::Field *vertLocs = &(subDomain->GetField("vertexLocations"));
    const ablate::domain::Field *rankLocs = &(subDomain->GetField("rank"));

    ablate::domain::Range cellRange, vertRange;
    solver.GetCellRangeWithoutGhost(cellRange);
    solver.GetRange(0, vertRange);


    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector();
    PetscScalar *auxArray = nullptr;

    PetscMPIInt  rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank) >> ablate::utilities::PetscUtilities::checkError;

    VecGetArray(auxVec, &auxArray);


    for (PetscInt c = cellRange.start; c < cellRange.end; ++c){
      const PetscInt cell = cellRange.GetPoint(c);
      PetscScalar *x;
      xDMPlexPointLocalRef(auxDM, cell, cellLocs->id, auxArray, &x) >> ablate::utilities::PetscUtilities::checkError;
      DMPlexComputeCellGeometryFVM(dm, cell, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;


      xDMPlexPointLocalRef(auxDM, cell, rankLocs->id, auxArray, &x) >> ablate::utilities::PetscUtilities::checkError;
      *x = rank;

    }


    for (PetscInt v = vertRange.start; v < vertRange.end; ++v){
      const PetscInt vert = vertRange.GetPoint(v);
      PetscScalar *x;
      xDMPlexPointLocalRef(auxDM, vert, vertLocs->id, auxArray, &x) >> ablate::utilities::PetscUtilities::checkError;
      DMPlexComputeCellGeometryFVM(dm, vert, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
    }

    VecRestoreArray(auxVec, &auxArray);

    solver.RestoreRange(cellRange);
    solver.RestoreRange(vertRange);
    PetscFunctionReturn(0);
}

ablate::finiteVolume::processes::locations::~locations() { }
REGISTER_WITHOUT_ARGUMENTS(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::locations, "saves vertex and cell locations");
