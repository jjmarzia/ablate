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

    ablate::domain::Range cellRange, vertRange;
    subDomain->GetCellRange(nullptr, cellRange);
    subDomain->GetRange(nullptr, 0, vertRange);


    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector();
    PetscScalar *auxArray = nullptr;

    VecGetArray(auxVec, &auxArray);

//    const ablate::domain::Field *eulerField = &(subDomain->GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD));
//    DM eulerDM = subDomain->GetFieldDM(*eulerField); // Get an euler-specific DM in case it's not in the same solution vector as the VOF field
//    const ablate::domain::Field *cellVelocity = &(subDomain->GetField("cellVelocity"));
//    const PetscInt dim = subDomain->GetDimensions();
//    const PetscScalar *xArray = nullptr;
//    VecGetArrayRead(locX, &xArray) >> utilities::PetscUtilities::checkError;

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c){
      const PetscInt cell = cellRange.GetPoint(c);

      PetscScalar *x;
      xDMPlexPointLocalRef(auxDM, cell, cellLocs->id, auxArray, &x) >> ablate::utilities::PetscUtilities::checkError;
      DMPlexComputeCellGeometryFVM(dm, cell, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;


//      const PetscScalar *euler = nullptr;
//      xDMPlexPointLocalRead(eulerDM, cell, eulerField->id, xArray, &euler) >> utilities::PetscUtilities::checkError;
//      const PetscScalar density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];

//      PetscScalar *vel;
//      xDMPlexPointLocalRef(auxDM, cell, cellVelocity->id, auxArray, &vel) >> ablate::utilities::PetscUtilities::checkError;

//      for (PetscInt d = 0; d < dim; ++d) {
//        vel[d] = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density;
//      }
    }

//    VecRestoreArrayRead(locX, &xArray) >> utilities::PetscUtilities::checkError;

    for (PetscInt v = vertRange.start; v < vertRange.end; ++v){
      const PetscInt vert = vertRange.GetPoint(v);

      PetscScalar *x;
      xDMPlexPointLocalRef(auxDM, vert, vertLocs->id, auxArray, &x) >> ablate::utilities::PetscUtilities::checkError;

      DMPlexComputeCellGeometryFVM(dm, vert, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
    }


    VecRestoreArray(auxVec, &auxArray);

    subDomain->RestoreRange(cellRange);
    subDomain->RestoreRange(vertRange);
    PetscFunctionReturn(0);
}

ablate::finiteVolume::processes::locations::~locations() { }


REGISTER_WITHOUT_ARGUMENTS(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::locations, "saves vertex and cell locations");
