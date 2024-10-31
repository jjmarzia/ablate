#include "locations.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include <fstream>

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

const auto &phiField = subDomain->GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);

    ablate::domain::Range cellRange, vertRange;
    subDomain->GetCellRange(nullptr, cellRange);
    subDomain->GetRange(nullptr, 0, vertRange);

//    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector();
    PetscScalar *auxArray = nullptr;

    VecGetArray(auxVec, &auxArray);

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c){
      const PetscInt cell = cellRange.GetPoint(c);
std::cout << cell<<"\n";
    }

    VecRestoreArray(auxVec, &auxArray);

    subDomain->RestoreRange(cellRange);
    subDomain->RestoreRange(vertRange);
    PetscFunctionReturn(0);
}

ablate::finiteVolume::processes::locations::~locations() { }


REGISTER_WITHOUT_ARGUMENTS(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::locations, "saves vertex and cell locations");
