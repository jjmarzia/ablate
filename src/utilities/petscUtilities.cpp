#include "petscUtilities.hpp"
#include "environment/runEnvironment.hpp"

void ablate::utilities::PetscUtilities::Initialize(const char help[]) {
    PetscInitialize(ablate::environment::RunEnvironment::GetArgCount(), ablate::environment::RunEnvironment::GetArgs(), nullptr, help) >> utilities::PetscUtilities::checkError;

    // register the cleanup
    ablate::environment::RunEnvironment::RegisterCleanUpFunction("ablate::utilities::PetscUtilities::Initialize", []() { PetscFinalize() >> utilities::PetscUtilities::checkError; });
}

void ablate::utilities::PetscUtilities::Set(const std::string& prefix, const std::map<std::string, std::string>& options, bool override) {
    // March over and set each option in the global petsc database
    for (const auto& optionPair : options) {
        std::string optionName = "-" + prefix + "" + optionPair.first;

        // If not override, check for use first
        if (!override) {
            PetscBool exists;
            PetscOptionsHasName(nullptr, nullptr, optionName.c_str(), &exists) >> utilities::PetscUtilities::checkError;
            if (exists) {
                continue;
            }
        }

        PetscOptionsSetValue(nullptr, optionName.c_str(), optionPair.second.empty() ? nullptr : optionPair.second.c_str()) >> utilities::PetscUtilities::checkError;
    }
}

void ablate::utilities::PetscUtilities::Set(const std::map<std::string, std::string>& options) {
    const std::string noPrefix;
    Set(noPrefix, options);
}

void ablate::utilities::PetscUtilities::Set(PetscOptions petscOptions, const std::map<std::string, std::string>& options) {
    for (const auto& optionPair : options) {
        std::string optionName = "-" + optionPair.first;
        PetscOptionsSetValue(petscOptions, optionName.c_str(), optionPair.second.empty() ? nullptr : optionPair.second.c_str()) >> utilities::PetscUtilities::checkError;
    }
}

void ablate::utilities::PetscUtilities::PetscOptionsDestroyAndCheck(const std::string& name, PetscOptions* options) {
    PetscInt nopt;
    PetscOptionsAllUsed(*options, &nopt) >> utilities::PetscUtilities::checkError;
    if (nopt) {
        PetscPrintf(PETSC_COMM_WORLD, "WARNING! There are options in %s you set that were not used!\n", name.c_str()) >> utilities::PetscUtilities::checkError;
        PetscPrintf(PETSC_COMM_WORLD, "WARNING! could be spelling mistake, etc!\n") >> utilities::PetscUtilities::checkError;
        if (nopt == 1) {
            PetscPrintf(PETSC_COMM_WORLD, "There is one unused database option. It is:\n") >> utilities::PetscUtilities::checkError;
        } else {
            PetscPrintf(PETSC_COMM_WORLD, "There are %" PetscInt_FMT "unused database options. They are:\n", nopt) >> utilities::PetscUtilities::checkError;
        }
    }
    PetscOptionsLeft(*options) >> utilities::PetscUtilities::checkError;
    PetscOptionsDestroy(options) >> utilities::PetscUtilities::checkError;
}
void ablate::utilities::PetscUtilities::Set(PetscOptions petscOptions, const char* name, const char* value, bool override) {
    // If not override, check for use first
    if (!override) {
        PetscBool exists;
        PetscOptionsHasName(petscOptions, nullptr, name, &exists) >> utilities::PetscUtilities::checkError;
        if (exists) {
            return;
        }
    }

    PetscOptionsSetValue(petscOptions, name, value) >> utilities::PetscUtilities::checkError;
}

// Given a base DM create a new one with a given DOF for a point range. This is useful when you want to create a new DM
//  to use single vectors containing a specific data type. This should reduce the communication overhead
void ablate::utilities::PetscUtilities::CopyDM(DM dm, const PetscInt pStart, const PetscInt pEnd, const PetscInt nDOF, DM *newDM) {

  PetscSection section;

  DM coordDM;
  DMGetCoordinateDM(dm, &coordDM) >> ablate::utilities::PetscUtilities::checkError;

  DMClone(dm, newDM) >> ablate::utilities::PetscUtilities::checkError;

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

namespace ablate::utilities {

std::istream& operator>>(std::istream& is, PetscDataType& v) {
    // get the string
    std::string enumString;
    is >> enumString;

    // ask petsc for the enum
    PetscBool found;
    PetscDataTypeFromString(enumString.c_str(), &v, &found) >> utilities::PetscUtilities::checkError;

    if (!found) {
        v = PETSC_DATATYPE_UNKNOWN;
    }

    return is;
}

std::ostream& operator<<(std::ostream& os, const PetscDataType& type) {
    os << PetscDataTypes[type];
    return os;
}

}  // namespace ablate::utilities
