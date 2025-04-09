#include "zeroDerBoundary.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

#include <petsc.h>
#include <memory>
#include <vector>
#include "domain/range.hpp"


ablate::finiteVolume::boundaryConditions::ZeroDerBoundary::ZeroDerBoundary(std::string boundaryName, std::vector<std::string> labelIds, std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction)
    : BoundaryCell(boundaryFunction->GetName(), boundaryName, labelIds), boundaryFunction(boundaryFunction) {

    if (!boundaryFunction) {
        throw std::invalid_argument("ZeroDerBoundary must be constructed with a valid boundary function");
    }
    if (boundaryFunction->GetSolutionField().GetPetscFunction() == nullptr) {
        throw std::invalid_argument(
            "ZeroDerBoundary must be constructed with a valid boundary function that has a solution field (i.e. mathFunction must not be null).");
    }



        PetscPrintf(PETSC_COMM_WORLD, "ZeroDerBoundary created: %s\n", boundaryName.c_str());
        PetscPrintf(PETSC_COMM_WORLD, "Associated labels: ");
        for (const auto& label : labelIds) {
            PetscPrintf(PETSC_COMM_WORLD, "%s ", label.c_str());
        }
        PetscPrintf(PETSC_COMM_WORLD, "\n");
    }

void ablate::finiteVolume::boundaryConditions::ZeroDerBoundary::updateFunction(PetscReal time, const PetscReal *x, PetscScalar *vals, PetscInt point) {

    // PetscPrintf(PETSC_COMM_WORLD, "prior to update function");
    // boundaryFunction->GetSolutionField().GetPetscFunction()(dim, time, x, fieldSize, vals, boundaryFunction->GetSolutionField().GetContext());

    //get the point in the region labeled as "domain" 


    DMLabel domainLabel;
    DMGetLabel(subDomain->GetDM(), "domain", &domainLabel) >> utilities::PetscUtilities::checkError;

    //print the domain label; not the name but the label itself
    PetscPrintf(PETSC_COMM_WORLD, "domainLabel= %p\n", (void *)domainLabel);
    //print the value of the label
    PetscInt value;
    DMLabelGetValue(domainLabel, 0, &value) >> utilities::PetscUtilities::checkError;
    PetscPrintf(PETSC_COMM_WORLD, "domainLabel value= %d\n", value);

    IS domainIS;
    DMLabelGetStratumIS(domainLabel, 1, &domainIS) >> utilities::PetscUtilities::checkError; //height 
    const PetscInt *domainPoints;
    PetscInt numDomainPoints;


    if (!domainIS) {
        throw std::runtime_error("error, domainIS is null");
    }
    ISGetLocalSize(domainIS, &numDomainPoints) >> utilities::PetscUtilities::checkError;
    ISGetIndices(domainIS, &domainPoints) >> utilities::PetscUtilities::checkError;

    PetscPrintf(PETSC_COMM_WORLD, "not getting to here yet\n");

    


    //find the closest cell in domain
    PetscInt closestCell = -1;
    PetscReal closestDistance = PETSC_MAX_REAL;
    for (PetscInt i = 0; i < numDomainPoints; i++) {
        PetscInt cell = domainPoints[i];
        PetscReal cellCentroid[3];
        DMPlexComputeCellGeometryFVM(subDomain->GetDM(), cell, nullptr, cellCentroid, nullptr) >> utilities::PetscUtilities::checkError;
        PetscReal distance = 0.0;
        for (int d = 0; d < dim; d++) {
            distance += (x[d] - cellCentroid[d]) * (x[d] - cellCentroid[d]);
        }
        distance = sqrt(distance);
        if (distance < closestDistance) {
            closestDistance = distance;
            closestCell = cell;
        }
    }

    //fetch the field variable from that closest Cell
    //let's fetch the entire solution vector
    Vec solVec;
    PetscScalar *solArray;
    DMGetGlobalVector(subDomain->GetDM(), &solVec) >> utilities::PetscUtilities::checkError;
    VecGetArray(solVec, &solArray) >> utilities::PetscUtilities::checkError;
    PetscScalar *domainFields = nullptr; 
    DMPlexPointLocalRef(subDomain->GetDM(), closestCell, solArray, &domainFields);
    PetscScalar *boundaryFields = nullptr; 
    DMPlexPointLocalRef(subDomain->GetDM(), point, solArray, &boundaryFields);

    boundaryFields[ablate::finiteVolume::CompressibleFlowFields::RHO] = domainFields[ablate::finiteVolume::CompressibleFlowFields::RHO];
    boundaryFields[ablate::finiteVolume::CompressibleFlowFields::RHOE] = domainFields[ablate::finiteVolume::CompressibleFlowFields::RHOE];
    boundaryFields[ablate::finiteVolume::CompressibleFlowFields::RHOU] = domainFields[ablate::finiteVolume::CompressibleFlowFields::RHOU];
    boundaryFields[ablate::finiteVolume::CompressibleFlowFields::RHOV] = domainFields[ablate::finiteVolume::CompressibleFlowFields::RHOV];
    boundaryFields[ablate::finiteVolume::CompressibleFlowFields::RHOW] = domainFields[ablate::finiteVolume::CompressibleFlowFields::RHOW];
    boundaryFields[subDomain->GetField(VOLUME_FRACTION_FIELD).offset] = domainFields[subDomain->GetField(VOLUME_FRACTION_FIELD).offset];
    boundaryFields[subDomain->GetField(DENSITY_VF_FIELD).offset] = domainFields[subDomain->GetField(DENSITY_VF_FIELD).offset];


//loop over 

    //get field IDs for rho, rhoE, rhoU, rhoV, rhoW
    // PetscInt fID = subDomain->GetField(GetFieldName()).id;
    // PetscInt fID2 = subDomain->GetField("rho").id;
    // PetscInt fID3 = subDomain->GetField("rhoE").id;
    // PetscInt fID4 = subDomain->GetField("rhoU").id;
    // PetscInt fID5 = subDomain->GetField("rhoV").id;
    // PetscInt fID6 = subDomain->GetField("rhoW").id;

    //loop over the length of the sol array in the closest cell and attribute them to the sol array in the boundary cell
    

    // xDMPlexPointLocalRead(subDomain->GetDM(), verts[1], fID, dataArray, &solVal)






    // PetscPrintf(PETSC_COMM_WORLD, "updateFunction called for boundary");
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::boundaryConditions::BoundaryCondition, ablate::finiteVolume::boundaryConditions::ZeroDerBoundary, "zero derivative (Neumann special case) for boundary cells created by adding a layer next to the domain. See boxMeshBoundaryCells for an example.",
         ARG(std::string, "boundaryName", "the name for this boundary condition"),
         ARG(std::vector<std::string>, "labelIds", "labels to apply this BC to"),
         ARG(ablate::mathFunctions::FieldFunction, "boundaryValue", "the field function used to describe the boundary"));
