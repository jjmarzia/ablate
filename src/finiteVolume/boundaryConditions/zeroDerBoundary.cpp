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
    // if (boundaryFunction->GetSolutionField().GetPetscFunction() == nullptr) {
    //     throw std::invalid_argument(
    //         "ZeroDerBoundary must be constructed with a valid boundary function that has a solution field (i.e. mathFunction must not be null).");
    // }



        PetscPrintf(PETSC_COMM_WORLD, "ZeroDerBoundary created: %s\n", boundaryName.c_str());
        PetscPrintf(PETSC_COMM_WORLD, "Associated labels: ");
        for (const auto& label : labelIds) {
            PetscPrintf(PETSC_COMM_WORLD, "%s ", label.c_str());
        }
        PetscPrintf(PETSC_COMM_WORLD, "\n");


    }

    

void ablate::finiteVolume::boundaryConditions::ZeroDerBoundary::ExtraSetup() {
    PetscPrintf(PETSC_COMM_WORLD, "ZeroDerBoundary ExtraSetup called\n");
    DM dm = subDomain->GetDM();

    //loop over all cells with label outletCells
    PetscInt pStart, pEnd;
    DMPlexGetHeightStratum(dm, 0, &pStart, &pEnd) >> utilities::PetscUtilities::checkError;

    
    DMLabel domainLabel;
    DMGetLabel(subDomain->GetDM(), "domain", &domainLabel) >> utilities::PetscUtilities::checkError;
    DMLabel outletLabel;
    DMGetLabel(subDomain->GetDM(), "outletCells", &outletLabel) >> utilities::PetscUtilities::checkError;

    IS outletvaluesIS;
    DMLabelGetNonEmptyStratumValuesIS(outletLabel, &outletvaluesIS);
    IS domainvaluesIS;
    DMLabelGetNonEmptyStratumValuesIS(domainLabel, &domainvaluesIS);

    PetscPrintf(PETSC_COMM_WORLD, "outletvaluesIS: %p\n", (void *)outletvaluesIS);
    PetscPrintf(PETSC_COMM_WORLD, "domainvaluesIS: %p\n", (void *)domainvaluesIS);

    PetscInt outletnumValues;
    const PetscInt *outletvalues;
    PetscInt domainnumValues;
    const PetscInt *domainvalues;

    ISGetSize(outletvaluesIS, &outletnumValues) >> utilities::PetscUtilities::checkError;
    ISGetIndices(outletvaluesIS, &outletvalues) >> utilities::PetscUtilities::checkError;
    ISGetSize(domainvaluesIS, &domainnumValues) >> utilities::PetscUtilities::checkError;
    ISGetIndices(domainvaluesIS, &domainvalues) >> utilities::PetscUtilities::checkError;
    std::vector<IS> outletsubISs(outletnumValues, nullptr);
    std::vector<IS> domainsubISs(domainnumValues, nullptr);

    //print outletnumValues
    PetscPrintf(PETSC_COMM_WORLD, "outletnumValues: %d\n", outletnumValues);
    PetscPrintf(PETSC_COMM_WORLD, "domainnumValues: %d\n", domainnumValues);

    //create an array to store the domain coordinates with zero length which will be appended to later
    PetscReal *domainVertexCoords = new PetscReal[0];
    PetscReal *domainCellCoords = new PetscReal[0];
    PetscReal *outletVertexCoords = new PetscReal[0];
    PetscReal *outletCellCoords = new PetscReal[0];
    
    PetscInt *outletCellID = new PetscInt[0];
    PetscInt *domainCellID = new PetscInt[0];
    
    PetscInt domainCoordsSize = 0;
    PetscInt outletCoordsSize = 0;
    

    //loop over the domain cells
    for (PetscInt v = 0; v < domainnumValues; ++v) {
        DMGetStratumIS(dm, "domain", domainvalues[v], &domainsubISs[v]) >> utilities::PetscUtilities::checkError;
        PetscPrintf(PETSC_COMM_WORLD, "-->\n");
        PetscInt ndomainPoints;
        ISGetLocalSize(domainsubISs[v], &ndomainPoints) >> utilities::PetscUtilities::checkError;
        //print the label associated with this region
        PetscPrintf(PETSC_COMM_WORLD, "--> Label %s with value %d has %d points\n", "domain", domainvalues[v], ndomainPoints);
        const PetscInt *points;
        ISGetIndices(domainsubISs[v], &points) >> utilities::PetscUtilities::checkError;
        // Vec domainsolVec;
        // PetscScalar *domainsolArray;
        // DMGetGlobalVector(subDomain->GetDM(), &domainsolVec) >> utilities::PetscUtilities::checkError;
        // VecGetArray(domainsolVec, &domainsolArray) >> utilities::PetscUtilities::checkError;
        //create a global array containing the coordinates of all the points in the domain
        for (PetscInt p = 0; p < ndomainPoints; ++p) {
            PetscInt point = points[p];
            PetscReal centroid[3];
            DMPlexComputeCellGeometryFVM(dm, point, nullptr, centroid, nullptr) >> utilities::PetscUtilities::checkError;

            PetscInt nVerts;
            PetscInt *verts;
            DMPlexCellGetVertices(dm, point, &nVerts, &verts) >> utilities::PetscUtilities::checkError;
            //store each vertex location (and corresponding cell) in the domainCoords arrays
            for (PetscInt v = 0; v < nVerts; ++v) {
                PetscInt vertex = verts[v];
                PetscReal vertexCoords[3];
                DMPlexComputeCellGeometryFVM(dm, vertex, nullptr, vertexCoords, nullptr) >> utilities::PetscUtilities::checkError;
                domainVertexCoords[domainCoordsSize * 3] = vertexCoords[0];
                domainVertexCoords[domainCoordsSize * 3 + 1] = vertexCoords[1];
                domainVertexCoords[domainCoordsSize * 3 + 2] = vertexCoords[2];
                domainCellCoords[domainCoordsSize * 3] = centroid[0];
                domainCellCoords[domainCoordsSize * 3 + 1] = centroid[1];
                domainCellCoords[domainCoordsSize * 3 + 2] = centroid[2];
                domainCellID[domainCoordsSize] = point;
                domainCoordsSize++;
            }
            DMPlexCellRestoreVertices(dm, point, &nVerts, &verts) >> utilities::PetscUtilities::checkError;
        }
    }

    //loop over the outlet cells
    for (PetscInt v=0; v<outletnumValues; ++v){
        DMGetStratumIS(dm, "outletCells", outletvalues[v], &outletsubISs[v]) >> utilities::PetscUtilities::checkError;
        PetscInt noutletPoints;
        ISGetLocalSize(outletsubISs[v], &noutletPoints) >> utilities::PetscUtilities::checkError;
        PetscPrintf(PETSC_COMM_WORLD, "--> Label %s with value %d has %d points\n", "outlet", outletvalues[v], noutletPoints);
        const PetscInt *points;
        ISGetIndices(outletsubISs[v], &points) >> utilities::PetscUtilities::checkError;
        // Vec outletsolVec;
        // PetscScalar *outletsolArray;
        // DMGetGlobalVector(subDomain->GetDM(), &outletsolVec) >> utilities::PetscUtilities::checkError;
        // VecGetArray(outletsolVec, &outletsolArray) >> utilities::PetscUtilities::checkError;

        //create a global array containing the coordinates of all the points in the outlet
        for (PetscInt p = 0; p < noutletPoints; ++p) {
            PetscInt point = points[p];
            PetscReal centroid[3];
            DMPlexComputeCellGeometryFVM(dm, point, nullptr, centroid, nullptr) >> utilities::PetscUtilities::checkError;

            PetscInt nVerts;
            PetscInt *verts;
            DMPlexCellGetVertices(dm, point, &nVerts, &verts) >> utilities::PetscUtilities::checkError;
            //store each vertex location (and corresponding cell) in the outletCoords arrays
            for (PetscInt vt = 0; vt < nVerts; ++vt) {
                PetscInt vertex = verts[vt];
                PetscReal vertexCoords[3];
                DMPlexComputeCellGeometryFVM(dm, vertex, nullptr, vertexCoords, nullptr) >> utilities::PetscUtilities::checkError;
                outletVertexCoords[outletCoordsSize * 3] = vertexCoords[0];
                outletVertexCoords[outletCoordsSize * 3 + 1] = vertexCoords[1];
                outletVertexCoords[outletCoordsSize * 3 + 2] = vertexCoords[2];
                outletCellCoords[outletCoordsSize * 3] = centroid[0];
                outletCellCoords[outletCoordsSize * 3 + 1] = centroid[1];
                outletCellCoords[outletCoordsSize * 3 + 2] = centroid[2];
                outletCellID[outletCoordsSize] = point;
                outletCoordsSize++;
            }
            DMPlexCellRestoreVertices(dm, point, &nVerts, &verts) >> utilities::PetscUtilities::checkError;
        }
    }


    // Vec solVec;
    // PetscScalar *solArray;
    // DMGetGlobalVector(subDomain->GetDM(), &solVec);
    // VecGetArray(solVec, &solArray);
    PetscInt *correspondingDomainCellID = new PetscInt[outletCoordsSize];
    // PetscInt sharedSize = 0;

    for (PetscInt j=0; j<domainCoordsSize; j++){
        
    }

    //loop over outletCoordsSize/all vertices in the outlet
    for(PetscInt i=0; i<outletCoordsSize; i++){

        // if (PetscAbs(outletVertexCoords[i*3] - 1.5e-3) < 1e-5){
        //     PetscPrintf(PETSC_COMM_WORLD, "outletVertexCoords[%d]: %g %g %g\n", i, outletVertexCoords[i*3], outletVertexCoords[i*3+1], outletVertexCoords[i*3+2]);
        // }

        for(PetscInt j=0; j<domainCoordsSize; j++){

            if (PetscAbs(domainVertexCoords[j*3] - 1.5e-3) < 1e-5){
                PetscPrintf(PETSC_COMM_WORLD, "-> domainVertexCoords[%d]: %g %g %g\n", j, domainVertexCoords[j*3], domainVertexCoords[j*3+1], domainVertexCoords[j*3+2]);
            }

            std::cout << "";

            PetscReal tol1 = 1e-5;
            PetscBool bool1 = (PetscAbsReal(outletVertexCoords[i*3] - domainVertexCoords[j * 3]) < tol1 &&
                   PetscAbsReal(outletVertexCoords[i*3+1] - domainVertexCoords[j * 3 + 1]) < tol1 &&
                   PetscAbsReal(outletVertexCoords[i*3+2] - domainVertexCoords[j * 3 + 2]) < tol1) ? PETSC_TRUE : PETSC_FALSE;

            // PetscReal tol2 = 0.0000075;
            // bool bool1 = (fabs(outletVertexCoords[i*3] - domainVertexCoords[j*3]) < tol1 && fabs(outletVertexCoords[i*3+1] - domainVertexCoords[j*3+1]) < tol1 && fabs(outletVertexCoords[i*3+2] - domainVertexCoords[j*3+2]) < tol1);
            // bool bool2 = ( fabs(outletCellCoords[i*3+1] - domainCellCoords[j*3+1]) < tol2 );

            if (bool1 == PETSC_TRUE){
                correspondingDomainCellID[i] = domainCellID[j];
            }
            else{
                correspondingDomainCellID[i] = -1;
            }

            
        }
    }

    //print each element of correspondingDomainCellID
    PetscPrintf(PETSC_COMM_WORLD, "domainCoordsSize: %d\n", domainCoordsSize);
    PetscPrintf(PETSC_COMM_WORLD, "outletCoordsSize: %d\n", outletCoordsSize);
    //print size of correspondingDomainCellID

    PetscPrintf(PETSC_COMM_WORLD, "correspondingDomainCellID: \n");
    for(PetscInt i=0; i<outletCoordsSize; i++){
        PetscPrintf(PETSC_COMM_WORLD, "correspondingDomainCellID[%d]: %d\n", i, correspondingDomainCellID[i]);
    }

    //free the domainCoords array
    delete[] domainVertexCoords;
    delete[] domainCellCoords;
    delete[] outletVertexCoords;
    delete[] outletCellCoords;
    delete[] domainCellID;
    delete[] outletCellID;
    delete[] correspondingDomainCellID;
    

    //free the shareVertexCount array
    // delete[] shareVertexCount;
    
    

    
}

void ablate::finiteVolume::boundaryConditions::ZeroDerBoundary::updateFunction(PetscReal time, const PetscReal *x, PetscScalar *vals, PetscInt point) {

    // PetscPrintf(PETSC_COMM_WORLD, "prior to update function");
    boundaryFunction->GetSolutionField().GetPetscFunction()(dim, time, x, fieldSize, vals, boundaryFunction->GetSolutionField().GetContext());

    //get the point in the region labeled as "domain" 


    // DMLabel domainLabel;
    // DMGetLabel(subDomain->GetDM(), "domain", &domainLabel) >> utilities::PetscUtilities::checkError;

    // //print the domain label; not the name but the label itself
    // PetscPrintf(PETSC_COMM_WORLD, "domainLabel= %p\n", (void *)domainLabel);
    // //print the value of the label
    // PetscInt value;
    // DMLabelGetValue(domainLabel, 0, &value) >> utilities::PetscUtilities::checkError;
    // PetscPrintf(PETSC_COMM_WORLD, "domainLabel value= %d\n", value);

    // IS domainIS;
    // DMLabelGetStratumIS(domainLabel, 1, &domainIS) >> utilities::PetscUtilities::checkError; //height 
    // const PetscInt *domainPoints;
    // PetscInt numDomainPoints;


    // if (!domainIS) {
    //     throw std::runtime_error("error, domainIS is null");
    // }
    // ISGetLocalSize(domainIS, &numDomainPoints) >> utilities::PetscUtilities::checkError;
    // ISGetIndices(domainIS, &domainPoints) >> utilities::PetscUtilities::checkError;

    // PetscPrintf(PETSC_COMM_WORLD, "not getting to here yet\n");

    


    // //find the closest cell in domain
    // PetscInt closestCell = -1;
    // PetscReal closestDistance = PETSC_MAX_REAL;
    // for (PetscInt i = 0; i < numDomainPoints; i++) {
    //     PetscInt cell = domainPoints[i];
    //     PetscReal cellCentroid[3];
    //     DMPlexComputeCellGeometryFVM(subDomain->GetDM(), cell, nullptr, cellCentroid, nullptr) >> utilities::PetscUtilities::checkError;
    //     PetscReal distance = 0.0;
    //     for (int d = 0; d < dim; d++) {
    //         distance += (x[d] - cellCentroid[d]) * (x[d] - cellCentroid[d]);
    //     }
    //     distance = sqrt(distance);
    //     if (distance < closestDistance) {
    //         closestDistance = distance;
    //         closestCell = cell;
    //     }
    // }

    // //fetch the field variable from that closest Cell
    // //let's fetch the entire solution vector
    // Vec solVec;
    // PetscScalar *solArray;
    // DMGetGlobalVector(subDomain->GetDM(), &solVec) >> utilities::PetscUtilities::checkError;
    // VecGetArray(solVec, &solArray) >> utilities::PetscUtilities::checkError;
    // // PetscScalar *domainFields = nullptr; 
    // // DMPlexPointLocalRef(subDomain->GetDM(), closestCell, solArray, &domainFields);
    // PetscScalar *boundaryFields = nullptr; 
    // DMPlexPointLocalRef(subDomain->GetDM(), point, solArray, &boundaryFields);

    // boundaryFields[ablate::finiteVolume::CompressibleFlowFields::RHO] = domainFields[ablate::finiteVolume::CompressibleFlowFields::RHO];
    // boundaryFields[ablate::finiteVolume::CompressibleFlowFields::RHOE] = domainFields[ablate::finiteVolume::CompressibleFlowFields::RHOE];
    // boundaryFields[ablate::finiteVolume::CompressibleFlowFields::RHOU] = domainFields[ablate::finiteVolume::CompressibleFlowFields::RHOU];
    // boundaryFields[ablate::finiteVolume::CompressibleFlowFields::RHOV] = domainFields[ablate::finiteVolume::CompressibleFlowFields::RHOV];
    // boundaryFields[ablate::finiteVolume::CompressibleFlowFields::RHOW] = domainFields[ablate::finiteVolume::CompressibleFlowFields::RHOW];
    // boundaryFields[subDomain->GetField(VOLUME_FRACTION_FIELD).offset] = domainFields[subDomain->GetField(VOLUME_FRACTION_FIELD).offset];
    // boundaryFields[subDomain->GetField(DENSITY_VF_FIELD).offset] = domainFields[subDomain->GetField(DENSITY_VF_FIELD).offset];


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
