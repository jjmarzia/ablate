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
    PetscReal *domainCoords = new PetscReal[0];
    PetscInt domainCoordsSize = 0;

    for (PetscInt v = 0; v < domainnumValues; ++v) {
        DMGetStratumIS(dm, "domain", domainvalues[v], &domainsubISs[v]) >> utilities::PetscUtilities::checkError;
        PetscPrintf(PETSC_COMM_WORLD, "-->\n");
        PetscInt ndomainPoints;
        ISGetLocalSize(domainsubISs[v], &ndomainPoints) >> utilities::PetscUtilities::checkError;
        //print the label associated with this region
        PetscPrintf(PETSC_COMM_WORLD, "--> Label %s with value %d has %d points\n", "domain", domainvalues[v], ndomainPoints);
        const PetscInt *points;
        ISGetIndices(domainsubISs[v], &points) >> utilities::PetscUtilities::checkError;
        Vec domainsolVec;
        PetscScalar *domainsolArray;
        DMGetGlobalVector(subDomain->GetDM(), &domainsolVec) >> utilities::PetscUtilities::checkError;
        VecGetArray(domainsolVec, &domainsolArray) >> utilities::PetscUtilities::checkError;
        //create a global array containing the coordinates of all the points in the domain
        for (PetscInt p = 0; p < ndomainPoints; ++p) {
            PetscInt point = points[p];
            PetscReal centroid[3];
            DMPlexComputeCellGeometryFVM(dm, point, nullptr, centroid, nullptr) >> utilities::PetscUtilities::checkError;

            //loop over the vertices and store each vertex location in the domainCoords array
            PetscInt nVerts;
            PetscInt *verts;
            DMPlexCellGetVertices(dm, point, &nVerts, &verts) >> utilities::PetscUtilities::checkError;
            //store each vertex location in the domainCoords array
            for (PetscInt v = 0; v < nVerts; ++v) {
                PetscInt vertex = verts[v];
                PetscReal vertexCoords[3];
                DMPlexComputeCellGeometryFVM(dm, vertex, nullptr, vertexCoords, nullptr) >> utilities::PetscUtilities::checkError;
                //append the vertex coords to the domainCoords array
                // domainCoords = (PetscReal *)realloc(domainCoords, (domainCoordsSize + 1) * 3 * sizeof(PetscReal));
                domainCoords[domainCoordsSize * 3] = vertexCoords[0];
                domainCoords[domainCoordsSize * 3 + 1] = vertexCoords[1];
                domainCoords[domainCoordsSize * 3 + 2] = vertexCoords[2];
                domainCoordsSize++;
            }
        }
    }

    //loop over the outlet cells
    for (PetscInt v=0; v<outletnumValues; ++v){
        DMGetStratumIS(dm, "outletCells", outletvalues[v], &outletsubISs[v]) >> utilities::PetscUtilities::checkError;
        PetscInt noutletPoints;
        ISGetLocalSize(outletsubISs[v], &noutletPoints) >> utilities::PetscUtilities::checkError;
        const PetscInt *points;
        ISGetIndices(outletsubISs[v], &points) >> utilities::PetscUtilities::checkError;
        Vec outletsolVec;
        PetscScalar *outletsolArray;
        DMGetGlobalVector(subDomain->GetDM(), &outletsolVec) >> utilities::PetscUtilities::checkError;
        VecGetArray(outletsolVec, &outletsolArray) >> utilities::PetscUtilities::checkError;

        //create an array marking the outlet cells of size noutletPoints that share a vertex with the domain
        PetscInt *shareVertexCount = new PetscInt[noutletPoints];

        for (PetscInt p = 0; p < noutletPoints; ++p) {
            PetscInt point = points[p];
            // PetscReal centroid[3];
            //get the vertices associated with this cell
            PetscInt nVerts;
            PetscInt *verts;
            DMPlexCellGetVertices(dm, point, &nVerts, &verts) >> utilities::PetscUtilities::checkError;
            //loop over the vertices and check to see if the vertex location is anywhere inside the domainCoords array with a tolerance of 0.0001
            for (PetscInt v = 0; v < nVerts; ++v) {
                PetscInt vertex = verts[v];
                PetscReal vertexCoords[3];
                DMPlexComputeCellGeometryFVM(dm, vertex, nullptr, vertexCoords, nullptr) >> utilities::PetscUtilities::checkError;
                //check to see if the vertexCoords are in the domainCoords array
                for (PetscInt d = 0; d < domainCoordsSize; ++d) {
                    PetscReal tol = 0.0000001;
                    if (PetscAbsReal(vertexCoords[0] - domainCoords[d * 3]) < tol && PetscAbsReal(vertexCoords[1] - domainCoords[d * 3 + 1]) < tol && PetscAbsReal(vertexCoords[2] - domainCoords[d * 3 + 2]) < tol) {
                        //print the coordinates of the vertex
                        // PetscPrintf(PETSC_COMM_WORLD, "vertexCoords: %g %g %g\n", vertexCoords[0], vertexCoords[1], vertexCoords[2]);
                        //mark the outlet cell as sharing a vertex with the domain by adding 1 to an array containing the number of vertices shared with the domain
                        shareVertexCount[p]++;
                        break;
                    }
                }
            }
            
        }

        //print all elements of the shareVertexCount array
        for (PetscInt p = 0; p < noutletPoints; ++p) {
            PetscPrintf(PETSC_COMM_WORLD, "shareVertexCount[%d]: %d\n", p, shareVertexCount[p]);
        }
        //free the shareVertexCount array
        delete[] shareVertexCount;
    }

    //free the domainCoords array
    delete[] domainCoords;
    //free the shareVertexCount array
    // delete[] shareVertexCount;
    
    PetscPrintf(PETSC_COMM_WORLD, "domainCoordsSize: %d\n", domainCoordsSize);

    for (PetscInt v = 0; v < outletnumValues; ++v) {

        DMGetStratumIS(dm, "outletCells", outletvalues[v], &outletsubISs[v]) >> utilities::PetscUtilities::checkError;
        PetscInt noutletPoints;
        ISGetLocalSize(outletsubISs[v], &noutletPoints) >> utilities::PetscUtilities::checkError;
        //print the label associated with this region
        PetscPrintf(PETSC_COMM_WORLD, "--> Label %s with value %d has %d points\n", "outletCells", outletvalues[v], noutletPoints);
        const PetscInt *points;
        ISGetIndices(outletsubISs[v], &points) >> utilities::PetscUtilities::checkError;
        Vec outletsolVec;
        PetscScalar *outletsolArray;
        DMGetGlobalVector(subDomain->GetDM(), &outletsolVec) >> utilities::PetscUtilities::checkError;
        PetscPrintf(PETSC_COMM_WORLD, "--> vec\n");
        VecGetArray(outletsolVec, &outletsolArray) >> utilities::PetscUtilities::checkError;
        PetscPrintf(PETSC_COMM_WORLD, "--> array\n");

        for (PetscInt p = 0; p < noutletPoints; ++p) {
            PetscInt point = points[p];


            // PetscReal centroid[3];
            // DMPlexComputeCellGeometryFVM(dm, point, nullptr, centroid, nullptr) >> utilities::PetscUtilities::checkError;
            // if ( centroid[0] < (0.0015 + 0.001515)/2 - (0.0015 - 0.001515)/4 ) {
            //     PetscInt neighbor;
            //     PetscReal neighborcentroid[3] = {0.001499, centroid[1], centroid[2]};
            //     DMPlexFindCell(dm, neighborcentroid, (0.0015 - 0.001515)/2, &neighbor) >> utilities::PetscUtilities::checkError;
            //     PetscPrintf(PETSC_COMM_WORLD, "point %d: (%g, %g, %g) neighbor centroid: %g %g %g neighbor: %d\n", point, centroid[0], centroid[1], centroid[2], neighborcentroid[0], neighborcentroid[1], neighborcentroid[2], neighbor);

            // }

            //using petscsupport.cpp functions get the vertices associated with this cell
            PetscInt nVerts;
            PetscInt *verts;
            DMPlexCellGetVertices(dm, point, &nVerts, &verts) >> utilities::PetscUtilities::checkError;
            //loop over the vertices and get the cells corresponding to each vertex
            for (PetscInt v = 0; v < nVerts; ++v) {
                PetscInt vertex = verts[v];
                PetscInt nCells;
                PetscInt *cells;
                DMPlexVertexGetCells(dm, vertex, &nCells, &cells) >> utilities::PetscUtilities::checkError;
                //print the number of cell associated with this vertex;
                // PetscPrintf(PETSC_COMM_WORLD, "vertex %d has %d cells\n", vertex, nCells);
                //for each cell IF the x coordinate of the cell is less than 0.0015, then print the cell
                for (PetscInt c = 0; c < nCells; ++c) {
                    //get the centroid of cells[c]
                    PetscReal centroid[3];
                    DMPlexComputeCellGeometryFVM(dm, cells[c], nullptr, centroid, nullptr) >> utilities::PetscUtilities::checkError;
                    //if the x coordinate of the centroid is less than 0.0015, then print the cell
                    if (centroid[0] < 0.0015) {
                        PetscPrintf(PETSC_COMM_WORLD, "--> pt %d cell neighbor is x=%lf y=%lf  cell %d\n", point, centroid[0], centroid[1], cells[c]);
                    }
                }
                DMPlexVertexRestoreCells(dm, vertex, &nCells, &cells) >> utilities::PetscUtilities::checkError;
            }
            DMPlexCellRestoreVertices(dm, point, &nVerts, &verts) >> utilities::PetscUtilities::checkError;


            // PetscInt nNeighbors, *neighbors;
            // DMPlexGetNeighbors(dm, point, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors) >> utilities::PetscUtilities::checkError;
            // //print the neighbors
            // PetscPrintf(PETSC_COMM_WORLD, "point %d has %d neighbors\n", point, nNeighbors);
            // for (PetscInt n = 0; n < nNeighbors; ++n) {
            //     PetscPrintf(PETSC_COMM_WORLD, "--> pt %d x=%lf y=%lf  neighbor %d\n", point, centroid[0], centroid[1], neighbors[n]);
            // }
            // DMPlexRestoreNeighbors(dm, point, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors) >> utilities::PetscUtilities::checkError;

            // PetscScalar *boundaryFields = nullptr; 
            // DMPlexPointLocalRef(subDomain->GetDM(), point, solArray, &boundaryFields);
            //     boundaryFields[ablate::finiteVolume::CompressibleFlowFields::RHO] = 5;
            //     boundaryFields[ablate::finiteVolume::CompressibleFlowFields::RHOE] = 5;
            //     boundaryFields[ablate::finiteVolume::CompressibleFlowFields::RHOU] = 5;
            //     boundaryFields[ablate::finiteVolume::CompressibleFlowFields::RHOV] = 5;
            //     boundaryFields[ablate::finiteVolume::CompressibleFlowFields::RHOW] = 5;
            //     boundaryFields[subDomain->GetField(VOLUME_FRACTION_FIELD).offset] = 5;
            //     boundaryFields[subDomain->GetField(DENSITY_VF_FIELD).offset] = 5;

            // PetscPrintf(PETSC_COMM_WORLD, "point %d: (%g, %g, %g)\n", point, centroid[0], centroid[1], centroid[2]);
        }
        // DMRestoreGlobalVector(subDomain->GetDM(), &solVec) >> utilities::PetscUtilities::checkError;
        // VecRestoreArray(solVec, &solArray) >> utilities::PetscUtilities::checkError;
        ISRestoreIndices(outletsubISs[v], &points) >> utilities::PetscUtilities::checkError;

    }


    //print the number of cells based on pstart, pend
    // PetscPrintf(PETSC_COMM_WORLD, "pStart: %d, pEnd: %d\n", pStart, pEnd);
    //print the number of cells betweeen pstart and pend
    // PetscPrintf(PETSC_COMM_WORLD, "Number of cells: %d\n", pEnd - pStart);
    // for (PetscInt i = pStart; i < pEnd; ++i) {
    //     PetscInt cell = i;
    //     PetscInt nNeighbors, *neighbors;
    //     DMPlexGetNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors) >> utilities::PetscUtilities::checkError;
    //     //print the number of neighbors
    //     // PetscPrintf(PETSC_COMM_WORLD, "cell: %d, nNeighbors: %d\n", cell, nNeighbors);
    //     // cellNeighbors[cell] = std::vector<PetscInt>(neighbors, neighbors + nNeighbors);

    //     DMPlexRestoreNeighbors(dm, cell, 1, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors) >> utilities::PetscUtilities::checkError;
    // }
    

    
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
