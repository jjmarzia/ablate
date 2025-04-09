#include "extrudeLabel.hpp"
#include <petsc/private/dmimpl.h>
#include <petsc/private/petscfeimpl.h>
#include <petscdmplextransform.h>
#include <utility>
#include "tagLabelInterface.hpp"
#include "utilities/petscUtilities.hpp"
#include "utilities/petscSupport.hpp"


ablate::domain::modifiers::ExtrudeLabel::ExtrudeLabel(std::vector<std::shared_ptr<domain::Region>> regions, std::shared_ptr<domain::Region> boundaryRegion,
                                                      std::shared_ptr<domain::Region> originalRegion, std::shared_ptr<domain::Region> extrudedRegion, double thickness)
    : regions(std::move(regions)), boundaryRegion(std::move(std::move(boundaryRegion))), originalRegion(std::move(originalRegion)), extrudedRegion(std::move(extrudedRegion)), thickness(thickness) {}

std::string ablate::domain::modifiers::ExtrudeLabel::ToString() const {
    std::string string = "ablate::domain::modifiers::ExtrudeLabel: (";
    for (const auto &region : regions) {
        string += region->ToString() + ",";
    }
    string.back() = ')';
    return string;
}

void ablate::domain::modifiers::ExtrudeLabel::Modify(DM &dm) {
    // create a temporary label to hold adapt information
    //adaptation label makes which cells need to be extruded
    DMLabel adaptLabel;
    DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", &adaptLabel) >> utilities::PetscUtilities::checkError;

    // loop through regions ie inlet, outlet, slab etc
    for (const auto &region : regions) {
        region->CheckForLabel(dm, PetscObjectComm((PetscObject)dm));
        DMLabel regionLabel;
        PetscInt regionValue;
        domain::Region::GetLabel(region, dm, regionLabel, regionValue);

// print the name of the label associated with the region
        if (!regionLabel) {
            PetscPrintf(PETSC_COMM_SELF, "Warning: Region %s does not exist on the mesh, skipping extrude label for this region.\n", region->ToString().c_str());
            continue;
        } else {
            PetscPrintf(PETSC_COMM_SELF, "Extruding region: %s with label value: %d\n", region->ToString().c_str(), regionValue);
        }

        // If this label exists on this domain then process its set of points/cells to extrude
        if (regionLabel) {
            IS bdIS;
            const PetscInt *points;
            PetscInt n, i;

            //gets the points in the region
            DMLabelGetStratumIS(regionLabel, regionValue, &bdIS) >> utilities::PetscUtilities::checkError;
            if (!bdIS) {
                continue;
            }
            

            ISGetLocalSize(bdIS, &n) >> utilities::PetscUtilities::checkError;
            ISGetIndices(bdIS, &points) >> utilities::PetscUtilities::checkError;
            for (i = 0; i < n; ++i) {
                DMLabelSetValue(adaptLabel, points[i], DM_ADAPT_REFINE) >> utilities::PetscUtilities::checkError;

                // //print the point being extruded
                // if (i < 10) {  // only print the first 10 for readability
                //     PetscPrintf(PETSC_COMM_SELF, "Marking point %d for extrusion in adaptLabel\n", points[i]);
                // }
            }
            ISRestoreIndices(bdIS, &points) >> utilities::PetscUtilities::checkError;
            ISDestroy(&bdIS) >> utilities::PetscUtilities::checkError;
        }
    }

    // set the options for the transform
    PetscOptions transformOptions;
    PetscOptionsCreate(&transformOptions) >> utilities::PetscUtilities::checkError;
    PetscOptionsInsertString(transformOptions, "-dm_plex_transform_type extrude");
    PetscOptionsInsertString(transformOptions, "-dm_plex_transform_extrude_use_tensor 0");

    // determine if the thickness needs to be computed
    PetscReal extrudeThickness = thickness;
    if (extrudeThickness == 0.0) {
        // Get the fv geom
        //if no thickness specified then just make the thickness twice the minimum radius of the mesh (cell center size)
        DMPlexGetGeometryFVM(dm, nullptr, nullptr, &extrudeThickness) >> utilities::PetscUtilities::checkError;
        extrudeThickness *= 2.0;  // double the thickness
    }
    const auto extrudeThicknessString = std::to_string(extrudeThickness);
    PetscOptionsSetValue(transformOptions, "-dm_plex_transform_extrude_thickness", extrudeThicknessString.c_str());

    // extrude the mesh
    //create new mesh with extruded cells
    DM dmAdapt;
    DMPlexTransformAdaptLabel(dm, nullptr, adaptLabel, nullptr, transformOptions, &dmAdapt) >> utilities::PetscUtilities::checkError;

    PetscPrintf(PETSC_COMM_WORLD, "Mesh extrusion completed.\n"); //this works
    if (!dmAdapt) {
        PetscPrintf(PETSC_COMM_WORLD,
                     "Error: DMPlexTransformAdaptLabel failed to produce an adapted mesh. Check the input mesh and adaptation label.\n");
    }

    
    if (dmAdapt) {
        (dmAdapt)->prealloc_only = dm->prealloc_only; // preserve the preallocation settings of the original dm
        PetscFree((dmAdapt)->vectype);
        PetscStrallocpy(dm->vectype, (char **)&(dmAdapt)->vectype); // preserve the vector type of the original dm
        PetscFree((dmAdapt)->mattype);
        PetscStrallocpy(dm->mattype, (char **)&(dmAdapt)->mattype); // preserve the matrix type of the original dm
    }

    // create hew new labels for each region (on the new adapted dm)
    DMLabel originalRegionLabel, extrudedRegionLabel;
    PetscInt originalRegionValue, extrudedRegionValue;
    originalRegion->CreateLabel(dmAdapt, originalRegionLabel, originalRegionValue);
    extrudedRegion->CreateLabel(dmAdapt, extrudedRegionLabel, extrudedRegionValue);

    //confirm the labels were created
    PetscPrintf(PETSC_COMM_WORLD,
                 "Created original region label on adapted mesh: %s with value: %d\n",
                 originalRegion->ToString().c_str(), originalRegionValue);
    PetscPrintf(PETSC_COMM_WORLD,
                 "Created extruded region label on adapted mesh: %s with value: %d\n",
                 extrudedRegion->ToString().c_str(), extrudedRegionValue);



    PetscInt cOriginalStart, cOriginalEnd;
    DMPlexGetHeightStratum(dm, 0, &cOriginalStart, &cOriginalEnd) >> utilities::PetscUtilities::checkError;
        PetscInt cAdaptStart, cAdaptEnd;
    DMPlexGetHeightStratum(dmAdapt, 0, &cAdaptStart, &cAdaptEnd) >> utilities::PetscUtilities::checkError;
        //adapted mesh = boundaryCells
// get the cell range of the original mesh
PetscPrintf(PETSC_COMM_WORLD, "Orig mesh cell range: [%d, %d)\n", cOriginalStart, cOriginalEnd);
// get the cell range of the adapted mesh
    PetscPrintf(PETSC_COMM_WORLD, "Adapted mesh cell range: [%d, %d)\n", cAdaptStart, cAdaptEnd);

    // store all original mesh x,y,z coordinates into an array
    std::vector<PetscReal> originalCellCentroids(cOriginalEnd - cOriginalStart);
    PetscReal *originalCellCentroidPtr = originalCellCentroids.data();
    for (PetscInt cell = cOriginalStart; cell < cOriginalEnd; ++cell) {
        PetscReal vol;
        PetscReal centroid[3];
        DMPlexComputeCellGeometryFVM(dm, cell, &vol, centroid, nullptr) >> utilities::PetscUtilities::checkError;
        originalCellCentroidPtr[cell - cOriginalStart] = centroid[0]; 
        originalCellCentroidPtr[cell - cOriginalStart + 1] = centroid[1];
        originalCellCentroidPtr[cell - cOriginalStart + 2] = centroid[2]; 
    }
    //now store the coordinates of the adapted mesh cells for comparison
    std::vector<PetscReal> adaptedCellCentroids(cAdaptEnd - cAdaptStart);
    PetscReal *adaptedCellCentroidPtr = adaptedCellCentroids.data();
    for (PetscInt cell = cAdaptStart; cell < cAdaptEnd; ++cell) {
        PetscReal vol;
        PetscReal centroid[3];
        DMPlexComputeCellGeometryFVM(dmAdapt, cell, &vol, centroid, nullptr) >> utilities::PetscUtilities::checkError;
        adaptedCellCentroidPtr[cell - cAdaptStart] = centroid[0];
        adaptedCellCentroidPtr[cell - cAdaptStart + 1] = centroid[1]; 
        adaptedCellCentroidPtr[cell - cAdaptStart + 2] = centroid[2]; 
    }
    //now check for each adapted cell if it corresponds to an original cell;
    //PRINT adapted cell if it is not corresponding to an original cell
    //also count the total number of adapted cells that do not correspond to original cells
    //let correspondence be within a tolerance of 1e-6 in x,y,z coordinates
    PetscInt numCorrespondingCells = 0;
    for (PetscInt cell = cAdaptStart; cell < cAdaptEnd; ++cell) {
        PetscBool foundCorrespondingCell = PETSC_FALSE;
        PetscReal *adaptedCentroid = adaptedCellCentroidPtr + (cell - cAdaptStart) * 3;

        for (PetscInt originalCellIndex = 0; originalCellIndex < (cOriginalEnd - cOriginalStart); ++originalCellIndex) {
            PetscReal *originalCentroid = originalCellCentroidPtr + originalCellIndex * 3;

            // check if the coordinates match within tolerance
            if (PetscAbsReal(adaptedCentroid[0] - originalCentroid[0]) < 1e-6 &&
                PetscAbsReal(adaptedCentroid[1] - originalCentroid[1]) < 1e-6 &&
                PetscAbsReal(adaptedCentroid[2] - originalCentroid[2]) < 1e-6) {
                foundCorrespondingCell = PETSC_TRUE;
                numCorrespondingCells++;
                break;
            }
        }

        if (!foundCorrespondingCell or foundCorrespondingCell) { //print all cells for now whether they're original or adapted
            PetscPrintf(PETSC_COMM_WORLD, "Adapted cell %d does not correspond to any original cell. Coordinates: (%g, %g, %g)\n", cell,
                        adaptedCentroid[0], adaptedCentroid[1], adaptedCentroid[2]);
        }
    }
    PetscPrintf(PETSC_COMM_WORLD,
                 "Total number of adapted cells: %d, Number of corresponding original cells: %d\n",
                 cAdaptEnd - cAdaptStart, numCorrespondingCells);




    // cell the depths of the cell layer
    //depth = the number of layers of cells in the original mesh that were extruded
    PetscInt cellDepth;
    DMPlexGetDepth(dm, &cellDepth) >> utilities::PetscUtilities::checkError;
    DMLabel depthAdaptLabel, ctAdaptLabel, ctLabel;
    DMPlexGetDepthLabel(dmAdapt, &depthAdaptLabel) >> utilities::PetscUtilities::checkError;
    DMPlexGetCellTypeLabel(dmAdapt, &ctAdaptLabel) >> utilities::PetscUtilities::checkError;
    DMPlexGetCellTypeLabel(dm, &ctLabel) >> utilities::PetscUtilities::checkError;

    // because the new cells can be intertwined with the old cells for mixed use we need to do this cell type by cell type
    for (PetscInt cellType = 0; cellType < DM_NUM_POLYTOPES; ++cellType) {
        auto ict = (DMPolytopeType)cellType;

        // get the new range for this cell type
        PetscInt tAdaptStart, tAdaptEnd;
        DMLabelGetStratumBounds(ctAdaptLabel, ict, &tAdaptStart, &tAdaptEnd) >> utilities::PetscUtilities::checkError;

        // only check if there are cell of this type
        if (tAdaptStart < 0) {
            continue;
        }
        // determine the depth of this cell type
        PetscInt cellTypeDepth;
        DMLabelGetValue(depthAdaptLabel, tAdaptStart, &cellTypeDepth) >> utilities::PetscUtilities::checkError;
        if (cellTypeDepth != cellDepth) {
            continue;
        }

        // Get the original range for this cell type
        PetscInt tStart, tEnd;
        DMLabelGetStratumBounds(ctLabel, ict, &tStart, &tEnd) >> utilities::PetscUtilities::checkError;
        PetscInt numberOldCells = tStart >= 0 ? tEnd - tStart : 0;

        // march over each new cell
        for (PetscInt c = tAdaptStart; c < tAdaptEnd; ++c) {
            if ((c - tAdaptStart) < numberOldCells) {
                DMLabelSetValue(originalRegionLabel, c, originalRegionValue) >> utilities::PetscUtilities::checkError;
            } else {
                DMLabelSetValue(extrudedRegionLabel, c, extrudedRegionValue) >> utilities::PetscUtilities::checkError;
            }
        }
    }

    // complete the labels
    DMPlexLabelComplete(dmAdapt, originalRegionLabel);
    DMPlexLabelComplete(dmAdapt, extrudedRegionLabel);

    // tag the interface between the faces (reuse modifier)
    TagLabelInterface(originalRegion, extrudedRegion, boundaryRegion).Modify(dmAdapt);

    // replace the dm
    ReplaceDm(dm, dmAdapt);

    // cleanup
    PetscOptionsDestroy(&transformOptions) >> utilities::PetscUtilities::checkError;
    DMLabelDestroy(&adaptLabel) >> utilities::PetscUtilities::checkError;
}

PetscErrorCode ablate::domain::modifiers::ExtrudeLabel::DMPlexTransformAdaptLabel(DM dm, Vec, DMLabel adaptLabel, DMLabel, PetscOptions transformOptions, DM *rdm) {
    DMPlexTransform tr;
    DM cdm, rcdm;

    PetscFunctionBegin;
    PetscCall(DMPlexTransformCreate(PetscObjectComm((PetscObject)dm), &tr));
    PetscCall(PetscObjectSetOptions((PetscObject)tr, transformOptions));
    PetscCall(DMPlexTransformSetDM(tr, dm));
    PetscCall(DMPlexTransformSetFromOptions(tr));
    PetscCall(DMPlexTransformSetActive(tr, adaptLabel));
    PetscCall(DMPlexTransformSetUp(tr));
    PetscCall(PetscObjectViewFromOptions((PetscObject)tr, nullptr, "-dm_plex_transform_view"));
    PetscCall(DMPlexTransformApply(tr, dm, rdm));
    PetscCall(DMCopyDisc(dm, *rdm));
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetCoordinateDM(*rdm, &rcdm));
    PetscCall(DMCopyDisc(cdm, rcdm));
    PetscCall(DMPlexTransformCreateDiscLabels(tr, *rdm));
    PetscCall(DMCopyDisc(dm, *rdm));
    PetscCall(DMPlexTransformDestroy(&tr));
    ((DM_Plex *)(*rdm)->data)->useHashLocation = ((DM_Plex *)dm->data)->useHashLocation;
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::ExtrudeLabel, "Extrudes a layer of cells based upon the region provided",
         ARG(std::vector<ablate::domain::Region>, "regions", "the region(s) describing the boundary cells"),
         ARG(ablate::domain::Region, "boundaryRegion", "the new label describing the faces between the original and extruded regions"),
         ARG(ablate::domain::Region, "originalRegion", "the region describing the original mesh"), ARG(ablate::domain::Region, "extrudedRegion", "the region describing the new extruded cells"),
         OPT(double, "thickness", "thickness for the extruded cells. If default (0) the 2 * minimum cell radius is used"));