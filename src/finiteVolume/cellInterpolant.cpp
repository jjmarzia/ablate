#include "cellInterpolant.hpp"
#include "utilities/petscSupport.hpp"
#include <petsc/private/dmpleximpl.h>
#include <utility>

ablate::finiteVolume::CellInterpolant::CellInterpolant(std::shared_ptr<ablate::domain::SubDomain> subDomainIn, const std::shared_ptr<domain::Region>& solverRegion, Vec faceGeomVec, Vec cellGeomVec)
    : subDomain(std::move(std::move(subDomainIn))) {
    auto getGradientDm = [this, solverRegion, faceGeomVec, cellGeomVec](const domain::Field& fieldInfo, std::vector<DM>& gradDMs) {
        auto petscField = subDomain->GetPetscFieldObject(fieldInfo);
        auto petscFieldFV = (PetscFV)petscField;

        PetscBool computeGradients;
        PetscFVGetComputeGradients(petscFieldFV, &computeGradients) >> utilities::PetscUtilities::checkError;

        if (computeGradients) {
            DM dmGradInt;

            DMLabel regionLabel = nullptr;
            PetscInt regionValue = PETSC_DECIDE;
            domain::Region::GetLabel(solverRegion, subDomain->GetDM(), regionLabel, regionValue);

            ComputeGradientFVM(subDomain->GetFieldDM(fieldInfo), regionLabel, regionValue, petscFieldFV, faceGeomVec, cellGeomVec, &dmGradInt) >> utilities::PetscUtilities::checkError;
            gradDMs.push_back(dmGradInt);
        } else {
            gradDMs.push_back(nullptr);
        }
    };

    // Compute the gradient dm for each field that supports it
    for (const auto& fieldInfo : subDomain->GetFields()) {
        getGradientDm(fieldInfo, gradientCellDms);
    }
}

ablate::finiteVolume::CellInterpolant::~CellInterpolant() {
    for (auto& dm : gradientCellDms) {
        if (dm) {
            DMDestroy(&dm) >> utilities::PetscUtilities::checkError;
        }
    }
}

void ablate::finiteVolume::CellInterpolant::ComputeRHS(PetscReal time, Vec locXVec, Vec locAuxVec, Vec locFVec, const std::shared_ptr<domain::Region>& solverRegion,
                                                       std::vector<CellInterpolant::DiscontinuousFluxFunctionDescription>& rhsFunctions, const ablate::domain::Range& faceRange,
                                                       const ablate::domain::Range& cellRange, Vec cellGeomVec, Vec faceGeomVec) {
    auto dm = subDomain->GetDM();
    auto dmAux = subDomain->GetAuxDM();

    /* 1: Get sizes from dm and dmAux */
    PetscSection section = nullptr;
    DMGetLocalSection(dm, &section) >> utilities::PetscUtilities::checkError;

    // Get the ds from he subDomain and required info
    auto ds = subDomain->GetDiscreteSystem();
    PetscInt nf, totDim;
    PetscDSGetNumFields(ds, &nf) >> utilities::PetscUtilities::checkError;
    PetscDSGetTotalDimension(ds, &totDim) >> utilities::PetscUtilities::checkError;

    // Check to see if the dm has an auxVec/auxDM associated with it.  If it does, extract it
    PetscDS dsAux = subDomain->GetAuxDiscreteSystem();
    PetscInt naf = 0, totDimAux = 0;
    if (locAuxVec) {
        PetscDSGetTotalDimension(dsAux, &totDimAux) >> utilities::PetscUtilities::checkError;
        PetscDSGetNumFields(dsAux, &naf) >> utilities::PetscUtilities::checkError;
    }

    /* 2: Get geometric data */
    // We can use a single call for the geometry data because it does not depend on the fv object
    const PetscScalar* cellGeomArray = nullptr;
    const PetscScalar* faceGeomArray = nullptr;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
    VecGetArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;
    DM faceDM, cellDM;
    VecGetDM(faceGeomVec, &faceDM) >> utilities::PetscUtilities::checkError;
    VecGetDM(cellGeomVec, &cellDM) >> utilities::PetscUtilities::checkError;

    // Get raw access to the computed values
    const PetscScalar *xArray, *auxArray = nullptr;
    VecGetArrayRead(locXVec, &xArray) >> utilities::PetscUtilities::checkError;
    if (locAuxVec) {
        VecGetArrayRead(locAuxVec, &auxArray) >> utilities::PetscUtilities::checkError;
    }

    // get raw access to the locF
    PetscScalar* locFArray;
    VecGetArray(locFVec, &locFArray) >> utilities::PetscUtilities::checkError;

    // there must be a separate gradient vector/dm for field because they can be different sizes
    std::vector<Vec> locGradVecs(nf, nullptr);

    /* Reconstruct and limit cell gradients */
    // for each field compute the gradient in the localGrads vector
    for (const auto& field : subDomain->GetFields()) {
        ComputeFieldGradients(field, locXVec, locGradVecs[field.subId], gradientCellDms[field.subId], cellGeomVec, faceGeomVec, faceRange, cellRange);
    }

    std::vector<const PetscScalar*> locGradArrays(nf, nullptr);
    for (const auto& field : subDomain->GetFields()) {
        if (locGradVecs[field.subId]) {
            VecGetArrayRead(locGradVecs[field.subId], &locGradArrays[field.subId]) >> utilities::PetscUtilities::checkError;
        }
    }

    ComputeFluxSourceTerms(dm,
                           ds,
                           totDim,
                           xArray,
                           dmAux,
                           dsAux,
                           totDimAux,
                           auxArray,
                           faceDM,
                           faceGeomArray,
                           cellDM,
                           cellGeomArray,
                           gradientCellDms,
                           locGradArrays,
                           locFArray,
                           solverRegion,
                           rhsFunctions,
                           faceRange,
                           cellRange);

    // clean up cell grads
    for (const auto& field : subDomain->GetFields()) {
        if (locGradVecs[field.subId]) {
            VecRestoreArrayRead(locGradVecs[field.subId], &locGradArrays[field.subId]) >> utilities::PetscUtilities::checkError;
            DMRestoreLocalVector(gradientCellDms[field.subId], &locGradVecs[field.subId]) >> utilities::PetscUtilities::checkError;
        }
    }

    // cleanup (restore access to locGradVecs, locAuxGradVecs with DMRestoreLocalVector)
    VecRestoreArrayRead(locXVec, &xArray) >> utilities::PetscUtilities::checkError;
    if (locAuxVec) {
        VecRestoreArrayRead(locAuxVec, &auxArray) >> utilities::PetscUtilities::checkError;
    }

    VecRestoreArray(locFVec, &locFArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(faceGeomVec, (const PetscScalar**)&faceGeomArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(cellGeomVec, (const PetscScalar**)&cellGeomArray) >> utilities::PetscUtilities::checkError;
}

void ablate::finiteVolume::CellInterpolant::ComputeRHS(PetscReal time, Vec locXVec, Vec locAuxVec, Vec locFVec, const std::shared_ptr<domain::Region>& solverRegion,
                                                       std::vector<CellInterpolant::PointFunctionDescription>& rhsFunctions, const ablate::domain::Range& cellRange, Vec cellGeomVec) {
    auto dm = subDomain->GetDM();
    auto dmAux = subDomain->GetAuxDM();

    /* 1: Get sizes from dm and dmAux */
    PetscSection section = nullptr;
    DMGetLocalSection(dm, &section) >> utilities::PetscUtilities::checkError;

    // Get the ds from he subDomain and required info
    auto ds = subDomain->GetDiscreteSystem();
    PetscInt nf, totDim;
    PetscDSGetNumFields(ds, &nf) >> utilities::PetscUtilities::checkError;
    PetscDSGetTotalDimension(ds, &totDim) >> utilities::PetscUtilities::checkError;

    // Check to see if the dm has an auxVec/auxDM associated with it.  If it does, extract it
    PetscDS dsAux = subDomain->GetAuxDiscreteSystem();
    PetscInt naf = 0, totDimAux = 0;
    if (locAuxVec) {
        PetscDSGetTotalDimension(dsAux, &totDimAux) >> utilities::PetscUtilities::checkError;
        PetscDSGetNumFields(dsAux, &naf) >> utilities::PetscUtilities::checkError;
    }

    // We can use a single call for the geometry data because it does not depend on the fv object
    const PetscScalar* cellGeomArray = nullptr;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
    DM cellDM;
    VecGetDM(cellGeomVec, &cellDM) >> utilities::PetscUtilities::checkError;

    // Get raw access to the computed values
    const PetscScalar *xArray, *auxArray = nullptr;
    VecGetArrayRead(locXVec, &xArray) >> utilities::PetscUtilities::checkError;
    if (locAuxVec) {
        VecGetArrayRead(locAuxVec, &auxArray) >> utilities::PetscUtilities::checkError;
    }

    // get raw access to the locF
    PetscScalar* locFArray;
    VecGetArray(locFVec, &locFArray) >> utilities::PetscUtilities::checkError;

    // Compute the source terms from flux across the interface for cell based gradient functions
    // Precompute the offsets to pass into the rhsFluxFunctionDescriptions
    std::vector<std::vector<PetscInt>> fluxComponentSize(rhsFunctions.size());
    std::vector<std::vector<PetscInt>> fluxComponentOffset(rhsFunctions.size());
    std::vector<std::vector<PetscInt>> uOff(rhsFunctions.size());
    std::vector<std::vector<PetscInt>> aOff(rhsFunctions.size());

    // Get the full set of offsets from the ds
    PetscInt* uOffTotal;
    PetscDSGetComponentOffsets(ds, &uOffTotal) >> utilities::PetscUtilities::checkError;

    for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
        for (std::size_t f = 0; f < rhsFunctions[fun].fields.size(); f++) {
            const auto& field = subDomain->GetField(rhsFunctions[fun].fields[f]);

            PetscInt fieldSize, fieldOffset;
            PetscDSGetFieldSize(ds, field.subId, &fieldSize) >> utilities::PetscUtilities::checkError;
            PetscDSGetFieldOffset(ds, field.subId, &fieldOffset) >> utilities::PetscUtilities::checkError;
            fluxComponentSize[fun].push_back(fieldSize);
            fluxComponentOffset[fun].push_back(fieldOffset);
        }

        for (std::size_t f = 0; f < rhsFunctions[fun].inputFields.size(); f++) {
            uOff[fun].push_back(uOffTotal[rhsFunctions[fun].inputFields[f]]);
        }
    }

    if (dsAux) {
        PetscInt* auxOffTotal;
        PetscDSGetComponentOffsets(dsAux, &auxOffTotal) >> utilities::PetscUtilities::checkError;
        for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
            for (std::size_t f = 0; f < rhsFunctions[fun].auxFields.size(); f++) {
                aOff[fun].push_back(auxOffTotal[rhsFunctions[fun].auxFields[f]]);
            }
        }
    }

    // check to see if there is a ghost label
    DMLabel ghostLabel;
    DMGetLabel(dm, "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

    PetscInt dim = subDomain->GetDimensions();

    // Size up a scratch variable
    PetscScalar fScratch[totDim];

    // March over each cell
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cellRange.points ? cellRange.points[c] : c;

        // make sure that this is not a ghost cell
        if (ghostLabel) {
            PetscInt ghostVal;

            DMLabelGetValue(ghostLabel, cell, &ghostVal) >> utilities::PetscUtilities::checkError;
            if (ghostVal > 0) continue;
        }

        // extract the point locations for this cell
        const PetscFVCellGeom* cg;
        const PetscScalar* u;
        PetscScalar* rhs;
        DMPlexPointLocalRead(cellDM, cell, cellGeomArray, &cg) >> utilities::PetscUtilities::checkError;
        DMPlexPointLocalRead(dm, cell, xArray, &u) >> utilities::PetscUtilities::checkError;
        DMPlexPointLocalRef(dm, cell, locFArray, &rhs) >> utilities::PetscUtilities::checkError;

        // if there is an aux field, get it
        const PetscScalar* a = nullptr;
        if (auxArray) {
            DMPlexPointLocalRead(dmAux, cell, auxArray, &a) >> utilities::PetscUtilities::checkError;
        }

        // March over each functionDescriptions
        for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
            rhsFunctions[fun].function(dim, time, cg, uOff[fun].data(), u, aOff[fun].data(), a, fScratch, rhsFunctions[fun].context) >> utilities::PetscUtilities::checkError;

            // copy over each result flux field
            PetscInt r = 0;
            for (std::size_t ff = 0; ff < rhsFunctions[fun].fields.size(); ff++) {
                for (PetscInt d = 0; d < fluxComponentSize[fun][ff]; ++d) {
                    rhs[fluxComponentOffset[fun][ff] + d] += fScratch[r++];
                }
            }
        }
    }

    // cleanup (restore access to locGradVecs, locAuxGradVecs with DMRestoreLocalVector)
    VecRestoreArrayRead(locXVec, &xArray) >> utilities::PetscUtilities::checkError;
    if (locAuxVec) {
        VecRestoreArrayRead(locAuxVec, &auxArray) >> utilities::PetscUtilities::checkError;
    }

    VecRestoreArray(locFVec, &locFArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
}

/**
 * This is a duplication of PETSC that we don't have access to
 */
static PetscErrorCode DMPlexApplyLimiter_Internal(DM dm, DM dmCell, PetscLimiter lim, PetscInt dim, PetscInt dof, PetscInt cell, PetscInt field, PetscInt face, PetscInt fStart, PetscInt fEnd,
                                                  PetscReal* cellPhi, const PetscScalar* x, const PetscScalar* cellgeom, const PetscFVCellGeom* cg, const PetscScalar* cx, const PetscScalar* cgrad) {
    const PetscInt* children;
    PetscInt numChildren;

    PetscFunctionBegin;
    PetscCall(DMPlexGetTreeChildren(dm, face, &numChildren, &children));
    if (numChildren) {
        PetscInt c;

        for (c = 0; c < numChildren; c++) {
            PetscInt childFace = children[c];

            if (childFace >= fStart && childFace < fEnd) {
                PetscCall(DMPlexApplyLimiter_Internal(dm, dmCell, lim, dim, dof, cell, field, childFace, fStart, fEnd, cellPhi, x, cellgeom, cg, cx, cgrad));
            }
        }
    } else {
        PetscScalar* ncx;
        PetscFVCellGeom* ncg;
        const PetscInt* fcells;
        PetscInt ncell, d;
        PetscReal v[3];

        PetscCall(DMPlexGetSupport(dm, face, &fcells));
        ncell = cell == fcells[0] ? fcells[1] : fcells[0];
        if (field >= 0) {
            PetscCall(DMPlexPointLocalFieldRead(dm, ncell, field, x, &ncx));
        } else {
            PetscCall(DMPlexPointLocalRead(dm, ncell, x, &ncx));
        }
        PetscCall(DMPlexPointLocalRead(dmCell, ncell, cellgeom, &ncg));
        DMPlex_WaxpyD_Internal(dim, -1, cg->centroid, ncg->centroid, v);
        for (d = 0; d < dof; ++d) {
            /* We use the symmetric slope limited form of Berger, Aftosmis, and Murman 2005 */
            PetscReal denom = DMPlex_DotD_Internal(dim, &cgrad[d * dim], v);
            PetscReal phi, flim = 0.5 * PetscRealPart(ncx[d] - cx[d]) / denom;

            PetscCall(PetscLimiterLimit(lim, flim, &phi));
            cellPhi[d] = PetscMin(cellPhi[d], phi);
        }
    }
    PetscFunctionReturn(0);
}

void ablate::finiteVolume::CellInterpolant::ComputeFieldGradients(const domain::Field& field, Vec xLocalVec, Vec& gradLocVec, DM& dmGrad, Vec cellGeomVec, Vec faceGeomVec,
                                                                  const ablate::domain::Range& faceRange, const ablate::domain::Range& cellRange) {
    // get the FVM petsc field associated with this field
    auto fvm = (PetscFV)subDomain->GetPetscFieldObject(field);
    auto dm = subDomain->GetFieldDM(field);

    // Get the dm for this grad field
    // If there is no grad, return
    if (!dmGrad) {
        // PetscPrintf(PETSC_COMM_WORLD, "dmGrad is nullptr\n");
        return;
    }

    // Create a gradLocVec
    DMGetLocalVector(dmGrad, &gradLocVec) >> utilities::PetscUtilities::checkError;

    // Get the correct sized vec (gradient for this field)
    Vec gradGlobVec;
    DMGetGlobalVector(dmGrad, &gradGlobVec) >> utilities::PetscUtilities::checkError;
    VecZeroEntries(gradGlobVec) >> utilities::PetscUtilities::checkError; 

    // check to see if there is a ghost label
    DMLabel ghostLabel;
    DMGetLabel(dm, "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

    // Get the face geometry
    DM dmFace;
    const PetscScalar* faceGeometryArray;
    VecGetDM(faceGeomVec, &dmFace) >> utilities::PetscUtilities::checkError;
    VecGetArrayRead(faceGeomVec, &faceGeometryArray);

    // extract the local x array
    const PetscScalar* xLocalArray;
    VecGetArrayRead(xLocalVec, &xLocalArray);

    // extract the global grad array
    PetscScalar* gradGlobArray;
    VecGetArray(gradGlobVec, &gradGlobArray);

    // Get the dof and dim
    PetscInt dim = subDomain->GetDimensions();
    PetscInt dof = field.numberComponents;

    for (PetscInt f = faceRange.start; f < faceRange.end; ++f) {
        PetscInt face = faceRange.points ? faceRange.points[f] : f;

        // make sure that this is a face we should use
        PetscBool boundary;
        PetscInt ghost = -1;
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, face, &ghost);
        }
        DMIsBoundaryPoint(dm, face, &boundary);
        PetscInt numChildren, numCells;
        DMPlexGetSupportSize(dm, face, &numCells);
        DMPlexGetTreeChildren(dm, face, &numChildren, nullptr);
        if (ghost >= 0 || boundary || numChildren || numCells!=2) continue;

        // Do a sanity check on the number of cells connected to this face
        if (numCells != 2) {
//PetscFVFaceGeom* fg;
//DMPlexPointLocalRead(dmFace, face, faceGeometryArray, &fg);
//printf("%+f\t%+f\n", fg->centroid[0], fg->centroid[1]);
//printf("cellInterpolant::393\n");
//exit(0);

            throw std::runtime_error("face " + std::to_string(face) + " has " + std::to_string(numCells) + " support points (cells): expected 2");
        }

        // add in the contributions from this face
        const PetscInt* cells;
        PetscFVFaceGeom* fg;
        PetscScalar* cx[2];
        PetscScalar* cgrad[2];

        DMPlexGetSupport(dm, face, &cells);
        DMPlexPointLocalRead(dmFace, face, faceGeometryArray, &fg);
        for (PetscInt c = 0; c < 2; ++c) {
            DMPlexPointLocalFieldRead(dm, cells[c], field.id, xLocalArray, &cx[c]) >> utilities::PetscUtilities::checkError;
            DMPlexPointGlobalRef(dmGrad, cells[c], gradGlobArray, &cgrad[c]) >> utilities::PetscUtilities::checkError;
        }
        for (PetscInt pd = 0; pd < dof; ++pd) {
            PetscScalar delta = cx[1][pd] - cx[0][pd];

            for (PetscInt d = 0; d < dim; ++d) {
                if (cgrad[0]) cgrad[0][pd * dim + d] += fg->grad[0][d] * delta;
                if (cgrad[1]) cgrad[1][pd * dim + d] -= fg->grad[1][d] * delta;
            }
        }
    }

    // Check for a limiter the limiter
    PetscLimiter lim;
    PetscFVGetLimiter(fvm, &lim) >> utilities::PetscUtilities::checkError;
    PetscPrintf(PETSC_COMM_WORLD, "lim = %p\n", lim);
    if (lim) {
        /* Limit interior gradients (using cell-based loop because it generalizes better to vector limiters) */
        // Get the cell geometry
        DM dmCell;
        const PetscScalar* cellGeometryArray;
        VecGetDM(cellGeomVec, &dmCell) >> utilities::PetscUtilities::checkError;
        VecGetArrayRead(cellGeomVec, &cellGeometryArray);

        // create a temp work array
        PetscReal* cellPhi;
        DMGetWorkArray(dm, dof, MPIU_REAL, &cellPhi) >> utilities::PetscUtilities::checkError;

        for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
            PetscInt cell = cellRange.points ? cellRange.points[c] : c;

            const PetscInt* cellFaces;
            PetscScalar* cx;
            PetscFVCellGeom* cg;
            PetscScalar* cgrad;
            PetscInt coneSize;

            DMPlexGetConeSize(dm, cell, &coneSize) >> utilities::PetscUtilities::checkError;
            DMPlexGetCone(dm, cell, &cellFaces) >> utilities::PetscUtilities::checkError;
            DMPlexPointLocalFieldRead(dm, cell, field.id, xLocalArray, &cx) >> utilities::PetscUtilities::checkError;
            DMPlexPointLocalRead(dmCell, cell, cellGeometryArray, &cg) >> utilities::PetscUtilities::checkError;
            DMPlexPointGlobalRef(dmGrad, cell, gradGlobArray, &cgrad) >> utilities::PetscUtilities::checkError;

            if (!cgrad) {
                /* Unowned overlap cell, we do not compute */
                continue;
            }
            /* Limiter will be minimum value over all neighbors */
            for (PetscInt d = 0; d < dof; ++d) {
                cellPhi[d] = PETSC_MAX_REAL;
            }
            for (PetscInt f = 0; f < coneSize; ++f) {
                DMPlexApplyLimiter_Internal(dm, dmCell, lim, dim, dof, cell, field.id, cellFaces[f], faceRange.start, faceRange.end, cellPhi, xLocalArray, cellGeometryArray, cg, cx, cgrad) >>
                    utilities::PetscUtilities::checkError;
            }
            /* Apply limiter to gradient */
            for (PetscInt pd = 0; pd < dof; ++pd) {
                /* Scalar limiter applied to each component separately */
                for (PetscInt d = 0; d < dim; ++d) {
                    cgrad[pd * dim + d] *= cellPhi[pd];
                }
            }
        }

        // clean up the limiter work
        DMRestoreWorkArray(dm, dof, MPIU_REAL, &cellPhi) >> utilities::PetscUtilities::checkError;
        VecRestoreArrayRead(cellGeomVec, &cellGeometryArray);
    }
    // Communicate gradient values
    VecRestoreArray(gradGlobVec, &gradGlobArray) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocalBegin(dmGrad, gradGlobVec, INSERT_VALUES, gradLocVec) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocalEnd(dmGrad, gradGlobVec, INSERT_VALUES, gradLocVec) >> utilities::PetscUtilities::checkError;

    // cleanup
    VecRestoreArrayRead(xLocalVec, &xLocalArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(faceGeomVec, &faceGeometryArray) >> utilities::PetscUtilities::checkError;
    DMRestoreGlobalVector(dmGrad, &gradGlobVec) >> utilities::PetscUtilities::checkError;
}


static PetscInt cnt = 0;

void ablate::finiteVolume::CellInterpolant::ComputeFluxSourceTerms(DM dm, PetscDS ds, PetscInt totDim, const PetscScalar* xArray, DM dmAux, PetscDS dsAux, PetscInt totDimAux,
                                                                   const PetscScalar* auxArray, DM faceDM, const PetscScalar* faceGeomArray, DM cellDM, const PetscScalar* cellGeomArray,
                                                                   std::vector<DM>& dmGrads, std::vector<const PetscScalar*>& locGradArrays, PetscScalar* locFArray,
                                                                   const std::shared_ptr<domain::Region>& solverRegion,
                                                                   std::vector<CellInterpolant::DiscontinuousFluxFunctionDescription>& rhsFunctions, const ablate::domain::Range& faceRange,
                                                                   const ablate::domain::Range& cellRange) {
    PetscInt dim = subDomain->GetDimensions();

    // Size up the work arrays (uL, uR, gradL, gradR, auxL, auxR, gradAuxL, gradAuxR), these are only sized for one face at a time
    PetscScalar* flux;
    DMGetWorkArray(dm, totDim, MPIU_SCALAR, &flux) >> utilities::PetscUtilities::checkError;

    PetscScalar *uL, *uR;
    DMGetWorkArray(dm, totDim, MPIU_SCALAR, &uL) >> utilities::PetscUtilities::checkError;
    DMGetWorkArray(dm, totDim, MPIU_SCALAR, &uR) >> utilities::PetscUtilities::checkError;

    PetscScalar *gradL, *gradR;
    DMGetWorkArray(dm, dim * totDim, MPIU_SCALAR, &gradL) >> utilities::PetscUtilities::checkError;
    DMGetWorkArray(dm, dim * totDim, MPIU_SCALAR, &gradR) >> utilities::PetscUtilities::checkError;

    // size up the aux variables
    PetscScalar *auxL = nullptr, *auxR = nullptr;

    // Precompute the offsets to pass into the rhsFluxFunctionDescriptions
    std::vector<PetscInt> fluxComponentSize(rhsFunctions.size());
    std::vector<PetscInt> fluxId(rhsFunctions.size());
    std::vector<std::vector<PetscInt>> uOff(rhsFunctions.size());
    std::vector<std::vector<PetscInt>> aOff(rhsFunctions.size());

    // Get the full set of offsets from the ds
    PetscInt* uOffTotal;
    PetscDSGetComponentOffsets(ds, &uOffTotal) >> utilities::PetscUtilities::checkError;

    for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
        const auto& field = subDomain->GetField(rhsFunctions[fun].field);
        fluxComponentSize[fun] = field.numberComponents;
        fluxId[fun] = field.id;

        for (std::size_t f = 0; f < rhsFunctions[fun].inputFields.size(); f++) {
            uOff[fun].push_back(uOffTotal[rhsFunctions[fun].inputFields[f]]);
        }
    }

    if (dsAux) {
        PetscInt* auxOffTotal;
        PetscDSGetComponentOffsets(dsAux, &auxOffTotal) >> utilities::PetscUtilities::checkError;
        for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
            for (std::size_t f = 0; f < rhsFunctions[fun].auxFields.size(); f++) {
                aOff[fun].push_back(auxOffTotal[rhsFunctions[fun].auxFields[f]]);
            }
        }
    }
    // check for ghost cells
    DMLabel ghostLabel;
    DMGetLabel(dm, "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

    // get the label for this region
    DMLabel regionLabel = nullptr;
    PetscInt regionValue = 0;
    domain::Region::GetLabel(solverRegion, subDomain->GetDM(), regionLabel, regionValue);

//PetscMPIInt  rank;
//MPI_Comm_rank(PETSC_COMM_WORLD, &rank) >> ablate::utilities::PetscUtilities::checkError;

++cnt;

//static PetscReal flux3104 = 0;

    // March over each face in this region
    for (PetscInt f = faceRange.start; f < faceRange.end; ++f) {
        const PetscInt face = faceRange.points ? faceRange.points[f] : f;

        // make sure that this is a valid face
        PetscInt ghost, nsupp, nchild;
        DMLabelGetValue(ghostLabel, face, &ghost) >> utilities::PetscUtilities::checkError;
        DMPlexGetSupportSize(dm, face, &nsupp) >> utilities::PetscUtilities::checkError;
        DMPlexGetTreeChildren(dm, face, &nchild, nullptr) >> utilities::PetscUtilities::checkError;
        if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;

        // Get the face geometry
        const PetscInt* faceCells;
        PetscFVFaceGeom* fg;
        PetscFVCellGeom *cgL, *cgR;
        DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg) >> utilities::PetscUtilities::checkError;
        DMPlexGetSupport(dm, face, &faceCells) >> utilities::PetscUtilities::checkError;
        DMPlexPointLocalRead(cellDM, faceCells[0], cellGeomArray, &cgL) >> utilities::PetscUtilities::checkError;
        DMPlexPointLocalRead(cellDM, faceCells[1], cellGeomArray, &cgR) >> utilities::PetscUtilities::checkError;

        PetscInt leftFlowLabelValue = regionValue;
        PetscInt rightFlowLabelValue = regionValue;
        if (regionLabel) {
            DMLabelGetValue(regionLabel, faceCells[0], &leftFlowLabelValue);
            DMLabelGetValue(regionLabel, faceCells[1], &rightFlowLabelValue);
        }
        // compute the left/right face values
        ProjectToFace(subDomain->GetFields(), ds, *fg, faceCells[0], *cgL, dm, xArray, dmGrads, locGradArrays, uL, gradL, leftFlowLabelValue == regionValue); // && false);
        ProjectToFace(subDomain->GetFields(), ds, *fg, faceCells[1], *cgR, dm, xArray, dmGrads, locGradArrays, uR, gradR, rightFlowLabelValue == regionValue); // && false);

        // determine the left/right cells
        if (auxArray) {
            // Get the field values at this cell
            DMPlexPointLocalRead(dmAux, faceCells[0], auxArray, &auxL) >> utilities::PetscUtilities::checkError;
            DMPlexPointLocalRead(dmAux, faceCells[1], auxArray, &auxR) >> utilities::PetscUtilities::checkError;
        }

        // March over each source function
        for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
            PetscArrayzero(flux, totDim) >> utilities::PetscUtilities::checkError;
            const auto& rhsFluxFunctionDescription = rhsFunctions[fun];
            rhsFluxFunctionDescription.function(dim, fg, uOff[fun].data(), uL, uR, aOff[fun].data(), auxL, auxR, flux, rhsFluxFunctionDescription.context) >> utilities::PetscUtilities::checkError;

            // add the flux back to the cell
            PetscScalar *fL = nullptr, *fR = nullptr;
            PetscInt cellLabelValue = regionValue;
            DMLabelGetValue(ghostLabel, faceCells[0], &ghost) >> utilities::PetscUtilities::checkError;
            if (regionLabel) {
                DMLabelGetValue(regionLabel, faceCells[0], &cellLabelValue) >> utilities::PetscUtilities::checkError;
            }
            if (ghost <= 0 && regionValue == cellLabelValue) {
                DMPlexPointLocalFieldRef(dm, faceCells[0], fluxId[fun], locFArray, &fL) >> utilities::PetscUtilities::checkError;
            }

            cellLabelValue = regionValue;
            DMLabelGetValue(ghostLabel, faceCells[1], &ghost) >> utilities::PetscUtilities::checkError;
            if (regionLabel) {
                DMLabelGetValue(regionLabel, faceCells[1], &cellLabelValue) >> utilities::PetscUtilities::checkError;
            }
            if (ghost <= 0 && regionValue == cellLabelValue) {
                DMPlexPointLocalFieldRef(dm, faceCells[1], fluxId[fun], locFArray, &fR) >> utilities::PetscUtilities::checkError;
            }

            for (PetscInt d = 0; d < fluxComponentSize[fun]; ++d) {
                if (fL) fL[d] -= flux[d] / cgL->volume;
                if (fR) fR[d] += flux[d] / cgR->volume;
            }
        }
    }

    // cleanup
    DMRestoreWorkArray(dm, totDim, MPIU_SCALAR, &flux) >> utilities::PetscUtilities::checkError;
    DMRestoreWorkArray(dm, totDim, MPIU_SCALAR, &uL) >> utilities::PetscUtilities::checkError;
    DMRestoreWorkArray(dm, totDim, MPIU_SCALAR, &uR) >> utilities::PetscUtilities::checkError;
    DMRestoreWorkArray(dm, dim * totDim, MPIU_SCALAR, &gradL) >> utilities::PetscUtilities::checkError;
    DMRestoreWorkArray(dm, dim * totDim, MPIU_SCALAR, &gradR) >> utilities::PetscUtilities::checkError;
}

void ablate::finiteVolume::CellInterpolant::ProjectToFace(const std::vector<domain::Field>& fields, PetscDS ds, const PetscFVFaceGeom& faceGeom, PetscInt cellId, const PetscFVCellGeom& cellGeom,
                                                          DM dm, const PetscScalar* xArray, const std::vector<DM>& dmGrads, const std::vector<const PetscScalar*>& gradArrays, PetscScalar* u,
                                                          PetscScalar* grad, bool projectField) {
    const auto dim = subDomain->GetDimensions();

    // Keep track of derivative offset
    PetscInt* offsets;
    PetscInt* dirOffsets;
    PetscDSGetComponentOffsets(ds, &offsets) >> utilities::PetscUtilities::checkError;
    PetscDSGetComponentDerivativeOffsets(ds, &dirOffsets) >> utilities::PetscUtilities::checkError;

    // March over each field
    for (const auto& field : fields) {
        PetscReal dx[3];
        PetscScalar* xCell;
        PetscScalar* gradCell;

        // Get the field values at this cell
        DMPlexPointLocalFieldRead(dm, cellId, field.subId, xArray, &xCell) >> utilities::PetscUtilities::checkError;

        // If we need to project the field
        if (projectField && dmGrads[field.subId]) {
            DMPlexPointLocalRead(dmGrads[field.subId], cellId, gradArrays[field.subId], &gradCell) >> utilities::PetscUtilities::checkError;
            DMPlex_WaxpyD_Internal(dim, -1, cellGeom.centroid, faceGeom.centroid, dx);

            // Project the cell centered value onto the face
            for (PetscInt c = 0; c < field.numberComponents; ++c) {
                u[offsets[field.subId] + c] = xCell[c] + DMPlex_DotD_Internal(dim, &gradCell[c * dim], dx);

                // copy the gradient into the grad vector
                for (PetscInt d = 0; d < dim; d++) {
                    grad[dirOffsets[field.subId] + c * dim + d] = gradCell[c * dim + d];
                }
            }

        } else if (dmGrads[field.subId]) {
            // PetscPrintf(PETSC_COMM_WORLD, "Projecting to face %d\n", field.subId);
            // Project the cell centered value onto the face
            DMPlexPointLocalRead(dmGrads[field.subId], cellId, gradArrays[field.subId], &gradCell) >> utilities::PetscUtilities::checkError;
            // Project the cell centered value onto the face
            for (PetscInt c = 0; c < field.numberComponents; ++c) {
                u[offsets[field.subId] + c] = xCell[c];

                // copy the gradient into the grad vector
                for (PetscInt d = 0; d < dim; d++) {
                    grad[dirOffsets[field.subId] + c * dim + d] = gradCell[c * dim + d];
                }
            }

        } else {
            // PetscPrintf(PETSC_COMM_WORLD, "Not projecting to face %d\n", field.subId);
            // Just copy the cell centered value on to the face
            for (PetscInt c = 0; c < field.numberComponents; ++c) {
                u[offsets[field.subId] + c] = xCell[c];

                // fill the grad with NAN to prevent use
                for (PetscInt d = 0; d < dim; d++) {
                    grad[dirOffsets[field.subId] + c * dim + d] = NAN;
                }
            }
        }
    }
}
