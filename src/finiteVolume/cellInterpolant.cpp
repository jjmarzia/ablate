#include "cellInterpolant.hpp"
#include <petsc/private/dmpleximpl.h>
#include <utility>
#include "finiteVolume/nPhaseFlowFields.hpp"

ablate::finiteVolume::CellInterpolant::CellInterpolant(std::shared_ptr<ablate::domain::SubDomain> subDomainIn, const std::shared_ptr<domain::Region>& solverRegion, Vec faceGeomVec, Vec cellGeomVec,
                                                       double maxGradIn)
    : subDomain(std::move(std::move(subDomainIn))), maxLimGrad(maxGradIn) {
    // Initialize slope limiter
    slopeLimiter = std::make_unique<SlopeLimiter>();
    
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
        //print field and dmgrad corresponding to field
        // PetscPrintf(PETSC_COMM_WORLD, "Field: %s, DMGrad: %p\n", field.name.c_str(), gradientCellDms[field.subId]);
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
    if (!locAuxVec) {
        PetscPrintf(PETSC_COMM_WORLD, "maxLimGrad = %g\n", maxLimGrad);
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
// static PetscErrorCode DMPlexApplyLimiter_Internal(DM dm, DM dmCell, PetscLimiter lim, PetscInt dim, PetscInt dof, PetscInt cell, PetscInt field, PetscInt face, PetscInt fStart, PetscInt fEnd,
//                                                   PetscReal* cellPhi, const PetscScalar* x, const PetscScalar* cellgeom, const PetscFVCellGeom* cg, const PetscScalar* cx, const PetscScalar* cgrad) {
//     const PetscInt* children;
//     PetscInt numChildren;

//     PetscFunctionBegin;
//     PetscCall(DMPlexGetTreeChildren(dm, face, &numChildren, &children));
//     if (numChildren) {  // if the tree contains children
//         PetscInt c;

//         for (c = 0; c < numChildren; c++) {
//             PetscInt childFace = children[c];

//             if (childFace >= fStart && childFace < fEnd) {
//                 PetscCall(DMPlexApplyLimiter_Internal(dm, dmCell, lim, dim, dof, cell, field, childFace, fStart, fEnd, cellPhi, x, cellgeom, cg, cx, cgrad));
//             }
//         }
//     } else {                     // if the tree doesn't contain children
//         PetscScalar* ncx;        // neighbor cell centered values
//         PetscFVCellGeom* ncg;    // neighbor cell geometry
//         const PetscInt* fcells;  // cells attached to this face
//         PetscInt ncell, d;       // neighbor cell and for loop index
//         PetscReal v[3];          // centr

//         PetscCall(DMPlexGetSupport(dm, face, &fcells));
//         ncell = cell == fcells[0] ? fcells[1] : fcells[0];  // figure out which cell is the neighbor and not this cell
//         // Read in the neighbor cell information
//         if (field >= 0) {
//             PetscCall(DMPlexPointLocalFieldRead(dm, ncell, field, x, &ncx));
//         } else {
//             PetscCall(DMPlexPointLocalRead(dm, ncell, x, &ncx));
//         }
//         PetscCall(DMPlexPointLocalRead(dmCell, ncell, cellgeom, &ncg));
//         // Calculate the distance between the neighbor cell and this cell
//         DMPlex_WaxpyD_Internal(dim, -1, cg->centroid, ncg->centroid, v);  // v_i = NeighborCentroid_i - ThisCentroid_i = dx_i
//         for (d = 0; d < dof; ++d) {
//             /* We use the symmetric slope limited form of Berger, Aftosmis, and Murman 2005 */
//             PetscReal denom = DMPlex_DotD_Internal(dim, &cgrad[d * dim], v);    // denominator = \grad u \cdot dx
//             PetscReal phi, flim = 0.5 * PetscRealPart(ncx[d] - cx[d]) / denom;  // f = 1/2 * (u_i+1-u_i)/(\Delta u from cell gradient dot with \delta x)
//             // What the above means is that if any cell face has no change, but there was ample enough change close to it such that
//             //  the cell gradient dot with delta x is not 0, there is no limiting... i.e f = 0 and all limiters = 0
//             PetscCall(PetscLimiterLimit(lim, flim, &phi));
//             cellPhi[d] = PetscMin(cellPhi[d], phi);
//         }
//     }
//     PetscFunctionReturn(0);
// }

void ablate::finiteVolume::CellInterpolant::ComputeFieldGradients(const domain::Field& field, Vec xLocalVec, Vec& gradLocVec, DM& dmGrad, Vec cellGeomVec, Vec faceGeomVec,
                                                                  const ablate::domain::Range& faceRange, const ablate::domain::Range& cellRange) {
    // get the FVM petsc field associated with this field
    // auto fvm = (PetscFV)subDomain->GetPetscFieldObject(field);
    auto dm = subDomain->GetFieldDM(field);

    // Get the dm for this grad field
    // If there is no grad, return
    if (!dmGrad) {
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

    // Setup slope limiter if not already done and gradients are being computed
    if (dmGrad && !slopeLimiter->IsSetup()) {
        slopeLimiter->Setup(dm, cellRange);
    }

    // Compute initial gradients using least squares
    for (PetscInt f = faceRange.start; f < faceRange.end; ++f) {
        PetscInt face = faceRange.points ? faceRange.points[f] : f;

        // make sure that this is a face we should use
        PetscBool boundary;
        PetscInt ghost = -1;
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, face, &ghost);
        }
        DMIsBoundaryPoint(dm, face, &boundary);
        PetscInt numChildren;
        DMPlexGetTreeChildren(dm, face, &numChildren, nullptr);
        if (ghost >= 0 || boundary || numChildren) continue;

        // Do a sanity check on the number of cells connected to this face
        PetscInt numCells;
        DMPlexGetSupportSize(dm, face, &numCells);
        if (numCells != 2) {
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

    // Apply slope limiting to the gradients only if gradients are being computed
    if (dmGrad) {
        slopeLimiter->ApplyLimiter(dm, dim, field, cellRange, xLocalArray, gradGlobArray);
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
    std::vector<std::vector<PetscInt>> fluxComponentSize(rhsFunctions.size());
    std::vector<std::vector<PetscInt>> fluxId(rhsFunctions.size());
    std::vector<std::vector<PetscInt>> uOff(rhsFunctions.size());
    std::vector<std::vector<PetscInt>> aOff(rhsFunctions.size());

    // Get the full set of offsets from the ds
    PetscInt* uOffTotal;
    PetscDSGetComponentOffsets(ds, &uOffTotal) >> utilities::PetscUtilities::checkError;

    for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
        for (std::size_t f = 0; f < rhsFunctions[fun].updateFields.size(); f++) {
            const auto& field = subDomain->GetField(rhsFunctions[fun].updateFields[f]);
            fluxComponentSize[fun].push_back(field.numberComponents);
            fluxId[fun].push_back(field.id);
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
    // check for ghost cells
    DMLabel ghostLabel;
    DMGetLabel(dm, "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

    // get the label for this region
    DMLabel regionLabel = nullptr;
    PetscInt regionValue = 0;
    domain::Region::GetLabel(solverRegion, subDomain->GetDM(), regionLabel, regionValue);
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
        ProjectToFace(subDomain->GetFields(), ds, *fg, faceCells[0], *cgL, dm, xArray, dmGrads, locGradArrays, uL, gradL, leftFlowLabelValue == regionValue);
        ProjectToFace(subDomain->GetFields(), ds, *fg, faceCells[1], *cgR, dm, xArray, dmGrads, locGradArrays, uR, gradR, rightFlowLabelValue == regionValue);

        // Post-process alphakrhok values to use face-averaged rhok
        // Find alphak and alphakrhok field indices
        PetscInt alphakFieldId = -1;
        PetscInt alphakrhokFieldId = -1;
        const auto& fields = subDomain->GetFields();
        for (PetscInt i = 0; i < (PetscInt)fields.size(); ++i) {
            if (fields[i].name == "alphak") {
                alphakFieldId = i;
            } else if (fields[i].name == "alphakrhok") {
                alphakrhokFieldId = i;
            }
        }
        
        // Debug prints for field identification
        if (faceCells[0] >= 48 && faceCells[0] <= 52) {
            //PetscPrintf(PETSC_COMM_WORLD, "=== ALPHAKRHOK POST-PROCESSING DEBUG ===\n");
            //PetscPrintf(PETSC_COMM_WORLD, "Face %d between cells %d and %d\n", face, faceCells[0], faceCells[1]);
            //PetscPrintf(PETSC_COMM_WORLD, "alphakFieldId: %d, alphakrhokFieldId: %d\n", alphakFieldId, alphakrhokFieldId);
            //PetscPrintf(PETSC_COMM_WORLD, "Total fields: %zu\n", fields.size());
        }
        
        // If both fields are found, compute face-averaged rhok for alphakrhok
        if (alphakFieldId >= 0 && alphakrhokFieldId >= 0) {
            if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                PetscPrintf(PETSC_COMM_WORLD, "Both fields found, proceeding with post-processing\n");
            }
            
            // Get cell-centered alphak values for both cells
            PetscScalar* alphakL, *alphakR;
            DMPlexPointLocalFieldRead(dm, faceCells[0], fields[alphakFieldId].id, xArray, &alphakL) >> utilities::PetscUtilities::checkError;
            DMPlexPointLocalFieldRead(dm, faceCells[1], fields[alphakFieldId].id, xArray, &alphakR) >> utilities::PetscUtilities::checkError;
            
            if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                PetscPrintf(PETSC_COMM_WORLD, "alphakL pointer: %p, alphakR pointer: %p\n", (void*)alphakL, (void*)alphakR);
                if (alphakL) {
                    PetscPrintf(PETSC_COMM_WORLD, "alphakL values: ");
                    for (PetscInt c = 0; c < fields[alphakFieldId].numberComponents; ++c) {
                        PetscPrintf(PETSC_COMM_WORLD, "%g ", alphakL[c]);
                    }
                    PetscPrintf(PETSC_COMM_WORLD, "\n");
                }
                if (alphakR) {
                    PetscPrintf(PETSC_COMM_WORLD, "alphakR values: ");
                    for (PetscInt c = 0; c < fields[alphakFieldId].numberComponents; ++c) {
                        PetscPrintf(PETSC_COMM_WORLD, "%g ", alphakR[c]);
                    }
                    PetscPrintf(PETSC_COMM_WORLD, "\n");
                }
            }
            
            // Get the reconstructed alphak values at the face
            PetscScalar* alphakFaceL = &uL[uOffTotal[fields[alphakFieldId].subId]];
            PetscScalar* alphakFaceR = &uR[uOffTotal[fields[alphakFieldId].subId]];
            
            if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                PetscPrintf(PETSC_COMM_WORLD, "alphakFaceL pointer: %p, alphakFaceR pointer: %p\n", (void*)alphakFaceL, (void*)alphakFaceR);
                PetscPrintf(PETSC_COMM_WORLD, "uOffTotal[%d]: %d\n", fields[alphakFieldId].subId, uOffTotal[fields[alphakFieldId].subId]);
                
                if (alphakFaceL) {
                    PetscPrintf(PETSC_COMM_WORLD, "alphakFaceL values: ");
                    for (PetscInt c = 0; c < fields[alphakFieldId].numberComponents; ++c) {
                        PetscPrintf(PETSC_COMM_WORLD, "%g ", alphakFaceL[c]);
                    }
                    PetscPrintf(PETSC_COMM_WORLD, "\n");
                }
                if (alphakFaceR) {
                    PetscPrintf(PETSC_COMM_WORLD, "alphakFaceR values: ");
                    for (PetscInt c = 0; c < fields[alphakFieldId].numberComponents; ++c) {
                        PetscPrintf(PETSC_COMM_WORLD, "%g ", alphakFaceR[c]);
                    }
                    PetscPrintf(PETSC_COMM_WORLD, "\n");
                }
            }
            
            // Get the alphakrhok arrays to modify
            PetscScalar* alphakrhokFaceL = &uL[uOffTotal[fields[alphakrhokFieldId].subId]];
            PetscScalar* alphakrhokFaceR = &uR[uOffTotal[fields[alphakrhokFieldId].subId]];
            
            if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                PetscPrintf(PETSC_COMM_WORLD, "alphakrhokFaceL pointer: %p, alphakrhokFaceR pointer: %p\n", (void*)alphakrhokFaceL, (void*)alphakrhokFaceR);
                PetscPrintf(PETSC_COMM_WORLD, "uOffTotal[%d]: %d\n", fields[alphakrhokFieldId].subId, uOffTotal[fields[alphakrhokFieldId].subId]);
                
                if (alphakrhokFaceL) {
                    PetscPrintf(PETSC_COMM_WORLD, "alphakrhokFaceL values (before): ");
                    for (PetscInt c = 0; c < fields[alphakrhokFieldId].numberComponents; ++c) {
                        PetscPrintf(PETSC_COMM_WORLD, "%g ", alphakrhokFaceL[c]);
                    }
                    PetscPrintf(PETSC_COMM_WORLD, "\n");
                }
                if (alphakrhokFaceR) {
                    PetscPrintf(PETSC_COMM_WORLD, "alphakrhokFaceR values (before): ");
                    for (PetscInt c = 0; c < fields[alphakrhokFieldId].numberComponents; ++c) {
                        PetscPrintf(PETSC_COMM_WORLD, "%g ", alphakrhokFaceR[c]);
                    }
                    PetscPrintf(PETSC_COMM_WORLD, "\n");
                }
            }
            
            // Get aux fields for both cells
            PetscScalar* auxL, *auxR;
            DMPlexPointLocalRead(dmAux, faceCells[0], auxArray, &auxL) >> utilities::PetscUtilities::checkError;
            DMPlexPointLocalRead(dmAux, faceCells[1], auxArray, &auxR) >> utilities::PetscUtilities::checkError;
            
            if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                PetscPrintf(PETSC_COMM_WORLD, "auxL pointer: %p, auxR pointer: %p\n", (void*)auxL, (void*)auxR);
            }
            
            // Find RHOK field index in aux fields
            PetscInt rhokFieldId = -1;
            PetscInt rhokOffset = -1;
            if (subDomain->ContainsField(NPhaseFlowFields::RHOK) && 
                subDomain->GetField(NPhaseFlowFields::RHOK).location == ablate::domain::FieldLocation::AUX) {
                rhokFieldId = subDomain->GetField(NPhaseFlowFields::RHOK).id;
                rhokOffset = subDomain->GetField(NPhaseFlowFields::RHOK).offset;
                if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                    PetscPrintf(PETSC_COMM_WORLD, "Found rhok field with id %d, offset %d\n", rhokFieldId, rhokOffset);
                }
            } else {
                if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                    PetscPrintf(PETSC_COMM_WORLD, "rhok field not found in aux fields\n");
                }
            }
            
            // Debug: Check pressure field ID for comparison
            PetscInt pressureFieldId = -1;
            PetscInt pressureOffset = -1;
            if (subDomain->ContainsField("p")) {
                pressureFieldId = subDomain->GetField("p").id;
                pressureOffset = subDomain->GetField("p").offset;
                if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                    PetscPrintf(PETSC_COMM_WORLD, "Pressure field 'p' has id %d, offset %d, location: %d\n", 
                               pressureFieldId, pressureOffset, (int)subDomain->GetField("p").location);
                }
            } else {
                if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                    PetscPrintf(PETSC_COMM_WORLD, "Pressure field 'p' not found\n");
                }
            }
            
            // Get the proper auxiliary field offset for rhok
            PetscInt rhokAuxOffset = -1;
            if (dmAux && rhokFieldId >= 0) {
                PetscDS dsAux;
                DMGetDS(dmAux, &dsAux) >> utilities::PetscUtilities::checkError;
                PetscInt* auxOffTotal;
                PetscDSGetComponentOffsets(dsAux, &auxOffTotal) >> utilities::PetscUtilities::checkError;
                rhokAuxOffset = auxOffTotal[rhokFieldId];
                if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                    PetscPrintf(PETSC_COMM_WORLD, "rhokAuxOffset = auxOffTotal[%d] = %d\n", rhokFieldId, rhokAuxOffset);
                }
            }
            
            // Compute face-averaged rhok and update alphakrhok face values
            for (PetscInt c = 0; c < fields[alphakrhokFieldId].numberComponents; ++c) {
                if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                    PetscPrintf(PETSC_COMM_WORLD, "Processing component %d:\n", c);
                }
                
                // Get rhok values from aux fields for both cells
                PetscReal rhokL = 0.0, rhokR = 0.0;
                
                if (rhokFieldId >= 0 && auxL && auxR) {
                    rhokL = auxL[rhokAuxOffset + c];  // rhok is per-phase
                    rhokR = auxR[rhokAuxOffset + c];
                    if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                        PetscPrintf(PETSC_COMM_WORLD, "  rhokL = auxL[%d] = %g\n", rhokAuxOffset + c, rhokL);
                        PetscPrintf(PETSC_COMM_WORLD, "  rhokR = auxR[%d] = %g\n", rhokAuxOffset + c, rhokR);
                    }
                } else {
                    if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                        PetscPrintf(PETSC_COMM_WORLD, "  rhokFieldId < 0 or aux pointers NULL, using default values\n");
                    }
                }
                
                // Compute face-averaged rhok
                PetscReal rhokFace = 0.5 * (rhokL + rhokR);
                if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                    PetscPrintf(PETSC_COMM_WORLD, "  rhokFace = 0.5 * (%g + %g) = %g\n", rhokL, rhokR, rhokFace);
                }
                
                // Update alphakrhok face values using reconstructed alphak * face-averaged rhok
                if (alphakrhokFaceL && alphakrhokFaceR && alphakFaceL && alphakFaceR) {
                    PetscReal oldL = alphakrhokFaceL[c];
                    PetscReal oldR = alphakrhokFaceR[c];
                    alphakrhokFaceL[c] = alphakFaceL[c] * rhokFace;
                    alphakrhokFaceR[c] = alphakFaceR[c] * rhokFace;
                    // alphakrhokFaceL[c] += 0 * alphakFaceL[c] * rhokFace;
                    // alphakrhokFaceR[c] += 0 * alphakFaceR[c] * rhokFace;
                    if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                        PetscPrintf(PETSC_COMM_WORLD, "  alphakFaceL[%d] = %g, alphakFaceR[%d] = %g\n", c, alphakFaceL[c], c, alphakFaceR[c]);
                        PetscPrintf(PETSC_COMM_WORLD, "  alphakrhokFaceL[%d]: %g -> %g\n", c, oldL, alphakrhokFaceL[c]);
                        PetscPrintf(PETSC_COMM_WORLD, "  alphakrhokFaceR[%d]: %g -> %g\n", c, oldR, alphakrhokFaceR[c]);
                    }
                } else {
                    if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                        PetscPrintf(PETSC_COMM_WORLD, "  WARNING: One or more face value pointers are NULL!\n");
                    }
                }
            }
            
            if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                PetscPrintf(PETSC_COMM_WORLD, "=== END ALPHAKRHOK POST-PROCESSING DEBUG ===\n");
            }
        } else {
            if (faceCells[0] >= 48 && faceCells[0] <= 52) {
                //PetscPrintf(PETSC_COMM_WORLD, "One or both fields not found, skipping post-processing\n");
                //PetscPrintf(PETSC_COMM_WORLD, "=== END ALPHAKRHOK POST-PROCESSING DEBUG ===\n");
            }
        }

        // determine the left/right cells
        if (auxArray) {
            // Get the field values at this cell
            DMPlexPointLocalRead(dmAux, faceCells[0], auxArray, &auxL) >> utilities::PetscUtilities::checkError;
            DMPlexPointLocalRead(dmAux, faceCells[1], auxArray, &auxR) >> utilities::PetscUtilities::checkError;
        }

        // March over each source function
        for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
            PetscInt fluxOffset = 0;  // Flux offset for the function ( Currently calculated by just adding the number of components of the previous fields)
            PetscArrayzero(flux, totDim) >> utilities::PetscUtilities::checkError;
            const auto& rhsFluxFunctionDescription = rhsFunctions[fun];
            rhsFluxFunctionDescription.function(dim, fg, uOff[fun].data(), uL, uR, aOff[fun].data(), auxL, auxR, flux, rhsFluxFunctionDescription.context) >> utilities::PetscUtilities::checkError;
            // add the fluxes back to the cell
            for (std::size_t updateFieldIdx = 0; updateFieldIdx < rhsFunctions[fun].updateFields.size(); updateFieldIdx++) {
                PetscInt cellLabelValue = regionValue;
                PetscScalar *fL = nullptr, *fR = nullptr;
                DMLabelGetValue(ghostLabel, faceCells[0], &ghost) >> utilities::PetscUtilities::checkError;
                if (regionLabel) {
                    DMLabelGetValue(regionLabel, faceCells[0], &cellLabelValue) >> utilities::PetscUtilities::checkError;
                }
                if (ghost <= 0 && regionValue == cellLabelValue) {
                    DMPlexPointLocalFieldRef(dm, faceCells[0], fluxId[fun][updateFieldIdx], locFArray, &fL) >> utilities::PetscUtilities::checkError;
                }

                cellLabelValue = regionValue;
                DMLabelGetValue(ghostLabel, faceCells[1], &ghost) >> utilities::PetscUtilities::checkError;
                if (regionLabel) {
                    DMLabelGetValue(regionLabel, faceCells[1], &cellLabelValue) >> utilities::PetscUtilities::checkError;
                }
                if (ghost <= 0 && regionValue == cellLabelValue) {
                    DMPlexPointLocalFieldRef(dm, faceCells[1], fluxId[fun][updateFieldIdx], locFArray, &fR) >> utilities::PetscUtilities::checkError;
                }

                for (PetscInt d = 0; d < (fluxComponentSize[fun][updateFieldIdx]); ++d) {
                    if (fL) fL[d] -= flux[fluxOffset + d] / cgL->volume;
                    if (fR) fR[d] += flux[fluxOffset + d] / cgR->volume;
                }
                fluxOffset += fluxComponentSize[fun][updateFieldIdx];
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

static PetscErrorCode BuildGradientReconstruction_Internal(DM dm, DMLabel regionLabel, PetscInt regionValue, PetscFV fvm, DM dmFace, PetscScalar* fgeom, DM dmCell, PetscScalar* cgeom) {
    DMLabel ghostLabel;
    PetscScalar *dx, *grad, **gref;
    PetscInt dim, cStart, cEnd, c, cEndInterior, maxNumFaces;

    PetscFunctionBegin;
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(DMPlexGetCellTypeStratum(dm, DM_POLYTOPE_FV_GHOST, &cEndInterior, nullptr));
    cEndInterior = cEndInterior < 0 ? cEnd : cEndInterior;
    PetscCall(DMPlexGetMaxSizes(dm, &maxNumFaces, nullptr));
    PetscCall(PetscFVLeastSquaresSetMaxFaces(fvm, maxNumFaces));
    PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
    PetscCall(PetscMalloc3(maxNumFaces * dim, &dx, maxNumFaces * dim, &grad, maxNumFaces, &gref));
    for (c = cStart; c < cEndInterior; c++) {
        const PetscInt* faces;
        PetscInt numFaces, usedFaces, f, d;
        PetscFVCellGeom* cg;
        PetscBool boundary;
        PetscInt ghost;
        PetscInt labelValue;

        // do not attempt to compute a gradient reconstruction stencil in a ghost cell.  It will never be used
        PetscCall(DMLabelGetValue(ghostLabel, c, &ghost));
        if (ghost >= 0) continue;

        if (regionLabel) {
            PetscCall(DMLabelGetValue(regionLabel, c, &labelValue));
            if (labelValue != regionValue) continue;
        }

        PetscCall(DMPlexPointLocalRead(dmCell, c, cgeom, &cg));
        PetscCall(DMPlexGetConeSize(dm, c, &numFaces));
        PetscCall(DMPlexGetCone(dm, c, &faces));
        PetscCheck(!(numFaces < dim), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cell %" PetscInt_FMT " has only %" PetscInt_FMT " faces, not enough for gradient reconstruction", c, numFaces);
        for (f = 0, usedFaces = 0; f < numFaces; ++f) {
            PetscFVCellGeom* cg1;
            PetscFVFaceGeom* fg;
            const PetscInt* fcells;
            PetscInt ncell, side;

            if (regionLabel) {
                PetscCall(DMLabelGetValue(regionLabel, faces[f], &labelValue));
                if (labelValue != regionValue) continue;
            }

            PetscCall(DMLabelGetValue(ghostLabel, faces[f], &ghost));
            PetscCall(DMIsBoundaryPoint(dm, faces[f], &boundary));
            if ((ghost >= 0) || boundary) continue;
            PetscCall(DMPlexGetSupport(dm, faces[f], &fcells));
            side = (c != fcells[0]); /* c is on left=0 or right=1 of face */
            ncell = fcells[!side];   /* the neighbor */
            PetscCall(DMPlexPointLocalRef(dmFace, faces[f], fgeom, &fg));
            PetscCall(DMPlexPointLocalRead(dmCell, ncell, cgeom, &cg1));
            for (d = 0; d < dim; ++d) dx[usedFaces * dim + d] = cg1->centroid[d] - cg->centroid[d];
            gref[usedFaces++] = fg->grad[side]; /* Gradient reconstruction term will go here */
        }
        PetscCheck(usedFaces, PETSC_COMM_SELF, PETSC_ERR_USER, "Mesh contains isolated cell (no neighbors). Is it intentional?");
        PetscCall(PetscFVComputeGradient(fvm, usedFaces, dx, grad));
        for (f = 0, usedFaces = 0; f < numFaces; ++f) {
            if (regionLabel) {
                PetscCall(DMLabelGetValue(regionLabel, faces[f], &labelValue));
                if (labelValue != regionValue) continue;
            }
            PetscCall(DMLabelGetValue(ghostLabel, faces[f], &ghost));
            PetscCall(DMIsBoundaryPoint(dm, faces[f], &boundary));
            if ((ghost >= 0) || boundary) continue;
            for (d = 0; d < dim; ++d) gref[usedFaces][d] = grad[usedFaces * dim + d];
            ++usedFaces;
        }
    }
    // Free the memory allocated earlier with PetscMalloc3
    PetscCall(PetscFree3(dx, grad, gref));
    PetscFunctionReturn(0);
}

static PetscErrorCode BuildGradientReconstruction_Internal_Tree(DM dm, DMLabel regionLabel, PetscInt regionValue, PetscFV fvm, DM dmFace, PetscScalar* fgeom, DM dmCell, PetscScalar* cgeom) {
    DMLabel ghostLabel;
    PetscScalar *dx, *grad, **gref;
    PetscInt dim, cStart, cEnd, c, cEndInterior, fStart, fEnd, f, nStart, nEnd, maxNumFaces = 0;
    PetscSection neighSec;
    PetscInt(*neighbors)[2];
    PetscInt* counter;

    PetscFunctionBegin;
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(DMPlexGetCellTypeStratum(dm, DM_POLYTOPE_FV_GHOST, &cEndInterior, nullptr));
    if (cEndInterior < 0) cEndInterior = cEnd;
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &neighSec));
    PetscCall(PetscSectionSetChart(neighSec, cStart, cEndInterior));
    PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
    PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
    for (f = fStart; f < fEnd; f++) {
        const PetscInt* fcells;
        PetscBool boundary;
        PetscInt ghost = -1;
        PetscInt numChildren, numCells, labelValue;

        if (ghostLabel) PetscCall(DMLabelGetValue(ghostLabel, f, &ghost));
        PetscCall(DMIsBoundaryPoint(dm, f, &boundary));
        PetscCall(DMPlexGetTreeChildren(dm, f, &numChildren, nullptr));
        if ((ghost >= 0) || boundary || numChildren) continue;

        if (regionLabel) {
            PetscCall(DMLabelGetValue(regionLabel, f, &labelValue));
            if (labelValue != regionValue) continue;
        }

        PetscCall(DMPlexGetSupportSize(dm, f, &numCells));
        if (numCells == 2) {
            PetscCall(DMPlexGetSupport(dm, f, &fcells));
            for (c = 0; c < 2; c++) {
                PetscInt cell = fcells[c];

                if (cell >= cStart && cell < cEndInterior) {
                    PetscCall(PetscSectionAddDof(neighSec, cell, 1));
                }
            }
        }
    }
    PetscCall(PetscSectionSetUp(neighSec));
    PetscCall(PetscSectionGetMaxDof(neighSec, &maxNumFaces));
    PetscCall(PetscFVLeastSquaresSetMaxFaces(fvm, maxNumFaces));
    nStart = 0;
    PetscCall(PetscSectionGetStorageSize(neighSec, &nEnd));
    PetscCall(PetscMalloc1((nEnd - nStart), &neighbors));
    PetscCall(PetscCalloc1((cEndInterior - cStart), &counter));
    for (f = fStart; f < fEnd; f++) {
        const PetscInt* fcells;
        PetscBool boundary;
        PetscInt ghost = -1;
        PetscInt numChildren, numCells, labelValue;

        if (ghostLabel) PetscCall(DMLabelGetValue(ghostLabel, f, &ghost));
        PetscCall(DMIsBoundaryPoint(dm, f, &boundary));
        PetscCall(DMPlexGetTreeChildren(dm, f, &numChildren, nullptr));
        if ((ghost >= 0) || boundary || numChildren) continue;

        if (regionLabel) {
            PetscCall(DMLabelGetValue(regionLabel, f, &labelValue));
            if (labelValue != regionValue) continue;
        }

        PetscCall(DMPlexGetSupportSize(dm, f, &numCells));
        if (numCells == 2) {
            PetscCall(DMPlexGetSupport(dm, f, &fcells));
            for (c = 0; c < 2; c++) {
                PetscInt cell = fcells[c], off;

                if (regionLabel) {
                    PetscCall(DMLabelGetValue(regionLabel, c, &labelValue));
                    if (labelValue != regionValue) continue;
                }

                if (cell >= cStart && cell < cEndInterior) {
                    PetscCall(PetscSectionGetOffset(neighSec, cell, &off));
                    off += counter[cell - cStart]++;
                    neighbors[off][0] = f;
                    neighbors[off][1] = fcells[1 - c];
                }
            }
        }
    }
    PetscCall(PetscFree(counter));
    PetscCall(PetscMalloc3(maxNumFaces * dim, &dx, maxNumFaces * dim, &grad, maxNumFaces, &gref));
    for (c = cStart; c < cEndInterior; c++) {
        PetscInt numFaces, d, off, labelValue, ghost = -1;
        PetscFVCellGeom* cg;

        PetscCall(DMPlexPointLocalRead(dmCell, c, cgeom, &cg));
        PetscCall(PetscSectionGetDof(neighSec, c, &numFaces));
        PetscCall(PetscSectionGetOffset(neighSec, c, &off));

        if (regionLabel) {
            PetscCall(DMLabelGetValue(regionLabel, c, &labelValue));
            if (labelValue != regionValue) continue;
        }

        // do not attempt to compute a gradient reconstruction stencil in a ghost cell.  It will never be used
        if (ghostLabel) PetscCall(DMLabelGetValue(ghostLabel, c, &ghost));
        if (ghost >= 0) continue;

        PetscCheck(!(numFaces < dim), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cell %" PetscInt_FMT " has only %" PetscInt_FMT " faces, not enough for gradient reconstruction", c, numFaces);
        for (f = 0; f < numFaces; ++f) {
            PetscFVCellGeom* cg1;
            PetscFVFaceGeom* fg;
            const PetscInt* fcells;
            PetscInt ncell, side, nface;

            if (regionLabel) {
                PetscCall(DMLabelGetValue(regionLabel, f, &labelValue));
                if (labelValue != regionValue) continue;
            }

            nface = neighbors[off + f][0];
            ncell = neighbors[off + f][1];
            PetscCall(DMPlexGetSupport(dm, nface, &fcells));
            side = (c != fcells[0]);
            PetscCall(DMPlexPointLocalRef(dmFace, nface, fgeom, &fg));
            PetscCall(DMPlexPointLocalRead(dmCell, ncell, cgeom, &cg1));
            for (d = 0; d < dim; ++d) dx[f * dim + d] = cg1->centroid[d] - cg->centroid[d];
            gref[f] = fg->grad[side]; /* Gradient reconstruction term will go here */
        }
        PetscCall(PetscFVComputeGradient(fvm, numFaces, dx, grad));
        for (f = 0; f < numFaces; ++f) {
            for (d = 0; d < dim; ++d) gref[f][d] = grad[f * dim + d];
        }
    }
    PetscCall(PetscFree3(dx, grad, gref));
    PetscCall(PetscSectionDestroy(&neighSec));
    PetscCall(PetscFree(neighbors));
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::CellInterpolant::ComputeGradientFVM(DM dm, DMLabel regionLabel, PetscInt regionValue, PetscFV fvm, Vec faceGeometry, Vec cellGeometry, DM* dmGrad) {
    DM dmFace, dmCell;
    PetscScalar *fgeom, *cgeom;
    PetscSection sectionGrad, parentSection;
    PetscInt dim, pdim, cStart, cEnd, cEndInterior, c;

    PetscFunctionBegin;
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(PetscFVGetNumComponents(fvm, &pdim));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(DMPlexGetCellTypeStratum(dm, DM_POLYTOPE_FV_GHOST, &cEndInterior, nullptr));
    /* Construct the interpolant corresponding to each face from the least-square solution over the cell neighborhood */
    PetscCall(VecGetDM(faceGeometry, &dmFace));
    PetscCall(VecGetDM(cellGeometry, &dmCell));
    PetscCall(VecGetArray(faceGeometry, &fgeom));
    PetscCall(VecGetArray(cellGeometry, &cgeom));
    PetscCall(DMPlexGetTree(dm, &parentSection, nullptr, nullptr, nullptr, nullptr));
    if (!parentSection) {
        PetscCall(BuildGradientReconstruction_Internal(dm, regionLabel, regionValue, fvm, dmFace, fgeom, dmCell, cgeom));
    } else {
        PetscCall(BuildGradientReconstruction_Internal_Tree(dm, regionLabel, regionValue, fvm, dmFace, fgeom, dmCell, cgeom));
    }
    PetscCall(VecRestoreArray(faceGeometry, &fgeom));
    PetscCall(VecRestoreArray(cellGeometry, &cgeom));
    /* Create storage for gradients */
    PetscCall(DMClone(dm, dmGrad));
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &sectionGrad));
    PetscCall(PetscSectionSetChart(sectionGrad, cStart, cEnd));
    for (c = cStart; c < cEnd; ++c) PetscCall(PetscSectionSetDof(sectionGrad, c, pdim * dim));
    PetscCall(PetscSectionSetUp(sectionGrad));
    PetscCall(DMSetLocalSection(*dmGrad, sectionGrad));
    PetscCall(PetscSectionDestroy(&sectionGrad));
    PetscFunctionReturn(0);
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