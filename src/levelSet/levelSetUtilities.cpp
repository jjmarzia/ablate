#include "levelSetUtilities.hpp"
#include <petsc.h>
#include <memory>
#include "LS-VOF.hpp"
#include "cellGrad.hpp"
#include "domain/range.hpp"
#include "domain/reverseRange.hpp"
#include "mathFunctions/functionWrapper.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"

void ablate::levelSet::Utilities::CellValGrad(DM dm, const PetscInt p, PetscReal *c, PetscReal *c0, PetscReal *g) {
    DMPolytopeType ct;
    PetscInt Nc;
    PetscReal *coords = NULL;
    const PetscScalar *array;
    PetscBool isDG;
    PetscReal x0[3];

    // Coordinates of the cell vertices
    DMPlexGetCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

    // Center of the cell
    DMPlexComputeCellGeometryFVM(dm, p, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

    // Get the cell type and call appropriate VOF function
    DMPlexGetCellType(dm, p, &ct) >> ablate::utilities::PetscUtilities::checkError;
    switch (ct) {
        case DM_POLYTOPE_SEGMENT:
            Grad_1D(x0, coords, c, c0, g) >> ablate::utilities::PetscUtilities::checkError;
            break;
        case DM_POLYTOPE_TRIANGLE:
            Grad_2D_Tri(x0, coords, c, c0, g) >> ablate::utilities::PetscUtilities::checkError;
            break;
        case DM_POLYTOPE_QUADRILATERAL:
            Grad_2D_Quad(x0, coords, c, c0, g) >> ablate::utilities::PetscUtilities::checkError;
            break;
        case DM_POLYTOPE_TETRAHEDRON:
            Grad_3D_Tetra(x0, coords, c, c0, g) >> ablate::utilities::PetscUtilities::checkError;
            break;
        case DM_POLYTOPE_HEXAHEDRON:
            Grad_3D_Hex(x0, coords, c, c0, g) >> ablate::utilities::PetscUtilities::checkError;
            break;
        default:
            throw std::invalid_argument("No element geometry for cell " + std::to_string(p) + " with type " + DMPolytopeTypes[ct]);
    }

    DMPlexRestoreCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;
}

void ablate::levelSet::Utilities::CellValGrad(DM dm, const PetscInt fid, const PetscInt p, Vec f, PetscReal *c0, PetscReal *g) {
    PetscInt nv, *verts;
    const PetscScalar *fvals, *v;
    PetscScalar *c;

    DMPlexCellGetVertices(dm, p, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

    DMGetWorkArray(dm, nv, MPIU_SCALAR, &c) >> ablate::utilities::PetscUtilities::checkError;

    VecGetArrayRead(f, &fvals) >> utilities::PetscUtilities::checkError;

    for (PetscInt i = 0; i < nv; ++i) {
        // DMPlexPointLocalFieldRead isn't behaving like I would expect. If I don't make f a pointer then it just returns zero.
        //    Additionally, it looks like it allows for the editing of the value.
        if (fid >= 0) {
            DMPlexPointLocalFieldRead(dm, verts[i], fid, fvals, &v) >> utilities::PetscUtilities::checkError;
        } else {
            DMPlexPointLocalRead(dm, verts[i], fvals, &v) >> utilities::PetscUtilities::checkError;
        }

        c[i] = *v;
    }

    DMPlexCellRestoreVertices(dm, p, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

    ablate::levelSet::Utilities::CellValGrad(dm, p, c, c0, g);

    DMRestoreWorkArray(dm, nv, MPIU_SCALAR, &c) >> ablate::utilities::PetscUtilities::checkError;
}

void ablate::levelSet::Utilities::CellValGrad(std::shared_ptr<ablate::domain::SubDomain> subDomain, const ablate::domain::Field *field, const PetscInt p, PetscReal *c0, PetscReal *g) {
    DM dm = subDomain->GetFieldDM(*field);
    Vec f = subDomain->GetVec(*field);
    ablate::levelSet::Utilities::CellValGrad(dm, field->id, p, f, c0, g);
}

void ablate::levelSet::Utilities::VertexToVertexGrad(std::shared_ptr<ablate::domain::SubDomain> subDomain, const ablate::domain::Field *field, const PetscInt p, PetscReal *g) {
    // Given a field determine the gradient at a vertex

    DM dm = subDomain->GetFieldDM(*field);
    Vec vec = subDomain->GetVec(*field);

    DMPlexVertexGradFromVertex(dm, p, vec, field->id, 0, g) >> ablate::utilities::PetscUtilities::checkError;
}

// Given a level set and normal at the cell center compute the level set values at the vertices assuming a straight interface
void ablate::levelSet::Utilities::VertexLevelSet_LS(DM dm, const PetscInt p, const PetscReal c0, const PetscReal *n, PetscReal **c) {
    PetscInt dim, Nc, nVerts, i, j;
    PetscReal x0[3] = {0.0, 0.0, 0.0};
    PetscReal *coords = NULL;
    const PetscScalar *array;
    PetscBool isDG;

    DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

    // The cell center
    DMPlexComputeCellGeometryFVM(dm, p, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

    // Coordinates of the cell vertices
    DMPlexGetCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

    // Number of vertices
    nVerts = Nc / dim;

    if (*c == NULL) {
        PetscMalloc1(nVerts, c) >> ablate::utilities::PetscUtilities::checkError;
    }

    // The level set value of each vertex. This assumes that the interface is a line/plane
    //    with the given unit normal.
    for (i = 0; i < nVerts; ++i) {
        (*c)[i] = c0;
        for (j = 0; j < dim; ++j) {
            (*c)[i] += n[j] * (coords[i * dim + j] - x0[j]);
        }
    }

    DMPlexRestoreCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;
}

// Given a cell VOF and normal at the cell center compute the level set values at the vertices assuming a straight interface
void ablate::levelSet::Utilities::VertexLevelSet_VOF(DM dm, const PetscInt p, const PetscReal targetVOF, const PetscReal *n, PetscReal **c) {
    PetscReal vof;         // current VOF of the cell
    PetscReal area;        // length (2D) or area (3D) of the cell face
    PetscReal cellVolume;  // Area (2D) or volume (3D) of the cell
    const PetscReal tol = 1e-8;
    PetscInt i;
    PetscReal offset;
    PetscReal vofError;
    PetscInt nv;

    // Get the number of vertices for the cell
    DMPlexCellGetNumVertices(dm, p, &nv) >> ablate::utilities::PetscUtilities::checkError;

    // Get an initial guess at the vertex level set values assuming that the interface passes through the cell-center.
    // Also allocates c if c==NULL on entry
    ablate::levelSet::Utilities::VertexLevelSet_LS(dm, p, 0.0, n, c);

    // Get the resulting VOF from the initial guess
    ablate::levelSet::Utilities::VOF(dm, p, *c, &vof, &area, &cellVolume);
    vofError = targetVOF - vof;

    while (fabs(vofError) > tol) {
        // The amount the center level set value needs to shift by.
        offset = vofError * cellVolume / area;

        // If this isn't damped then it will overshoot and there will be no interface in the cell
        offset *= 0.5;

        for (i = 0; i < nv; ++i) {
            (*c)[i] -= offset;
        }

        ablate::levelSet::Utilities::VOF(dm, p, *c, &vof, &area, NULL);
        vofError = targetVOF - vof;
    };
}

// Returns the VOF for a given cell using the level-set values at the cell vertices.
// Refer to "Quadrature rules for triangular and tetrahedral elements with generalized functions"
//  by Holdych, Noble, and Secor, Int. J. Numer. Meth. Engng 2008; 73:1310-1327.
void ablate::levelSet::Utilities::VOF(DM dm, const PetscInt p, PetscReal *c, PetscReal *vof, PetscReal *area, PetscReal *vol) {
    DMPolytopeType ct;
    PetscInt Nc;
    PetscReal *coords = NULL;
    const PetscScalar *array;
    PetscBool isDG;

    // Coordinates of the cell vertices
    DMPlexGetCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

    // Get the cell type and call appropriate VOF function
    DMPlexGetCellType(dm, p, &ct) >> ablate::utilities::PetscUtilities::checkError;
    switch (ct) {
        case DM_POLYTOPE_SEGMENT:
            VOF_1D(coords, c, vof, area, vol);
            break;
        case DM_POLYTOPE_TRIANGLE:
            VOF_2D_Tri(coords, c, vof, area, vol);
            break;
        case DM_POLYTOPE_QUADRILATERAL:
            VOF_2D_Quad(coords, c, vof, area, vol);
            break;
        case DM_POLYTOPE_TETRAHEDRON:
            VOF_3D_Tetra(coords, c, vof, area, vol);
            break;
        case DM_POLYTOPE_HEXAHEDRON:
            VOF_3D_Hex(coords, c, vof, area, vol);
            break;
        default:
            throw std::invalid_argument("No element geometry for cell " + std::to_string(p) + " with type " + DMPolytopeTypes[ct]);
    }

    DMPlexRestoreCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;
}

// Returns the VOF for a given cell with a known level set value (c0) and normal (nIn).
//  This computes the level-set values at the vertices by approximating the interface as a straight-line with the same normal
//  as provided
void ablate::levelSet::Utilities::VOF(DM dm, const PetscInt p, const PetscReal c0, const PetscReal *nIn, PetscReal *vof, PetscReal *area, PetscReal *vol) {
    PetscReal *c = NULL;
    ablate::levelSet::Utilities::VertexLevelSet_LS(dm, p, c0, nIn, &c);

    ablate::levelSet::Utilities::VOF(dm, p, c, vof, area, vol);  // Do the actual calculation.

    PetscFree(c) >> ablate::utilities::PetscUtilities::checkError;
}

// Returns the VOF for a given cell using an analytic level set equation
// Refer to "Quadrature rules for triangular and tetrahedral elements with generalized functions"
void ablate::levelSet::Utilities::VOF(DM dm, PetscInt p, const std::shared_ptr<ablate::mathFunctions::MathFunction> &phi, PetscReal *vof, PetscReal *area, PetscReal *vol) {
    PetscInt dim, Nc, nVerts, i;
    PetscReal *c = NULL, *coords = NULL;
    const PetscScalar *array;
    PetscBool isDG;

    DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

    // Coordinates of the cell vertices
    DMPlexGetCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

    // Number of vertices
    nVerts = Nc / dim;

    PetscMalloc1(nVerts, &c) >> ablate::utilities::PetscUtilities::checkError;

    // The level set value of each vertex. This assumes that the interface is a line/plane
    //    with the given unit normal.
    for (i = 0; i < nVerts; ++i) {
        c[i] = phi->Eval(&coords[i * dim], dim, 0.0);
    }

    DMPlexRestoreCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

    ablate::levelSet::Utilities::VOF(dm, p, c, vof, area, vol);  // Do the actual calculation.

    PetscFree(c) >> ablate::utilities::PetscUtilities::checkError;
}

// Return the VOF in a cell where the level set is defined at vertices
void ablate::levelSet::Utilities::VOF(std::shared_ptr<ablate::domain::SubDomain> subDomain, PetscInt cell, const ablate::domain::Field *lsField, PetscReal *vof, PetscReal *area, PetscReal *vol) {
    DM dm = subDomain->GetFieldDM(*lsField);
    Vec vec = subDomain->GetVec(*lsField);
    const PetscScalar *array;
    PetscReal *c;

    PetscInt nv, *verts;
    DMPlexCellGetVertices(dm, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
    DMGetWorkArray(dm, nv, MPI_REAL, &c);

    VecGetArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
    for (PetscInt i = 0; i < nv; ++i) {
        const PetscReal *val;
        xDMPlexPointLocalRead(dm, verts[i], lsField->id, array, &val) >> ablate::utilities::PetscUtilities::checkError;
        c[i] = *val;
    }
    VecRestoreArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;

    ablate::levelSet::Utilities::VOF(dm, cell, c, vof, area, vol);

    DMRestoreWorkArray(dm, nv, MPI_REAL, &c);
    DMPlexCellRestoreVertices(dm, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
}
