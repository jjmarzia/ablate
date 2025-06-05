#include "nPhaseSurfaceForce.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "utilities/petscUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "finiteVolume/nPhaseFlowFields.hpp"
#include <petsc/private/dmpleximpl.h>
#include <cmath>
#include <algorithm>

namespace ablate::finiteVolume::processes {

NPhaseSurfaceForce::NPhaseSurfaceForce(const std::vector<std::vector<PetscReal>>& sigma, PetscReal C, PetscReal N)
    : sigma(sigma), C(C), N(N)  {
    // Validate surface tension matrix
    nPhases = sigma.size();
    if (nPhases < 2) {
        throw std::invalid_argument("NPhaseSurfaceForce requires at least 2 phases");
    }

    // Check matrix size and symmetry

    for (PetscInt i = 0; i < nPhases; i++) {
        for (PetscInt j = i + 1; j < nPhases; j++) {
            if (std::abs(sigma[i][j] - sigma[j][i]) > 1e-10) {
                throw std::invalid_argument("Surface tension matrix must be symmetric");
            }
        }
    }
}

void NPhaseSurfaceForce::Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) {
    // Get the subdomain
    // subDomain = std::make_shared<ablate::domain::SubDomain>(flow.GetSubDomain());

    auto subDomain = const_cast<FiniteVolumeSolver&>(flow).GetSubDomainPtr();
    if (!subDomain) {
        throw std::runtime_error("SubDomain not set in solver");
    }

    // Get the DM and dimension
    DM dm = subDomain->GetDM();
    DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;

    // Setup connectivity
    SetupConnectivity(dm);

    // Register curvature fields
    RegisterKappaFields(dm);

    // Register the source term using RHSArbitraryFunction
    flow.RegisterRHSFunction(ComputeSource, this);
}

void NPhaseSurfaceForce::Initialize(ablate::finiteVolume::FiniteVolumeSolver& flow) {
    // Get vertex DM from subdomain
    DM dm = subDomain->GetDM();
    DMGetCoordinateDM(dm, &vertexDM) >> utilities::PetscUtilities::checkError;
}

void NPhaseSurfaceForce::SetupConnectivity(DM dm) {
    // Get cell and face ranges
    DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;
    DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd) >> utilities::PetscUtilities::checkError;

    // Build cell-to-faces connectivity
    for (PetscInt c = cStart; c < cEnd; c++) {
        PetscInt nFaces;
        const PetscInt* faces;
        DMPlexGetConeSize(dm, c, &nFaces) >> utilities::PetscUtilities::checkError;
        DMPlexGetCone(dm, c, &faces) >> utilities::PetscUtilities::checkError;
        cellToFaces[c] = std::vector<PetscInt>(faces, faces + nFaces);
    }

    // Build face-to-cells connectivity
    for (PetscInt f = fStart; f < fEnd; f++) {
        PetscInt nCells;
        const PetscInt* cells;
        DMPlexGetSupportSize(dm, f, &nCells) >> utilities::PetscUtilities::checkError;
        DMPlexGetSupport(dm, f, &cells) >> utilities::PetscUtilities::checkError;
        faceToCells[f] = std::vector<PetscInt>(cells, cells + nCells);
    }

    // Build cell-to-vertices connectivity
    for (PetscInt c = cStart; c < cEnd; c++) {
        PetscInt nVertices;
        const PetscInt* vertices;
        DMPlexGetConeSize(dm, c, &nVertices) >> utilities::PetscUtilities::checkError;
        DMPlexGetCone(dm, c, &vertices) >> utilities::PetscUtilities::checkError;
        cellToVertices[c] = std::vector<PetscInt>(vertices, vertices + nVertices);

        // Add to vertex-to-cells map
        for (PetscInt v : cellToVertices[c]) {
            vertexToCells[v].push_back(c);
        }
    }

    // Get vertex range
    DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd) >> utilities::PetscUtilities::checkError;
}

// Helper function to compute Gaussian weight
static inline void ComputeGaussianWeight(PetscReal d, PetscReal s, PetscReal* weight) {
    *weight = std::exp(-0.5 * (d * d) / (s * s));
}

// Helper function to get cell center coordinates
static inline void GetCellCenter(DM dm, PetscInt cell, PetscReal* x, PetscReal* y, PetscReal* z) {
    PetscInt nVertices;
    const PetscInt* vertices;
    DMPlexGetConeSize(dm, cell, &nVertices) >> utilities::PetscUtilities::checkError;
    DMPlexGetCone(dm, cell, &vertices) >> utilities::PetscUtilities::checkError;

    // Initialize centroid
    PetscReal centroid[3] = {0.0, 0.0, 0.0};
    PetscInt dim;
    DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;

    // Get array access for coordinates
    Vec coordVec;
    DMGetCoordinatesLocal(dm, &coordVec) >> utilities::PetscUtilities::checkError;
    const PetscScalar* coordArray;
    VecGetArrayRead(coordVec, &coordArray) >> utilities::PetscUtilities::checkError;

    // Sum vertex coordinates
    for (PetscInt v = 0; v < nVertices; v++) {
        const PetscScalar* vCoord;
        // Use field ID 0 for coordinates
        xDMPlexPointLocalRead(dm, vertices[v], 0, coordArray, &vCoord) >> utilities::PetscUtilities::checkError;
        if (!vCoord) continue;
        for (PetscInt d = 0; d < dim; d++) {
            centroid[d] += vCoord[d];
        }
    }

    // Restore array access
    VecRestoreArrayRead(coordVec, &coordArray) >> utilities::PetscUtilities::checkError;

    // Average to get center
    for (PetscInt d = 0; d < dim; d++) {
        centroid[d] /= nVertices;
    }

    // Copy to output
    *x = centroid[0];
    *y = (dim > 1) ? centroid[1] : 0.0;
    *z = (dim > 2) ? centroid[2] : 0.0;
}

// Helper function to compute distance between points
static inline PetscReal ComputeDistance(PetscInt dim, const PetscReal* x1, const PetscReal* x2) {
    PetscReal d = 0.0;
    for (PetscInt i = 0; i < dim; i++) {
        d += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }
    return std::sqrt(d);
}

void NPhaseSurfaceForce::ComputeSmoothFields(DM dm, const Vec& locX, std::vector<std::vector<PetscReal>>& alphaTilde) {
    // Get field offsets
    PetscInt alphaField;
    const auto& fields = subDomain->GetFields();
    for (const auto& field : fields) {
        if (field.name == ALPHAK) {
            alphaField = field.id;
            break;
        }
    }

    // Get array access
    const PetscScalar* locXArray;
    VecGetArrayRead(locX, &locXArray) >> utilities::PetscUtilities::checkError;

    // Initialize alphaTilde for each phase
    alphaTilde.resize(nPhases);
    for (auto& phase : alphaTilde) {
        phase.resize(cEnd - cStart, 0.0);
    }

    // For each cell
    for (PetscInt c = cStart; c < cEnd; c++) {
        // Get cell's volume fractions
        const PetscScalar* cellValues;
        xDMPlexPointLocalRead(dm, c, alphaField, locXArray, &cellValues) >> utilities::PetscUtilities::checkError;
        if (!cellValues) continue;

        // For each phase, compute smoothed field
        for (PetscInt k = 0; k < nPhases; k++) {
            PetscReal alphaC = cellValues[k];
            PetscReal weightSum = 1.0;  // Include self with weight 1.0
            PetscReal weightedSum = alphaC;

            // Get cell center
            PetscReal xC[3] = {0.0, 0.0, 0.0};
            GetCellCenter(dm, c, &xC[0], &xC[1], &xC[2]);

            // For each neighbor
            for (PetscInt n : cellToFaces[c]) {
                PetscInt neighborCell = (faceToCells[n][0] == c) ? faceToCells[n][1] : faceToCells[n][0];
                if (neighborCell < cStart || neighborCell >= cEnd) continue;

                // Get neighbor's volume fraction
                const PetscScalar* neighborValues;
                xDMPlexPointLocalRead(dm, neighborCell, alphaField, locXArray, &neighborValues) >> utilities::PetscUtilities::checkError;
                if (!neighborValues) continue;

                PetscReal alphaN = neighborValues[k];

                // Get neighbor center
                PetscReal xN[3] = {0.0, 0.0, 0.0};
                GetCellCenter(dm, neighborCell, &xN[0], &xN[1], &xN[2]);

                // Compute distance and weight
                PetscReal d = ComputeDistance(dim, xC, xN);
                PetscReal weight;
                ComputeGaussianWeight(d, C, &weight);

                // Accumulate weighted sum
                weightedSum += weight * alphaN;
                weightSum += weight;
            }

            // Store smoothed value
            alphaTilde[k][c - cStart] = weightedSum / weightSum;
        }
    }

    VecRestoreArrayRead(locX, &locXArray);
}

void NPhaseSurfaceForce::ComputeGradients(DM dm, const std::vector<std::vector<PetscReal>>& alphaTilde,
                                        std::vector<std::vector<std::vector<PetscReal>>>& gradAlphaTilde) {
    // Initialize gradient arrays
    gradAlphaTilde.resize(nPhases);
    for (PetscInt k = 0; k < nPhases; k++) {
        gradAlphaTilde[k].resize(cEnd - cStart);
        for (PetscInt c = cStart; c < cEnd; c++) {
            gradAlphaTilde[k][c - cStart].resize(dim, 0.0);
        }
    }

    // For each cell
    for (PetscInt c = cStart; c < cEnd; c++) {
        // Get cell center
        PetscReal xc[3];
        GetCellCenter(dm, c, &xc[0], &xc[1], &xc[2]);

        for (PetscInt k = 0; k < nPhases; k++) {
            // Initialize gradient components
            std::vector<PetscReal> grad(dim, 0.0);
            PetscReal den = 0.0;

            // Loop over neighboring cells
            for (PetscInt f : cellToFaces[c]) {
                for (PetscInt neighborCell : faceToCells[f]) {
                    if (neighborCell == c) continue;  // Skip self

                    // Get neighbor center
                    PetscReal xn[3];
                    GetCellCenter(dm, neighborCell, &xn[0], &xn[1], &xn[2]);

                    // Compute distance
                    PetscReal d = ComputeDistance(dim, xc, xn);
                    if (d > N * C) continue;  // Skip if beyond smoothing radius

                    // Get neighbor's smoothed value
                    PetscReal alphaN = alphaTilde[k][neighborCell - cStart];
                    PetscReal alphaC = alphaTilde[k][c - cStart];

                    // Compute Gaussian weight
                    PetscReal weight;
                    ComputeGaussianWeight(d, C, &weight);

                    // Accumulate gradient
                    for (PetscInt d = 0; d < dim; d++) {
                        grad[d] += weight * (alphaN - alphaC) * (xn[d] - xc[d]) / (d * d);
                    }
                    den += weight;
                }
            }

            // Normalize gradient
            if (den > 1e-10) {
                for (PetscInt d = 0; d < dim; d++) {
                    gradAlphaTilde[k][c - cStart][d] = grad[d] / den;
                }
            }
        }
    }
}

void NPhaseSurfaceForce::RegisterKappaFields(DM dm) {
    // Clear any existing fields
    for (auto& field : kappaFields) {
        if (field) {
            VecDestroy(&field) >> utilities::PetscUtilities::checkError;
        }
    }
    kappaFields.clear();
    kappaFieldNames.clear();

    // Create a field for each unique phase pair
    // PetscInt pairIdx = 0;
    for (PetscInt i = 0; i < nPhases; i++) {
        for (PetscInt j = i + 1; j < nPhases; j++) {
            // Create field name
            std::string fieldName = KAPPA_FIELD_PREFIX + std::to_string(i) + "_" + std::to_string(j);
            kappaFieldNames.push_back(fieldName);

            // Create the field
            Vec field;
            DMCreateLocalVector(dm, &field) >> utilities::PetscUtilities::checkError;
            VecSet(field, 0.0) >> utilities::PetscUtilities::checkError;
            kappaFields.push_back(field);

            // pairIdx++;
        }
    }
}

void NPhaseSurfaceForce::ComputeCurvature(DM dm, const std::vector<std::vector<std::vector<PetscReal>>>& gradAlphaTilde,
                                        std::vector<std::vector<PetscReal>>& kappa) {
    // Initialize curvature array for each unique phase pair
    PetscInt nPairs = (nPhases * (nPhases - 1)) / 2;
    kappa.resize(nPairs);
    for (PetscInt p = 0; p < nPairs; p++) {
        kappa[p].resize(cEnd - cStart, 0.0);
    }

    // For each cell
    for (PetscInt c = cStart; c < cEnd; c++) {
        // Get cell center
        PetscReal xc[3];
        GetCellCenter(dm, c, &xc[0], &xc[1], &xc[2]);

        // For each unique phase pair (i < j)
        PetscInt pairIdx = 0;
        for (PetscInt i = 0; i < nPhases; i++) {
            for (PetscInt j = i + 1; j < nPhases; j++) {
                // Compute interface indicator function gradient
                std::vector<PetscReal> gradPhi(dim, 0.0);
                PetscReal den = 0.0;
                for (PetscInt d = 0; d < dim; d++) {
                    gradPhi[d] = gradAlphaTilde[i][c - cStart][d] - gradAlphaTilde[j][c - cStart][d];
                    den += gradPhi[d] * gradPhi[d];
                }

                if (den > 1e-10) {
                    PetscReal norm = std::sqrt(den);
                    // Normalize gradient
                    for (PetscInt d = 0; d < dim; d++) {
                        gradPhi[d] /= norm;
                    }

                    // Compute divergence of normalized gradient
                    PetscReal divGradPhi = 0.0;
                    for (PetscInt f : cellToFaces[c]) {
                        for (PetscInt neighborCell : faceToCells[f]) {
                            if (neighborCell == c) continue;  // Skip self

                            // Get neighbor center
                            PetscReal xn[3];
                            GetCellCenter(dm, neighborCell, &xn[0], &xn[1], &xn[2]);

                            // Compute distance
                            PetscReal d = ComputeDistance(dim, xc, xn);
                            if (d > N * C) continue;  // Skip if beyond smoothing radius

                            // Get neighbor's gradient
                            std::vector<PetscReal> gradPhiN(dim, 0.0);
                            PetscReal denN = 0.0;
                            for (PetscInt d = 0; d < dim; d++) {
                                gradPhiN[d] = gradAlphaTilde[i][neighborCell - cStart][d] - gradAlphaTilde[j][neighborCell - cStart][d];
                                denN += gradPhiN[d] * gradPhiN[d];
                            }

                            if (denN > 1e-10) {
                                PetscReal normN = std::sqrt(denN);
                                // Normalize neighbor gradient
                                for (PetscInt d = 0; d < dim; d++) {
                                    gradPhiN[d] /= normN;
                                }

                                // Compute Gaussian weight
                                PetscReal weight;
                                ComputeGaussianWeight(d, C, &weight);

                                // Accumulate divergence
                                for (PetscInt d = 0; d < dim; d++) {
                                    divGradPhi += weight * (gradPhiN[d] - gradPhi[d]) * (xn[d] - xc[d]) / (d * d);
                                }
                            }
                        }
                    }

                    // Store curvature
                    kappa[pairIdx][c - cStart] = -divGradPhi;
                }
                pairIdx++;
            }
        }
    }

    // Update curvature fields
    PetscInt pairIdx = 0;
    for (PetscInt i = 0; i < nPhases; i++) {
        for (PetscInt j = i + 1; j < nPhases; j++) {
            // Get array access for the field
            PetscScalar* kappaArray;
            VecGetArray(kappaFields[pairIdx], &kappaArray) >> utilities::PetscUtilities::checkError;

            // Copy curvature values
            for (PetscInt c = cStart; c < cEnd; c++) {
                kappaArray[c - cStart] = kappa[pairIdx][c - cStart];
            }

            // Restore array access
            VecRestoreArray(kappaFields[pairIdx], &kappaArray) >> utilities::PetscUtilities::checkError;

            pairIdx++;
        }
    }
}

void NPhaseSurfaceForce::ComputeSourceTerms(DM dm, const Vec& locX, const std::vector<std::vector<PetscReal>>& kappa,
                                          const std::vector<std::vector<std::vector<PetscReal>>>& gradAlphaTilde) {
    // Get field offsets
    PetscInt allaireField, alphakrhokField;
    const auto& fields = subDomain->GetFields();
    for (const auto& field : fields) {
        if (field.name == NPhaseFlowFields::ALLAIRE_FIELD) {
            allaireField = field.id;
        } else if (field.name == NPhaseFlowFields::ALPHAKRHOK) {
            alphakrhokField = field.id;
        }
    }

    // Get array access
    const PetscScalar* locXArray;
    VecGetArrayRead(locX, &locXArray) >> utilities::PetscUtilities::checkError;
    PetscScalar* locFArray;
    VecGetArray(locX, &locFArray) >> utilities::PetscUtilities::checkError;

    // For each cell, compute source terms
    for (PetscInt c = cStart; c < cEnd; c++) {
        // Get cell values
        const PetscScalar* cellValues;
        xDMPlexPointLocalRead(dm, c, allaireField, locXArray, &cellValues) >> utilities::PetscUtilities::checkError;
        if (!cellValues) continue;

        // Get mixture velocity
        std::vector<PetscReal> u(dim);
        for (PetscInt d = 0; d < dim; d++) {
            u[d] = cellValues[NPhaseFlowFields::RHOU + d];
        }

        // Get phase densities (alphakrhok)
        const PetscScalar* alphakrhokValues;
        xDMPlexPointLocalRead(dm, c, alphakrhokField, locXArray, &alphakrhokValues) >> utilities::PetscUtilities::checkError;
        if (!alphakrhokValues) continue;

        // Compute total density as sum of phase densities
        PetscReal rho = 0.0;
        for (PetscInt k = 0; k < nPhases; k++) {
            rho += alphakrhokValues[k];
        }

        // Normalize velocity by total density
        if (rho > 0.0) {
            for (PetscInt d = 0; d < dim; d++) {
                u[d] /= rho;
            }
        }

        // Initialize source terms
        std::vector<PetscReal> momentumSource(dim, 0.0);
        PetscReal energySource = 0.0;

        // For each unique phase pair
        PetscInt pairIdx = 0;
        for (PetscInt i = 0; i < nPhases; i++) {
            for (PetscInt j = i + 1; j < nPhases; j++) {
                // Get gradient of interface indicator
                const auto& gradPhi = gradAlphaTilde[pairIdx][c - cStart];

                // Compute momentum and energy source terms
                PetscReal sigma_ij = sigma[i][j];
                PetscReal kappa_ij = kappa[pairIdx][c - cStart];
                for (PetscInt d = 0; d < dim; d++) {
                    momentumSource[d] += sigma_ij * kappa_ij * gradPhi[d];
                }

                // Compute energy source term
                PetscReal uDotGradPhi = 0.0;
                for (PetscInt d = 0; d < dim; d++) {
                    uDotGradPhi += u[d] * gradPhi[d];
                }
                energySource += sigma_ij * kappa_ij * uDotGradPhi;

                pairIdx++;
            }
        }

        // Add source terms to RHS
        PetscScalar* cellRHS;
        xDMPlexPointLocalRef(dm, c, allaireField, locFArray, &cellRHS) >> utilities::PetscUtilities::checkError;
        if (!cellRHS) continue;

        // Add momentum source terms
        for (PetscInt d = 0; d < dim; d++) {
            cellRHS[NPhaseFlowFields::RHOU + d] += momentumSource[d];
        }

        // Add energy source term
        cellRHS[NPhaseFlowFields::RHOE] += energySource;
    }

    VecRestoreArrayRead(locX, &locXArray);
    VecRestoreArray(locX, &locFArray);
}

PetscErrorCode NPhaseSurfaceForce::ComputeSource(const FiniteVolumeSolver& solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void* ctx) {
    PetscFunctionBeginUser;
    auto surfaceForce = (NPhaseSurfaceForce*)ctx;

    // Get local arrays
    const PetscScalar* locXArray;
    VecGetArrayRead(locX, &locXArray) >> utilities::PetscUtilities::checkError;
    PetscScalar* locFArray;
    VecGetArray(locFVec, &locFArray) >> utilities::PetscUtilities::checkError;

    // Compute smooth fields
    std::vector<std::vector<PetscReal>> alphaTilde;
    surfaceForce->ComputeSmoothFields(dm, locX, alphaTilde);

    // Compute gradients
    std::vector<std::vector<std::vector<PetscReal>>> gradAlphaTilde;
    surfaceForce->ComputeGradients(dm, alphaTilde, gradAlphaTilde);

    // Compute curvature
    std::vector<std::vector<PetscReal>> kappa;
    surfaceForce->ComputeCurvature(dm, gradAlphaTilde, kappa);

    // Compute source terms
    surfaceForce->ComputeSourceTerms(dm, locX, kappa, gradAlphaTilde);

    // Restore arrays
    VecRestoreArrayRead(locX, &locXArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArray(locFVec, &locFArray) >> utilities::PetscUtilities::checkError;

    PetscFunctionReturn(0);
}

}  // namespace ablate::finiteVolume::processes

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::NPhaseSurfaceForce, "N-phase surface force process",
         ARG(std::vector<std::vector<PetscReal>>, "sigma", "Surface tension coefficients between phases (symmetric matrix)"),
         ARG(PetscReal, "C", "Smoothing parameter (standard deviation)"),
         ARG(PetscReal, "N", "Number of standard deviations for smoothing"));
        //  OPT(bool, "flipPhiTilde", "Whether to flip the smoothed field (default: false)")); 