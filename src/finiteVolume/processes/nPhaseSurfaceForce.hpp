#pragma once

#include <petsc.h>
#include <memory>
#include <vector>
#include <map>
#include "process.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "finiteVolume/nPhaseFlowFields.hpp"
#include "eos/nPhase.hpp"

namespace ablate::finiteVolume::processes {

class NPhaseSurfaceForce : public Process {
   private:
    // Surface tension parameters
    std::vector<std::vector<PetscReal>> sigma;  // Surface tension coefficients between phases [i][j]
    PetscReal C;                                // Smoothing parameter (standard deviation)
    PetscReal N;                                // Number of standard deviations for smoothing
    // bool flipPhiTilde;                          // Whether to flip the smoothed field

    // Domain and connectivity
    DM vertexDM{};
    std::shared_ptr<ablate::domain::SubDomain> subDomain;
    PetscInt dim;  // Problem dimension

    // Connectivity maps
    std::map<PetscInt, std::vector<PetscInt>> cellToFaces;  // For each cell, list of its faces
    std::map<PetscInt, std::vector<PetscInt>> faceToCells;  // For each face, list of its cells (should always be 2)
    std::map<PetscInt, std::vector<PetscInt>> cellToVertices;  // For each cell, list of its vertices
    std::map<PetscInt, std::vector<PetscInt>> vertexToCells;   // For each vertex, list of cells containing it

    // Ranges
    PetscInt cStart, cEnd;  // Cell range
    PetscInt fStart, fEnd;  // Face range
    PetscInt vStart, vEnd;  // Vertex range

    // Number of phases
    PetscInt nPhases;

    // Curvature fields
    std::vector<Vec> kappaFields;  // One field per unique phase pair
    std::vector<std::string> kappaFieldNames;  // Names for the curvature fields

    // Helper functions
    void SetupConnectivity(DM dm);
    void ComputeSmoothFields(DM dm, const Vec& locX, std::vector<std::vector<PetscReal>>& alphaTilde);
    void ComputeGradients(DM dm, const std::vector<std::vector<PetscReal>>& alphaTilde, 
                         std::vector<std::vector<std::vector<PetscReal>>>& gradAlphaTilde);
    void ComputeCurvature(DM dm, const std::vector<std::vector<std::vector<PetscReal>>>& gradAlphaTilde,
                         std::vector<std::vector<PetscReal>>& kappa);
    void ComputeSourceTerms(DM dm, const Vec& locX, const std::vector<std::vector<PetscReal>>& kappa,
                          const std::vector<std::vector<std::vector<PetscReal>>>& gradAlphaTilde);
    void RegisterKappaFields(DM dm);

    // Constants
    inline const static std::string ALLAIRE_FIELD = NPhaseFlowFields::ALLAIRE_FIELD;
    inline const static std::string ALPHAK = eos::NPhase::ALPHAK;
    inline const static std::string KAPPA_FIELD_PREFIX = "kappa_";  // Prefix for curvature field names

   public:
    /**
     * Constructor for N-phase surface force
     * @param sigma Surface tension coefficients between phases (symmetric matrix)
     * @param C Smoothing parameter (standard deviation)
     * @param N Number of standard deviations for smoothing
     */
    explicit NPhaseSurfaceForce(const std::vector<std::vector<PetscReal>>& sigma, PetscReal C, PetscReal N); //, bool flipPhiTilde = false);

    /**
     * Setup the surface force process
     * @param flow The finite volume solver
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) override;

    /**
     * Initialize the surface force process
     * @param flow The finite volume solver
     */
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver& flow) override;

    /**
     * Compute the source terms for the surface force
     * @param solver The finite volume solver
     * @param dm The DM
     * @param time The current time
     * @param locX The local solution vector
     * @param locFVec The local RHS vector
     * @param ctx The context (this)
     */
    static PetscErrorCode ComputeSource(const FiniteVolumeSolver& solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void* ctx);

    /**
     * Get the names of the curvature fields
     * @return Vector of curvature field names
     */
    const std::vector<std::string>& GetKappaFieldNames() const { return kappaFieldNames; }
};

}  // namespace ablate::finiteVolume::processes 