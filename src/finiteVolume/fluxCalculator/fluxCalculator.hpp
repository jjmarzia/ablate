#ifndef ABLATELIBRARY_FLUXCALCULATOR_HPP
#define ABLATELIBRARY_FLUXCALCULATOR_HPP
#include <petsc.h>

namespace ablate::finiteVolume::fluxCalculator {

/**
 * Structure to hold the full flux vector components
 */
struct FullFluxVector {
    PetscReal massFlux;           // Mass flux
    PetscReal pressureFlux;       // Pressure flux (p12)
    PetscReal momentumFlux[3];    // Momentum flux components
    PetscReal energyFlux;         // Energy flux
    std::vector<PetscReal> alphakRhokFlux;  // Alpha*rho flux for each phase
    std::vector<PetscReal> alphakFlux;      // Alpha flux for each phase

    // Constructor to initialize vectors with given size
    explicit FullFluxVector(size_t nPhases = 0) : alphakRhokFlux(nPhases, 0.0), alphakFlux(nPhases, 0.0) {}
};

/**
 * This function returns the flow direction
 * > 0 left to right
 * < 0 right to left
 */
enum Direction { LEFT = 1, RIGHT = 2, NA = 0 };
using FluxCalculatorFunction = Direction (*)(void* ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal* massFlux,
                                             PetscReal* p12);

class FluxCalculator {
   public:
    FluxCalculator() = default;
    FluxCalculator(FluxCalculator const&) = delete;
    FluxCalculator& operator=(FluxCalculator const&) = delete;
    virtual ~FluxCalculator() = default;

    // Original interface for backward compatibility
    virtual FluxCalculatorFunction GetFluxCalculatorFunction() = 0;
    virtual void* GetFluxCalculatorContext() { return nullptr; }

    // New interface for full flux vector computation
    virtual bool SupportsFullFluxVector() const { return false; }
    
    /**
     * Compute the full flux vector across a face
     * @param ctx The calculator context
     * @param uL Left state velocity (normal component)
     * @param aL Left state speed of sound
     * @param rhoL Left state density
     * @param pL Left state pressure
     * @param uR Right state velocity (normal component)
     * @param aR Right state speed of sound
     * @param rhoR Right state density
     * @param pR Right state pressure
     * @param dim Number of dimensions
     * @param normal Face normal vector
     * @param areaMag Face area magnitude
     * @param velocityL Left state velocity vector
     * @param velocityR Right state velocity vector
     * @param internalEnergyL Left state internal energy
     * @param internalEnergyR Right state internal energy
     * @param fluxVector Output flux vector structure
     * @return true if the full flux vector was computed, false if falling back to basic interface
     */
    virtual bool ComputeFullFluxVector(void* ctx, 
                                     PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL,
                                     PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR,
                                     PetscInt dim, const PetscReal* normal, PetscReal areaMag,
                                     const PetscReal* velocityL, const PetscReal* velocityR,
                                     PetscReal internalEnergyL, PetscReal internalEnergyR,
                                     const std::vector<PetscReal>& alphakL,
                                     const std::vector<PetscReal>& alphakR,
                                     const std::vector<PetscReal>& alphakRhokL,
                                     const std::vector<PetscReal>& alphakRhokR,
                                     FullFluxVector* fluxVector) {
        return false;  // Default implementation returns false
    }
};
}  // namespace ablate::finiteVolume::fluxCalculator
#endif  // ABLATELIBRARY_FLUXCALCULATOR_HPP
