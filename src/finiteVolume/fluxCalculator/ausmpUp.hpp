#ifndef ABLATELIBRARY_AUSMPUP_HPP
#define ABLATELIBRARY_AUSMPUP_HPP

#include "finiteVolume/processes/pressureGradientScaling.hpp"
#include "fluxCalculator.hpp"
namespace ablate::finiteVolume::fluxCalculator {

/**
 * A sequel to AUSM, Part II: AUSM+-up for all speeds
 */
class AusmpUp : public fluxCalculator::FluxCalculator {
   private:
    // AusmUp uses a pgs if provided
    const std::shared_ptr<ablate::finiteVolume::processes::PressureGradientScaling> pgs;

    static Direction AusmpUpFunction(void*, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal* massFlux, PetscReal* p12);

    static PetscReal M1Plus(PetscReal m);
    static PetscReal M2Plus(PetscReal m);
    static PetscReal M1Minus(PetscReal m);
    static PetscReal M2Minus(PetscReal m);
    const inline static PetscReal beta = 1.e+0 / 8.e+0;
    const inline static PetscReal Kp = 0.25;
    const inline static PetscReal Ku = 0.75;
    const inline static PetscReal sigma = 0.25;

    // The reference infinity mach number
    const double mInf;

   public:
    explicit AusmpUp(double mInf, std::shared_ptr<ablate::finiteVolume::processes::PressureGradientScaling> = {});
    AusmpUp(AusmpUp const&) = delete;
    AusmpUp& operator=(AusmpUp const&) = delete;
    ~AusmpUp() override = default;

    FluxCalculatorFunction GetFluxCalculatorFunction() override { return AusmpUpFunction; }
    void* GetFluxCalculatorContext() override { return this; }

    /**
     * Support calls
     * @param m
     * @return
     */
    static PetscReal M4Plus(PetscReal m);
    static PetscReal M4Minus(PetscReal m);
    static PetscReal P5Plus(PetscReal m, double fa);
    static PetscReal P5Minus(PetscReal m, double fa);

    // Override the new interface methods
    bool SupportsFullFluxVector() const override { return true; }
    
    bool ComputeFullFluxVector(void* ctx, 
                             PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL,
                             PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR,
                             PetscInt dim, const PetscReal* normal, PetscReal areaMag,
                             const PetscReal* velocityL, const PetscReal* velocityR,
                             PetscReal internalEnergyL, PetscReal internalEnergyR,
                             const std::vector<PetscReal>& alphakL,
                             const std::vector<PetscReal>& alphakR,
                             const std::vector<PetscReal>& alphakRhokL,
                             const std::vector<PetscReal>& alphakRhokR,
                             fluxCalculator::FullFluxVector* fluxVector) override;
};

}  // namespace ablate::finiteVolume::fluxCalculator
#endif  // ABLATELIBRARY_AUSMPUP_HPP