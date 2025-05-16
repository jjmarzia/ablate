#include "ausmpUp.hpp"

ablate::finiteVolume::fluxCalculator::AusmpUp::AusmpUp(double mInf, std::shared_ptr<ablate::finiteVolume::processes::PressureGradientScaling> pgs) : pgs(pgs), mInf(mInf) {}

ablate::finiteVolume::fluxCalculator::Direction ablate::finiteVolume::fluxCalculator::AusmpUp::AusmpUpFunction(void* ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR,
                                                                                                               PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal* massFlux, PetscReal* p12) {
    // extract pgs/minf if provided
    auto ausmUp = (ablate::finiteVolume::fluxCalculator::AusmpUp*)ctx;
    PetscReal pgsAlpha = ausmUp->pgs ? ausmUp->pgs->GetAlpha() : 1.0;
    PetscReal mInf = ausmUp->mInf;

    // Compute the density at the interface
    PetscReal rho12 = (0.5) * (rhoL + rhoR);

    // compute the speed of sound at a12
    PetscReal a12 = 0.5 * (aL + aR) / pgsAlpha;  // Simple average of aL and aR.  This can be replaced with eq. 30;

    // Compute the left and right mach numbers
    PetscReal mL = uL / a12;
    PetscReal mR = uR / a12;

    // Compute mBar2 (eq 70)
    PetscReal mBar2 = (PetscSqr(uL) + PetscSqr(uR)) / (2.0 * a12 * a12);

    // compute mInf2 or set fa to unity
    PetscReal fa = 1.0;
    if (mInf > 0) {
        PetscReal mInf2 = PetscSqr(mInf);

        PetscReal mO2 = PetscMin(1.0, PetscMax(mBar2, mInf2));
        PetscReal mO = PetscSqrtReal(mO2);
        fa = mO * (2.0 - mO);
    }
    // compute the mach number on the interface
    PetscReal m12 = M4Plus(mL) + M4Minus(mR) - (Kp / fa) * PetscMax(1.0 - (sigma * mBar2), 0) * (pR - pL) / (rho12 * a12 * a12 * pgsAlpha * pgsAlpha);

    // store the mass flux;
    Direction direction;
    if (m12 > 0) {
        direction = LEFT;
        *massFlux = a12 * m12 * rhoL;
    } else {
        direction = RIGHT;
        *massFlux = a12 * m12 * rhoR;
    }

    // Pressure
    if (p12) {
        double p5Plus = P5Plus(mL, fa);
        double p5Minus = P5Minus(mR, fa);

        *p12 = p5Plus * pL + p5Minus * pR - Ku * p5Plus * p5Minus * rho12 * fa * a12 * a12 * pgsAlpha * pgsAlpha * (mR - mL);
        *p12 /= PetscSqr(pgsAlpha);
    }
    return direction;
}

PetscReal ablate::finiteVolume::fluxCalculator::AusmpUp::M1Plus(PetscReal m) { return 0.5 * (m + PetscAbs(m)); }

PetscReal ablate::finiteVolume::fluxCalculator::AusmpUp::M2Plus(PetscReal m) { return 0.25 * PetscSqr(m + 1); }

PetscReal ablate::finiteVolume::fluxCalculator::AusmpUp::M1Minus(PetscReal m) { return 0.5 * (m - PetscAbs(m)); }
PetscReal ablate::finiteVolume::fluxCalculator::AusmpUp::M2Minus(PetscReal m) { return -0.25 * PetscSqr(m - 1); }

PetscReal ablate::finiteVolume::fluxCalculator::AusmpUp::M4Plus(PetscReal m) {
    if (PetscAbs(m) >= 1.0) {
        return M1Plus(m);
    } else {
        return M2Plus(m) * (1.0 - 16.0 * beta * M2Minus(m));
    }
}
PetscReal ablate::finiteVolume::fluxCalculator::AusmpUp::M4Minus(PetscReal m) {
    if (PetscAbs(m) >= 1.0) {
        return M1Minus(m);
    } else {
        return M2Minus(m) * (1.0 + 16.0 * beta * M2Plus(m));
    }
}
PetscReal ablate::finiteVolume::fluxCalculator::AusmpUp::P5Plus(PetscReal m, double fa) {
    if (PetscAbs(m) >= 1.0) {
        return (M1Plus(m) / (m + 1E-30));
    } else {
        // compute alpha
        double alpha = 3.0 / 16.0 * (-4.0 + 5 * fa * fa);

        return (M2Plus(m) * ((2.0 - m) - 16. * alpha * m * M2Minus(m)));
    }
}
PetscReal ablate::finiteVolume::fluxCalculator::AusmpUp::P5Minus(PetscReal m, double fa) {
    if (PetscAbs(m) >= 1.0) {
        return (M1Minus(m) / (m + 1E-30));
    } else {
        double alpha = 3.0 / 16.0 * (-4.0 + 5 * fa * fa);
        return (M2Minus(m) * ((-2.0 - m) + 16. * alpha * m * M2Plus(m)));
    }
}

bool ablate::finiteVolume::fluxCalculator::AusmpUp::ComputeFullFluxVector(void* ctx, 
                                                                      PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL,
                                                                      PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR,
                                                                      PetscInt dim, const PetscReal* normal, PetscReal areaMag,
                                                                      const PetscReal* velocityL, const PetscReal* velocityR,
                                                                      PetscReal internalEnergyL, PetscReal internalEnergyR,
                                                                      const std::vector<PetscReal>& alphakL,
                                                                      const std::vector<PetscReal>& alphakR,
                                                                      const std::vector<PetscReal>& alphakRhokL,
                                                                      const std::vector<PetscReal>& alphakRhokR,
                                                                      fluxCalculator::FullFluxVector* fluxVector) {

    // Extract parameters from context
    auto ausmUp = (ablate::finiteVolume::fluxCalculator::AusmpUp*)ctx;
    PetscReal pgsAlpha = ausmUp->pgs ? ausmUp->pgs->GetAlpha() : 1.0;
    PetscReal mInf = ausmUp->mInf;

    // Compute the density at the interface
    PetscReal rho12 = 0.5 * (rhoL + rhoR);

    // Compute the speed of sound at a12
    PetscReal a12 = 0.5 * (aL + aR) / pgsAlpha;

    // Compute the left and right mach numbers
    PetscReal mL = uL / a12;
    PetscReal mR = uR / a12;

    // Compute mBar2 (eq 70)
    PetscReal mBar2 = (PetscSqr(uL) + PetscSqr(uR)) / (2.0 * a12 * a12);

    // Compute mInf2 or set fa to unity
    PetscReal fa = 1.0;
    if (mInf > 0) {
        PetscReal mInf2 = PetscSqr(mInf);
        PetscReal mO2 = PetscMin(1.0, PetscMax(mBar2, mInf2));
        PetscReal mO = PetscSqrtReal(mO2);
        fa = mO * (2.0 - mO);
    }

    // Compute the mach number on the interface
    PetscReal m12 = M4Plus(mL) + M4Minus(mR) - (Kp / fa) * PetscMax(1.0 - (sigma * mBar2), 0) * (pR - pL) / (rho12 * a12 * a12 * pgsAlpha * pgsAlpha);

    // Compute the Riemann velocity
    PetscReal vRiem = a12 * m12;

    // Split the velocity
    PetscReal lPlus = 0.5 * (vRiem + PetscAbs(vRiem));
    PetscReal lMinus = 0.5 * (vRiem - PetscAbs(vRiem));

    // Compute pressure flux
    PetscReal p5Plus = P5Plus(mL, fa);
    PetscReal p5Minus = P5Minus(mR, fa);
    PetscReal p12 = (p5Plus * pL + p5Minus * pR - Ku * p5Plus * p5Minus * rho12 * fa * a12 * a12 * pgsAlpha * pgsAlpha * (mR - mL)) / (pgsAlpha * pgsAlpha);

    // Set the basic flux components
    fluxVector->massFlux = (lPlus * rhoL + lMinus * rhoR) * areaMag;
    fluxVector->pressureFlux = p12 * areaMag;

    // Compute momentum flux components
    for (PetscInt d = 0; d < dim; d++) {
        // Compute momentum flux using split velocities and full velocity vectors
        fluxVector->momentumFlux[d] = (lPlus * rhoL * velocityL[d] + lMinus * rhoR * velocityR[d]) * areaMag;
        
        // Add pressure contribution
        fluxVector->momentumFlux[d] += p12 * normal[d] * areaMag;
    }

    // Compute total energy (internal + kinetic)
    PetscReal EL = internalEnergyL;
    PetscReal ER = internalEnergyR;
    for (PetscInt d = 0; d < dim; d++) {
        EL += 0.5 * velocityL[d] * velocityL[d];
        ER += 0.5 * velocityR[d] * velocityR[d];
    }

    // Compute energy flux using split velocities
    fluxVector->energyFlux = (lPlus * rhoL * EL + lMinus * rhoR * ER) * areaMag;

    // Add pressure work term using average velocity
    PetscReal pWork = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        PetscReal vAvg = 0.5 * (velocityL[d] + velocityR[d]);
        pWork += p12 * vAvg * normal[d];
    }
    fluxVector->energyFlux += pWork * areaMag;



    // Compute phase-specific fluxes, and loop through the number of phases
    for (std::size_t k = 0; k < alphakRhokL.size(); k++) {
        fluxVector->alphakRhokFlux[k] = (lPlus * alphakRhokL[k] + lMinus * alphakRhokR[k]) * areaMag;
        fluxVector->alphakFlux[k] = (lPlus * alphakL[k] + lMinus * alphakR[k]) * areaMag;
    }


    return true;
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::fluxCalculator::FluxCalculator, ablate::finiteVolume::fluxCalculator::AusmpUp, "A sequel to AUSM, Part II: AUSM+-up for all speeds, Meng-Sing Liou, Pages 137-170, 2006",
         OPT(double, "mInf", "the reference mach number"),
         OPT(ablate::finiteVolume::processes::PressureGradientScaling, "pgs", "Pressure gradient scaling is used to scale the acoustic propagation speed and increase time step for low speed flows"));