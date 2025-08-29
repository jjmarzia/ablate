target_sources(ablateLibrary
        PRIVATE
        # auxFieldInterpolant.cpp 
        flowProcess.cpp
        navierStokesTransport.cpp
        speciesTransport.cpp
        evTransport.cpp
        twoPhaseEulerAdvection.cpp
        nPhaseAllaireAdvection.cpp
        nPhaseSurfaceTension.cpp
        nPhaseNonconservativeRHS.cpp
        nPhaseIntSharp.cpp
        # intSharp.cpp
        issf.cpp
        locations.cpp
        gravity.cpp
        pressureGradientScaling.cpp
        arbitrarySource.cpp
        buoyancy.cpp
        constantPressureFix.cpp
        les.cpp
        chemistry.cpp
        radiationLoss.cpp
        thermophoreticDiffusion.cpp
        # surfaceForce.cpp
        soot.cpp
        compactCompressibleNSSpeciesTransport.cpp
        compactCompressibleNSSpeciesSingleProgressTransport.cpp
        locations.cpp

        PUBLIC
        process.hpp
        # auxFieldInterpolant.hpp 
        flowProcess.hpp
        navierStokesTransport.hpp
        speciesTransport.hpp
        evTransport.hpp
        twoPhaseEulerAdvection.hpp
        nPhaseAllaireAdvection.hpp
        nPhaseSurfaceTension.hpp
        nPhaseNonconservativeRHS.hpp
        nPhaseIntSharp.hpp
        # intSharp.hpp
        issf.hpp
        locations.hpp
        gravity.hpp
        pressureGradientScaling.hpp
        arbitrarySource.hpp
        buoyancy.hpp
        constantPressureFix.hpp
        les.hpp
        chemistry.hpp
        radiationLoss.hpp
        thermophoreticDiffusion.hpp
        # surfaceForce.hpp
        compactCompressibleNSSpeciesTransport.hpp
        compactCompressibleNSSpeciesSingleProgressTransport.hpp
        soot.hpp
        locations.hpp
        )
