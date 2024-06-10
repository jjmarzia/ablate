target_sources(ablateLibrary
        PRIVATE
        flowProcess.cpp
        navierStokesTransport.cpp
        speciesTransport.cpp
        evTransport.cpp
        twoPhaseEulerAdvection.cpp
        gravity.cpp
        pressureGradientScaling.cpp
        arbitrarySource.cpp
        buoyancy.cpp
        constantPressureFix.cpp
        les.cpp
        chemistry.cpp
        radiationLoss.cpp
        thermophoreticDiffusion.cpp
        surfaceForce.cpp
        soot.cpp
        intSharp.cpp
        #        duplicate symbol 'GetVertexRange(_p_DM*, std::__1::shared_ptr<ablate::domain::Region> const&, ablate::domain::Range&)' in:
        #        CMakeFiles/ablateLibrary.dir/src/finiteVolume/processes/surfaceForce.cpp.o
        #        CMakeFiles/ablateLibrary.dir/src/finiteVolume/processes/intSharp.cpp.o
        #        ld: 1 duplicate symbol for architecture arm64

        PUBLIC
        process.hpp
        flowProcess.hpp
        navierStokesTransport.hpp
        speciesTransport.hpp
        evTransport.hpp
        twoPhaseEulerAdvection.hpp
        gravity.hpp
        pressureGradientScaling.hpp
        arbitrarySource.hpp
        buoyancy.hpp
        constantPressureFix.hpp
        les.hpp
        chemistry.hpp
        radiationLoss.hpp
        thermophoreticDiffusion.hpp
        surfaceForce.hpp
        soot.hpp
        intSharp.hpp

        )
