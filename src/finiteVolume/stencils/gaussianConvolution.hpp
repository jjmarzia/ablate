#ifndef ABLATELIBRARY_GAUSSIANCONVOLUTION_HPP
#define ABLATELIBRARY_GAUSSIANCONVOLUTION_HPP

#include "domain/RBF/ga.hpp"
#include "domain/RBF/hybrid.hpp"
#include "domain/RBF/imq.hpp"
#include "domain/RBF/mq.hpp"
#include "domain/RBF/intMQ.hpp"
#include "domain/RBF/phs.hpp"
#include "domain/RBF/rbf.hpp"
#include "domain/subDomain.hpp"
#include "utilities/petscUtilities.hpp"



#include "domain/range.hpp"
#include "domain/reverseRange.hpp"

// Cell-based gaussian convolution

namespace ablate::finiteVolume::stencil {

  class GaussianConvolution {

    private:

      PetscInt cStart;
      PetscInt cEnd;

      void BuildList(const PetscInt p);

      // Number of points in the 1D quadrature
      PetscInt nLayers = -1;

      // The weights for each point
      PetscReal *weights = nullptr;

      // The standard deviation distance
      PetscReal sigma = 1.0;

      // Used for cell-centers
      Vec cellGeomVec = nullptr;

      // List of cells necessary to do the integration.
      PetscInt **cellList = nullptr;
      PetscInt *nCellList = nullptr;

      // Weights of each cell
      PetscReal **cellWeights = nullptr;

      DM geomDM = nullptr;

       // Used for periodicity
      PetscReal maxDist[3] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
      PetscReal sideLen[3] = {0, 0, 0};


    public:
      void Evaluate(DM dm, const PetscInt p, const PetscInt fid, const PetscScalar *array, PetscInt offset, const PetscInt nDof, PetscReal vals[]);
      void Evaluate(DM dm, const PetscInt p, const PetscInt fid, Vec fVec, PetscInt offset, const PetscInt nDof, PetscReal vals[]);

      PetscInt GetCellList(const PetscInt p, const PetscInt **cellListOut);

      void FormAllLists();
      GaussianConvolution(DM geomDM, const PetscInt nLayers, const PetscInt sigmaFactor);


      ~GaussianConvolution();

  };



}  // namespace ablate::levelSet
#endif  // ABLATELIBRARY_GAUSSIANCONVOLUTION_HPP
