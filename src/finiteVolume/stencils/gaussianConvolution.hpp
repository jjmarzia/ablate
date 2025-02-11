#ifndef ABLATELIBRARY_GAUSSIANCONVOLUTION_HPP
#define ABLATELIBRARY_GAUSSIANCONVOLUTION_HPP

#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"

// Cell-based gaussian convolution
namespace ablate::finiteVolume::stencil {

  class GaussianConvolution {

    private:

      PetscInt rangeStart;
      PetscInt rangeEnd;

      void BuildList(const PetscInt p);

      // The weights for each point
      PetscReal *weights = nullptr;

      // The standard deviation distance
      PetscReal sigma = 1.0;

      // Used for cell-centers
      Vec cellGeomVec = nullptr;

      // List of cells necessary to do the integration.
      PetscInt **cellList = nullptr;
      PetscInt *nCellList = nullptr;
      PetscReal **cellDist = nullptr;

      // Weights of each cell
      PetscReal **cellWeights = nullptr;

      DM geomDM = nullptr;

       // Used for periodicity
      PetscReal maxDist[3] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
      PetscReal sideLen[3] = {0, 0, 0};


    public:
      void Evaluate(const PetscInt p, const PetscInt dx[], DM dataDM, const PetscInt fid, const PetscScalar *array, PetscInt offset, const PetscInt nDof, PetscReal *vals);
      void Evaluate(const PetscInt p, const PetscInt dx[], DM dataDM, const PetscInt fid, Vec fVec, PetscInt offset, const PetscInt nDof, PetscReal *vals);

      PetscInt GetCellList(const PetscInt p, const PetscInt **cellListOut);

      void FormAllLists();

      typedef enum { DEPTH, HEIGHT } DepthOrHeight;

      GaussianConvolution(DM geomDM, const PetscInt sigmaFactor, const PetscInt loc, DepthOrHeight doh);



      ~GaussianConvolution();

  };



}  // namespace ablate::levelSet
#endif  // ABLATELIBRARY_GAUSSIANCONVOLUTION_HPP
