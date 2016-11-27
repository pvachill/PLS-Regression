/* This class contains the variables and the methods for a complete PLS Regression */

#ifndef PLS_HPP
#define PLS_HPP

#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

// start of class definition
class PLSR
{
 public:

  PLSR( const int patterns, const int varsX, const int varsY ){;};

  PLSR( const mat& X, const mat& Y);

  void PLSRegression( const mat& X, const mat& Y, const int comp );

  void PLS1( const mat& X, const mat& Y, const int comp );

  cube LOOCV_Residuals( const mat& X, const mat& Y, const int comp );

  mat VarExp( const mat& X, const mat& Y, int comp );

  rowvec TSS( const mat& Y, const int comp );

  rowvec SSE( const mat& X, const mat& Y, const int comp);

  mat Residuals( const mat& x, const mat& Y, const int comp);

  mat FittedValues( const mat& X, const int comp);

  mat Coefficients( const int comp);

  const cube LOOCV( const mat& X, const mat& Y, const int comp );

  mat LatentVec() const { return T; }

  mat LoadingsX() const { return P; }

  mat LoadingsY() const { return Q; }

  mat ScoresY() const { return U; }

  mat RegressionWeights() const { return B; }


 private:
  //! The latent vectors or score matrix of X.
  mat T;

  //! The loadings of X.
  mat P;

  //! The score matrix of Y.
  mat U;

  //! The regression weights of Y.
  mat B;

  //! The weight matrix or the loadings of Y.
  mat Q;

  //! something from pls1
  mat W;

  //! eror factor?
  mat B0;

  //! The number of patterns (data)
  int patterns;

  //! The numver of X-variables
  int varsX;

  //! The number of Y-variables
  int varsY; 
}; // End of class definition

#include "pls_impl.hpp"
#endif