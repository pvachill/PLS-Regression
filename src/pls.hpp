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

  PLSR(const int comp = 10);

  PLSR( const mat& X, const mat& Y, const int comp = 10 );

  void PLSRegression( const mat& X, const mat& Y, int comp = -1 );
  void PLSRegression( const int comp = -1) { PLSRegression(X, Y, comp); };

  void PLS1( const mat& X, const mat& Y, int comp = -1 );
  void PLS1( const int comp = -1) { PLS1(X, Y, comp); };

  cube LOOCV_Residuals( const mat& X, const mat& Y, int comp = -1 );
  cube LOOCV_Residuals( const int comp = -1 ) { return LOOCV_Residuals(X, Y, comp); };

  mat VarExp( const mat& X, const mat& Y, const int comp = -1 );
  mat VarExp( const int comp = -1 ) { return VarExp(X, Y, comp); };

  rowvec TSS( const mat& Y);

  rowvec SSE( const mat& X, const mat& Y, const int comp = -1);
  rowvec SSE( const int comp = -1 ) { return SSE(X, Y, comp); };

  mat Residuals( const mat& x, const mat& Y, int comp = -1);
  mat Residuals( const int comp = -1 ) { return Residuals(X, Y, comp); };

  mat FittedValues( const mat& X, int comp = -1);
  mat FittedValues( const int comp = -1 ) { return FittedValues(X, comp); };

  mat Coefficients( const int comp = -1);

  const cube LOOCV( const mat& X, const mat& Y, int comp = -1 );
  const cube LOOCV( const int comp = -1 ) { return LOOCV(X, Y, comp); };

  void ComponentCheck( const int vars, const int comp);


  mat LatentVec( const int comp = -1 ) const { return T; }

  mat LoadingsX( const int comp = -1 ) const { return P; }

  mat LoadingsY( const int comp = -1 ) const { return Q; }

  mat ScoresY( const int comp = -1  ) const { return U; }

  mat RegressionWeights( const int comp = -1  ) const { return B; }


 private:


  const mat& X = NULL;
  const mat& Y = NULL;
  int components;
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