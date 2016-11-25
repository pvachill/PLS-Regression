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

  PLSR( const mat & X, const mat & Y);

  void PLSRegression();

  mat VarExp(  int n );

  const mat &  Predictors() const { return X; }

  const mat &  Observations() const { return Y; }  

  mat LatentVec() const { return T; }

  mat LoadingsX() const { return P; }

  mat LoadingsY() const { return Q; }

  mat ScoresY() const { return U; }

  mat RegressionWeights() const { return B; }


 private:
  //! The predictors matrix.
  const mat & X;

  //! The observations matrix.
  const mat & Y;

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

  //! The number of patterns (data)
  
  int patterns;

  //! The numver of X-variables
  int varsX;

  //! The number of Y-variables
  int varsY; 
}; // End of class definition

#include "pls_impl.hpp"
#endif