/* This class contains the variables and the methods for a complete PLS Regression */

#ifndef PLS_H
#include PLS_H

#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

// start of class definition
class PLSR
{
 public:

  PLSR(mat X, mat Y);

  void PLSRegression();

  mat Predictors() const { return X; }

  mat Observations() const { return Y; }

  mat LatentVec() const { return T; }

  mat LoadingsX() const { return P; }

  mat LoadingsY() const { return C; }

  mat ScoreY() const { return U; }

  mat RegressionWeights() const { return B; }


 private:
  //! The predictors matrix.
  mat X;

  //! The observations matrix.
  mat Y;

  //! The latent vectors or score matrix of X.
  mat T;

  //! The loadings of X.
  mat P;

  //! The score matrix of Y.
  mat U;

  //! The regression weights of Y.
  mat B;

  //! The weight matrix or the loadings of Y.
  mat C;
}; // End of class definition