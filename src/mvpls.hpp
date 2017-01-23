/**
 * @file mvpls.hpp
 * @author Pasias Axilleas
 *
 * Implementation of Multivariate Partial Least Squares (MVPLS) 
 * Thie implementation is taken from the book 
 * "Encyclopedia for research methods for the social sciences" of Hervé Abdi .
 */
#ifndef MVPLS_HPP
#define MVPLS_HPP

#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

/**
* MVPLS is a simple method that performs a multivariate regression. 
* MVPLS estimates an orthogonal matrix T (Latent vectors)  and
* an orthogonal matrix 	U (Scores of Y). It also estimates the Loadings
* of X and Y , P and Q respectively.
*/
class MVPLS: public PLSR{
 public:
  MVPLS( const mat& X, const mat& Y, const int comp = 10, const double tolerance = 0.00001 ): PLSR(X, Y, comp, tolerance) {}

  MVPLS( const int comp = 10, const double tolerance = 0.00001 ): PLSR(comp, tolerance) {}
  
  void PLSRegression( const mat& X, const mat& Y, int comp = -1 );
  
  /**
  * In MVPLS (Multivariate PLS) method, the Coefficients are computed
  * according to this formula: \f$ BPLS = P^{-1}*B*Q^T\f$.
  */
  mat Coefficients( const int comp = -1);

  //! Get the first comp Regression Weights (B)
  mat RegressionWeights( int comp = -1  ) { 
    if( comp == -1) comp = components;
  	//ComponentCheck(varsX, comp);
  	return B.cols(0,comp).rows(0,comp); }

 private:
  //! Regression Weights
  mat B;
};

#include "mvpls_impl.hpp"
#endif