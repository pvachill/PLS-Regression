/**
 * @file pls1.hpp
 * @author Pasias Axilleas
 *
 * Implementation of Partial Least Squares1 (PLS1)
 *
 */
#ifndef PLS1_HPP
#define PLS1_HPP

#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

/**
* PLS1 is mainly used for the vector case of predictions Y. It estimates
* an orthonormal matrix T (Latent Vectors) without the need of centering of X and Y.
* The deflation of Matrix X (Obsevations) is necessery, but the deflation of Matrix Y (Prediction)
* is not performed.
*/
class PLS1: public PLSR{
 public:
  
  PLS1( const mat& X, const mat& Y, const int comp = 10, const double tolerance = 0.00001 ): PLSR(X, Y, comp, tolerance) {}
  
  PLS1( const int comp = 10, const double tolerance = 0.00001 ): PLSR(comp, tolerance) {}
  
  void PLSRegression( const mat& X, const mat& Y, int comp = -1 );
  
  /**
  * In PLS1 method the Coefficients are computed according to this
  * formula: \f$ BPLS = W(P^TW)^{-1}q\f$ 
  */
  mat Coefficients( const int comp = -1);
};

#include "pls1_impl.hpp"
#endif