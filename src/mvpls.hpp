/**
 * @file pls1.hpp
 * @author Pasias Axilleas
 *
 * Implementation of Multivariate Partial Least Squares1 (MVPLS) 
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
  MVPLS( const mat& X, const mat& Y, const int comp = 10 ): PLSR(X, Y, comp) {}

  MVPLS( const int comp = 10 ): PLSR(comp) {}
  
  void PLSRegression( const mat& X, const mat& Y, int comp = -1 );
  
  /**
  * In MVPLS (Multivariate PLS) method, the Coefficients are computed
  * according to this formula: \f$ BPLS = P^{-1}*B*Q^T\f$.
  */
  mat Coefficients( const int comp = -1);
};

#include "mvpls_impl.hpp"
#endif