#ifndef PLS1_HPP
#define PLS1_HPP

#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

class PLS1: public PLSR{
 public:
  PLS1( const mat& X, const mat& Y, const int comp = 10 ): PLSR(X, Y, comp) {}
  PLS1(const int comp = 10): PLSR(comp) {}
  void PLSRegression( const mat& X, const mat& Y, int comp = -1 );
  mat Coefficients( const int comp = -1);
};

#include "pls1_impl.hpp"
#endif