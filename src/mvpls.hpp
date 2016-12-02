#ifndef MVPLS_HPP
#define MVPLS_HPP

#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

class MVPLS: public PLSR{
 public:
  MVPLS( const mat& X, const mat& Y, const int comp = 10 ): PLSR(X, Y, comp) {}
  MVPLS(const int comp = 10): PLSR(comp) {}
  void PLSRegression( const mat& X, const mat& Y, int comp = -1 );
  mat Coefficients( const int comp = -1);
};

#include "mvpls_impl.hpp"
#endif