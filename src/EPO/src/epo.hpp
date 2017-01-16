#ifndef EPO_HPP
#define EPO_HPP

#include <armadillo>
#include <iostream>

using namespace std;
using namespace arma;

class 	EPO
{
 public:

 	EPO( const mat& X, const mat& Y, const int comp = 10);

 	mat ProjectionEPO( const int comp = -1);
 	mat ProjectionEPO( const mat& X, const mat& Y, const int comp = -1);

 	mat TransformedSpectra( const int comp = -1);
 	mat TransformedSpectra( const mat& X, const int comp = -1);

 	mat DifferenceSpectra();
 	mat DifferenceSpectra( const mat& X, const mat& Y);

 	mat Coefficients(const int comp = -1);
 	
 	mat Projection( const int comp = -1);

 	void ComponentCheck( const int vars, const int comp);
 	


 private:

 	int components;

 	int patterns;

 	int varsX;

    mat coeff;

 	const mat& X;

 	const mat& Y;
};

#include "epo_impl.hpp"
#endif