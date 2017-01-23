#ifndef EPO_IMPL_HPP
#define EPO_IMPL_HPP

#include "epo.hpp"

EPO::EPO( const mat& X, const mat& Y, const int comp ):
	components(comp),
	X(X),
	Y(Y)
{
	if( (patterns = X.n_rows ) != Y.n_rows ) {
		cout << "The number of Predictors (X) does not match the number of Observations (Y)" << endl;
	//	exit(0);
	}

	varsX = X.n_cols;
} // End of constructor

mat EPO::ProjectionEPO( const mat& X, const mat& Y, int comp )
{	

	varsX = X.n_cols;
	if( comp = -1 ) comp = components;
//	ComponentCheck(varsX, comp);

	mat D = X-Y;
	coeff = princomp(D.t()*D);
	return Projection(comp);
}

mat EPO::ProjectionEPO( const int comp )
{
	return ProjectionEPO(X, Y, comp);
}

mat EPO::TransformedSpectra( const mat& X, const int comp )
{
	return X*Projection(comp);
}

mat EPO::TransformedSpectra( const int comp )
{
	return TransformedSpectra(X, comp);
}

mat EPO::Projection( const int comp )
{
	mat temp = Coefficients(comp);
	return eye(varsX, varsX) - temp*temp.t();
}


mat EPO::Coefficients(const int comp )
{
//	ComponentCheck(varsX, comp);
	return coeff.cols(0,comp-1);
}

void EPO::ComponentCheck( const int vars, const int comp)
{
	if( comp >= vars || comp <= 0){
		 cout << "Wrong number of components. Check again your values" << endl;
	    exit(0);
	}	
}

#endif

