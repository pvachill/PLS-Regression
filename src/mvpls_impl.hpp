
#ifndef MVPLS_IMPL_HPP
#define MVPLS_IMPL_HPP
// In case it hasn't included yet.
#include "mvpls.hpp"
void MVPLS::PLSRegression(const mat& X, const mat& Y,  int comp ) {
	
	patterns = X.n_rows;
	varsX = X.n_cols;
	varsY = Y.n_cols;

	if( patterns != Y.n_rows ) {
		cout << "The number of Predictors (X) does not match the number of Observations (Y)" << endl;
		exit(0);
	}
	

	if( comp == -1 ) comp = components;
	ComponentCheck(varsX, comp);

    T = zeros<mat>(patterns, varsX);
	P = zeros<mat>(varsX, varsX);
	U = zeros<mat>(patterns, varsX);
	Q = zeros<mat>(varsY, varsX);
	B.zeros(varsX, varsX);

	mat E = X;// We dont want to change X permanently
	mat F = Y;// We dont want to change Y permanently
	vec u(patterns, fill::randu);
	vec t(patterns, fill::zeros);
	vec q(varsY, fill::ones);
	vec p(patterns, fill::zeros);
	vec last_t(patterns, fill::zeros);
	vec w(patterns, fill::zeros);
	
	for( int j = 0; j < comp; j++ ) { 
		// Here we find the PLS components and loadings
		srand(time(NULL));
		u=F.col(rand()%varsY);	
		// Find jth weights according to this method
		do {

		
			last_t =t;
			w = normalise(E.t()*u);
			t = normalise(E*w);
			q = normalise(F.t()*t);
			u = F*q;
		
		} // End of while loop
		while( all(last_t - t > 0.0001) );

		p = E.t()*t;

		B.row(j).col(j) =	t.t()*u;
		E -= t*p.t(); // Remove the effect of jth latent vector from Predictors (X)
		//F -= B(j,j)*u*q.t(); // Remove the effect of jth latent vector from Observations (Y)
		
		// Update PLS components
		T.col(j) = t;
		P.col(j) = p;
		U.col(j) = u;
		Q.col(j) = q;

		if( all(vectorise(E) < 0.00001) ) break;
	} // End of for loop	
	return;
}

mat MVPLS::Coefficients( int comp )
{	
	if( comp == -1 ) comp = components;
	ComponentCheck(varsX, comp);
	mat a = P.t();
	mat Pin = pinv(a);
	return Pin.cols(0,comp-1)*B.rows(0,comp-1).cols(0,comp-1)*Q.cols(0,comp-1).t();
}

#endif