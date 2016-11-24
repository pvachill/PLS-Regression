#ifndef PLS_IMPL_HPP
#define PLS_IMPL_HPP
// In case it hasn't included yet.
#include "pls.hpp"

// 	Constructor implemenation 
PLSR::PLSR(const mat & X, const mat & Y):
	X(X),
	Y(Y)
{
	if( (patterns = X.n_rows ) != Y.n_rows ) {
		cout << "The number of Predictors (X) does not match the number of Observations (Y)" << endl;
		exit(0);
	}

	varsX = X.n_cols;
	varsY = Y.n_cols;
} // End of constructor.

void PLSR::PLSRegression() {


	vec u(patterns, fill::zeros);
	vec t(patterns, fill::zeros);
	vec q(varsY, fill::zeros);
	vec p(10, fill::zeros);
	vec change(patterns);
	
	change.fill(0.001); //tolerance
	u = Y.col(0);
	
	vec last_t(patterns, fill::zeros);
	// Find jth weights according to this method
	do {
		last_t = t;
		p = X.t()*u;
		t = X*p;
		q = Y.t()*t;
		u = Y*q;
	}
	while( sum(arma::abs(last_t -t) > change) );
	

	return;
}

#endif
