
#ifndef PLS1_IMPL_HPP
#define PLS1_IMPL_HPP
// In case it hasn't included yet.
#include "pls1.hpp"
void PLS1::PLSRegression(const mat& X, const mat& Y,  int comp )
{
	patterns = X.n_rows;
	varsX = X.n_cols;
	varsY = Y.n_cols;

	if( patterns != Y.n_rows ) {
		cout << "The number of Predictors (X) does not match the number of Observations (Y)" << endl;
		exit(0);
	}


	if( comp == -1 ) comp = components;
	ComponentCheck(varsX, comp);

	vec t(patterns, fill::zeros);
	vec q(varsY, fill::zeros);
	vec p(varsX, fill::zeros);
	vec w(varsX, fill::zeros);

	T = zeros<mat>(patterns, varsX);
	P = zeros<mat>(varsX, varsX);
	Q = zeros<mat>(varsY, varsX);
	W = zeros<mat>(varsX, varsX);
	
	mat E = X; // We dont want to change X permanently
	mat F = Y; // We dont want to change Y permanently
	
	w = normalise(X.t()*Y);
	t = X*w;

	for( int i = 0; i < comp; i++ ) {
		double tk = conv_to<double>::from(t.t()*t);
		t /= tk;
		p = E.t()*t;
		q = Y.t()*t;
		
		W.col(i) = w;
		P.col(i) = p;
		Q.col(i) = q;
		T.col(i) = t;
		if( norm(q) == 0 ) 
			break;

		E -= tk*t*p.t();
		w = E.t()*Y;
		t = E*w;
		 
		if( all(vectorise(E) < 0.00001) )
			break;
	} // End of for loop

}

mat PLS1::Coefficients( int comp )
{	
	if( comp == -1 ) comp = components;
	ComponentCheck(varsX, comp);
	mat tem = P.t()*W;
	tem = pinv(tem);
	B = W.cols(0,comp-1)*tem.rows(0,comp-1).cols(0,comp-1)*Q.cols(0,comp-1).t();
	return B;
}

#endif