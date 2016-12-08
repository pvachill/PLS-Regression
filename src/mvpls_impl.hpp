/**
 * @file mvpls_impl.hpp
 * @author Pasias Axilleas
 *
 * Implementation of Multivariate Partial Least Squares1 (MVPLS) 
 * Thie implementation is taken from the book 
 * "Encyclopedia for research methods for the social sciences" of Hervé Abdi .
 */
#ifndef MVPLS_IMPL_HPP
#define MVPLS_IMPL_HPP
// In case it hasn't included yet.
#include "mvpls.hpp"

// MVPLS Regression
void MVPLS::PLSRegression(const mat& X, const mat& Y,  int comp ) {
	
	patterns = X.n_rows; // Patterns 
	varsX = X.n_cols; // X variables
	varsY = Y.n_cols; // Y variables

	if( patterns != Y.n_rows ) {
		cout << "The number of Predictors (X) does not match the number of Observations (Y)" << endl;
		exit(0);
	}

	// Check the number of components
	if( comp == -1 ) comp = components;
	ComponentCheck(varsX, comp);

	// Score and Loading vectors used in algorithm
	vec t(patterns, fill::zeros);
	vec q(varsY, fill::zeros);
	vec p(varsX, fill::zeros);
	vec w(varsX, fill::zeros);
	vec u(patterns, fill::randu);
	vec last_t(patterns, fill::zeros);


	// Initialize Loadings and Scores
	U = zeros<mat>(patterns, varsX);
	T = zeros<mat>(patterns, varsX);
	P = zeros<mat>(varsX, varsX);
	Q = zeros<mat>(varsY, varsX);
	W = zeros<mat>(varsX, varsX);
	B = zeros<mat>(varsX, varsX);


	mat E = X;// We dont want to change X permanently
	mat F = Y;// We dont want to change Y permanently
	
	// Find comp number of components
	for( int j = 0; j < comp; j++ ) { 
		// The algorithm stops when comp components found or when E is a null matrix.
		srand(time(NULL));
		u=F.col(rand()%varsY); // Initialization of u with a random variable of Y

		// Find jth weights 
		do {
			// Iterate until t progress is less that the tolerance
			last_t =t; // last score of X
			w = normalise(E.t()*u); // jth weight
			t = normalise(E*w); // jth score of X
			q = normalise(F.t()*t); // fth loading of Y
			u = F*q; // jth score of Y
		} // End of while loop
		while( all(abs(last_t - t) > tolerance) );

		p = E.t()*t; // jth loading of X

		B.row(j).col(j) =	t.t()*u; // jth Regression Weight
		E -= t*p.t();  // Deflation of E
		
		// Update PLS components
		T.col(j) = t;
		P.col(j) = p;
		U.col(j) = u;
		Q.col(j) = q;
		
		// If E become a "null" matrix, stop iteration 
		if( all(vectorise(E) < tolerance) )
			break;
	} // End of for loop	
	return;
} // End of PLSRegression

// Return the coefficinets
mat MVPLS::Coefficients( int comp )
{	// Check the numbe of components	
	if( comp == -1 ) comp = components;
	ComponentCheck(varsX, comp);

	mat a = P.t(); // Transpose Loadings of X
	mat Pin = pinv(a); // Pseudo inverse of a
	return Pin.cols(0,comp-1)*B.rows(0,comp-1).cols(0,comp-1)*Q.cols(0,comp-1).t(); // Coefficients
} // End of Coefficients

#endif