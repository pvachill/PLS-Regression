/**
 * @file pls1_impl.hpp
 * @author Pasias Axilleas
 *
 * Implementation of Partial Least Squares1 (PLS1)
 *
 */
#ifndef PLS1_IMPL_HPP
#define PLS1_IMPL_HPP
// In case it hasn't been included yet.
#include "pls1.hpp"

// PLS1 Regression
void PLS1::PLSRegression(const mat& X, const mat& Y,  int comp )
{
	patterns = X.n_rows; // Patterns 
	varsX = X.n_cols; // X variables
	varsY = 1; // Y variables

	if( patterns != Y.n_rows ) {
		cout << "The number of Predictors (X) does not match the number of Observations (Y)" << endl;
		exit(0);
	}

	if( Y.n_cols != 1) {
		cout << "PLS1 works with the case when  Y is a vector" << endl;
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

	// Initialize Loadings and Scores
	U = zeros<mat>(patterns, varsX);
	T = zeros<mat>(patterns, varsX);
	P = zeros<mat>(varsX, varsX);
	Q = zeros<mat>(varsY, varsX);
	W = zeros<mat>(varsX, varsX);

	
	mat E = X; // We dont want to change X permanently
	mat F = Y; // We dont want to change Y permanently
	

	// Beggining of the algorithm
	w = normalise(X.t()*Y); // Initial estimate of w
	t = X*w; // First score of X

	// Find comp number of components
	for( int i = 0; i < comp; i++ ) {
		// The algorithm stops when comp components found or when E is a null matrix.
		double tk = conv_to<double>::from(t.t()*t); // Squared L2 norm of t
		t /=tk; // divide t by tk
		p = E.t()*t; // jth loading of X
		q = Y.t()*t; // jth loading of Y
		
		if( norm(q) == 0)break;
		// Update PLS components
		W.col(i) = w;
		P.col(i) = p;
		Q.col(i) = q;
		T.col(i) = t;

		E -= tk*t*p.t(); // Deflation of E
		// If E become a "null" matrix, stop iteration 
		if( all(vectorise(E) < tolerance) )
			break;
	
		w = E.t()*Y; // jth + weight
		t = E*w; // jth +1 score of X
	} // End of regression loop
} // End of PLSRegression

// Returns the coefficients
mat PLS1::Coefficients( int comp )
{	// Check the number of components
	if( comp == -1 ) comp = components;
	ComponentCheck(varsX, comp);
	
	mat tem = P.t()*W; 
	tem= pinv(tem); // pseudo inverse of temp
	return W.cols(0,comp-1)*tem.rows(0,comp-1).cols(0,comp-1)*Q.cols(0,comp-1).t(); // Coefficients
} // End of Coefficients

#endif