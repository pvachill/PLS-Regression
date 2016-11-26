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
	
	for( int j = 0; j < varsX; j++ ) { 
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

void PLSR::PLS1()
{

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

	for( int i = 0; i < varsX; i++ ) {
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
} // End of PLS1

mat PLSR::VarExp(  int comp)
{
	return 1 - (SSE(X, Y, comp) / TSS(Y, comp));
}

rowvec PLSR::SSE( const mat& X, const mat& Y, const int comp)
{
	mat res = Residuals(X, Y, comp);
	rowvec e(varsY, fill::zeros);

	for( int i = 0; i< varsY; i++ )
		e.col(i) = res.col(i).t()*res.col(i);

	return e;
}

rowvec PLSR::TSS( const mat& Y, const int comp )
{
	rowvec tss(varsY, fill::zeros);

	for( int i = 0; i < varsY; i++) 
		tss.col(i) = sum(pow(( Y.col(i) - mean(Y.col(i)) ), 2));

	return tss;

}

mat PLSR::Residuals( const mat& X, const mat& Y, const int comp )
{	
	return Y - FittedValues(X, comp);
}

mat PLSR::FittedValues( const mat& X, const int comp )
{	
	return X*Coefficients(comp);
	//return U*Q.t();
	}


mat PLSR::Coefficients( const int comp )
{
	//mat a = P.t();
	//mat in = pinv(a);
	//return in.cols(0,comp)*B.rows(0,comp).cols(0,comp)*Q.cols(0,comp).t();
	mat tem = P.t()*W;
	tem = pinv(tem);
	B = W.cols(0,comp)*tem.rows(0,comp).cols(0,comp)*Q.cols(0,comp).t();
	return B;
}

#endif
