#ifndef PLS_IMPL_HPP
#define PLS_IMPL_HPP
// In case it hasn't included yet.
#include "pls.hpp"

// 	Constructor implemenation 
PLSR::PLSR(const mat & X, const mat & Y, const int comp ):
	X(X),
	Y(Y),
	components(comp)

{
	if( (patterns = X.n_rows ) != Y.n_rows ) {
		cout << "The number of Predictors (X) does not match the number of Observations (Y)" << endl;
		exit(0);
	}


	varsX = X.n_cols;
	varsY = Y.n_cols;
} // End of constructor.

PLSR::PLSR( const int comp ):
	X(Null),
	Y(Null),
	components(comp)
{ 	varsX = varsY = patterns = 0; 
	Null = mat(0,0);
}	

/*void PLSR::PLSRegression(const mat& X, const mat& Y,  int comp ) {
	
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
}*/

void PLSR::PLS1( const mat& X, const mat& Y, int comp )
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
} // End of PLS1

// Percentage of variance explained
mat PLSR::VarExp( const mat& X, const mat& Y, int comp)
{
	return 1 - (SSE(X, Y, comp) / TSS(Y));
}

// Sum of squared errors
rowvec PLSR::SSE( const mat& X, const mat& Y, const int comp)
{
	mat res = Residuals(X, Y, comp);
	rowvec e(Y.n_cols, fill::zeros);

	for( int i = 0; i< varsY; i++ )
		e.col(i) = res.col(i).t()*res.col(i);

	return e;
}

// Total sum of squares
rowvec PLSR::TSS( const mat& Y )
{
	rowvec tss(varsY, fill::zeros);

	for( int i = 0; i < varsY; i++) 
		tss.col(i) = sum(pow(( Y.col(i) - mean(Y.col(i)) ), 2));

	return tss;

}

// Residual space (error)
mat PLSR::Residuals( const mat& X, const mat& Y, int comp )
{	
	if( comp == -1 ) comp = components;
	ComponentCheck(varsX, comp);
	return Y - FittedValues(X, comp);
}

// Predicted Values
mat PLSR::FittedValues( const mat& X, int comp )
{	
	if( comp == -1 ) comp = components;
	ComponentCheck(varsX, comp);
	return X*Coefficients(comp);
	//return U*Q.t();
	}

/*// Compute coefficients (BPLS) 	
mat PLSR::Coefficients( int comp )
{	
	if( comp == -1 ) comp = components;
	ComponentCheck(varsX, comp);
	mat a = P.t();
	mat in = pinv(a);
	//a.print();
	return in.cols(0,comp-1)*B.rows(0,comp-1).cols(0,comp-1)*Q.cols(0,comp-1).t();
	//mat tem = P.t()*W;
	//tem = pinv(tem);
	//B = W.cols(0,comp)*tem.rows(0,comp).cols(0,comp)*Q.cols(0,comp).t();
	//return B;
}*/

cube PLSR::LOOCV_Residuals( const mat& X, const mat& Y, const int comp )
{	
	


	cube res(patterns, varsY, comp);
	mat Xtr = X.rows(1, patterns -1);
	mat Ytr = Y.rows(1, patterns -1);

	//PLSR cvModel(Xtr, Ytr, comp);

	for( register int i = 0; i < patterns; i++) {
		PLSRegression(Xtr, Ytr, comp);
		for( register int j = 0; j< comp; j++){
			rowvec tempRes = Residuals( X.row(i), Y.row(i), j+1);
			res.slice(j).row(i) = tempRes;
		} // End of Residuals for

		if( i < Xtr.n_rows){
			Xtr(i) = X(i);
			Ytr(i) = Y(i);
		}

	} // End of cross validation for
	return res;
} // End of leave one out cross validation function

const cube  PLSR::LOOCV( const mat& X, const mat& Y, int comp )
{
	if( comp == -1 ) comp = components;
	ComponentCheck(varsX, comp);

	cube res = LOOCV_Residuals(X, Y, comp);
	cube statistics(comp, varsY, 4, fill::zeros);
	mat SSE(comp, varsY,fill::zeros);
	mat RMSE(comp, varsY, fill::zeros);
	mat MSE(comp ,varsY, fill::zeros);
	mat R2(comp, varsY, fill::zeros);

	for( int i = 0; i < comp; i++)
		for( int j = 0; j < varsY; j++){
			vec temp = res.slice(i).col(j);
			SSE(i,j) = dot(temp, temp);
			vec temp1 = Y.col(j)-temp;
			R2(i,j) = 1 - SSE(i,j)/dot(temp1,temp1);
		}
	
	MSE = SSE/patterns;
	RMSE = sqrt(MSE);

	statistics.slice(0) = SSE;
	statistics.slice(1) = MSE;
	statistics.slice(2) = RMSE;
	statistics.slice(3) = R2;
	
	return statistics;
	}

void PLSR::ComponentCheck( const int vars, const int comp)
{
	if( comp > varsX || comp <= 0){
		 cout << "Wrong number of components. Check again your values PLS" << endl;
	    exit(0);
	}	
}



#endif

