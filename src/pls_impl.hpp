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


    T = randu<mat>(patterns, varsX);
	P = randu<mat>(varsX, varsX);
	U = randu<mat>(patterns, varsX);
	Q = randu<mat>(varsY, varsX);
	B.eye(varsX, varsX);

	mat E = X; // We dont want to change X permanently
	mat F = Y; // We dont want to change Y permanently
	vec u(patterns, fill::randu);
	vec t(patterns, fill::randu);
	vec q(varsY, fill::ones);
	vec p(patterns, fill::randu);
	vec last_t(patterns, fill::zeros);
	vec change(patterns);
	
	change.fill(0.000001); //tolerance
	
	for( int j = 0; j < varsX; j++ ) { 
	// Here we find the PLS components and loadings
	u = F.col(0	);
	srand(time(NULL));
	//t = E.col(rand()%X.n_cols);	
	int i = 0;

	// Find jth weights according to this method
	do {

		i++;
		last_t =t;
		p = E.t()*u;
		if( norm(p) )
		p = p/norm(p);
		t = E*p;
		//q = F.t()*t;
		if(norm(q))
		q=q/norm(q);
		u = F*q;
		
	} // End of while loop
	while( sum(arma::abs(last_t -t) > change) );


	E -= t*p.t(); // Remove the effect of jth latent vector from Predictors (X)
	F -= u*q.t(); // Remove the effect of jth latent vector from Observations (Y)
	
	// Update PLS components
	T.col(j) = t;
	P.col(j) = p;
	U.col(j) = u;
	Q.col(j) = q;
	//B(j,j)= conv_to<double>::from(t.t()*u);
	//B.print();
	} // End of for loop
	// Calculate Regression weights
	//T.print();
	//B = pinv(T)*U;
	B = solve(T,U);  
	B.print();
	cout<<endl;
	U.print();
	cout << endl;
	(T*B).print();
	cout << endl;
	Q.print();
	cout<<endl<<endl<<"\n";

	return;
}

mat PLSR::VarExp(  int n )
{
	
	mat totalVar = var(Y);

	mat predictions = T*B*Q.t();//T.cols(0,n)*B.rows(0,n)*Q.t();
	mat residualVar = var(Y- predictions);		
	mat explaindVar = totalVar - residualVar;


	explaindVar/=totalVar;	
	return explaindVar;
}

#endif
