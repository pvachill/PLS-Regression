/**
 * @file pls_impl.hpp
 * @author Pasias Axilleas
 *
 * Implementation of Partial Least Squares (PLS)
 *
 */
#ifndef PLS_IMPL_HPP
#define PLS_IMPL_HPP
// In case it hasn't been included yet.
#include "pls.hpp"

// Constructor1 implemenation 
PLSR::PLSR(const mat & X, const mat & Y, const int comp, const double tolerance ):
	X(X),
	Y(Y),
	components(comp),
	tolerance(tolerance)

{
	if( (patterns = X.n_rows ) != Y.n_rows ) {
		cout << "The number of Predictors (X) does not match the number of Observations (Y)" << endl;
		exit(0);
	}


	varsX = X.n_cols;
	varsY = Y.n_cols;
} // End of Constructor1 implemenation.

// Constructor2 implemenation
PLSR::PLSR( const int comp, const double tolerance ):
	X(Null),
	Y(Null),
	components(comp),
	tolerance(tolerance)
{ 	
	varsX = varsY = patterns = 0; 
	Null = mat(0,0);
} // End of Constructor2 implementation	

// Percentage of variance explained
mat PLSR::VarExp( const mat& X, const mat& Y, int comp)
{
	return 1 - (SSE(X, Y, comp) / TSS(Y));
} // End of VarExp implementation

// Sum of squared errors
rowvec PLSR::SSE( const mat& X, const mat& Y, const int comp)
{
	mat res = Residuals(X, Y, comp); // Calculate the residuals
	rowvec e(Y.n_cols, fill::zeros);

	// For every variable of y calculate the SSE
	for( int i = 0; i< varsY; i++ )
		e.col(i) = res.col(i).t()*res.col(i); // SSE of ith variable.

	return e;
} // End of SSE implemenation

// Total sum of squares
rowvec PLSR::TSS( const mat& Y )
{
	rowvec tss(varsY, fill::zeros); // TSS values

	// For every variable of Y calculate the TSS
	for( int i = 0; i < varsY; i++) 
		tss.col(i) = sum(pow(( Y.col(i) - mean(Y.col(i)) ), 2)); // TSS if ith variable.

	return tss;
} // End of TSS

// Residual space (error)
mat PLSR::Residuals( const mat& X, const mat& Y, int comp )
{	
	// Check the number of components
	if( comp == -1 ) comp = components;
	ComponentCheck(varsX, comp);
	// Return the errors (Residual Space)
	return Y - FittedValues(X, comp);
} // End of Residuals

// Predicted Values
mat PLSR::FittedValues( const mat& X, int comp )
{	
	// Check the number of components
	if( comp == -1 ) comp = components;
	ComponentCheck(varsX, comp);
	// Return the Fitted Values
	return X*Coefficients(comp);
	} // End of FittedValues

// Leave One Out Cross Validation to acquire the Residuals
cube PLSR::LOOCV_Residuals( const mat& X, const mat& Y, const int comp )
{	
	cube res(patterns, varsY, comp); // Residuals
	mat Xtr = X.rows(1, patterns -1); // Trainning Observations 
	mat Ytr = Y.rows(1, patterns -1); // Trainning Predictions

	// Validate the residuals for every pattern and every comp combination
	for( register int i = 0; i < patterns; i++) {
	
		PLSRegression(Xtr, Ytr, comp); // Train
		patterns++; // Xtr has one less pattern and patterns is changed in PLSRegression
					// we adjust the value by adding +1 at every iteration.

		// For every number of components calculate the residuals
		for( register int j = 0; j< comp; j++){
		
			rowvec tempRes = Residuals( X.row(i), Y.row(i), j+1); // Residuals for j number of Components and ith pattern
			res.slice(j).row(i) = tempRes; // Save the residuals
		} // End of Residuals for

		// If cross validation finished continue. 
		if( i < Xtr.n_rows){
			Xtr(i) = X(i);
			Ytr(i) = Y(i);
		}
		
	} // End of cross validation for
	return res;
} // End of LOOCV_Residuals

// Leave One Out Cross Validation to acquire usefull statistics
const cube  PLSR::LOOCV( const mat& X, const mat& Y, int comp )
{
	// Check the number of components
	if( comp == -1 ) comp = components; 
	ComponentCheck(varsX, comp);

	cube res = LOOCV_Residuals(X, Y, comp); // Acquire the Residuals
	cube statistics(comp, varsY, 4, fill::zeros); // Statistics
	mat SSE(comp, varsY,fill::zeros); // Sum of Squared Errors
	mat RMSE(comp, varsY, fill::zeros); // Root Mean Squared Errors
	mat MSE(comp ,varsY, fill::zeros); // Mean Squared Errors
	mat R2(comp, varsY, fill::zeros); // R Squared

	// For every number of components
	for( int i = 0; i < comp; i++)
		// For every variable of Y
		for( int j = 0; j < varsY; j++){
			// Calculate the SSE and R2
			vec temp = res.slice(i).col(j);
			SSE(i,j) = dot(temp, temp); 
			R2(i,j) = 1 - SSE(i,j)/dot(Y.col(j),Y.col(j));
		}
	
	MSE = SSE/patterns;
	RMSE = sqrt(MSE);
	cout<<patterns;

	// Statistics
	statistics.slice(0) = SSE;
	statistics.slice(1) = MSE;
	statistics.slice(2) = RMSE;
	statistics.slice(3) = R2;
	
	return statistics;
} // End of LOOCV

// Check the number of Components
void PLSR::ComponentCheck( const int vars, const int comp)
{	
	// If the number of components are not valid, print a message and exit
	if( comp > varsX || comp <= 0){
		 cout << "Wrong number of components. Check again your values PLS" << endl;
	    exit(0);
	}	
} // End of ComponentCheck

#endif