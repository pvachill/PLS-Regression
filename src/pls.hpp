/**
 * @file pls.hpp
 * @author Pasias Axilleas
 *
 * Implementation of Partial Least Squares (PLS)
 *
 */
#ifndef PLS_HPP
#define PLS_HPP

#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

/**
* The idea of PLS method is to express both the design matrix X and the response
* matrix Y like in Principal Components Analysis (PCA).
* 
* The independent variables are decomposed as \f$X = T*P^T\f$ and \f$Y = U*C^T\f$. 
* Matrices T and U are the score matrices and P and C are the loadings by analogy to PCA.
* Y is estimated as \f$Y' = TBPLS\f$ where B is the regression coefficients.
*/
class PLSR
{
 public:
 /**
 * Construct the PLS regression model and keep the address of the data.
 * Also store the number of Observations (patterns) and the number of
 * variables of the Observations (X) and the Responses (Y).
 * 
 * @param X Matrix that holds the Observations.
 * @param Y Matrix that holds the Responses.
 * @param comp Number of Latent vectors (PLS Components).
 * @param tolerance The tolerance for termination.
 */
  PLSR( const mat& X, const mat& Y, const int comp = 10, const double tolerance = 0.00001 );
  
  /**
  * Construct the PLS regression model without initial values,
  * optionally the  Number of latennt components can be set.
  * 
  * @param comp Number of Latent vectors (PLS Components).
  * @param tolerance The tolerance for termination.
  */ 
  PLSR( const int comp = 10, const double tolerance = 0.00001 );

  /**
  * A virtual function that performs the regression between X and Y.
  *
  * @param X Matrix that holds the Observations.
  * @param Y Matrix that holds the Responses.
  * @param comp Number of Latent vectors (PLS Components).
  */
  virtual void PLSRegression( const mat& X, const mat& Y, int comp = -1 ) =0;

  /**
  * A wrapper for PLSRegression function that optionally needs only  the
  * number of Components. The regression is then implemented with the 
  * default Observations (X) and Predictions (Y). 
  *
  * @param comp Number of Latent vectors (PLS Components).
  */
  void PLSRegression( const int comp = -1 ) { PLSRegression(X, Y, comp); };

  /**
  * Peforms a Leave One Out Cross Validation  and calculates the residuals for
  * evry combination of Components given as input.
  *
  * @param X Matrix that holds the Observations.
  * @param Y Matrix that holds the Responses.
  * @param comp Number of Latent vectors (PLS Components).
  */
  cube LOOCV_Residuals( const mat& X, const mat& Y, int comp = -1 );
  
  /**
  * A wrapper for LOOCV_Residuls function that optionally needs only  the
  * number of Components. The cross-validatin is then implemented with the 
  * default Observations (X) and Predictions (Y). 
  *
  * @param comp Number of Latent vectors (PLS Components).
  */
  cube LOOCV_Residuals( const int comp = -1 ) { return LOOCV_Residuals(X, Y, comp); };
  
  //! Calculates the portion of variance explained using comp number of latent vectors.
  /**
  * The percentage of variance explained is given by this equation: \f$VarExp = 1 - \frac{SSE}{TSS}\f$
  * where SSE is the Sum of Squared Errors and SST is the Total Sum of Squares.
  *
  * @param X Matrix that holds the Observations.
  * @param Y Matrix that holds the Responses.
  * @param comp Number of Latent vectors (PLS Components).
  */
  mat VarExp( const mat& X, const mat& Y, const int comp = -1 );
  
  /**
  * A wrapper for VarExp function that optionally needs only  the
  * number of Components. The Variance explanation is then calculated for the 
  * default Observations (X) and Predictions (Y). 
  *
  * @param comp Number of Latent vectors (PLS Components).
  */  
  mat VarExp( const int comp = -1 ) { return VarExp(X, Y, comp); };

  //! Calculates the Total Sum of Squares.
  /** 
  * The Total Sum of Squares (TSS or SST) calculates the difference of 
  * of the dependent data (Y) and its mean Y': 
  * \f[
  * TSS = \sum_{i = 0}^{n} (Y_i - Y')^2
  *	\f]
  * 
  * @param Y Matrix that holds the Responses.
  */
  rowvec TSS( const mat& Y );

  
  /**
  * The Sum of Squared Error or (Residual Sum of Squares) calculates the
  * difference of the dependtend data (Y) and the predictions Y':
  * \f[
  * SSE = \sum_{i = 0}^{n} (Y - Y')^2
  * \f]
  *  
  * @param X Matrix that holds the Observations.
  * @param Y Matrix that holds the Responses.
  * @param comp Number of Latent vectors (PLS Components).
  */
  rowvec SSE( const mat& X, const mat& Y, const int comp = -1 );
  
  /**
  * A wrapper for SSE function that optionally needs only  the
  * number of Components. The Sum of squared errors are then calculated for the 
  * default Observations (X) and Predictions (Y). 
  *
  * @param comp Number of Latent vectors (PLS Components).
  */
  rowvec SSE( const int comp = -1 ) { return SSE(X, Y, comp); };
 
  /**
  * \brief Calculates the residuals
  *
  * @param X Matrix that holds the Observations.
  * @param Y Matrix that holds the Responses.
  * @param comp Number of Latent vectors (PLS Components).
  */
  mat Residuals( const mat& X, const mat& Y, int comp = -1 );
  
  /**
  * A wrapper for Residuals function that optionally needs only  the
  * number of Components. The Residuals are then calculated for the 
  * default Observations (X) and Predictions (Y). 
  *
  * @param comp Number of Latent vectors (PLS Components).
  */
  mat Residuals( const int comp = -1 ) { return Residuals(X, Y, comp); };

  /**
  * /brief Predicts the dependent variables.
  * The function predicts the dependent values with respect to the number of 
  * components and Observations given. The model should have been trained at least once,
  * else a logical error will occur.
  *
  * @param X Matrix that holds the Observations.
  * @param Y Matrix that holds the Responses.
  * @param comp Number of Latent vectors (PLS Components).
  */
  mat FittedValues( const mat& X, int comp = -1 );

  /**
  * A wrapper for FittedValues function that optionally needs only  the
  * number of Components. The predictions are then calculated for the 
  * default Observations (X). 
  *
  * @param comp Number of Latent vectors (PLS Components).
  */
  mat FittedValues( const int comp = -1 ) { return FittedValues(X, comp); };

  /**
  * Returns the Coefficients  with respect to the number of components.
  * 
  * @param comp Number of Latent vectors (PLS Components).
  */
  virtual mat Coefficients( const int comp = -1 ) =0;

  /**
  * Performs a Leave One Out Cross Validations and calculates some statistics,
  * for every combination of components given as input. 
  * The statistics are:
  * slice(0) - SSE (Sum of Squared Errors)
  * slice(1) - MSE (Mean Squared Errors)
  * slice(2) - RMSE (Root Mean Squared Errors)
  * slice(3) - R2 (R Squared)
  *
  * @param X Matrix that holds the Observations.
  * @param Y Matrix that holds the Responses.
  * @param comp Number of Latent vectors (PLS Components).
  */
  const cube LOOCV( const mat& X, const mat& Y, int comp = -1 );
  
  /**
  * A wrapper for LOOCV function that optionally needs only  the
  * number of Components. The Cross Validation is then implemented for the 
  * default Observations (X) and Predictions (Y). 
  *
  * @param comp Number of Latent vectors (PLS Components).
  */
  const cube LOOCV( const int comp = -1 ) { return LOOCV(X, Y, comp); };

  //! Checks if the number of Components given is valid.
  void ComponentCheck( const int vars, const int comp );

  //! Get the tolerance for termination
  double Tolerance() const { return tolerance; };
 
  //! Modify the tolerance for termination
  double& Tolerance() { return tolerance; };

  //! Get the first comp Scores (T) of Observations (X).
  mat LatentVec(  int comp = -1 ) { 
  	if( comp == -1) comp = components;
  	ComponentCheck(varsX, comp);
  	return T.cols(0,comp); }

  //! Get the the first comp Loadings (P) of Observations (X).  
  mat LoadingsX( int comp = -1 ) { 
  	if( comp == -1) comp = components;
  	ComponentCheck(varsX, comp);
  	return P.rows(0,comp); }

  // Get the first comp Loadings (Q) of Predictions (Y).
  mat LoadingsY( int comp = -1 ) { 
  	if( comp == -1) comp = components;
  	ComponentCheck(varsX, comp);
  	return Q.rows(0,comp); }

  //! Get the Score Matrix (U) of Predictions (Y).
  mat ScoresY(  int comp = -1  ) { 
  	if( comp == -1) comp = components;
  	ComponentCheck(varsX, comp);
  	return U.cols(0,comp); }

 protected:

  //! Used to initialize Observations and Predictions.
  mat Null;

  //! Observations.
  const mat& X;

  //! Predictions.
  const mat& Y;
  
  //! Number of components to use (By default is 10)
  int components;

  //! The latent vectors or score matrix of X.
  mat T;

  //! The loadings of X.
  mat P;

  //! The score matrix of Y.
  mat U;

  //! The weight matrix or the loadings of Y.
  mat Q;

  //! Weights
  mat W;

  //! The number of patterns (data)
  int patterns;

  //! The numver of X-variables
  int varsX;

  //! The number of Y-variables
  int varsY; 

  //! The tolerance for terminationx
  double tolerance;
}; // End of class definition

#include "pls_impl.hpp"
#include "pls1.hpp"
#include "mvpls.hpp"
#endif