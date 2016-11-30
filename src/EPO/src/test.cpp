#include "epo.hpp"
#include <iostream>
#include "../../PLS-Regression/src/pls.hpp"

using namespace arma;
using namespace std;

int main( int argc, char *argv[] ) {


	mat s30, s40, s50, s60, s70, cont;
	s30.load("dataset/spec30.csv", csv_ascii);	
	s40.load("dataset/spec40.csv", csv_ascii);	
    s50.load("dataset/spec50.csv", csv_ascii);	
    s60.load("dataset/spec60.csv", csv_ascii);	
    s70.load("dataset/spec70.csv", csv_ascii);
    cont.load("dataset/conc.csv", csv_ascii);


    mat X;
    X = s40 +s60;
    X = X/2;


    EPO epo(X, s30,20);
    epo.ProjectionEPO(X,s30,20);

    //mat x1 = epo.TransformedSpectra(s30.rows(0,15),3);
    //mat x2 = epo.TransformedSpectra(s30.rows(16,21),3);
   
    //PLSR model(s30, cont.col(1));
   // model.PLSRegression(s30, cont.col(1), 511);
  // model.Residuals(s50,cont.col(1),10).print();
    cout<<endl;

   s30 = epo.TransformedSpectra(s30,10);
    
    PLSR model(s30.rows(0,15), cont.col(1).rows(0,15));
    model.PLSRegression(s30.rows(0,15), cont.rows(0,15).col(1), 511);

   // model.Residuals(s60,cont.col(1),10).print();
    cout<<endl;

    mat pre = model.FittedValues(s60, 15);

    pre.print();
    cout<<endl;
    cont.col(1).print();
    cout<<endl;


    s70= epo.TransformedSpectra(s70,20);
    for( int i = 0; i<30; i++)
	 	model.VarExp(s70.rows(16,21), cont.rows(16,21).col(1), i).print();

	//model.Residuals(s40, cont.col(0), 7).print();

	return 0;

}