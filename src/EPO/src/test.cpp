#include "epo.hpp"
#include <iostream>
#include "../../pls.hpp"

using namespace arma;
using namespace std;

int main( int argc, char *argv[] ) {

    wall_clock a;
	mat s30, s40, s50, s60, s70, cont;
	s30.load("dataset/spec30.csv", csv_ascii);	
	s40.load("dataset/spec40.csv", csv_ascii);	
    s50.load("dataset/spec50.csv", csv_ascii);	
    s60.load("dataset/spec60.csv", csv_ascii);	
    s70.load("dataset/spec70.csv", csv_ascii);
    cont.load("dataset/conc.csv", csv_ascii);

    mat X;
    mat Y;
  
    s30 = normalise(s30.each_row() - mean(s30,0));
    s40 = normalise(s40.each_row() - mean(s40,0));
    s50 = normalise(s50.each_row() - mean(s50,0));
    s60 = normalise(s60.each_row() - mean(s60,0));
    s70 = normalise(s70.each_row() - mean(s70,0));

    cont = normalise(cont.each_row() - mean(cont,0));
mat st = s70;
    a.tic();
int end = s30.n_rows -1;
	
	PLS1 model(st.rows(0,end), cont.col(0).rows(0,end), 11, 0.00001);
	
    model.PLSRegression(st.rows(0,end), cont.col(0).rows(0,end), 11);
	
	for(int i =0; i<10; i++){
		model.VarExp(st.rows(0,end), cont.rows(0,end).col(0), i+1).print();
		cout<<endl;	
	}


	cube stat = model.LOOCV(st.rows(0,end), cont.rows(0,end).col(0), 15);
    stat.print();	
    double n = a.toc();
    cout <<endl<<"Seconds: "<<n<<endl;



    X = s40 +s60;
    X = X/2;


    EPO epo(X, s30,20);
    epo.ProjectionEPO(X,s30,20);
   st= epo.TransformedSpectra(st, 11);
  	PLS1 model1(st.rows(0,end), cont.col(0).rows(0,end), 11, 0.00001);

	
    model1.PLSRegression(st.rows(0,end), cont.col(0).rows(0,end), 11);
	
	for(int i =0; i<10; i++){
		model1.VarExp(st.rows(0,end), cont.rows(0,end).col(0), i+1).print();
		cout<<endl;	
	}
 stat = model.LOOCV(st.rows(0,end), cont.rows(0,end).col(0), 15);
    stat.print();
	return 0;

}