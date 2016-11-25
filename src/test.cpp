#include "pls.hpp"
#include <iostream>

using namespace std;
using namespace arma;

int main( int argc, char *argv[] ) {
	if (argc < 3) { fprintf(stderr, "Usage: ./pls X_data.csv Y_data.csv \n"); exit(100); }
    std::string x_filename(argv[1]);
    std::string y_filename(argv[2]);

    mat X;
    X.load(x_filename, csv_ascii);	
    mat P;
    P.load(y_filename, csv_ascii);
	mat Y = P.col(0);

	//X = X.each_col()- mean(X,1);
	//P = P.each_row() -mean(P,0);
	PLSR model(X,Y);

	const mat & D = model.Observations();

	model.PLSRegression();

	mat T = model.LatentVec();


	//double lvar,var;
	//var = var(Y.col(0));
	//lvar = var(T.)
	for( int i = 1; i<15	; i++)
	 model.VarExp(i);//.print();

	

	return 0;
}