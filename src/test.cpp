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

	
	X = normalise(X.each_row() - mean(X,0));
	P = normalise(P.each_row() - mean(P,0));
	mat Y = P.col(0);

	PLSR model(X,P);

	const mat & D = model.Observations();

	//model.PLS1();
	model.PLS1();

	mat T = model.LatentVec();


	//double lvar,var;
	//var = var(Y.col(0));
	//lvar = var(T.)
	for( int i = 0; i<15	; i++)
	 model.VarExp(i	).print();
	// model.VarExp(	).print();

	

	return 0;
}