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

	PLSR model(X,P);


	//model.PLS1();
	wall_clock timer;
	timer.tic();
	model.PLSRegression( X,P,14);

	//mat T = model.LatentVec();


	//double lvar,var;
	//var = var(Y.col(0));
	//lvar = var(T.)
	for( int i = 0; i<5; i++)
	 model.VarExp( X,P,i).print();
		double n = timer.toc();
	cout << "Seconds per regression: " << n <<endl;


	// model.VarExp(	).print();
	/*cube A = model.LOOCV(X,P,14);
	cout<<endl;
	for(int i =0; i<4;i++){
		A.slice(i).print();
		cout <<endl;
	}*/

	return 0;
}