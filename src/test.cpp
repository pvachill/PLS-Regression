#include "pls.hpp"
#include <iostream>

using namespace std;
using namespace arma;

int main() {

	mat X(10,10, fill::randu);
	mat Y(10,1, fill::randu);

	PLSR model(X,Y);

	const mat & D = model.Observations();
	
	for( int i =0; i<10;i++) 
		cout << D(i) << endl;

	model.PLSRegression();

	

	return 0;
}