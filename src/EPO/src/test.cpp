#include "epo.hpp"
#include <iostream>

using namespace arma;
using namespace std;

int main( int argc, char *argv[] ) {

	mat A(10,11,fill::randu);
	mat B(10,11,fill::randu);
	A.print();
	cout<<endl;
	EPO a(A, B);

	mat coeff = a.ProjectionEPO();
	coeff.print();

	return 0;

}