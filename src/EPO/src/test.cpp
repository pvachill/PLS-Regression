#include "epo.hpp"
#include <iostream>
#include <iomanip>
#include "../../pls.hpp"

#define EVALUATION 1
using namespace arma;
using namespace std;



int main( int argc, char *argv[] ) {

    mat X1,X2,X3,X4,REF,Y;
    X4.load("dataset/smoothDataset/4kPa_tr");
    X3.load("dataset/smoothDataset/10kPa_tr");
    X2.load("dataset/smoothDataset/100kPa_tr");
    X1.load("dataset/smoothDataset/300kPa_tr");
    REF.load("dataset/smoothDataset/600kPa_tr");
    Y.load("dataset/smoothDataset/OM_tr");

    mat st = X4;
    
    Y = normalise(Y.each_row() - mean(Y,0));
    st = normalise(st.each_row() - mean(st,0));

    mat X1ev,X2ev,X3ev,X4ev,REFev,Yev;
    X4ev.load("dataset/smoothDataset/4kPa_ev");
    X3ev.load("dataset/smoothDataset/10kPa_ev");
    X2ev.load("dataset/smoothDataset/100kPa_ev");
    X1ev.load("dataset/smoothDataset/300kPa_ev");
    REFev.load("dataset/smoothDataset/600kPa_ev");
    Yev.load("dataset/smoothDataset/OM_ev");

    mat ev = X4ev;
    
    Yev = normalise(Yev.each_row() - mean(Yev,0));
    ev = normalise(ev.each_row() - mean(ev,0));
    
    int end = st.n_rows -1;
    int endev = ev.n_rows-1;
    double tolerance = 0.01;
    int comp = 15;
    int width = 25;
    mat varexp(comp,1); 

    PLS1 model(st.rows(0,end), Y.col(0).rows(0,end), comp, tolerance);
    
    model.PLSRegression(st.rows(0,end), Y.col(0).rows(0,end), comp);
    
    cout << "PLS Without EPO\n"<<endl;
    cout <<left<<setw(width)<< "Variance EXP"<<setw(width)<<"SSE"<<setw(width)<<"MSE"<<setw(width)<<"RMSE"<<setw(width)<<"RSquared"<<endl;
  
    for(int i =0; i<comp; i++)
        varexp.row(i) = model.VarExp(st.rows(0,end), Y.rows(0,end).col(0), i+1);   

    cube stat = model.LOOCV(st.rows(0,end), Y.rows(0,end).col(0), comp);
    
    for( int i = 0; i<comp; i++)
        cout << setw(width)<< varexp(i) << setw(width) << stat(i,0,0) << setw(width) << stat(i,0,1)<< setw(width) << stat(i,0,2) << setw(width) <<stat(i,0,3)<< endl;

    model.PLSRegression(st.rows(0,end), Y.col(0).rows(0,end), comp);
    if(EVALUATION){
        for(int i =0; i<comp; i++)
            varexp.row(i) = model.VarExp(ev, Yev.rows(0,endev).col(0), i+1);

        cout <<"\n\nEVALUATION\n";

        for( int i = 0; i<comp; i++)
         cout <<setw(width)<< varexp(i);
    } 
 
 /* IN THIS SECTION WE MAKE USE OF EXTERNAL PARAMETER ORTHOGONALIZATION METHOD */

    mat X=X4+X2+X1+X3;
    X/=4;

    int compEPO=25;
    
    for(int j  =9;j<compEPO;j++){
        EPO epo(X, REF,j+1);
        epo.ProjectionEPO(X,REF,j+1);
        st= epo.TransformedSpectra(X4, j+1);

        st = normalise(st.each_row() - mean(st,0));
   
   
        model.PLSRegression(st.rows(0,end), Y.col(0).rows(0,end), comp);
    
        cout << endl<<"PLS With EPO\n"<<endl;
        cout <<left<<setw(width)<< "Variance EXP"<<setw(width)<<"SSE"<<setw(width)<<"MSE"<<setw(width)<<"RMSE"<<setw(width)<<"RSquared"<<endl;
    
        for(int i =0; i<comp; i++)
            varexp.row(i) = model.VarExp(st.rows(0,end), Y.rows(0,end).col(0), i+1);   

        stat = model.LOOCV(st.rows(0,end), Y.rows(0,end).col(0), comp);
        
        for( int i = 0; i<comp; i++)
            cout <<setw(width)<< varexp(i) << setw(width) << stat(i,0,0) << setw(width) << stat(i,0,1)<< setw(width) << stat(i,0,2) << setw(width) <<stat(i,0,3)<< endl;
        
            model.PLSRegression(st.rows(0,end), Y.col(0).rows(0,end), comp);

        if(EVALUATION){
            ev= epo.TransformedSpectra(X4ev, j+1);
            ev = normalise(ev.each_row() - mean(ev,0));
            for(int i =0; i<comp; i++){
                varexp.row(i) = model.VarExp(ev.rows(0,endev), Yev.rows(0,endev).col(0), i+1);
            }
            cout<<"\n\nEVALUATION\n";
            for( int i = 0; i<comp; i++){
                cout <<endl<<setw(width)<< varexp(i);//    cout << setw(width)<< varexp(i) << setw(width) << stat(i,0,0) << setw(width) << stat(i,0,1)<< setw(width) << stat(i,0,2) << setw(width) <<stat(i,0,3)<< endl;
            }
        }
    }
	return 0;

}