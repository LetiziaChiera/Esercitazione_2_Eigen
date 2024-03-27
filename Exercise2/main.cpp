#include <iostream>
#include <vector>
#include <string>
#include "Eigen/Eigen"
using namespace std;
using namespace Eigen;

// Creo funzione per la decomposizione PALU
VectorXd decomposition_PALU(const MatrixXd& A, const VectorXd& b)
{
    PartialPivLU<MatrixXd> lu(A);
    MatrixXd P = lu.permutationP();
    MatrixXd L1 = lu.matrixLU().triangularView<StrictlyLower>(); // <StrictlyLower> viene esclusa la diagonale
    MatrixXd U = lu.matrixLU().triangularView<Upper>();
    MatrixXd I = MatrixXd::Identity(A.rows(), A.cols());
    MatrixXd L = L1 + I; // aggiungo alla matrice matrice L1 la diagonale unitaria

// Calcolo del vettore y risolvendo il sistema Ly = Pb
    VectorXd Pb = P*b;
    VectorXd y = L.triangularView<Lower>().solve(Pb);
    // trangularView per ottenere la vista delle parti triangolari delle funzioni ( altrimenti vista matrice LU completa)

// Calcolo del vettore x risolvendo i sistema Ux = y
    VectorXd x = U.triangularView<Upper>().solve(y);
    return x;
}

VectorXd decomposition_QR(const MatrixXd& A, const VectorXd& b)
{
    HouseholderQR<MatrixXd> qr_decomposition(A);
    MatrixXd Q = qr_decomposition.householderQ();
    MatrixXd R = qr_decomposition.matrixQR().triangularView<Upper>();
    // Calcolo y = Q'b
    VectorXd y = Q.transpose()*b;
    // Risolvo il sistema Rx = y
    VectorXd x = R.triangularView<Upper>().solve(y);
    return x;
}
double errRel(const VectorXd& x1, const VectorXd& x2)
{
    double diff = (x1-x2).norm();
    double x = x1.norm();
    return diff/x;
}
int main()
{
    // soluzione di tutti i sistemi
    VectorXd x_esatto(2);
    x_esatto<< -1.0e+0, -1.0e+00 ;

    // Primo sistema
    cout << "Primo sistema"<<endl;
    VectorXd b1(2);
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    MatrixXd A1(2,2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01, -9.992887623566787e-01;
    VectorXd x1PALU = decomposition_PALU(A1,b1);
    VectorXd x1QR = decomposition_QR(A1, b1);
    double errPALU1 = errRel(x_esatto,x1PALU);
    double errQR1 = errRel(x_esatto, x1QR);

    cout << "Errore relativo decomposizione PALU:" << errPALU1 << endl;
    cout << "Errore relativo decomposizione QR:" << errQR1 << endl;

    // secondo sistema
    cout << "Secondo sistema"<<endl;
    VectorXd b2(2);
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    MatrixXd A2(2,2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01;
    VectorXd x2PALU = decomposition_PALU(A2,b2);
    VectorXd x2QR = decomposition_QR(A2, b2);
    double errPALU2 = errRel(x_esatto,x2PALU);
    double errQR2 = errRel(x_esatto, x2QR);

    cout << "Errore relativo decomposizione PALU:" << errPALU2 << endl;
    cout << "Errore relativo decomposizione QR:" << errQR2 << endl;

    // terzo sistema
    cout << "terzo sistema"<<endl;
    VectorXd b3(2);
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    MatrixXd A3(2,2);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01,
        -8.320502947645361e-01;
    VectorXd x3PALU = decomposition_PALU(A3,b3);
    VectorXd x3QR = decomposition_QR(A3, b3);
    double errPALU3 = errRel(x_esatto,x3PALU);
    double errQR3 = errRel(x_esatto, x3QR);

    cout << "Errore relativo decomposizione PALU:" << errPALU3 << endl;
    cout << "Errore relativo decomposizione QR:" << errQR3 << endl;



  return 0;
}
