#include<iostream>
#include <Eigen/SVD>
#include <Eigen/Core>
#include <algorithm>
#include <algorithm>
#include <array>
#include <stdlib.h>
#include <string>
#include <bitset> 
#include <aris.hpp>
#include<iomanip>
using namespace std;

const double PI = 3.1415926535;

// 利用Eigen库，采用SVD分解的方法求解矩阵伪逆，默认误差er为0
Eigen::MatrixXd pinv_eigen_based(Eigen::MatrixXd& origin, const float er = 0) {
    // 进行svd分解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_holder(origin,
        Eigen::ComputeThinU |
        Eigen::ComputeThinV);
    // 构建SVD分解结果
    Eigen::MatrixXd U = svd_holder.matrixU();
    Eigen::MatrixXd V = svd_holder.matrixV();
    Eigen::MatrixXd D = svd_holder.singularValues();

    // 构建S矩阵
    Eigen::MatrixXd S(V.cols(), U.cols());
    S.setZero();

    for (unsigned int i = 0; i < D.size(); ++i) {

        if (D(i, 0) > er) {
            S(i, i) = 1 / D(i, 0);
        }
        else {
            S(i, i) = 0;
        }
    }

    // pinv_matrix = V * S * U^T
    return V * S * U.transpose();
}

bool IsEqual(double* arr1, double* arr2) {
    for (int i = 0; i < 9; i++) {
        if (arr1[i] != arr2[i])return false;
    }
    return true;
}

double s_pm_dot_pm(double *pm1, double *pm2, double *pm_out) {
    pm_out[0] = pm1[0] * pm2[0] + pm1[1] * pm2[4] + pm1[2] * pm2[8];
    pm_out[1] = pm1[0] * pm2[1] + pm1[1] * pm2[5] + pm1[2] * pm2[9];
    pm_out[2] = pm1[0] * pm2[2] + pm1[1] * pm2[6] + pm1[2] * pm2[10];
    pm_out[3] = pm1[0] * pm2[3] + pm1[1] * pm2[7] + pm1[2] * pm2[11] + pm1[3];

    pm_out[4] = pm1[4] * pm2[0] + pm1[5] * pm2[4] + pm1[6] * pm2[8];
    pm_out[5] = pm1[4] * pm2[1] + pm1[5] * pm2[5] + pm1[6] * pm2[9];
    pm_out[6] = pm1[4] * pm2[2] + pm1[5] * pm2[6] + pm1[6] * pm2[10];
    pm_out[7] = pm1[4] * pm2[3] + pm1[5] * pm2[7] + pm1[6] * pm2[11] + pm1[7];

    pm_out[8] = pm1[8] * pm2[0] + pm1[9] * pm2[4] + pm1[10] * pm2[8];
    pm_out[9] = pm1[8] * pm2[1] + pm1[9] * pm2[5] + pm1[10] * pm2[9];
    pm_out[10] = pm1[8] * pm2[2] + pm1[9] * pm2[6] + pm1[10] * pm2[10];
    pm_out[11] = pm1[8] * pm2[3] + pm1[9] * pm2[7] + pm1[10] * pm2[11] + pm1[11];

    pm_out[12] = 0;
    pm_out[13] = 0;
    pm_out[14] = 0;
    pm_out[15] = 1;

    return 0;
}

bool NearZero(double near) {
    if (near < 1e-6) { return 1; }
    else { return 0; }
}

double* vec2so3(double* a) {//a is omga(w)
    static double so3mat[9];
    so3mat[0] = 0.0;    so3mat[1] = -a[2];    so3mat[2] = a[1];
    so3mat[3] = a[2];    so3mat[4] = 0.0;    so3mat[5] = -a[0];
    so3mat[6] = -a[1];    so3mat[7] = a[0];    so3mat[8] = 0.0;
    return so3mat;
}

double* so32vec(double* a) {
    static double w[3];
    w[0] = a[7];
    w[1] = a[2];
    w[2] = a[3];
    return w;
}

double* vec2se3(double v[6]) {
    static double se3mat[16];
    se3mat[0] = 0; se3mat[1] = -v[2]; se3mat[2] = v[1];  se3mat[3] = v[3];
    se3mat[4] = v[2]; se3mat[5] = 0; se3mat[6] = -v[0];  se3mat[7] = v[4];
    se3mat[8] = -v[1]; se3mat[9] = v[0]; se3mat[10] = 0; se3mat[11] = v[5];
    se3mat[12] = 0; se3mat[13] = 0; se3mat[14] = 0; se3mat[15] = 0;
    return se3mat;
}

double* matrixexp3(double so3[9]) {
    static double R[9];
    double* omgtheta = so32vec(so3);
    //    std::std::cout << "omgtheta: " << omgtheta[0] << " " << omgtheta[1] << " " << omgtheta[2] << std::endl;
    //    double theta = sqrt(omgtheta[0]*omgtheta[0] + omgtheta[1]*omgtheta[1] + omgtheta[2]*omgtheta[2]);
    double theta = aris::dynamic::s_norm(3, omgtheta);
    //    std::std::cout << "theta" << theta <<std::endl;
    if (std::abs(theta) <= 0.000001) {
        R[0] = 1.0; R[1] = 0.0; R[2] = 0.0;
        R[3] = 0.0; R[4] = 1.0; R[5] = 0.0;
        R[6] = 0.0; R[7] = 0.0; R[8] = 1.0;
    }
    else {
        double omgmat[9];
        omgmat[0] = 0.0; omgmat[1] = so3[1] / theta; omgmat[2] = so3[2] / theta;
        omgmat[3] = so3[3] / theta; omgmat[4] = 0.0; omgmat[5] = so3[5] / theta;
        omgmat[6] = so3[6] / theta; omgmat[7] = so3[7] / theta; omgmat[8] = 0.0;
        //        std::std::cout << std::endl;
        //        for(int i=0; i<9; i++){
        //            std::std::cout << "so3omgmat" << i << ": " << omgmat[i] << "  ";
        //        }
        //        std::std::cout << std::endl;
        double a = sin(theta); double b = 1.0 - cos(theta);
        R[0] = 1.0 + (omgmat[1] * omgmat[3] + omgmat[2] * omgmat[6]) * b; R[1] = omgmat[1] * a + omgmat[2] * omgmat[7] * b;                 R[2] = omgmat[2] * a + omgmat[1] * omgmat[5] * b;
        R[3] = omgmat[3] * a + omgmat[5] * omgmat[6] * b;                 R[4] = 1.0 + (omgmat[1] * omgmat[3] + omgmat[5] * omgmat[7]) * b; R[5] = omgmat[5] * a + omgmat[3] * omgmat[2] * b;
        R[6] = omgmat[6] * a + omgmat[7] * omgmat[3] * b;                 R[7] = omgmat[7] * a + omgmat[6] * omgmat[1] * b;                 R[8] = 1.0 + (omgmat[6] * omgmat[2] + omgmat[7] * omgmat[5]) * b;
    }
    return R;
}

double* matrixexp6(double se3[16]) {
    static double T[16];
    double omgtheta[3];
    omgtheta[0] = se3[9];
    omgtheta[1] = se3[2];
    omgtheta[2] = se3[4];
    //    double norm = sqrt(omgtheta[0] * omgtheta[0] + omgtheta[1] * omgtheta[1] + omgtheta[2] * omgtheta[2]);
    double norm = aris::dynamic::s_norm(3, omgtheta);
    if (std::abs(norm) <= 0.000001) {
        T[0] = 1.0; T[1] = 0.0; T[2] = 0.0; T[3] = 0.0;
        T[4] = 0.0; T[5] = 1.0; T[6] = 0.0; T[7] = 0.0;
        T[8] = 0.0; T[9] = 0.0; T[10] = 1.0; T[11] = 0.0;
        T[12] = 0.0; T[13] = 0.0; T[14] = 0.0; T[15] = 1.0;
    }
    else {
        double theta = norm;
        //        std::std::cout << "norm: " << norm << std::endl;
        double a = 1.0 - cos(theta); double b = theta - sin(theta);
        //        double omghat[3];
        //        omghat[0] = omgtheta[0] / theta; omghat[1] = omgtheta[1] / theta; omghat[2] = omgtheta[2] / theta;
        double omgmat[9];
        omgmat[0] = 0.0; omgmat[1] = se3[1] / theta; omgmat[2] = se3[2] / theta;
        omgmat[3] = se3[4] / theta; omgmat[4] = 0.0; omgmat[5] = se3[6] / theta;
        omgmat[6] = se3[8] / theta; omgmat[7] = se3[9] / theta; omgmat[8] = 0.0;
        //        for(int i=0; i<9; i++){
        //            std::std::cout << "omgmat" << i << ": " << omgmat[i] << "  ";
        //        }
        double R[9];
        R[0] = 0.0; R[1] = se3[1]; R[2] = se3[2];
        R[3] = se3[4]; R[4] = 0.0; R[5] = se3[6];
        R[6] = se3[8]; R[7] = se3[9]; R[8] = 0.0;
        double* Rt = matrixexp3(R);
        //        std::std::cout << std::endl;
        T[0] = Rt[0]; T[1] = Rt[1]; T[2] = Rt[2]; T[3] = ((omgmat[1] * omgmat[3] + omgmat[2] * omgmat[6]) * b + theta) * se3[3] / theta + ((omgmat[2] * omgmat[7]) * b + omgmat[1] * a) * se3[7] / theta + ((omgmat[1] * omgmat[5]) * b + omgmat[2] * a) * se3[11] / theta;
        T[4] = Rt[3]; T[5] = Rt[4]; T[6] = Rt[5]; T[7] = ((omgmat[5] * omgmat[6]) * b + omgmat[3] * a) * se3[3] / theta + ((omgmat[3] * omgmat[1] + omgmat[5] * omgmat[7]) * b + theta) * se3[7] / theta + ((omgmat[3] * omgmat[2]) * b + omgmat[5] * a) * se3[11] / theta;
        T[8] = Rt[6]; T[9] = Rt[7]; T[10] = Rt[8]; T[11] = ((omgmat[7] * omgmat[3]) * b + omgmat[6] * a) * se3[3] / theta + ((omgmat[6] * omgmat[1]) * b + omgmat[7] * a) * se3[7] / theta + ((omgmat[6] * omgmat[2] + omgmat[7] * omgmat[5]) * b + theta) * se3[11] / theta;
        T[12] = 0.0; T[13] = 0.0; T[14] = 0.0; T[15] = 10.;
    }
    return T;
}

double* transtorp(double* T) {
    double* rp = new double[12];
    rp[0] = T[0]; rp[1] = T[1]; rp[2] = T[2];
    rp[3] = T[4]; rp[4] = T[5]; rp[5] = T[6];
    rp[6] = T[8]; rp[7] = T[9]; rp[8] = T[10];
    rp[9] = T[3]; rp[10] = T[7]; rp[11] = T[11];
    return rp;
}

double* transinv(double* T) {
    double* inv = new double[16];
    //    double *rp = transtorp(T);
    inv[0] = T[0]; inv[1] = T[4]; inv[2] = T[8]; inv[3] = -(T[0] * T[3] + T[4] * T[7] + T[8] * T[11]);
    inv[4] = T[1]; inv[5] = T[5]; inv[6] = T[9]; inv[7] = -(T[1] * T[3] + T[5] * T[7] + T[9] * T[11]);
    inv[8] = T[2]; inv[9] = T[6]; inv[10] = T[10]; inv[11] = -(T[2] * T[3] + T[6] * T[7] + T[10] * T[11]);
    inv[12] = 0.0; inv[13] = 0.0; inv[14] = 0.0; inv[15] = 1.0;
    //    std::std::cout << "here" ;
    return inv;
}

double trace(double R[9]) {
    double trace = R[0] + R[4] + R[8];
    return trace;
}

double* matrixlog3(double R[9]) {
    //    aris::dynamic::dsp(3,3,R);
    double acosinput = (trace(R) - 1.0) / 2.0;
    //    std::std::cout << "acosinput: " << acosinput << std::endl;
    static double so3mat[9];
    double omg[3] = { 0.0, 0.0, 0.0 };
    if (acosinput >= 1.0) {
        for (int i = 0; i < 9; i++)so3mat[i] = 0.0;
    }
    else if (acosinput <= -1.0) {
        if (!(std::abs(1 + R[8]) <= 1e-6)) {
            double k = 1.0 / std::sqrt(2.0 * (1.0 + R[8]));
            omg[0] = k * R[2]; omg[1] = k * R[5]; omg[2] = k * (1.0 + R[8]);
        }
        else if (!(abs(1 + R[4]) <= 1e-6)) {
            double k = 1.0 / sqrt(2.0 * (1.0 + R[4]));
            omg[0] = k * R[1]; omg[1] = k * (R[4] + 1.0); omg[2] = k * R[7];
        }
        else {
            double k = 1.0 / sqrt(2.0 * (1.0 + R[0]));
            omg[0] = k * (R[0] + 1.0); omg[1] = k * R[3]; omg[2] = k * R[6];
        }
        for (int i = 0; i < 3; i++)omg[i] = PI * omg[i];
        //        so3mat = vec2so3(omg);
        double* so = vec2so3(omg);
        for (int i = 0; i < 9; i++)so3mat[i] = so[i];
    }
    else {
        double theta = acos(acosinput);
        //        std::std::cout << "acos(acosinput) =  " << theta << std::endl;
        double kk = theta / (2.0 * sin(theta));
        //        std::std::cout << "kk =  " << kk << std::endl;
        so3mat[0] = 0.0; so3mat[1] = (R[1] - R[3]) * kk; so3mat[2] = (R[2] - R[6]) * kk;
        so3mat[3] = (R[3] - R[1]) * kk; so3mat[4] = 0.0; so3mat[5] = (R[5] - R[7]) * kk;
        so3mat[6] = (R[6] - R[2]) * kk; so3mat[7] = (R[7] - R[5]) * kk; so3mat[8] = 0.0;
    }
    //    aris::dynamic::dsp(3,3,so3mat);
    return so3mat;
}

double* matrixlog6(double* T) {
    static double expmat[16];
    double* rp = transtorp(T);
    double R[9];
    double p[3];
    for (int i = 0; i < 9; i++)R[i] = rp[i];
    for (int i = 0; i < 3; i++)p[i] = rp[i + 9];
    double* omgmat = matrixlog3(R);
    double zeros[9];
    for (int i = 0; i < 9; i++)zeros[i] = 0.0;
    if (IsEqual(omgmat, zeros)) {
        std::fill_n(expmat, 16, 0.0);
        aris::dynamic::s_vc(3, T + 3, 4, expmat + 3, 4);
    }
    else {
        aris::dynamic::s_cm3(T, expmat);
        double theta;
        double tr = (trace(R) - 1.0) / 2.0;
        theta = std::acos(tr);
        double cottheta = std::cos(theta / 2.0) / std::sin(theta / 2.0);
        double k = (1.0 / theta - (cottheta) / 2.0) / theta;
        expmat[0] = omgmat[0]; expmat[1] = omgmat[1]; expmat[2] = omgmat[2]; expmat[3] = (1.0 - omgmat[0] / 2.0 + k * (omgmat[0] * omgmat[0] + omgmat[1] * omgmat[3] + omgmat[2] * omgmat[6])) * p[0] + ((-omgmat[1] / 2.0) + k * (omgmat[0] * omgmat[1] + omgmat[1] * omgmat[4] + omgmat[2] * omgmat[7])) * p[1] + ((-omgmat[2] / 2.0) + k * (omgmat[0] * omgmat[2] + omgmat[1] * omgmat[5] + omgmat[2] * omgmat[8])) * p[2];
        //        expmat[0] = omgmat[0]; expmat[1] = omgmat[1]; expmat[2]  = omgmat[2]; expmat[3]  = (1 - omgmat[0]/2 + (omgmat[0]*omgmat[0] + omgmat[1]*omgmat[3] + omgmat[2]*omgmat[6])*(1/theta - (1/tan(theta/2))/2)/theta) * p[0] + ((-omgmat[1]/2) + (omgmat[0]*omgmat[1] + omgmat[1]*omgmat[4] + omgmat[2]*omgmat[7])*(1/theta - (1/tan(theta/2))/2)/theta) * p[1] + ((-omgmat[2]/2) + (omgmat[0]*omgmat[2] + omgmat[1]*omgmat[5] + omgmat[2]*omgmat[8])*(1/theta - (1/tan(theta/2))/2)/theta) * p[2];
        expmat[4] = omgmat[3]; expmat[5] = omgmat[4]; expmat[6] = omgmat[5]; expmat[7] = (-omgmat[3] / 2.0 + k * (omgmat[3] * omgmat[0] + omgmat[4] * omgmat[3] + omgmat[5] * omgmat[6])) * p[0] + (1.0 - omgmat[4] / 2.0 + (omgmat[3] * omgmat[1] + omgmat[4] * omgmat[4] + omgmat[5] * omgmat[7])) * p[1] + (-omgmat[5] / 2.0 + k * (omgmat[3] * omgmat[2] + omgmat[4] * omgmat[5] + omgmat[5] * omgmat[8])) * p[2];
        expmat[8] = omgmat[6]; expmat[9] = omgmat[7]; expmat[10] = omgmat[8]; expmat[11] = (-omgmat[6] / 2.0 + k * (omgmat[6] * omgmat[0] + omgmat[7] * omgmat[3] + omgmat[8] * omgmat[6])) * p[0] + (-omgmat[7] / 2.0 + (omgmat[6] * omgmat[1] + omgmat[7] * omgmat[4] + omgmat[8] * omgmat[7])) * p[1] + (1.0 - omgmat[8] / 2.0 + k * (omgmat[6] * omgmat[2] + omgmat[7] * omgmat[5] + omgmat[8] * omgmat[8])) * p[2];
        expmat[12] = 0.0; expmat[13] = 0.0; expmat[14] = 0.0; expmat[15] = 0.0;
    }
    return expmat;
}

double* se3tovec(double se3mat[16]) {
    static double V[6];
    V[0] = se3mat[9];
    V[1] = se3mat[2];
    V[2] = se3mat[4];
    V[3] = se3mat[3];
    V[4] = se3mat[7];
    V[5] = se3mat[11];
    return V;
}

double* adjoint(double T[16]) {
    double* rp = transtorp(T);
    double p[3] = { rp[9], rp[10], rp[11] };
    double* s = vec2so3(p);
    static double adt[36];
    adt[0] = T[0]; adt[1] = T[1]; adt[2] = T[2];  adt[3] = 0.0; adt[4] = 0.0; adt[5] = 0.0;
    adt[6] = T[4]; adt[7] = T[5]; adt[8] = T[6];  adt[9] = 0.0; adt[10] = 0.0; adt[11] = 0.0;
    adt[12] = T[8]; adt[13] = T[9]; adt[14] = T[10]; adt[15] = 0.0; adt[16] = 0.0; adt[17] = 0.0;
    adt[18] = s[0] * rp[0] + s[1] * rp[3] + s[2] * rp[6]; adt[19] = s[0] * rp[1] + s[1] * rp[4] + s[2] * rp[7]; adt[20] = s[0] * rp[2] + s[1] * rp[5] + s[2] * rp[8]; adt[21] = rp[0]; adt[22] = rp[1]; adt[23] = rp[2];
    adt[24] = s[3] * rp[0] + s[4] * rp[3] + s[5] * rp[6]; adt[25] = s[3] * rp[1] + s[4] * rp[4] + s[5] * rp[7]; adt[26] = s[3] * rp[2] + s[4] * rp[5] + s[5] * rp[8]; adt[27] = rp[3]; adt[28] = rp[4]; adt[29] = rp[5];
    adt[30] = s[6] * rp[0] + s[7] * rp[3] + s[8] * rp[6]; adt[31] = s[6] * rp[1] + s[7] * rp[4] + s[8] * rp[7]; adt[32] = s[6] * rp[2] + s[7] * rp[5] + s[8] * rp[8]; adt[33] = rp[6]; adt[34] = rp[7]; adt[35] = rp[8];
    return adt;
}

double* jacobianbody(double blist[42], double thetalist[7]) {
    static double jb[42];
    double T[16] = { 1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0 };
    for (int i = 6; i >= 1; i--) {
        double vec[6];//blist(:,i)
        vec[0] = -blist[6 * i] * thetalist[i];
        vec[1] = -blist[6 * i + 1] * thetalist[i];
        vec[2] = -blist[6 * i + 2] * thetalist[i];
        vec[3] = -blist[6 * i + 3] * thetalist[i];
        vec[4] = -blist[6 * i + 4] * thetalist[i];
        vec[5] = -blist[6 * i + 5] * thetalist[i];
        double* F = matrixexp6(vec2se3(vec));
        double Tout[16];
        s_pm_dot_pm(T, F, Tout);
        //        aris::dynamic::dsp(4,4,Tout);
        for (int i = 0; i < 16; i++) { T[i] = Tout[i]; };
        double* adj = adjoint(Tout);
        jb[6 * (i - 1)] = adj[0] * blist[6 * (i - 1)] + adj[1] * blist[6 * (i - 1) + 1] + adj[2] * blist[6 * (i - 1) + 2] + adj[3] * blist[6 * (i - 1) + 3] + adj[4] * blist[6 * (i - 1) + 4] + adj[5] * blist[6 * (i - 1) + 5];
        jb[6 * (i - 1) + 1] = adj[6] * blist[6 * (i - 1)] + adj[7] * blist[6 * (i - 1) + 1] + adj[8] * blist[6 * (i - 1) + 2] + adj[9] * blist[6 * (i - 1) + 3] + adj[10] * blist[6 * (i - 1) + 4] + adj[11] * blist[6 * (i - 1) + 5];
        jb[6 * (i - 1) + 2] = adj[12] * blist[6 * (i - 1)] + adj[13] * blist[6 * (i - 1) + 1] + adj[14] * blist[6 * (i - 1) + 2] + adj[15] * blist[6 * (i - 1) + 3] + adj[16] * blist[6 * (i - 1) + 4] + adj[17] * blist[6 * (i - 1) + 5];
        jb[6 * (i - 1) + 3] = adj[18] * blist[6 * (i - 1)] + adj[19] * blist[6 * (i - 1) + 1] + adj[20] * blist[6 * (i - 1) + 2] + adj[21] * blist[6 * (i - 1) + 3] + adj[22] * blist[6 * (i - 1) + 4] + adj[23] * blist[6 * (i - 1) + 5];
        jb[6 * (i - 1) + 4] = adj[24] * blist[6 * (i - 1)] + adj[25] * blist[6 * (i - 1) + 1] + adj[26] * blist[6 * (i - 1) + 2] + adj[27] * blist[6 * (i - 1) + 3] + adj[28] * blist[6 * (i - 1) + 4] + adj[29] * blist[6 * (i - 1) + 5];
        jb[6 * (i - 1) + 5] = adj[30] * blist[6 * (i - 1)] + adj[31] * blist[6 * (i - 1) + 1] + adj[32] * blist[6 * (i - 1) + 2] + adj[33] * blist[6 * (i - 1) + 3] + adj[34] * blist[6 * (i - 1) + 4] + adj[35] * blist[6 * (i - 1) + 5];
    }
    jb[36] = blist[36];
    jb[37] = blist[37];
    jb[38] = blist[38];
    jb[39] = blist[39];
    jb[40] = blist[40];
    jb[41] = blist[41];
    return jb;
}

double* fkinbody(double M[16], double blist[42], double thetalist[7]) {
    static double T[16];
    for (int i = 0; i < 16; i++) {
        T[i] = M[i];
    }
    for (int i = 0; i <= 6; i++) {
        double vec[6];//blist(:,i)
//        std::std::cout << "i: " << i <<std::endl;
        vec[0] = blist[6 * i] * thetalist[i];
        vec[1] = blist[6 * i + 1] * thetalist[i];
        vec[2] = blist[6 * i + 2] * thetalist[i];
        vec[3] = blist[6 * i + 3] * thetalist[i];
        vec[4] = blist[6 * i + 4] * thetalist[i];
        vec[5] = blist[6 * i + 5] * thetalist[i];
        double* F = matrixexp6(vec2se3(vec));
        double Tout[16];
        s_pm_dot_pm(T, F, Tout);
        //        aris::dynamic::dsp(4,4,Tout);
        for (int i = 0; i < 16; i++) { T[i] = Tout[i]; };
    }
    return T;
}

double* ikbody(double blist[42], double M[16], double T[16], double thetalist0[7]) {
    static double ew = 0.01;
    static double ev = 0.001;
    double* thetalist = new double[7];
    double* blist_ = new double[42];
    double* vb = new double[6];
    std::copy(thetalist0, thetalist0 + 7, thetalist);
    std::copy(blist, blist + 42, blist_);
    double* Tcopy = new double[16];
    std::copy(T, T + 16, Tcopy);
    double pmout[16];
    s_pm_dot_pm(transinv(fkinbody(M, blist_, thetalist)), Tcopy, pmout);
    vb = se3tovec(matrixlog6(pmout));
    int i = 0;
    int maxinterations = 100;
    bool err = sqrt(vb[0] * vb[0] + vb[1] * vb[1] + vb[2] * vb[2]) > ew || sqrt(vb[3] * vb[3] + vb[4] * vb[4] + vb[5] * vb[5]) > ev;
    while (err && i <= maxinterations) {
        double* jacb = jacobianbody(blist_, thetalist);
        Eigen::MatrixXd A(6, 7);
        A << jacb[0], jacb[6], jacb[12], jacb[18], jacb[24], jacb[30], jacb[36],
            jacb[1], jacb[7], jacb[13], jacb[19], jacb[25], jacb[31], jacb[37],
            jacb[2], jacb[8], jacb[14], jacb[20], jacb[26], jacb[32], jacb[38],
            jacb[3], jacb[9], jacb[15], jacb[21], jacb[27], jacb[33], jacb[39],
            jacb[4], jacb[10], jacb[16], jacb[22], jacb[28], jacb[34], jacb[40],
            jacb[5], jacb[11], jacb[17], jacb[23], jacb[29], jacb[35], jacb[41];
        Eigen::MatrixXd pinv = pinv_eigen_based(A);
        thetalist[0] = thetalist[0] + pinv(0, 0) * vb[0] + pinv(0, 1) * vb[1] + pinv(0, 2) * vb[2] + pinv(0, 3) * vb[3] + pinv(0, 4) * vb[4] + pinv(0, 5) * vb[5];
        thetalist[1] = thetalist[1] + pinv(1, 0) * vb[0] + pinv(1, 1) * vb[1] + pinv(1, 2) * vb[2] + pinv(1, 3) * vb[3] + pinv(1, 4) * vb[4] + pinv(1, 5) * vb[5];
        thetalist[2] = thetalist[2] + pinv(2, 0) * vb[0] + pinv(2, 1) * vb[1] + pinv(2, 2) * vb[2] + pinv(2, 3) * vb[3] + pinv(2, 4) * vb[4] + pinv(2, 5) * vb[5];
        thetalist[3] = thetalist[3] + pinv(3, 0) * vb[0] + pinv(3, 1) * vb[1] + pinv(3, 2) * vb[2] + pinv(3, 3) * vb[3] + pinv(3, 4) * vb[4] + pinv(3, 5) * vb[5];
        thetalist[4] = thetalist[4] + pinv(4, 0) * vb[0] + pinv(4, 1) * vb[1] + pinv(4, 2) * vb[2] + pinv(4, 3) * vb[3] + pinv(4, 4) * vb[4] + pinv(4, 5) * vb[5];
        thetalist[5] = thetalist[5] + pinv(5, 0) * vb[0] + pinv(5, 1) * vb[1] + pinv(5, 2) * vb[2] + pinv(5, 3) * vb[3] + pinv(5, 4) * vb[4] + pinv(5, 5) * vb[5];
        thetalist[6] = thetalist[6] + pinv(6, 0) * vb[0] + pinv(6, 1) * vb[1] + pinv(6, 2) * vb[2] + pinv(6, 3) * vb[3] + pinv(6, 4) * vb[4] + pinv(6, 5) * vb[5];

        double* fk2 = fkinbody(M, blist, thetalist);
        double* trans2 = transinv(fk2);
        double Tout[16];
        s_pm_dot_pm(trans2, T, Tout);
        double* log62 = matrixlog6(Tout);
        vb = se3tovec(log62);
        err = std::sqrt(vb[0] * vb[0] + vb[1] * vb[1] + vb[2] * vb[2]) > ew || std::sqrt(vb[3] * vb[3] + vb[4] * vb[4] + vb[5] * vb[5]) > ev;
        i++;
    }
    return thetalist;
}

double* s_rm_dot_rm(double* rm1, double* rm2) {
    static double rm_out[9];
    rm_out[0] = rm1[0] * rm2[0] + rm1[1] * rm2[3] + rm1[2] * rm2[6]; rm_out[1] = rm1[0] * rm2[1] + rm1[1] * rm2[4] + rm1[2] * rm2[7]; rm_out[2] = rm1[0] * rm2[2] + rm1[1] * rm2[5] + rm1[2] * rm2[8];
    rm_out[3] = rm1[3] * rm2[0] + rm1[4] * rm2[3] + rm1[5] * rm2[6]; rm_out[4] = rm1[3] * rm2[1] + rm1[4] * rm2[4] + rm1[5] * rm2[7]; rm_out[5] = rm1[3] * rm2[2] + rm1[4] * rm2[5] + rm1[5] * rm2[8];
    rm_out[6] = rm1[6] * rm2[0] + rm1[7] * rm2[3] + rm1[8] * rm2[6]; rm_out[7] = rm1[6] * rm2[1] + rm1[7] * rm2[4] + rm1[8] * rm2[7]; rm_out[8] = rm1[6] * rm2[2] + rm1[7] * rm2[5] + rm1[8] * rm2[8];
    return rm_out;
}

double* point2point(double T_start[16], double T_end[16], double s) {
    static double T[16];
    double* rp_start = transtorp(T_start);
    //double*   rp_end = transtorp(T_end);
    //计算位置插补
    T[3] = T_start[3] + s * (T_end[3] - T_start[3]);
    T[7] = T_start[7] + s * (T_end[7] - T_start[7]);
    T[11] = T_start[11] + s * (T_end[11] - T_start[11]);
    //计算姿态插补
    double T_out[16];
    double T_start_T[16];
    aris::dynamic::s_mc(4, 4, T_start, aris::dynamic::RowMajor(4), T_start_T, aris::dynamic::ColMajor(4));
    s_pm_dot_pm(T_start_T, T_end, T_out);
    double* Rp_out = transtorp(T_out);
    double R_out[9];
    std::copy(Rp_out, Rp_out + 9, R_out);
    double* so3m = matrixlog3(R_out);
    for (int i = 0; i < 9; i++)so3m[i] = so3m[i] * s;
    double* matexp3 = matrixexp3(so3m);
    double* Rs = s_rm_dot_rm(rp_start, matexp3);
    T[0] = Rs[0]; T[1] = Rs[1]; T[2] = Rs[2];
    T[4] = Rs[3]; T[5] = Rs[4]; T[6] = Rs[5];
    T[8] = Rs[6]; T[9] = Rs[7]; T[10] = Rs[8];
    T[12] = 0; T[13] = 0; T[14] = 0; T[15] = 1;
    if (s/0.05 == 0) std::cout << "T[11]" << T[11] << std::endl;
    return T;
}

struct Quaternion
{
    double w, x, y, z;
};

Quaternion ToQuaternion(double yaw, double pitch, double roll) // yaw (Z), pitch (Y), roll (X)
{
    // Abbreviations for the various angular functions
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    Quaternion q;
    q.w = cy * cp * cr + sy * sp * sr;
    q.x = cy * cp * sr - sy * sp * cr;
    q.y = sy * cp * sr + cy * sp * cr;
    q.z = sy * cp * cr - cy * sp * sr;

    return q;
}
struct Quaternion Q = {
    Q.w = 0.0,
    Q.x = 0.0,
    Q.y = 0.0,
    Q.z = 0.0,
};

void line(double* p0, double* p1, double vec[3]) {
    vec[0] = p0[0] - p1[0];
    vec[1] = p0[1] - p1[1];
    vec[2] = p0[2] - p1[2];
    //  return 0;
}

double s_vec_dot_vec(double* vec1, double* vec2, int size) {
    double norm = 0.0;
    for (int i = 0; i < size; i++) {
        norm += vec1[i] * vec2[i];
    }
    return sqrt(norm);
}

double* turningzone(double* p0, double* p1, double* p2, double r, double t){
    //计算直线角theta
    double vec1[3], vec2[3];
    line(p0, p1, vec1);
    line(p2, p1, vec2);
    for (int i = 0; i < 3; i++) {
        std::cout << "vec1: " << vec1[i] << std::endl;
    }
    for (int i = 0; i < 3; i++) {
        std::cout << "vec2: " << vec2[i] << std::endl;
    }
    double n1 = aris::dynamic::s_norm(3, vec1);
    std::cout << "n1: " << n1 << std::endl;
    double n2 = aris::dynamic::s_norm(3, vec2);
    double theta = std::acos(s_vec_dot_vec(vec1, vec2, 3)/(n1 * n2));
    //求转接点
    std::cout << "theta: " << theta << std::endl;
    double k = (r / std::tan(theta / 2)) / n1;
    std::cout << "k: " << k << std::endl;
    double deltavec1[3], deltavec2[3];
    double pt1[3], pt2[3];
    for (int i = 0; i < 3; i++) {
        deltavec1[i] = k * vec1[i];
        deltavec2[i] = k * vec2[i];
        pt1[i] = p1[i] + deltavec1[i];
        pt2[i] = p1[i] + deltavec2[i];
    }
    std::cout << "pt1: " << pt1[0] << pt1[1] << pt1[2] << std::endl;
    std::cout << "pt2: " << pt2[0] << pt2[1] << pt2[2] << std::endl;
    //计算路径长度d1、弧长d2
    double L1[3] = { p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2] }; 
    double d1 = aris::dynamic::s_norm(3, L1);
    double d2 = (PI - theta) * r;
    //求圆心C
    //向量pt1m
    double vec_pt1m[3];
    for (int i = 0; i < 3; i++) {
        vec_pt1m[i] = 0.5 * (pt2[i] - pt1[i]);
    }
    std::cout << "vec_pt1m: " << vec_pt1m[0] << vec_pt1m[1] << vec_pt1m[2] << std::endl;
    //m坐标
    double m[3];
    for (int i = 0; i < 3; i++) {
        m[i] = pt1[i] + vec_pt1m[i];
    }
    std::cout << "m: " << m[0] << m[1] << m[2] << std::endl;
    //向量p1m及长度
    double vec_p1m[3];
    for (int i = 0; i < 3; i++) {
        vec_p1m[i] = m[i] - p1[i];
    }
    std::cout << "vec_p1m: " << vec_p1m[0] << vec_p1m[1] << vec_p1m[2] << std::endl;
    double p1m = aris::dynamic::s_norm(3, vec_p1m);
    //p1c长度、向量，圆心c坐标
    double p1c = r / sin(theta/2);
    std::cout << "p1c: " << p1c << std::endl;
    double vec_p1c[3], c[3];
    for (int i = 0; i < 3; i++) {
        vec_p1c[i] = (p1c/p1m) * vec_p1m[i];
        c[i] = p1[i] + vec_p1c[i];
    }
    std::cout << "vec_p1c: " << vec_p1c[0] << " " << vec_p1c[1] << " " << vec_p1c[2] << std::endl;
    std::cout << "c: " << c[0] << " " << c[1] << " " << c[2] << std::endl;
    return 0;
}



int main()
{

    static double M[16] = { 1,0,0,0,
                        0,1,0,-98,
                        0,0,1,150,
                        0,0,0,1 };
    static double blist[42] = { 0,0,1,98,0,0,
                               0,1,0,203,0,0,
                               0,0,1,232,0,0,
                               0,-1,0,247,0,0,
                               0,0,1,-98,0,0,
                               0,-1,0, -53,0,0,
                               0,0,1,0,0,0 };
    static double thetalist[7] = { 0,PI / 6,0,PI / 3,0,-PI * 2 / 3,0 };
    static double Tsd[16] = { 0,0,1,428,
                             0,1,0,-98,
                             -1,0,0,130,
                             0,0,0,1 };

    static double *ik = ikbody(blist, M, Tsd, thetalist);
    for (int i = 0; i < 7; i++) {
        std::cout << setprecision(15) << ik[i] << endl;
    }
    static double* fk = fkinbody(M, blist, ik);
    aris::dynamic::dsp(4, 4, fk);

    static double Tsd2[16] = { 0,0,1,428,
                             0,1,0,-98,
                             -1,0,0,230,
                             0,0,0,1 };
    for (int count = 1; count <= 10000; count++) {
        double s = count / 10000.0;
        double p2ppm[16];
        double* p2p = point2point(fk, Tsd2, s);
        std::copy(p2p, p2p + 16, p2ppm);
        if (count % 500 == 0) {
            aris::dynamic::dsp(4, 4, p2p);
        }
        static double ikbody2[7];
        if (count == 1) {
            std::copy(ik, ik + 7, ikbody2);
        }
        if (count % 500 == 0) {
            aris::dynamic::dsp(7, 1, ikbody2);
        }
        static double *ikbody3 = ikbody(blist, M, p2ppm, ikbody2);
        if (count % 500 == 0) {
            std::cout << "ikbody3: " << std::endl;
            aris::dynamic::dsp(7, 1, ikbody3);
            aris::dynamic::dsp(7, 1, ikbody3);
        }
        std::copy(ikbody3, ikbody3 + 7, ikbody2);
        if (count == 10000) {
            std::cout << "when s = 1, fk is: " << std::endl;
            double* fkpm = fkinbody(M, blist, ikbody2);
            aris::dynamic::dsp(4, 4, fkpm);               
        }
    }

    //double pe_out[6];
    //aris::dynamic::s_pm2pe(fk, pe_out);

    //double pq_out[7];
    //aris::dynamic::s_pe2pq(pe_out, pq_out);

    //double pe_out2[6] = {0,0,0,0,0,0};
    //aris::dynamic::s_pm2pe(fk, pe_out2, "321");

    //double pq_out2[7];   
    //Q = ToQuaternion(pe_out2[3], pe_out2[4], pe_out2[5]);

    //double p0[3] = { 1, 0, 0 };
    //double p1[3] = { 0, 0, 0 };
    //double p2[3] = { 0, 0, 1 };
    //turningzone(p0, p1, p2, 1, 0);

    return 0;
}