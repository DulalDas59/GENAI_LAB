#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <iomanip>

// ---------------------------
// Simple Matrix (row-major)
// ---------------------------
struct Mat {
    int r, c;
    std::vector<double> a; // size r*c

    Mat(int rows=0, int cols=0, double val=0.0) : r(rows), c(cols), a(rows*cols, val) {}

    double& operator()(int i, int j) { return a[i*c + j]; }
    double  operator()(int i, int j) const { return a[i*c + j]; }

    static Mat random(int r, int c, double stddev=0.1, unsigned seed=42) {
        Mat m(r,c);
        std::mt19937 gen(seed);
        std::normal_distribution<double> nd(0.0, stddev);
        for (auto &x : m.a) x = nd(gen);
        return m;
    }
};

// Matrix multiplication: (m x k) * (k x n) = (m x n)
Mat matmul(const Mat& A, const Mat& B) {
    assert(A.c == B.r);
    Mat C(A.r, B.c, 0.0);
    for (int i=0;i<A.r;i++) {
        for (int k=0;k<A.c;k++) {
            double aik = A(i,k);
            for (int j=0;j<B.c;j++) {
                C(i,j) += aik * B(k,j);
            }
        }
    }
    return C;
}

// Transpose
Mat T(const Mat& A) {
    Mat B(A.c, A.r);
    for (int i=0;i<A.r;i++)
        for (int j=0;j<A.c;j++)
            B(j,i) = A(i,j);
    return B;
}

// Add with row-broadcast bias: (m x n) + (1 x n)
Mat add_row_bias(const Mat& A, const Mat& b) {
    assert(b.r == 1 && b.c == A.c);
    Mat C(A.r, A.c);
    for (int i=0;i<A.r;i++)
        for (int j=0;j<A.c;j++)
            C(i,j) = A(i,j) + b(0,j);
    return C;
}

// Elementwise: C = A - B
Mat sub(const Mat& A, const Mat& B) {
    assert(A.r==B.r && A.c==B.c);
    Mat C(A.r, A.c);
    for (int i=0;i<A.r*A.c;i++) C.a[i] = A.a[i] - B.a[i];
    return C;
}

// Elementwise: C = A ⊙ B
Mat hadamard(const Mat& A, const Mat& B) {
    assert(A.r==B.r && A.c==B.c);
    Mat C(A.r, A.c);
    for (int i=0;i<A.r*A.c;i++) C.a[i] = A.a[i] * B.a[i];
    return C;
}

// Apply function elementwise
template <typename F>
Mat apply(const Mat& A, F f) {
    Mat C(A.r, A.c);
    for (int i=0;i<A.r*A.c;i++) C.a[i] = f(A.a[i]);
    return C;
}

// Sum over rows -> (1 x n)
Mat sum_rows(const Mat& A) {
    Mat s(1, A.c, 0.0);
    for (int i=0;i<A.r;i++)
        for (int j=0;j<A.c;j++)
            s(0,j) += A(i,j);
    return s;
}

// Scale matrix by scalar
Mat scale(const Mat& A, double k) {
    Mat C(A.r, A.c);
    for (int i=0;i<A.r*A.c;i++) C.a[i] = A.a[i] * k;
    return C;
}

// In-place: A -= lr * dA
void sgd_update(Mat& A, const Mat& dA, double lr) {
    assert(A.r==dA.r && A.c==dA.c);
    for (int i=0;i<A.r*A.c;i++) A.a[i] -= lr * dA.a[i];
}

// ---------------------------
// Activations
// ---------------------------
double sigmoid(double x) {
    // Stable-ish sigmoid
    if (x >= 0) {
        double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    } else {
        double z = std::exp(x);
        return z / (1.0 + z);
    }
}
double relu(double x) { return x > 0 ? x : 0; }
double relu_deriv(double x) { return x > 0 ? 1.0 : 0.0; }

// ---------------------------
// Loss: BCE
// ---------------------------
double bce_loss(const Mat& y, const Mat& a2) {
    assert(y.r==a2.r && y.c==a2.c);
    const double eps = 1e-12;
    double sum = 0.0;
    int m = y.r;
    for (int i=0;i<m;i++) {
        double yi = y(i,0);
        double pi = std::min(1.0 - eps, std::max(eps, a2(i,0)));
        sum += -(yi * std::log(pi) + (1.0 - yi) * std::log(1.0 - pi));
    }
    return sum / m;
}

// ---------------------------
// 2-layer MLP for XOR
// ---------------------------
struct Net {
    Mat W1, b1, W2, b2;
    // caches
    Mat Z1, A1, Z2, A2;

    Net(int in_dim, int hidden_dim, unsigned seed=42)
        : W1(Mat::random(in_dim, hidden_dim, 0.2, seed)),
          b1(1, hidden_dim, 0.0),
          W2(Mat::random(hidden_dim, 1, 0.2, seed+1)),
          b2(1, 1, 0.0) {}

    Mat forward(const Mat& X) {
        Z1 = add_row_bias(matmul(X, W1), b1);
        A1 = apply(Z1, relu);
        Z2 = add_row_bias(matmul(A1, W2), b2);
        A2 = apply(Z2, sigmoid);
        return A2;
    }

    void backward_and_update(const Mat& X, const Mat& Y, double lr) {
        int m = X.r;
        // dZ2 = A2 - Y  (for sigmoid + BCE)
        Mat dZ2 = sub(A2, Y);                 // (m x 1)

        // dW2 = (A1^T dZ2)/m
        Mat dW2 = scale(matmul(T(A1), dZ2), 1.0 / m);  // (h x 1)
        // db2 = sum_rows(dZ2)/m
        Mat db2 = scale(sum_rows(dZ2), 1.0 / m);       // (1 x 1)

        // dA1 = dZ2 W2^T
        Mat dA1 = matmul(dZ2, T(W2));         // (m x h)

        // dZ1 = dA1 ⊙ ReLU'(Z1)
        Mat reluPrime = apply(Z1, relu_deriv);
        Mat dZ1 = hadamard(dA1, reluPrime);   // (m x h)

        // dW1 = (X^T dZ1)/m
        Mat dW1 = scale(matmul(T(X), dZ1), 1.0 / m);   // (in x h)
        // db1 = sum_rows(dZ1)/m
        Mat db1 = scale(sum_rows(dZ1), 1.0 / m);       // (1 x h)

        // SGD updates
        sgd_update(W2, dW2, lr);
        sgd_update(b2, db2, lr);
        sgd_update(W1, dW1, lr);
        sgd_update(b1, db1, lr);
    }
};

// ---------------------------
// Demo: XOR
// ---------------------------
int main() {
    // XOR dataset (m=4)
    Mat X(4,2);
    X(0,0)=0; X(0,1)=0;
    X(1,0)=0; X(1,1)=1;
    X(2,0)=1; X(2,1)=0;
    X(3,0)=1; X(3,1)=1;

    Mat Y(4,1);
    Y(0,0)=0;
    Y(1,0)=1;
    Y(2,0)=1;
    Y(3,0)=0;

    Net net(/*in_dim=*/2, /*hidden=*/4, /*seed=*/123);

    double lr = 0.1;
    int epochs = 10000;

    for (int e=1;e<=epochs;e++) {
        Mat preds = net.forward(X);
        double loss = bce_loss(Y, preds);
        net.backward_and_update(X, Y, lr);

        if (e % 1000 == 0) {
            std::cout << "epoch " << std::setw(5) << e
                      << "  loss=" << std::fixed << std::setprecision(6) << loss
                      << "\n";
        }
    }

    Mat preds = net.forward(X);
    std::cout << "\nPredictions:\n";
    for (int i=0;i<4;i++) {
        double p = preds(i,0);
        int cls = (p >= 0.5) ? 1 : 0;
        std::cout << X(i,0) << " XOR " << X(i,1)
                  << " -> p=" << std::fixed << std::setprecision(4) << p
                  << "  class=" << cls << "\n";
    }
    return 0;
}
