template <typename Vector>
struct VectorTraits {
    typedef typename Vector::Scalar Scalar;
    typedef typename Vector::Index Index;

    static Vector zero(Index size);
    static Scalar dot(const Vector& v1, const Vector& v2);
    static Scalar norm(const Vector& v);
    // other vector operations...
};


template <>
struct VectorTraits<Eigen::VectorXd> {
    typedef double Scalar;
    typedef int Index;

    static Eigen::VectorXd zero(Index size) {
        return Eigen::VectorXd::Zero(size);
    }
    static Scalar dot(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) {
        return v1.dot(v2);
    }
    static Scalar norm(const Eigen::VectorXd& v) {
        return v.norm();
    }
};


template <typename Matrix, typename Vector>
struct MatrixTraits {
    typedef typename Matrix::Scalar Scalar;
    typedef typename Matrix::Index Index;

    static Matrix identity(Index rows, Index cols);
    static Matrix transpose(const Matrix& m);
    static Matrix inverse(const Matrix& m);
    static Vector solve(const Matrix& A, const Vector& b);
    static Vector operator*(const Matrix& A, const Vector& x);
    // other matrix operations...
};


template <>
struct MatrixTraits<Eigen::MatrixXd, Eigen::VectorXd> {
    typedef double Scalar;
    typedef int Index;

    static Eigen::MatrixXd identity(Index rows, Index cols) {
        return Eigen::MatrixXd::Identity(rows, cols);
    }
    static Eigen::MatrixXd transpose(const Eigen::MatrixXd& m) {
        return m.transpose();
    }
    static Eigen::MatrixXd inverse(const Eigen::MatrixXd& m) {
        return m.inverse();
    }
    static Eigen::VectorXd solve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
        return A.colPivHouseholderQr().solve(b);
    }
    static Eigen::VectorXd operator*(const Eigen::MatrixXd& A, const Eigen::VectorXd& x) {
        assert(A.cols() == x.rows());
        return A * x;
    }
};


template <>
struct MatrixTraits<Eigen::MatrixXd> {
    typedef double Scalar;
    typedef int Index;

    static Eigen::MatrixXd identity(Index rows, Index cols) {
        return Eigen::MatrixXd::Identity(rows, cols);
    }
    static Eigen::MatrixXd transpose(const Eigen::MatrixXd& m) {
        return m.transpose();
    }
    static Eigen::MatrixXd inverse(const Eigen::MatrixXd& m) {
        return m.inverse();
    }

    static Scalar maxValue(const Eigen::MatrixXd& m) {
        return m.maxCoeff();
    }
};


template <>
struct MatrixTraits<Armadillo::Mat<double>> {
    typedef double Scalar;
    typedef int Index;

    static Armadillo::Mat<double> identity(Index rows, Index cols) {
        return Armadillo::eye<Armadillo::Mat<double>>(rows, cols);
    }
    static Armadillo::Mat<double> transpose(const Armadillo::Mat<double>& m) {
        return m.t();
    }
    static Armadillo::Mat<double> inverse(const Armadillo::Mat<double>& m) {
        return m.i();
    }

    static Scalar maxValue(const Armadillo::Mat<double>& m) {
        return m.max();
    }
};