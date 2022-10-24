#ifndef CUBIC_SPLINE_HPP
#define CUBIC_SPLINE_HPP

#include "cubic_curve.hpp"

#include <Eigen/Eigen>

#include <cmath>
#include <vector>

namespace cubic_spline
{

    // The banded system class is used for solving
    // banded linear system Ax=b efficiently.
    // A is an N*N band matrix with lower band width lowerBw
    // and upper band width upperBw.
    // Banded LU factorization has O(N) time complexity.
    class BandedSystem
    {
    public:
        // The size of A, as well as the lower/upper
        // banded width p/q are needed
        inline void create(const int &n, const int &p, const int &q)
        {
            // In case of re-creating before destroying
            destroy();
            N = n;
            lowerBw = p;
            upperBw = q;
            int actualSize = N * (lowerBw + upperBw + 1);
            ptrData = new double[actualSize];
            std::fill_n(ptrData, actualSize, 0.0);
            return;
        }

        inline void destroy()
        {
            if (ptrData != nullptr)
            {
                delete[] ptrData;
                ptrData = nullptr;
            }
            return;
        }

    private:
        int N;
        int lowerBw;
        int upperBw;
        // Compulsory nullptr initialization here
        double *ptrData = nullptr;

    public:
        // Reset the matrix to zero
        inline void reset(void)
        {
            std::fill_n(ptrData, N * (lowerBw + upperBw + 1), 0.0);
            return;
        }

        // The band matrix is stored as suggested in "Matrix Computation"
        inline const double &operator()(const int &i, const int &j) const
        {
            return ptrData[(i - j + upperBw) * N + j];
        }

        inline double &operator()(const int &i, const int &j)
        {
            return ptrData[(i - j + upperBw) * N + j];
        }

        // This function conducts banded LU factorization in place
        // Note that NO PIVOT is applied on the matrix "A" for efficiency!!!
        inline void factorizeLU()
        {
            int iM, jM;
            double cVl;
            for (int k = 0; k <= N - 2; ++k)
            {
                iM = std::min(k + lowerBw, N - 1);
                cVl = operator()(k, k);
                for (int i = k + 1; i <= iM; ++i)
                {
                    if (operator()(i, k) != 0.0)
                    {
                        operator()(i, k) /= cVl;
                    }
                }
                jM = std::min(k + upperBw, N - 1);
                for (int j = k + 1; j <= jM; ++j)
                {
                    cVl = operator()(k, j);
                    if (cVl != 0.0)
                    {
                        for (int i = k + 1; i <= iM; ++i)
                        {
                            if (operator()(i, k) != 0.0)
                            {
                                operator()(i, j) -= operator()(i, k) * cVl;
                            }
                        }
                    }
                }
            }
            return;
        }

        // This function solves Ax=b, then stores x in b
        // The input b is required to be N*m, i.e.,
        // m vectors to be solved.
        template <typename EIGENMAT>
        inline void solve(EIGENMAT &b) const
        {
            int iM;
            for (int j = 0; j <= N - 1; ++j)
            {
                iM = std::min(j + lowerBw, N - 1);
                for (int i = j + 1; i <= iM; ++i)
                {
                    if (operator()(i, j) != 0.0)
                    {
                        b.row(i) -= operator()(i, j) * b.row(j);
                    }
                }
            }
            for (int j = N - 1; j >= 0; --j)
            {
                b.row(j) /= operator()(j, j);
                iM = std::max(0, j - upperBw);
                for (int i = iM; i <= j - 1; ++i)
                {
                    if (operator()(i, j) != 0.0)
                    {
                        b.row(i) -= operator()(i, j) * b.row(j);
                    }
                }
            }
            return;
        }

        // This function solves ATx=b, then stores x in b
        // The input b is required to be N*m, i.e.,
        // m vectors to be solved.
        template <typename EIGENMAT>
        inline void solveAdj(EIGENMAT &b) const
        {
            int iM;
            for (int j = 0; j <= N - 1; ++j)
            {
                b.row(j) /= operator()(j, j);
                iM = std::min(j + upperBw, N - 1);
                for (int i = j + 1; i <= iM; ++i)
                {
                    if (operator()(j, i) != 0.0)
                    {
                        b.row(i) -= operator()(j, i) * b.row(j);
                    }
                }
            }
            for (int j = N - 1; j >= 0; --j)
            {
                iM = std::max(0, j - lowerBw);
                for (int i = iM; i <= j - 1; ++i)
                {
                    if (operator()(j, i) != 0.0)
                    {
                        b.row(i) -= operator()(j, i) * b.row(j);
                    }
                }
            }
        }
    };

    class CubicSpline
    {
    public:
        CubicSpline() = default;
        ~CubicSpline() { A.destroy(); }

   public:
        // interpolated points
        Eigen::Matrix2Xd points_;
    private:
        int N;
        Eigen::Vector2d headP;
        Eigen::Vector2d tailP;
        BandedSystem A;
        Eigen::MatrixX2d b;

    public:
        inline void setConditions(const Eigen::Vector2d &headPos,
                                  const Eigen::Vector2d &tailPos,
                                  const int &pieceNum)
        {
           //TODO
            N = pieceNum - 1;  // dimension of A is N*N
            headP = headPos;
            tailP = tailPos;
            int p = 1;
            int q = 1;
            A.create(N, p, q);
            // TODO: Store value for matrix A
            A(0, 0) = 4;
            A(0, 1) = 1;
            for (size_t i = 1; i < N-1; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                {
                    // Jump over non-band parameters
                    if (j < i - q || j > i + p)
                    {
                        continue ;
                    }

                    if (i == j)
                    {
                        A(i, j) = 4;
                    }
                    else
                    {
                        A(i, j) = 1;
                    }
                }
            }
            A(N-1, N-2) = 1;
            A(N-1, N-1) = 4;

            return;
        }

        inline void setInnerPoints(const Eigen::Ref<const Eigen::Matrix2Xd> &inPs)
        {
            //TODO Set value for vector b
            points_ = inPs;
            b.resize(N, 2);
            for (size_t i = 0; i < b.rows(); ++i)
            {
                if (i == 0)
                {
                    b(i, 0) = 3 * (points_(0, i + 2 - 1) - headP(0));
                    b(i, 1) = 3 * (points_(1, i + 2 - 1) - headP(1));
                }
                else if (i < b.rows() - 1)
                {
                    b(i, 0) = 3 * (points_(0, i+2 - 1) - points_(0, i - 1));
                    b(i, 1) = 3 * (points_(1, i+2 - 1) - points_(1, i - 1));
                }
                else
                {
                    b(i, 0) = 3 * (tailP(0) - points_(0, i - 1));
                    b(i, 1) = 3 * (tailP(1) - points_(1, i - 1));
                }
            }

            // Solve matrix A
            A.solve<Eigen::MatrixX2d>(b);

            return;
        }

        inline void getCurve(CubicCurve &curve) const
        {
            // TODO
            int piece_num = N + 1;
            std::vector<Eigen::Matrix<double, 2, 4>> coef_vec;

            // The first piece's coefficients
            Eigen::Matrix<double, 2, 4> coef_mat0;
            // a0 = x0
            coef_mat0(0, 0) = headP(0);
            coef_mat0(1, 0) = headP(1);
            // b0 = D0
            coef_mat0(0, 1) = 0;
            coef_mat0(1, 1) = 0;
            // c0 = 3*(x1 - x0) - 2*D0 - D1
            coef_mat0(0, 2) = 3*(points_(0, 0) - headP(0)) - 2*0 - b(0, 0);
            coef_mat0(1, 2) = 3*(points_(1, 0) - headP(1)) - 2*0 - b(0, 1);
            // d0 = 2*(x0 - x1) + D0 + D1
            coef_mat0(0, 3) = 2*(headP(0) - points_(0, 0)) + 0 + b(0, 0);
            coef_mat0(1, 3) = 2*(headP(1) - points_(1, 0)) + 0 + b(0, 1);

            coef_vec.emplace_back(coef_mat0);

            // middle pieces
            for (size_t i = 1; i < piece_num-1; ++i)
            {
                Eigen::Matrix<double, 2, 4> coef_mat;
                // a_i = x_i
                coef_mat(0, 0) = points_(0, i-1);
                coef_mat(1, 0) = points_(1, i-1);
                // b_i = D_i
                coef_mat(0, 1) = b(i-1 , 0);
                coef_mat(1, 1) = b(i-1, 1);
                // c_i = 3*(x_{i+1} - x_i) - 2*D_i - D_{i+1}
                coef_mat(0, 2) = 3 * (points_(0, i) - points_(0, i-1)) - 2 * b(i-1, 0) - b(i, 0);
                coef_mat(1, 2) = 3 * (points_(1, i) - points_(1, i-1)) - 2 * b(i-1, 1) - b(i, 1);
                // d_i = 2*(x_i - x_{i+1}) + D_i + D_{i+1}
                coef_mat(0, 3) = 2 * (points_(0, i-1) - points_(0, i)) + b(i-1, 0) + b(i, 0);
                coef_mat(1, 3) = 2 * (points_(1, i-1) - points_(1, i)) + b(i-1, 1) + b(i, 1);

                coef_vec.emplace_back(coef_mat);
            }

            // Total points num (including start point and end point) is N+2, The last piece (start from x_{N} --> x_{N+1})
            Eigen::Matrix<double, 2, 4> coef_mat_last;
            // a_{N} = x_{N}
            coef_mat_last(0, 0) = points_(0, N-1);
            coef_mat_last(1, 0) = points_(1, N-1);
            // b_{N} = D_{N}
            coef_mat_last(0, 1) = b(N-1, 0);
            coef_mat_last(1, 1) = b(N-1, 1);
            // c_{N} = 3*(x_{N+1} - x_{N}) - 2*D_{N} - D_{N+1}
            coef_mat_last(0, 2) = 3 * (tailP(0) - points_(0, N-1)) - 2 * b(N-1, 0) - 0;
            coef_mat_last(1, 2) = 3 * (tailP(1) - points_(1, N-1)) - 2 * b(N-1, 1) - 0;
            // d_{N} = 2*(x_{N} - x_{N+1}) + D_{N} + D_{N+1}
            coef_mat_last(0, 3) = 2 * (points_(0, N-1) - tailP(0)) + b(N-1, 0) + 0;
            coef_mat_last(1, 3) = 2 * (points_(1, N-1) - tailP(1)) + b(N-1, 1) + 0;

            coef_vec.emplace_back(coef_mat_last);

            std::vector<double> duration(coef_vec.size(), 1.0);
            curve = CubicCurve(duration, coef_vec);

            return;
        }

        inline void getStretchEnergy(double &energy) const
        {
            // TODO
            // for a single segment, the energy is 4*(c1^2 + c2^2) + 12*(c1*d1 + c2*d2) + 12*(d1^2 + d2^2)
            CubicCurve curve;
            getCurve(curve);

            for (size_t i = 0; i < curve.getPieceNum(); ++i)
            {
                auto piece = curve[i];
                auto coef = piece.getCoeffMat();
                auto c1 = coef(0, 2);
                auto c2 = coef(1, 2);
                auto d1 = coef(0, 3);
                auto d2 = coef(1, 3);
                auto piece_energy = 4*(c1*c1 + c2*c2) + 12*(c1*d1 + c2*d2) +12*(d1*d1 + d2*d2);

                energy += piece_energy;
            }
            return;
        }

        inline const Eigen::MatrixX2d &getCoeffs(void) const
        {
            return b;
        }

        inline void getGrad(Eigen::Ref<Eigen::Matrix2Xd> gradByPoints) const
        {
            // TODO
            // Compute gradients of points: gradient = b_i + 2*c_i*t+3*d_i*t^2, t \in [0, 1]
            CubicCurve curve;
            getCurve(curve);

            for (size_t i = 0; i < gradByPoints.cols(); ++i)
            {
                auto piece = curve[i];
                auto coef = piece.getCoeffMat();
                // t = 1
                gradByPoints.col(i) = coef.col(1) + 2 * coef.col(2) + 3 * coef.col(3);
            }
        }
    };
}

#endif
