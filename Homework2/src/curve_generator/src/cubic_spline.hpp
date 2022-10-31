#pragma once

#include "Eigen/Core"
#include "cubic_curve.hpp"

class CubicSpline
{
   public:
    CubicSpline() = default;

    ~CubicSpline() = default;

    void Initialize(const Eigen::Vector2d& head_point, const Eigen::Vector2d& tail_point, int pieces_num)
    {
        head_point_ = head_point;
        tail_point_ = tail_point;
        pieces_num_ = static_cast<size_t>(pieces_num);

        // Init matrix A and inverse of A
        std::cout << "Init matrix A and inverse of A\n";
        inner_points_num_ = pieces_num_ - 1;
        A_.resize(inner_points_num_, inner_points_num_);
        inv_A_.resize(inner_points_num_, inner_points_num_);
        grad_B_.resize(inner_points_num_, inner_points_num_);
        grad_D_.resize(inner_points_num_, inner_points_num_);
        b_.resize(inner_points_num_, 2);

        // Set values for matrix A_
        size_t bandwidth = 1;
        for (size_t i = 0; i < inner_points_num_; ++i)
        {
            for (size_t j = 0; j < inner_points_num_; ++j)
            {
                if (j < i - bandwidth || j > i + bandwidth)
                {
                    A_(i, j) = 0.0;
                }
                else if (i == j)
                {
                    A_(i, j) = 4.0;
                }
                else
                {
                    // j == i - bandwidth or j == i + bandwidth
                    A_(i, j) = 1.0;
                }
            }
        }

        // Get the inverse matrix of A_
        inv_A_ = A_.inverse();

        // Get grad_B_ and grad_D_
        std::cout << "Get grad_B_ and grad_D_\n";
        for (size_t i = 0; i < inner_points_num_; ++i)
        {
            for (size_t j = 0; j < inner_points_num_; ++j)
            {
                if (j < i - bandwidth || j > i + bandwidth)
                {
                    grad_B_(i, j) = 0.0;
                }
                else if (i == j)
                {
                    grad_B_(i, j) = 0.0;
                }
                else if (j == i - bandwidth)
                {
                    grad_B_(i, j) = -3.0;
                }
                else
                {
                    // j == i + bandwidth
                    grad_B_(i, j) = 3.0;
                }
            }
        }

        grad_D_ = inv_A_ * grad_B_;  // matrix multiplication

        // Init grad_1_
        std::cout << "Init grad_1_\n";
        grad_1_.setZero(pieces_num_, inner_points_num_);
        for (size_t i = 0; i < pieces_num_; ++i)
        {
            for (size_t j = 0; j < inner_points_num_; ++j)
            {
                if (i == j)
                {
                    grad_1_(i, j) = -1.0;
                }
                else if (j < i)
                {
                    grad_1_(i, j) = 1.0;
                }
            }
        }

        // Init b_
        std::cout << "Init b_\n";
        b_.setZero();

        // Init gradients size for inner_points: [x1, x2, ..., x_{n-1}]
        std::cout << "Init gradients size for inner_points\n";
        inner_points_gradients_.resize(inner_points_num_, 2);
        inner_points_gradients_.setZero(inner_points_num_, 2);
    }

    void Update(const Eigen::Matrix2Xd& inner_points)
    {
        UpdateInnerPoints(inner_points);

        UpdateMatrixB();

        UpdateCubicCurve();
    }

    double GetStretchEnergy() const
    {
        // Must be called after Update() function is called
        double energy = 0.0;
        for(size_t i = 0; i < static_cast<size_t>(cubic_curve_.getPieceNum()); ++i)
        {
            const auto& piece = cubic_curve_[i];
            const auto& coef_mat = piece.getCoeffMat();
            auto ci = coef_mat.col(2);
            auto di = coef_mat.col(3);
            // 12di^T * di + 4ci^T * ci + 12ci^T*di
            energy += 12 * di.squaredNorm() + 4 * ci.squaredNorm() + 12 * (ci(0) * di(0) + ci(1) * di(1));
        }
        return energy;
    }

    const Eigen::Matrix2Xd& GetGradients()
    {
        return inner_points_gradients_;
    }

   private:
    void UpdateInnerPoints(const Eigen::Matrix2Xd& inner_points)
    {
        if (static_cast<size_t>(inner_points.cols()) != inner_points_num_)
        {
            ROS_ERROR("Size of inner_points not equal the initialized size!");
            return;
        }

        // Update inner_points
        inner_points_ = inner_points;
    }

    void UpdateMatrixB()
    {
        // Update b_, index of points is in 0 ~ n, 0 for head point, n for tail point, 1 ~ n-1 for inner points x1 ~
        // x_{n-1} D0 = Dn = (0, 0), D1 ~ D_{n-1} saved in b_

        for (size_t i = 0; i < inner_points_num_; ++i)
        {
            if (i == 0)
            {
                // D1 = 3*(x2 - x0)
                b_(i, 0) = 3 * (inner_points_(0, 1) - 0);
                b_(i, 1) = 3 * (inner_points_(1, 1) - 0);
            }
            else if (i < inner_points_num_ - 1)
            {
                // For Di = D2 ~ D_{n-2}, Di = 3*(x{i+2} - xi)
                size_t idx1 = i - 1;
                size_t idx2 = (i + 2) - 1;
                auto col1 = inner_points_.col(idx1);
                auto col2 = inner_points_.col(idx2);
                b_(i, 0) = 3 * (col2(0) - col1(0));
                b_(i, 1) = 3 * (col2(1) - col1(1));
            }
            else
            {
                // D_{n-1} = 3*(xn - x_{n-2}), i == inner_points_num_ - 1
                auto col = inner_points_.col(i - 1);  // x_{n-2}
                b_(i, 0) = 3 * (0 - col(0));
                b_(i, 1) = 3 * (0 - col(1));
            }
        }
    }

    void UpdateCubicCurve()
    {
        // Update coefficients for each piece of cubic_curve_
        std::vector<Eigen::Matrix<double, 2, 4>> coef_vec;

        // The first piece's coefficients
        Eigen::Matrix<double, 2, 4> coef_mat0;
        // a0 = x0
        coef_mat0(0, 0) = head_point_(0);
        coef_mat0(1, 0) = head_point_(1);
        // b0 = D0
        coef_mat0(0, 1) = 0;
        coef_mat0(1, 1) = 0;
        // c0 = 3*(x1 - x0) - 2*D0 - D1
        coef_mat0(0, 2) = 3 * (inner_points_(0, 0) - head_point_(0)) - 2 * 0 - b_(0, 0);
        coef_mat0(1, 2) = 3 * (inner_points_(1, 0) - head_point_(1)) - 2 * 0 - b_(0, 1);
        // d0 = 2*(x0 - x1) + D0 + D1
        coef_mat0(0, 3) = 2 * (head_point_(0) - inner_points_(0, 0)) + 0 + b_(0, 0);
        coef_mat0(1, 3) = 2 * (head_point_(1) - inner_points_(1, 0)) + 0 + b_(0, 1);
        coef_vec.emplace_back(coef_mat0);

        // middle pieces
        for (size_t i = 1; i < pieces_num_ - 1; ++i)
        {
            Eigen::Matrix<double, 2, 4> coef_mat;

            auto xi = inner_points_.col(i - 1);
            auto xi_next = inner_points_.col(i);
            auto Di = b_.row(i - 1);
            auto Di_next = b_.row(i);
            // a_i = x_i
            coef_mat(0, 0) = xi(0);
            coef_mat(1, 0) = xi(1);
            // b_i = D_i
            coef_mat(0, 1) = Di(0);
            coef_mat(1, 1) = Di(1);
            // c_i = 3*(x_{i+1} - x_i) - 2*D_i - D_{i+1}
            coef_mat(0, 2) = 3 * (xi_next(0) - xi(0)) - 2 * Di(0) - Di_next(0);
            coef_mat(1, 2) = 3 * (xi_next(1) - xi(1)) - 2 * Di(1) - Di_next(1);
            // d_i = 2*(x_i - x_{i+1}) + D_i + D_{i+1}
            coef_mat(0, 3) = 2 * (xi(0) - xi_next(0)) + Di(0) + Di_next(0);
            coef_mat(1, 3) = 2 * (xi(1) - xi_next(1)) + Di(1) + Di_next(1);

            coef_vec.emplace_back(coef_mat);
        }

        // Total points num (including start point and end point) is n+1, The last piece (start from x_{n-1} --> to end
        // x_{n})
        Eigen::Matrix<double, 2, 4> coef_mat_last;
        // x_{n-1}
        auto prev_xn = inner_points_.col(inner_points_num_ - 1);
        // D_{n-1}
        auto prev_Dn = b_.row(inner_points_num_ - 1);
        // a_{n-1} = x_{n-1}
        coef_mat_last(0, 0) = prev_xn(0);
        coef_mat_last(1, 0) = prev_xn(1);
        // b_{n-1} = D_{n-1}
        coef_mat_last(0, 1) = prev_Dn(0);
        coef_mat_last(1, 1) = prev_Dn(1);
        // c_{n-1} = 3*(x_n - x_{n-1}) - 2*D_{n-1} - D_n
        coef_mat_last(0, 2) = 3 * (tail_point_(0) - prev_xn(0)) - 2 * prev_Dn(0) - 0;
        coef_mat_last(1, 2) = 3 * (tail_point_(1) - prev_xn(1)) - 2 * prev_Dn(1) - 0;
        // d_{n-1} = 2*(x_{n-1} - x_{n}) + D_{n-1} + D_{n}
        coef_mat_last(0, 3) = 2 * (prev_xn(0) - tail_point_(0)) + prev_Dn(0) + 0;
        coef_mat_last(1, 3) = 2 * (prev_xn(1) - tail_point_(1)) + prev_Dn(1) + 0;

        coef_vec.emplace_back(coef_mat_last);

        std::vector<double> duration(coef_vec.size(), 1.0);
        cubic_curve_ = CubicCurve(duration, coef_vec);
    }

    void UpdateGradient()
    {
        // Calc gradients for [x1, x2, ..., x_{n-1}]
        inner_points_gradients_.setZero(inner_points_num_, 2);

        for(size_t i = 0; i < static_cast<size_t>(cubic_curve_.getPieceNum()); ++i)
        {
            const auto& piece = cubic_curve_[i];
            const auto& coef_mat = piece.getCoeffMat();
            Eigen::Vector2d ci = coef_mat.col(2);
            Eigen::Vector2d di = coef_mat.col(3);

            assert(ci.size() == di.size());
            /*
             For a piece,
                    gradient = 24 * grad(di)^T * di + 12 * grad(ci)^T * di + 12 * grad(di)^T * ci + 8 * grad(ci)^T * ci
             where,
                    g1 = d(xi - x_{i+1}) / dx, g2 = dDi/dx, g3 = dD_{i+1}/dx
                    grad(di) = 2 * g1 + g2 + g3
                    grad(ci) = -3 * g1 - 2 * g2 - g3
             */
            Eigen::VectorXd g1 = grad_1_.row(i);
            Eigen::VectorXd g2 = grad_D_.row(i);

            Eigen::VectorXd g3;
            g3.resize(inner_points_num_);
            if (i < static_cast<size_t>(cubic_curve_.getPieceNum()) - 1)
            {
                g3 = grad_D_.row(i+1);
            }
            else
            {
                g3.setZero();
            }

            auto grad_di = 2 * g1 + g2 + g3;
            auto grad_ci = -3 * g1 - 2 * g2 - g3;

            Eigen::MatrixX2d new_grad_di;
            new_grad_di.col(0) = grad_di;
            new_grad_di.col(1) = grad_di;

            Eigen::MatrixX2d new_grad_ci;
            new_grad_ci.col(0) << grad_ci;
            new_grad_ci.col(1) << grad_ci;

            std::cout << "ci.size(): " << ci.size();
            std::cout << "di.size(): " << di.size();
            std::cout << "new_grad_ci.size(): " << new_grad_ci.size();
            std::cout << "new_grad_di.size(): " << new_grad_di.size();

            Eigen::MatrixX2d gradient;
//            gradient.setZero(inner_points_num_, 2);
//            gradient = 24 * new_grad_di * di + 12 * new_grad_ci * di + 12 * new_grad_di * ci + 8 * grad_ci * ci;
        }
    }

   private:
    // n = pieces_num_
    size_t pieces_num_;
    // inner_points_num_ = n - 1
    size_t inner_points_num_;
    Eigen::Vector2d head_point_;
    Eigen::Vector2d tail_point_;
    // dimension: (n-1) * 2
    Eigen::MatrixXd inner_points_;
    // dimension: (n-1) * 2
    Eigen::MatrixXd inner_points_gradients_;
    // dimension:: (n-1) * (n-1)
    Eigen::MatrixXd A_;
    Eigen::MatrixXd inv_A_;
    Eigen::MatrixXd grad_B_;
    Eigen::MatrixXd grad_D_;
    // g1 = d(xi - x_{i+1})/dx, dimension: n*(n-1), n is piece num
    Eigen::MatrixXd grad_1_;
    // dimension: (n-1) * 2
    Eigen::MatrixXd b_;
    CubicCurve cubic_curve_;
};