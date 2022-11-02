#pragma once

#include "history_buffer.hpp"
#include "cubic_spline.hpp"
#include "potential_function.hpp"

#include <Eigen/Eigen>

#include <cfloat>
#include <climits>
#include <cmath>
#include <iostream>
#include <vector>

namespace path_smoother {

class PathSmoother
{
   public:
    PathSmoother(const Eigen::Vector2d& head_point, const Eigen::Vector2d& tail_point, const int& piece_num,
                 const Eigen::Matrix3Xd& disk_obstacles, const double penalty_weight)
        : piece_num_(static_cast<size_t>(piece_num)),
          disk_obstacles_(disk_obstacles),
          penalty_weight_(penalty_weight),
          head_point_(head_point),
          tail_point_(tail_point),
          cubic_spline_(head_point, tail_point, piece_num),
          potential_function_(disk_obstacles, penalty_weight, piece_num),
          c1_(1e-4),
          c2_(0.9),
          memory_size_(8),
          epsilon_(1e-3),
          delta_x_buffer_(memory_size_),
          delta_g_buffer_(memory_size_),
          rho_buffer_(memory_size_)
    {
        inner_points_.resize(2, piece_num_ - 1);
    }

    double GetPenaltyWeight() { return penalty_weight_; }

    CubicSpline& GetCubicSpline() { return cubic_spline_; }

    Eigen::Matrix3Xd GetDiskObstacle() { return disk_obstacles_; }

    inline double optimize(CubicCurve& curve, const Eigen::Matrix2Xd& init_inner_points, const double& rel_cost_tol)
    {
        ROS_INFO("Starting optimization");
        // Set init inner points
        epsilon_ = rel_cost_tol;
        inner_points_ = init_inner_points;
        UpdateInnerPoints(inner_points_);

        // Perform Optimization
        Eigen::MatrixXd g = GetGrad();
        Eigen::MatrixXd direction = g;
        int iteration = 0;

        ROS_INFO_STREAM("Before enter while loop g.norm() = " << g.norm() << " | cost --> " << GetCost());
        std::cout << "while loop condition: "<< static_cast<int>(g.norm() > epsilon_) << "\n";
        while (g.norm() > epsilon_)
        {
            double cost = GetCost();
            ROS_INFO_STREAM("iteration#" << iteration << ", cost --> " << cost);
            // inner_points_: 2*(n-1), direction: (n-1*2)
            double t = LewisOvertonLineSearch(inner_points_, direction);
            auto next_inner_points = inner_points_ + t * direction.transpose();
            // calc new grad
            UpdateInnerPoints(next_inner_points);
            auto next_g = GetGrad();

            // update search direction
            auto delta_g = next_g - g;
            auto delta_x = next_inner_points - inner_points_;
            auto next_direction = CautiousLimitedMemoryBFGSUpdate(g, delta_g, delta_x);

            // update g and direction
            g = next_g;
            direction = next_direction;
            inner_points_ = next_inner_points;
            iteration += 1;
        }

        UpdateInnerPoints(inner_points_);
        curve = cubic_spline_.GetCurve();
        double min_cost = GetCost();
        return min_cost;
    }

   private:
    void UpdateInnerPoints(const Eigen::Matrix2Xd& inner_points)
    {
        cubic_spline_.Update(inner_points);
        potential_function_.Update(inner_points);
    }

    double LewisOvertonLineSearch(const Eigen::Matrix2Xd& cur_inner_points, const Eigen::MatrixXd& direction)
    {
        double L = 0.0;
        double u = std::numeric_limits<double>::max();
        double alpha = 1.0;

        while (true)
        {
            UpdateInnerPoints(cur_inner_points);
            auto cost1 = GetCost();
            auto grad1 = GetGrad();  // Dimension :  (n-1) * 2

            auto new_inner_points = cur_inner_points + alpha * direction;
            UpdateInnerPoints(new_inner_points);
            auto cost2 = GetCost();
            auto grad2 = GetGrad();

            // element-wise production
            auto production1 = (direction.cwiseProduct(grad1)).sum();
            auto production2 = (direction.cwiseProduct(grad2)).sum();

            if (!S_Func(alpha, cost1, cost2, production1))
            {
                u = alpha;
            }
            else if (!C_Func(production1, production2))
            {
                L = alpha;
            }
            else
            {
                return alpha;
            }

            if (u < std::numeric_limits<double>::max())
            {
                alpha = (L + u) / 2;
            }
            else
            {
                alpha = 2 * L;
            }
        }
    }

    bool S_Func(double alpha, double cost1, double cost2, double production1)
    {
        // direction : 2*(n-1), grad1: (n-1) * 1
        return cost1 - cost2 >= -c1_ * alpha * production1;
    }

    bool C_Func(double production1, double production2) { return production2 >= c2_ * production1; }

    Eigen::MatrixXd CautiousLimitedMemoryBFGSUpdate(const Eigen::MatrixXd& g, const Eigen::MatrixXd& delta_g, const Eigen::MatrixXd& delta_x)
    {
        // L-BFGS two for loop update
        auto d = g;
        auto delta_x_history = delta_x_buffer_.GetHistory();
        auto delta_g_history = delta_g_buffer_.GetHistory();
        auto rho_history = rho_buffer_.GetHistory();
        size_t m = delta_x_history.size();
        std::vector<double> alpha_history(m, 0.0);

        for (int i = m-1; i >= 0; --i)
        {
            auto alpha = rho_history[i] * (delta_x_history[i].transpose().cwiseProduct(d).sum());
            d = d - alpha * delta_g_history[i];
            alpha_history[i] = alpha;
        }

        auto gamma = rho_history[m-1] * delta_g_history[m-1].norm();
        d = d / gamma;

        for (size_t i = 0; i < m; ++i)
        {
            auto beta = rho_history[i] * (delta_g_history[i].cwiseProduct(d).sum());
            d = d + delta_x_history[i].transpose() * (alpha_history[i] - beta);
        }

        // Update buffer
        delta_x_buffer_.AddData(delta_x);
        delta_g_buffer_.AddData(delta_g);
        auto rho = 1.0 / (delta_g.transpose().cwiseProduct(delta_x)).sum();
        rho_buffer_.AddData(rho);

        return d;
    }

    double GetCost()
    {
        double stretch_energy = cubic_spline_.GetStretchEnergy();
        double potential_cost = potential_function_.GetCost();
        ROS_INFO_STREAM("stretch_energy --> " << stretch_energy << " | potential_cost --> " << potential_cost);
        return stretch_energy + potential_cost;
    }

    Eigen::MatrixXd GetGrad()
    {
        // dimension: (n-1) * 1
        auto cubic_grad = cubic_spline_.GetGradients();
        ROS_INFO_STREAM("cubic_grad size: row = " << cubic_grad.rows() << " col = " << cubic_grad.cols());
        // (n-1) * 2
        Eigen::MatrixXd grad1(piece_num_-1, 2);
        grad1.col(0) << cubic_grad;
        grad1.col(1) << cubic_grad;

        ROS_INFO_STREAM("energy grad.norm() --> " << grad1.norm());

        // dimension: (n-1) * 2
        Eigen::MatrixXd grad2 = potential_function_.GetGradients();

        ROS_INFO_STREAM("potential grad.norm() --> " << grad2.norm());

        // dimension: (n-1) * 2
        Eigen::MatrixXd grad = grad1 + grad2;

        return grad;
    }

   private:
    size_t piece_num_;
    Eigen::Matrix3Xd disk_obstacles_;
    double penalty_weight_;
    Eigen::Vector2d head_point_;
    Eigen::Vector2d tail_point_;
    Eigen::Matrix2Xd inner_points_;

    CubicSpline cubic_spline_;
    PotentialFunction potential_function_;

    // coefficient weak Wolfe conditions
    double c1_;
    double c2_;
    // memory size for L-BFGS
    size_t memory_size_;
    // convergence limit
    double epsilon_;

    // L-BFGS buffer
    HistoryBuffer<Eigen::MatrixXd> delta_x_buffer_;
    HistoryBuffer<Eigen::MatrixXd> delta_g_buffer_;
    HistoryBuffer<double> rho_buffer_;
};

}  // namespace path_smoother