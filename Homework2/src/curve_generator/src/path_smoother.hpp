#pragma once

#include "L-BFGS.hpp"
#include "cubic_spline.hpp"
#include "potential_function.hpp"

#include <Eigen/Eigen>

#include <cfloat>
#include <cmath>
#include <iostream>
#include <vector>

namespace path_smoother {

class PathSmoother
{
   private:
    size_t piece_num_;
    Eigen::Matrix3Xd disk_obstacles_;
    double penalty_weight_;
    Eigen::Vector2d head_point_;
    Eigen::Vector2d tail_point_;
    Eigen::Matrix2Xd inner_points_;

    CubicSpline cubic_spline_;
    PotentialFunction potential_function_;

   private:
    double GetCost()
    {
        double stretch_energy = cubic_spline_.GetStretchEnergy();
        double potential_cost = potential_function_.GetCost();
        return stretch_energy + potential_cost;
    }

    Eigen::MatrixXd GetGrad()
    {
        // dimension: (n-1) * 1
        Eigen::MatrixXd cubic_spline_grad = cubic_spline_.GetGradients();
        Eigen::MatrixXd grad1(piece_num_-1, 2);
        grad1.col(0) = cubic_spline_grad;
        grad1.col(1) = cubic_spline_grad;

        // dimension: (n-1) * 1
        Eigen::MatrixXd grad2 = potential_function_.GetGradients();

        Eigen::MatrixXd grad = grad1 + grad2;

        return grad;
    }

   public:
    PathSmoother(const Eigen::Vector2d& head_point, const Eigen::Vector2d& tail_point, const int& piece_num,
                 const Eigen::Matrix3Xd& disk_obstacles, const double penalty_weight)
        : piece_num_(static_cast<size_t>(piece_num)),
          disk_obstacles_(disk_obstacles),
          penalty_weight_(penalty_weight),
          head_point_(head_point),
          tail_point_(tail_point),
          cubic_spline_(head_point, tail_point, piece_num),
          potential_function_(disk_obstacles, penalty_weight, piece_num)
    {
        inner_points_.resize(2, piece_num_ - 1);
    }

    double GetPenaltyWeight() { return penalty_weight_; }

    CubicSpline& GetCubicSpline() { return cubic_spline_; }

    Eigen::Matrix3Xd GetDiskObstacle() { return disk_obstacles_; }

    inline double optimize(CubicCurve& curve, const Eigen::Matrix2Xd& init_inner_points, const double& rel_cost_tol)
    {
        // Set init inner points
        cubic_spline_.Update(init_inner_points);
        potential_function_.Update(init_inner_points);

        // Perform Optimization

        double min_cost = 0.0;
        return min_cost;
    }
};

}  // namespace path_smoother