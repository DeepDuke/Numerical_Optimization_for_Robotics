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
    Eigen::Matrix2Xd inner_points_grad_;

    CubicSpline cubic_spline_;
    PotentialFunction potential_function_;

   private:
    static inline double costFunction()
    {
        double cost = 0.0;
        return cost;
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
        inner_points_grad_.resize(2, piece_num_ - 1);
    }

    double GetPenaltyWeight() { return penalty_weight_; }

    CubicSpline& GetCubicSpline() { return cubic_spline_; }

    Eigen::Matrix3Xd GetDiskObstacle() { return disk_obstacles_; }

    inline double optimize(CubicCurve& curve, const Eigen::Matrix2Xd& iniInPs, const double& relCostTol)
    {
        cubic_spline_.Update(iniInPs);
        // Perform Optimization

        double min_cost = 0.0;
        return min_cost;
    }
};

}  // namespace path_smoother