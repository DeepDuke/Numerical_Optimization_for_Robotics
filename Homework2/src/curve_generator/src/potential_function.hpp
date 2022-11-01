#pragma once

#include <Eigen/Eigen>
#include <algorithm>
#include <vector>

/**
 * @brief Potential(x1, x2, ..., x_{n-1}) = penalty_weight * \sum_{i=1}^{N-1}\sum{j=1}{M}max(r_j -|| x_i - o_j ||, 0)
 */
class PotentialFunction
{
   public:
    PotentialFunction(const Eigen::Matrix3Xd& disk_obstacles, double penalty_weight, size_t piece_num)
        : disk_obstacles_(disk_obstacles), penalty_weight_(penalty_weight), piece_num_(piece_num), cost_(0.0)
    {
        grad_.setZero(piece_num_ - 1, 2);
    }

    void Update(const Eigen::Matrix2Xd& inner_points)
    {
        // Calculate potential cost and gradient
        cost_ = 0.0;
        size_t obs_num = disk_obstacles_.rows();
        grad_.setZero(piece_num_ - 1, 2);

        for (size_t i = 0; i < piece_num_ - 1; ++i)
        {
            Eigen::Vector2d point;
            point.setZero();
            point(0) = inner_points(0, i);
            point(1) = inner_points(1, i);

            for (size_t j = 0; j < obs_num; ++j)
            {
                double r = disk_obstacles_.col(j)(2);

                Eigen::Vector2d obs;
                obs.setZero();
                obs(0) = disk_obstacles_(0, j);
                obs(1) = disk_obstacles_(1, j);
                double dist_to_obs = (point - obs).norm();

                cost_ += std::max<double>(r - dist_to_obs, 0);

                grad_(i, 0) += -penalty_weight_ * (point(0) - obs(0)) / dist_to_obs;
                grad_(i, 1) += -penalty_weight_ * (point(1) - obs(1)) / dist_to_obs;
            }
        }
    }

    double GetCost() { return cost_; }

    const Eigen::MatrixXd& GetGradients() { return grad_; }

   private:
    // disk shape obstacles in map
    Eigen::Matrix3Xd disk_obstacles_;
    // penalty coefficient
    double penalty_weight_;
    // piece num
    size_t piece_num_;
    // potential cost
    double cost_;
    // dimension (n-1) * 2, n is # of pieces
    Eigen::MatrixXd grad_;
};