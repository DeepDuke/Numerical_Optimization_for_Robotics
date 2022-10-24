#ifndef PATH_SMOOTHER_HPP
#define PATH_SMOOTHER_HPP

#include "cubic_spline.hpp"
#include "lbfgs.hpp"

#include <Eigen/Eigen>

#include <cmath>
#include <cfloat>
#include <iostream>
#include <vector>

namespace path_smoother
{

    class PathSmoother
    {
    private:
        cubic_spline::CubicSpline cubSpline;

        int pieceN;
        Eigen::Matrix3Xd diskObstacles;
        double penaltyWeight;
        Eigen::Vector2d headP;
        Eigen::Vector2d tailP;
        Eigen::Matrix2Xd points;
        Eigen::Matrix2Xd gradByPoints;

        lbfgs::lbfgs_parameter_t lbfgs_params;

    private:
        static inline double costFunction(void *ptr,
                                          const Eigen::VectorXd &x,
                                          Eigen::VectorXd &g)
        {
            //TODO
            PathSmoother& path_smoother = *(PathSmoother*)ptr;

            // Compute the energy of cubic spline curve
            double energy = 0.0;
            auto& cubic_spline = path_smoother.GetCubicSpline();

            Eigen::Matrix2Xd inPts(2, x.size()/2);
            std::cout << "x.size() --> " << x.size() << "\n";
            for (size_t i = 0; i < x.size(); i += 2)
            {
                inPts(0, i/2) = x(i);
                inPts(1, i/2) = x(i+1);
            }

            cubic_spline.setInnerPoints(inPts);
            cubic_spline.getStretchEnergy(energy);

            // Compute the obstacle potential
            auto obstacles = path_smoother.GetDiskObstacle();
            double potential = 0.0;
            for (size_t i = 0; i < x.size(); i += 2)
            {
                for (size_t j = 0; j < obstacles.cols(); ++j)
                {
                    double dist_to_obs = sqrt(pow(x[i] - obstacles(0, j), 2) + pow(x[i+1] - obstacles(1, j), 2));
                    potential += std::max<double>(obstacles(2, j) - dist_to_obs, 0);
                }
            }
            potential *= path_smoother.GetPenaltyWeight();

            // Compute the total cost
            double cost = energy + potential;

            return cost;
        }

    public:
        double GetPenaltyWeight() { return penaltyWeight; }

        cubic_spline::CubicSpline& GetCubicSpline()
        {
            return cubSpline;
        }

        Eigen::Matrix3Xd GetDiskObstacle()
        {
            return diskObstacles;
        }

            inline bool setup(const Eigen::Vector2d &initialP,
                          const Eigen::Vector2d &terminalP,
                          const int &pieceNum,
                          const Eigen::Matrix3Xd &diskObs,
                          const double penaWeight)
        {
            pieceN = pieceNum;
            diskObstacles = diskObs;
            penaltyWeight = penaWeight;
            headP = initialP;
            tailP = terminalP;

            cubSpline.setConditions(headP, tailP, pieceN);

            points.resize(2, pieceN - 1);
            gradByPoints.resize(2, pieceN - 1);

            return true;
        }

        inline double optimize(CubicCurve &curve,
                               const Eigen::Matrix2Xd &iniInPs,
                               const double &relCostTol)
        {
            //TODO
            double minCost = 0.0;

            lbfgs_params.mem_size = 18;
            lbfgs_params.g_epsilon = 0.0;
            lbfgs_params.min_step = 1.0e-32;
            lbfgs_params.past = 3;
            lbfgs_params.delta = 1.0e-7;

            points = iniInPs;
            Eigen::VectorXd x(points.size()*2);
            for (size_t i = 0; i < points.cols(); ++i)
            {
                x(i) = points(0, i);
                x(i+1) = points(1, i);
            }

            int ret = lbfgs::lbfgs_optimize(x,
                                            minCost,
                                            &PathSmoother::costFunction,
                                            nullptr,
                                            this,
                                            lbfgs_params);

            std::cout << "[path_smoother] the min_cost is: " << minCost << "\n";

            if (minCost < relCostTol)
            {
                std::cout << "optimize successfully, minCost=" << minCost << " is smaller than " << relCostTol;
            }
            else
            {
                std::cout << "optimize successfully, minCost=" << minCost << " is smaller than " << relCostTol;
            }

            // Update curve
            cubSpline.getCurve(curve);

            return minCost;
        }
    };

}

#endif
