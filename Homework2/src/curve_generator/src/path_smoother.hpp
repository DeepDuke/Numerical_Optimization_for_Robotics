#pragma once

#include "cubic_spline.hpp"
#include "L-BFGS.hpp"

#include <Eigen/Eigen>

#include <cmath>
#include <cfloat>
#include <iostream>
#include <vector>

namespace path_smoother {

class PathSmoother
{
   private:
    CubicSpline cubic_spline;

    int pieceN;
    Eigen::Matrix3Xd diskObstacles;
    double penaltyWeight;
    Eigen::Vector2d headP;
    Eigen::Vector2d tailP;
    Eigen::Matrix2Xd points;
    Eigen::Matrix2Xd gradByPoints;

   private:
    static inline double costFunction()
    {
        double cost = 0.0;
        return cost;
    }

   public:
    double GetPenaltyWeight() { return penaltyWeight; }

    CubicSpline& GetCubicSpline()
    {
        return cubic_spline;
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

        cubic_spline.Initialize(initialP, terminalP, pieceNum);

        points.resize(2, pieceN - 1);
        gradByPoints.resize(2, pieceN - 1);

        return true;
    }

    inline double optimize(CubicCurve &curve,
                           const Eigen::Matrix2Xd &iniInPs,
                           const double &relCostTol)
    {
        cubic_spline.Update(iniInPs);
        double min_cost = 0.0;
        return min_cost;
    }
};


} // namespace path_smoother