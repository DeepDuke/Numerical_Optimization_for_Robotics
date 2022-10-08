#include <cmath>
#include <iostream>
#include <map>
#include <vector>
#include "matplotlibcpp.h"
#include "tic_toc.h"

namespace plt = matplotlibcpp;

class Rosenbrock
{
   public:
    Rosenbrock(size_t n) : n_(n), c_(1e-4), eps_(1e-6) { Solve(); }

    Rosenbrock(size_t n, double c) : n_(n), c_(c), eps_(1e-6) { Solve(); }

    Rosenbrock(size_t n, double c, double eps) : n_(n), c_(c), eps_(eps) { Solve(); }

    /**
     * @brief perform line search gradient descent with Armijo condition
     */
    void Solve();

    /**
     * @brief Get mid-state of x during solving process
     */
    std::vector<std::vector<double>>& GetTrajectory();

    /**
     * @brief Get solution
     */
    std::vector<double>& GetSolution();

    /**
     * @brief get function value
     * @param x: input variable
     */
    double func(const std::vector<double>& x);

    /**
     * @brief Print optimal x
     */
    void PrintSolution();

    /**
     * @brief Print all mid-state of x during solving process
     */
    void PrintTrajectory();

   private:
    /**
     * @brief Calculate gradient at x
     * @param x
     */
    std::vector<double> CalcGradient(const std::vector<double>& x);

    /**
     * @brief Check if gradient descent algorithm is converged
     * @param grad: current gradient
     * @return: whether converged
     */
    bool IsConverged(const std::vector<double>& grad);

    /**
     * @brief Perform Armijo Line Search
     * @param x: current x
     * @param d: search direction
     */
    double LineSearch(const std::vector<double>& x, const std::vector<double>& d);

   private:
    // dimension
    size_t n_;
    // line search param
    double c_;
    // precision threshold
    double eps_;
    // trajectory
    std::vector<std::vector<double>> trajectory_;
    // solution
    std::vector<double> solution_;
    // final grad
    std::vector<double> final_grad_;
    // time for get optimal solution (unit: ms)
    double running_time_;
};

double Rosenbrock::func(const std::vector<double>& x)
{
    double value = 0.0;
    for (size_t offset = 0; offset <= static_cast<size_t>(n_ / 2) - 1; ++offset)
    {
        size_t i = offset + 1;
        // x_{2i-1}
        double a = x[(2 * i - 1) - 1];
        // x_{2i}
        double b = x[2 * i - 1];

        value += 100 * (a * a - b) * (a * a - b) + (a - 1) * (a - 1);
    }

    return value;
}

std::vector<double> Rosenbrock::CalcGradient(const std::vector<double>& x)
{
    std::vector<double> grad(n_, 0.0);
    for (size_t offset = 0; offset < n_; ++offset)
    {
        size_t i = offset + 1;
        // odd case
        if (i % 2 == 1)
        {
            // x_i
            double a = x[offset];
            // x_{i+1}
            double b = x[offset + 1];
            // gradient
            grad[offset] = 400 * a * a * a - 400 * a * b + 2 * a - 2;
        }
        // even case
        else
        {
            // x_i
            double a = x[offset];
            // x_{i-1}
            double b = x[offset - 1];
            // gradient
            grad[offset] = 200 * (a - b * b);
        }
    }

    return grad;
}

bool Rosenbrock::IsConverged(const std::vector<double>& grad)
{
    size_t cnt = 0;
    for (size_t i = 0; i < n_; ++i)
    {
        if (std::fabs(grad[i]) < eps_)
        {
            cnt += 1;
        }
    }

    return cnt == n_;
}

double Rosenbrock::LineSearch(const std::vector<double>& x, const std::vector<double>& d)
{
    double tau = 1.0;
    while (true)
    {
        // f(x + tau * d)
        std::vector<double> new_x(n_, 0.0);
        for (size_t i = 0; i < n_; ++i)
        {
            new_x[i] = x[i] + tau * d[i];
        }
        auto f_new_x = func(new_x);

        // f(x)
        auto f_x = func(x);

        // rhs term
        double rhs_term = 0.0;
        for (size_t i = 0; i < n_; ++i)
        {
            rhs_term += c_ * tau * d[i] * (-d[i]);
        }

        // check Armijo condition
        if (f_new_x > f_x + rhs_term)
        {
            tau *= 0.5;
        }
        else
        {
            break;
        }
    }

    return tau;
}

void Rosenbrock::Solve()
{
    TicToc timer;
    // check input
    if (n_ % 2 != 0 || n_ < 0)
    {
        std::cerr << "Input dimension is not positive even integer!!!\n";
        return;
    }

    // clear
    solution_.clear();
    trajectory_.clear();
    solution_.reserve(n_);

    // initial value
    std::vector<double> x(n_, 0.0);
    trajectory_.emplace_back(x);
    // search direction
    std::vector<double> d(n_, 0.0);

    while (true)
    {
        // calc gradient
        std::vector<double> grad = CalcGradient(x);

        // check convergence
        if (IsConverged(grad))
        {
            solution_ = x;
            final_grad_ = grad;
            break;
        }

        for (size_t i = 0; i < n_; ++i)
        {
            d[i] = -grad[i];
        }

        // find tau through line search
        auto tau = LineSearch(x, d);

        // update x
        for (size_t i = 0; i < n_; ++i)
        {
            x[i] = x[i] + tau * d[i];
        }
        trajectory_.emplace_back(x);
    }

    running_time_ = timer.toc();
}

std::vector<std::vector<double>>& Rosenbrock::GetTrajectory() { return trajectory_; }

std::vector<double>& Rosenbrock::GetSolution() { return solution_; }

void Rosenbrock::PrintSolution()
{
    if (solution_.empty())
    {
        std::cout << "Solution is empty!\n";
        return;
    }

    std::cout << "For n = " << n_ << ":\n"
              << "Solution is: [";

    for (size_t i = 0; i < n_; ++i)
    {
        std::cout << solution_[i];
        if (i < n_ - 1)
        {
            std::cout << ", ";
        }
        else
        {
            std::cout << "]\n";
        }
    }

    std::cout << "Converge Threshold is: epsilon --> " << eps_ << "\n";
    std::cout << "Final Gradient is: [";

    for (size_t i = 0; i < n_; ++i)
    {
        std::cout << final_grad_[i];
        if (i < n_ - 1)
        {
            std::cout << ", ";
        }
        else
        {
            std::cout << "]\n";
        }
    }

    std::cout << "Solving time: " << running_time_ << " ms\n\n";
}

void Rosenbrock::PrintTrajectory()
{
    if (trajectory_.empty())
    {
        std::cout << "Trajectory is empty!\n";
        return;
    }

    std::cout << "Trajectory is: \n\n[";

    for (size_t i = 0; i < trajectory_.size(); ++i)
    {
        auto x = trajectory_[i];

        std::cout << "iteration #" << i << ": [";

        for (size_t j = 0; j < n_; ++j)
        {
            std::cout << x[j];
            if (j < n_ - 1)
            {
                std::cout << ", ";
            }
            else
            {
                std::cout << "]\n\n";
            }
        }
    }
}

int main()
{
    size_t n = 2;
    Rosenbrock solver(n);

    solver.PrintTrajectory();
    solver.PrintSolution();

    // Plot for n = 2
    if (n == 2)
    {
        auto trajectory = solver.GetTrajectory();
        plt::figure();
        std::vector<double> X;
        std::vector<double> Y;

        for (const auto& ele : trajectory)
        {
            X.emplace_back(ele[0]);
            Y.emplace_back(ele[1]);
        }
        plt::plot(X, Y, "c--");

        std::vector<double> init_X = {0.0};
        std::vector<double> init_Y = {0.0};
        std::map<std::string, std::string> settings;

        plt::plot(init_X, init_Y, "ro");

        auto solution = solver.GetSolution();
        std::vector<double> end_X = {solution[0]};
        std::vector<double> end_Y = {solution[1]};
        plt::plot(end_X, end_Y, "g*");

        plt::title("iteration for N = 2");
        plt::save("iteration.png");
    }

    // n = 10
    size_t n2 = 10;
    Rosenbrock solver2(n2);

    solver2.PrintSolution();

    return 0;
}
