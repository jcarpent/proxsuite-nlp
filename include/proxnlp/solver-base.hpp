/**
 * @file solver-base.hpp
 * @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
 */
#pragma once

#include "proxnlp/fwd.hpp"
#include "proxnlp/problem-base.hpp"
#include "proxnlp/pdal.hpp"
#include "proxnlp/workspace.hpp"
#include "proxnlp/results.hpp"
#include "proxnlp/helpers-base.hpp"
#include "proxnlp/logger.hpp"
#include "proxnlp/bcl-params.hpp"

#include "proxnlp/modelling/costs/squared-distance.hpp"

#include "proxnlp/linesearch-base.hpp"

namespace proxnlp {

enum class MultiplierUpdateMode { NEWTON, PRIMAL, PRIMAL_DUAL };

enum class HessianApprox {
  /// Exact Hessian construction from provided function Hessians
  EXACT,
  /// Gauss-Newton (or rather SCQP) approximation
  GAUSS_NEWTON,
  // BFGS
};

template <typename _Scalar> class SolverTpl {
public:
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = ProblemTpl<Scalar>;
  using Workspace = WorkspaceTpl<Scalar>;
  using Results = ResultsTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using LinesearchOptions = typename Linesearch<Scalar>::Options;
  using CallbackPtr = shared_ptr<helpers::base_callback<Scalar>>;
  using ConstraintSet = ConstraintSetBase<Scalar>;

  enum InertiaFlag { INERTIA_OK = 0, INERTIA_BAD = 1, INERTIA_HAS_ZEROS = 2 };

  /// Manifold on which to optimize.
  shared_ptr<Problem> problem_;
  /// Merit function.
  PDALFunction<Scalar> merit_fun;
  /// Proximal regularization penalty.
  QuadraticDistanceCost<Scalar> prox_penalty;

  /// Level of verbosity of the solver.
  VerboseLevel verbose = QUIET;
  /// Use a Gauss-Newton approximation for the Lagrangian Hessian.
  HessianApprox hess_approx = HessianApprox::GAUSS_NEWTON;
  /// Linesearch strategy.
  LinesearchStrategy ls_strat = LinesearchStrategy::ARMIJO;
  MultiplierUpdateMode mul_up_mode = MultiplierUpdateMode::NEWTON;

  //// Algorithm proximal parameters

  Scalar inner_tol0 = 1.;
  Scalar prim_tol0 = 1.;
  Scalar inner_tol_ = inner_tol0;
  Scalar prim_tol_ = prim_tol0;
  Scalar rho_init_; // Initial primal proximal penalty parameter.
  Scalar mu_init_;  // Initial penalty parameter.
private:
  Scalar rho_ = rho_init_;   // Primal proximal penalty parameter.
  Scalar mu_ = mu_init_;     // Penalty parameter.
  Scalar mu_inv_ = 1. / mu_; // Inverse penalty parameter.
public:
  Scalar inner_tol_min = 1e-9; // Lower safeguard for the subproblem tolerance.
  Scalar mu_upper_ = 1.;
  Scalar mu_lower_ = 1e-9; // Lower safeguard for the penalty parameter.

  /// BCL strategy parameters.
  BCLParams<Scalar> bcl_params;

  /// Linesearch options.
  LinesearchOptions ls_options;

  /// Target tolerance for the problem.
  Scalar target_tol;

  /// Logger.
  BaseLogger logger{};

  //// Parameters for the inertia-correcting strategy.

  const Scalar del_inc_k = 8.;
  const Scalar del_inc_big = 100.;
  const Scalar del_dec_k = 1. / 3.;

  const Scalar DELTA_MIN = 1e-14; // Minimum nonzero regularization strength.
  const Scalar DELTA_MAX = 1e6;   // Maximum regularization strength.
  const Scalar DELTA_NONZERO_INIT = 1e-4;
  Scalar DELTA_INIT = 0.;

  /// Solver maximum number of iterations.
  std::size_t max_iters = 100;

  /// Callbacks
  std::vector<CallbackPtr> callbacks_;

  SolverTpl(shared_ptr<Problem> prob, const Scalar tol = 1e-6,
            const Scalar mu_eq_init = 1e-2, const Scalar rho_init = 0.,
            const VerboseLevel verbose = QUIET, const Scalar mu_lower = 1e-9,
            const Scalar prim_alpha = 0.1, const Scalar prim_beta = 0.9,
            const Scalar dual_alpha = 1., const Scalar dual_beta = 1.,
            const LinesearchOptions ls_options = LinesearchOptions());

  const Manifold &manifold() const { return *problem_->manifold_; }

  /**
   * @brief Solve the problem.
   *
   * @param workspace
   * @param results
   * @param x0    Initial guess.
   * @param lams0 Initial Lagrange multipliers given separately for each
   * constraint.
   *
   */
  ConvergenceFlag solve(Workspace &workspace, Results &results,
                        const ConstVectorRef &x0,
                        const std::vector<VectorRef> &lams0);

  /**
   * @copybrief solve().
   *
   * @param workspace
   * @param results
   * @param x0    Initial guess.
   * @param lams0 Initial Lagrange multipliers given separately for each
   * constraint.
   *
   */
  ConvergenceFlag solve(Workspace &workspace, Results &results,
                        const ConstVectorRef &x0, const ConstVectorRef &lams0);

  ConvergenceFlag solve(Workspace &workspace, Results &results,
                        const ConstVectorRef &x0);

  void solveInner(Workspace &workspace, Results &results);

  /// Update penalty parameter using the provided factor (with a safeguard
  /// SolverTpl::mu_lower).
  inline void updatePenalty();

  /// @brief Set penalty parameter, its inverse and update the merit function.
  /// @param new_mu The new penalty parameter.
  void setPenalty(const Scalar &new_mu) noexcept;

  /// Set proximal penalty parameter.
  void setProxParameter(const Scalar &new_rho) noexcept;

  /// @brief    Add a callback to the solver instance.
  inline void registerCallback(const CallbackPtr &cb) noexcept {
    callbacks_.push_back(cb);
  }

  /// @brief    Remove all callbacks from the instance.
  void clearCallbacks() noexcept { callbacks_.clear(); }

  /**
   * @brief Update primal-dual subproblem tolerances upon failure (insufficient
   * primal feasibility)
   *
   * This is called upon initialization of the solver.
   */
  void updateToleranceFailure() noexcept;

  /**
   * @brief Update primal-dual subproblem tolerances upon  successful outer-loop
   * iterate (good primal feasibility)
   */
  void updateToleranceSuccess() noexcept;

  inline void tolerancePostUpdate() noexcept;

  /// @brief  Accept Lagrange multiplier estimates.
  void acceptMultipliers(const Results &results, Workspace &workspace) const;

  /**
   * Evaluate the problem data, as well as the proximal/projection operators,
   * and the first-order & primal-dual multiplier estimates.
   *
   * @param inner_lams_data Inner (SQP) dual variables
   * @param workspace       Problem workspace.
   */
  void computeMultipliers(const ConstVectorRef &inner_lams_data,
                          Workspace &workspace) const;

  /**
   * Evaluate the derivatives (cost gradient, Hessian, constraint Jacobians,
   * vector-Hessian products) of the problem data.
   *
   * @param x         Primal variable
   * @param workspace Problem workspace.
   * @param second_order Whether to compute the second-order information; set to
   * false for e.g. linesearch.
   */
  void computeConstraintDerivatives(const ConstVectorRef &x,
                                    Workspace &workspace,
                                    bool second_order) const;

  /**
   * Take a trial step.
   *
   * @param manifold  Working space/manifold
   * @param workspace Workspace
   * @param results   Contains the previous primal-dual point
   * @param alpha     Step size
   */
  static void tryStep(const Manifold &manifold, Workspace &workspace,
                      const Results &results, Scalar alpha);

  void invokeCallbacks(Workspace &workspace, Results &results) {
    for (auto cb : callbacks_) {
      cb->call(workspace, results);
    }
  }

  /**
   * @brief Check the matrix has the desired inertia.
   * @param signature The computed inertia as a vector of ints valued -1, 0,
   * or 1.
   * @param delta     Scale factor for the identity matrix to add
   */
  InertiaFlag checkInertia(const Eigen::VectorXi &signature) const;
};

} // namespace proxnlp

#include "proxnlp/solver-base.hxx"
