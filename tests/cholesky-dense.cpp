/// @file
/// @author Wilson Jallet
/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#include "util.hpp"
#include "proxsuite-nlp/ldlt-allocator.hpp"

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/dataset.hpp>

namespace utf = boost::unit_test;
using namespace proxsuite::nlp;
using context::MatrixXs;
using context::VectorXs;

auto TEST_DIMS = utf::data::make({4, 8, 16, 32, 55, 67, 122, 240, 540, 813});
static constexpr std::size_t num_rhs_test = 100;
const Eigen::Index m = 4;

template <class Dec, class Rhs, class Sol>
void rhs_solve_test(const MatrixXs &M, const Dec &dec, const Rhs &rhs, Sol &sol,
                    double tol) {
  sol = rhs;
  dec.solveInPlace(sol);
  auto err = math::infty_norm(rhs - M * sol);
  BOOST_CHECK_LE(err, tol);
}

BOOST_DATA_TEST_CASE(llt_pos, TEST_DIMS, n) {
  Eigen::Rand::P8_mt19937_64 urng{42};
  auto Mgen = Eigen::Rand::makeWishartGen(n + 1, MatrixXs::Identity(n, n));
  MatrixXs M = Mgen.generate(urng);
  Eigen::LLT<MatrixXs> llt(M);
  auto rM = llt.reconstructedMatrix();
  BOOST_TEST(rM.isApprox(M, 1e-10));

  MatrixXs rhs(n, m);
  MatrixXs sol(n, m);
  for (std::size_t i = 0; i < num_rhs_test; i++) {
    rhs = Eigen::Rand::normal<VectorXs>(n, m, urng);
    rhs_solve_test(M, llt, rhs, sol, 1e-9);
  }
}

BOOST_DATA_TEST_CASE(ldlt_goe, TEST_DIMS, n) {
  Eigen::Rand::P8_mt19937_64 urng{42};
  MatrixXs M = sampleGaussianOrthogonalEnsemble(n);
  Eigen::LDLT<MatrixXs> ldlt(M);

  MatrixXs rhs(n, m);
  MatrixXs sol(n, m);
  for (std::size_t i = 0; i < num_rhs_test; i++) {
    rhs = Eigen::Rand::normal<VectorXs>(n, m, urng);
    rhs_solve_test(M, ldlt, rhs, sol, 1e-9);
  }
}

BOOST_DATA_TEST_CASE(bunchkaufman_goe, TEST_DIMS, n) {
  Eigen::Rand::P8_mt19937_64 urng{42};
  MatrixXs M = sampleGaussianOrthogonalEnsemble(n);
  Eigen::BunchKaufman<MatrixXs> ldlt(M);

  MatrixXs rhs(n, m);
  MatrixXs sol(n, m);
  for (std::size_t i = 0; i < num_rhs_test; i++) {
    rhs = Eigen::Rand::normal<VectorXs>(n, m, urng);
    rhs_solve_test(M, ldlt, rhs, sol, 1e-9);
  }
}
