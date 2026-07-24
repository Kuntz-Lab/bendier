// Numerical checks of the robot-level Jacobians (as opposed to the
// per-factor analytic Jacobians in JacobianTests.cpp). These Jacobians
// (e.g. TendonRobotMarginals::J_pose_tensions) are derived from a solved
// factor graph's marginal covariances (Sigma_TQ * Sigma_QQ^-1), not
// hand-written analytic derivatives, so they're validated here against
// gtsam::numericalDerivative applied to "solve the whole graph, read off
// the output" as a black-box function. Done in C++ (rather than Python)
// so the same gtsam::Pose3::Local/Retract chart is used on both sides of
// the comparison, with no separate reimplementation of the tangent-space
// convention and no Python/numpy round-tripping to introduce ambiguity.
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>

#include "parallel_robot/ParallelRobotSolver.h"
#include "rigid_robot/RigidJointTorqueFactor.h"
#include "rigid_robot/RigidRobotSolver.h"
#include "tendon_robot/TendonRobotSolver.h"

#include <cmath>
#include <functional>

using namespace gtsam;

namespace {

// ---------------------------------------------------------------------
// Tendon robot: mirrors python/scripts/tendon_robot/config.py's DEFAULTS.
// ---------------------------------------------------------------------

constexpr int kNumTendons = 4;

Matrix6 TendonRodKInv() {
  double d = 0.0012;
  double E = 40.0e9;
  double G = 15.0e9;

  double I = M_PI * std::pow(d, 4) / 64.0;
  double J = 2.0 * I;
  double A = M_PI * d * d / 4.0;

  Vector6 diag;
  diag << 1.0 / (E * I), 1.0 / (E * I), 1.0 / (G * J),
          1.0 / (G * A), 1.0 / (G * A), 1.0 / (E * A);
  return diag.asDiagonal();
}

TendonRoutingInput TendonInput() {
  TendonRoutingInput routing;
  routing.routing_radius = 0.01;
  routing.params = {
      RoutingFunctionParams{/*angle_offset=*/0.0,             /*total_angle=*/2.0 * M_PI},
      RoutingFunctionParams{/*angle_offset=*/M_PI,             /*total_angle=*/0.0},
      RoutingFunctionParams{/*angle_offset=*/3.0 * M_PI / 2.0, /*total_angle=*/0.0},
      RoutingFunctionParams{/*angle_offset=*/0.0,             /*total_angle=*/0.0},
  };
  return routing;
}

TendonRobotSolverConfig BuildTendonConfig() {
  return TendonRobotSolverConfig(
      /*rod_length=*/0.25, /*num_discs=*/9, /*num_between_nodes=*/3,
      TendonRodKInv(),
      /*sigma_strain_rot=*/0.1, /*sigma_strain_pos=*/0.01,
      /*sigma_small_force=*/1.0e-4, /*sigma_small_moment=*/1.0e-5,
      /*sigma_base_pose_pos=*/1.0e-4, /*sigma_base_pose_rot=*/1.0e-3,
      TendonInput());
}

Pose3 TendonSolveTipPose(
    TendonRobotSolver& solver,
    const std::optional<VectorXGaussian>& tensions,
    const std::optional<Vector6Gaussian>& tip_wrench,
    const std::optional<VectorXGaussian>& displacement_meas = std::nullopt)
{
  auto sol = solver.solve(tensions, tip_wrench, std::nullopt, displacement_meas);
  return Pose3(sol.marginals.rod.states.back().pose.mean);
}

// A displacement vector picked by hand isn't guaranteed physically
// consistent for this routing, which slows/destabilizes convergence and
// biases the finite-difference comparison. Instead derive a
// guaranteed-consistent displacement from an ordinary tension-commanded
// solve and read off its resulting displacement -- same approach as the
// prior Python test this replaces.
Vector TendonConsistentDisplacement0(
    const TendonRobotSolverConfig& config, const Vector& tensions0, const Vector6Gaussian& tip_wrench)
{
  TendonRobotSolver solver(config);
  Matrix tight_cov = 1e-16 * Matrix::Identity(kNumTendons, kNumTendons);
  auto sol = solver.solve(VectorXGaussian{tensions0, tight_cov}, tip_wrench, std::nullopt);
  return sol.marginals.displacements.mean;
}

// J_pose_displacements/J_tension_displacements have much larger-magnitude
// entries than J_pose_tensions (observed up to ~10^2-10^3 for this
// routing), so a single fixed *absolute* tolerance appropriate to one
// scale is badly mismatched to the other -- compare relative error
// against the Jacobian's own scale instead, same as the Python tests this
// replaces did.
double RelativeError(const Matrix& actual, const Matrix& expected) {
  return (actual - expected).norm() / expected.norm();
}

}  // namespace

TEST(TendonRobotModel, J_pose_tensions_matches_numerical_zero_wrench) {
  TendonRobotSolverConfig config = BuildTendonConfig();
  Matrix tight_cov = 1e-16 * Matrix::Identity(kNumTendons, kNumTendons);
  Vector6Gaussian tip_wrench{Vector6::Zero(), 1e-12 * Matrix6::Identity()};
  Vector tensions0 = (Vector(kNumTendons) << 1.0, 0.5, 0.3, 0.2).finished();

  std::function<Pose3(const Vector&)> h = [&](const Vector& tensions) -> Pose3 {
    TendonRobotSolver solver(config);  // fresh instance -- no warm-start bias
    return TendonSolveTipPose(solver, VectorXGaussian{tensions, tight_cov}, tip_wrench);
  };

  Matrix J_numeric = numericalDerivative11<Pose3, Vector, kNumTendons>(h, tensions0, 1e-6);

  TendonRobotSolver solver(config);
  auto sol0 = solver.solve(VectorXGaussian{tensions0, tight_cov}, tip_wrench, std::nullopt);
  Matrix J = sol0.marginals.J_pose_tensions;

  EXPECT(assert_equal(J_numeric, J, 1e-3));
}

TEST(TendonRobotModel, J_pose_tensions_matches_numerical_loaded_wrench) {
  TendonRobotSolverConfig config = BuildTendonConfig();
  Matrix tight_cov = 1e-16 * Matrix::Identity(kNumTendons, kNumTendons);

  // A genuine, nonzero tip load (same order of magnitude as
  // JacobianTests.cpp's wrench_1()/wrench_2()), pinned tightly -- exercises
  // the Jacobian at an operating point where the rod is actually bent by an
  // external wrench, not only at the unloaded configuration.
  Vector6 loaded_mean = (Vector6() << 0.003, -0.002, 0.0015, 0.02, 0.05, -0.03).finished();
  Vector6Gaussian tip_wrench{loaded_mean, 1e-12 * Matrix6::Identity()};
  Vector tensions0 = (Vector(kNumTendons) << 1.0, 0.5, 0.3, 0.2).finished();

  std::function<Pose3(const Vector&)> h = [&](const Vector& tensions) -> Pose3 {
    TendonRobotSolver solver(config);
    return TendonSolveTipPose(solver, VectorXGaussian{tensions, tight_cov}, tip_wrench);
  };

  Matrix J_numeric = numericalDerivative11<Pose3, Vector, kNumTendons>(h, tensions0, 1e-6);

  TendonRobotSolver solver(config);
  auto sol0 = solver.solve(VectorXGaussian{tensions0, tight_cov}, tip_wrench, std::nullopt);
  Matrix J = sol0.marginals.J_pose_tensions;

  EXPECT(assert_equal(J_numeric, J, 1e-3));
}

TEST(TendonRobotModel, J_pose_displacements_matches_numerical_zero_wrench) {
  TendonRobotSolverConfig config = BuildTendonConfig();
  Vector6Gaussian tip_wrench{Vector6::Zero(), 1e-12 * Matrix6::Identity()};
  Vector tensions0 = (Vector(kNumTendons) << 1.0, 0.5, 0.3, 0.2).finished();
  Vector displacement0 = TendonConsistentDisplacement0(config, tensions0, tip_wrench);

  Matrix tight_cov = 1e-16 * Matrix::Identity(kNumTendons, kNumTendons);
  Matrix loose_tensions_cov = 400.0 * Matrix::Identity(kNumTendons, kNumTendons);
  Vector zero_tensions = Vector::Zero(kNumTendons);

  TendonRobotSolver solver(config);  // reused/warm-started -- same convergence-speed rationale as before

  std::function<Pose3(const Vector&)> h = [&](const Vector& displacement) -> Pose3 {
    return TendonSolveTipPose(
        solver, VectorXGaussian{zero_tensions, loose_tensions_cov}, tip_wrench,
        VectorXGaussian{displacement, tight_cov});
  };

  Matrix J_numeric = numericalDerivative11<Pose3, Vector, kNumTendons>(h, displacement0, 1e-6);

  auto sol0 = solver.solve(
      VectorXGaussian{zero_tensions, loose_tensions_cov}, tip_wrench, std::nullopt,
      VectorXGaussian{displacement0, tight_cov});
  Matrix J = sol0.marginals.J_pose_displacements;

  EXPECT(RelativeError(J, J_numeric) < 1e-3);
}

TEST(TendonRobotModel, J_tension_displacements_matches_numerical_zero_wrench) {
  TendonRobotSolverConfig config = BuildTendonConfig();
  Vector6Gaussian tip_wrench{Vector6::Zero(), 1e-12 * Matrix6::Identity()};
  Vector tensions0 = (Vector(kNumTendons) << 1.0, 0.5, 0.3, 0.2).finished();
  Vector displacement0 = TendonConsistentDisplacement0(config, tensions0, tip_wrench);

  Matrix tight_cov = 1e-16 * Matrix::Identity(kNumTendons, kNumTendons);
  Matrix loose_tensions_cov = 400.0 * Matrix::Identity(kNumTendons, kNumTendons);
  Vector zero_tensions = Vector::Zero(kNumTendons);

  TendonRobotSolver solver(config);

  std::function<Vector(const Vector&)> h = [&](const Vector& displacement) -> Vector {
    auto sol = solver.solve(
        VectorXGaussian{zero_tensions, loose_tensions_cov}, tip_wrench, std::nullopt,
        VectorXGaussian{displacement, tight_cov});
    return sol.marginals.tensions.mean;
  };

  Matrix J_numeric = numericalDerivative11<Vector, Vector, kNumTendons>(h, displacement0, 1e-6);

  auto sol0 = solver.solve(
      VectorXGaussian{zero_tensions, loose_tensions_cov}, tip_wrench, std::nullopt,
      VectorXGaussian{displacement0, tight_cov});
  Matrix J = sol0.marginals.J_tension_displacements;

  EXPECT(RelativeError(J, J_numeric) < 1e-3);
}

namespace {

// ---------------------------------------------------------------------
// Rigid robot: a small synthetic serial chain (mixed revolute/prismatic,
// arbitrary nontrivial offsets/axes) rather than a real URDF -- the point
// is validating J_tip_joints' formula, not any specific robot's geometry,
// and there's no URDF loader on the C++ side of this repo.
// ---------------------------------------------------------------------

// At least 6 joints are needed for a torque-sensing solve to fully observe
// a 6-dof tip wrench (3 joints leave the wrench's linear system rank
// deficient/indeterminate) -- a 6R chain with varied, non-degenerate
// axes/offsets, loosely mirroring a typical anthropomorphic arm.
constexpr int kNumJoints = 6;

Pose3Gaussian TightPose3Gaussian(const Pose3& mean) {
  Pose3Gaussian g;
  g.mean = mean.matrix();
  g.cov = 1e-12 * Matrix6::Identity();
  return g;
}

std::vector<RigidJointSpec> RigidJoints() {
  return {
      RigidJointSpec{
          TightPose3Gaussian(Pose3(Rot3::Rodrigues(0.05, -0.02, 0.1), Point3(0.02, 0.01, 0.15))),
          Vector3(0, 0, 1), JointType::Revolute},
      RigidJointSpec{
          TightPose3Gaussian(Pose3(Rot3::Rodrigues(-0.1, 0.05, 0.2), Point3(0.01, -0.02, 0.10))),
          Vector3(0, 1, 0), JointType::Revolute},
      RigidJointSpec{
          TightPose3Gaussian(Pose3(Rot3::Rodrigues(0.02, 0.1, -0.05), Point3(0.03, 0.02, 0.12))),
          Vector3(0, 1, 0), JointType::Revolute},
      RigidJointSpec{
          TightPose3Gaussian(Pose3(Rot3::Rodrigues(-0.03, 0.08, 0.15), Point3(0.0, 0.01, 0.10))),
          Vector3(0, 0, 1), JointType::Revolute},
      RigidJointSpec{
          TightPose3Gaussian(Pose3(Rot3::Rodrigues(0.12, -0.05, -0.02), Point3(0.02, -0.01, 0.09))),
          Vector3(0, 1, 0), JointType::Revolute},
      RigidJointSpec{
          TightPose3Gaussian(Pose3(Rot3::Rodrigues(-0.07, 0.02, 0.09), Point3(0.0, 0.0, 0.05))),
          Vector3(0, 0, 1), JointType::Revolute},
  };
}

RigidRobotSolverConfig RigidConfig(bool enable_wrench_sensing = false) {
  Pose3Gaussian base_cal = TightPose3Gaussian(Pose3::Identity());
  Pose3Gaussian tip_cal = TightPose3Gaussian(Pose3(Rot3::Identity(), Point3(0, 0, 0.05)));
  return RigidRobotSolverConfig(
      RigidJoints(), base_cal, tip_cal,
      /*sigma_chain_rot=*/1e-6, /*sigma_chain_pos=*/1e-6,
      enable_wrench_sensing);
}

}  // namespace

TEST(RigidRobotModel, J_tip_joints_matches_numerical_unloaded) {
  RigidRobotSolverConfig config = RigidConfig();
  Vector q0 = (Vector(kNumJoints) << 0.3, -0.2, 0.5, 0.4, -0.3, 0.2).finished();
  Matrix tight_cov = 1e-16 * Matrix::Identity(kNumJoints, kNumJoints);

  std::function<Pose3(const Vector&)> h = [&](const Vector& q) -> Pose3 {
    RigidRobotSolver solver(config);  // fresh instance -- no warm-start bias
    auto sol = solver.solve(VectorXGaussian{q, tight_cov});
    return Pose3(sol.marginals.tip_pose.mean);
  };

  Matrix J_numeric = numericalDerivative11<Pose3, Vector, kNumJoints>(h, q0, 1e-6);

  RigidRobotSolver solver(config);
  auto sol0 = solver.solve(VectorXGaussian{q0, tight_cov});
  Matrix J = sol0.marginals.J_tip_joints;

  EXPECT(assert_equal(J_numeric, J, 1e-3));
}

namespace {

// Self-consistent per-joint torque "sensor readings" for a chosen
// world-frame wrench at the given joint configuration q, computed via the
// model's own RigidJointTorqueFactor (not a separate hand-derived
// reimplementation of its transport-and-project formula) so there's no
// risk of a sign/formula mismatch. Child-link poses depend on q (via the
// FK chain), so this must be recomputed at each perturbed q, exactly as a
// real sensor's readings would reflect the true (perturbed) configuration.
VectorXGaussian SyntheticJointTorques(
    const RigidRobotSolverConfig& kinematic_config,
    const std::vector<RigidJointSpec>& joints,
    const Vector& q,
    const Vector6& true_wrench,
    const Matrix& tight_cov)
{
  RigidRobotSolver unloaded_solver(kinematic_config);
  auto unloaded_sol = unloaded_solver.solve(VectorXGaussian{q, tight_cov});
  Pose3 tip_pose(unloaded_sol.marginals.tip_pose.mean);

  int num_joints = static_cast<int>(joints.size());
  Vector joint_torques(num_joints);
  Matrix torque_cov = 1e-12 * Matrix::Identity(num_joints, num_joints);
  for (int i = 0; i < num_joints; ++i) {
    Pose3 pose_child(unloaded_sol.marginals.links[i + 1].mean);
    RigidJointTorqueFactor factor(
        Symbol('T', 0), Symbol('T', 1), Symbol('W', 0),
        joints[i].axis, joints[i].type, /*torque_meas=*/0.0, noiseModel::Unit::Create(1));
    Values values;
    values.insert(Symbol('T', 0), tip_pose);
    values.insert(Symbol('T', 1), pose_child);
    values.insert(Symbol('W', 0), true_wrench);
    joint_torques(i) = factor.unwhitenedError(values)(0);
  }
  return VectorXGaussian{joint_torques, torque_cov};
}

}  // namespace

TEST(RigidRobotModel, J_tip_joints_matches_numerical_with_tip_wrench) {
  RigidRobotSolverConfig kinematic_config = RigidConfig(/*enable_wrench_sensing=*/false);
  RigidRobotSolverConfig config = RigidConfig(/*enable_wrench_sensing=*/true);
  std::vector<RigidJointSpec> joints = RigidJoints();

  Vector q0 = (Vector(kNumJoints) << 0.3, -0.2, 0.5, 0.4, -0.3, 0.2).finished();
  Matrix tight_cov = 1e-16 * Matrix::Identity(kNumJoints, kNumJoints);

  // A real, nonzero external tip wrench (same order of magnitude as
  // JacobianTests.cpp's wrench_1()) -- exercises J_tip_joints in the
  // presence of the tip_wrench variable and the RigidJointTorqueFactors
  // coupling it into the graph, not just the wrench-free baseline.
  Vector6 true_wrench = (Vector6() << 0.05, -0.03, 0.02, 1.0, -0.5, 2.0).finished();

  auto solve_loaded = [&](const Vector& q) {
    RigidRobotSolver solver(config);  // fresh instance -- no warm-start bias
    VectorXGaussian joint_torques = SyntheticJointTorques(kinematic_config, joints, q, true_wrench, tight_cov);
    return solver.solve(VectorXGaussian{q, tight_cov}, std::nullopt, joint_torques);
  };

  std::function<Pose3(const Vector&)> h = [&](const Vector& q) -> Pose3 {
    return Pose3(solve_loaded(q).marginals.tip_pose.mean);
  };

  Matrix J_numeric = numericalDerivative11<Pose3, Vector, kNumJoints>(h, q0, 1e-6);

  auto sol0 = solve_loaded(q0);
  Matrix J = sol0.marginals.J_tip_joints;

  EXPECT(assert_equal(J_numeric, J, 1e-3));
}

namespace {

// ---------------------------------------------------------------------
// Parallel robot: mirrors python/scripts/parallel_robot/config.py's
// DEFAULTS.
// ---------------------------------------------------------------------

constexpr int kNumRods = 6;

Matrix6 ParallelRodKInv() {
  double r = 0.0015 / 2.0;
  double I = 0.25 * M_PI * std::pow(r, 4);
  double A = M_PI * r * r;
  double J = 2.0 * I;
  double E = 207.0e9;
  double G = 79.3e9;

  Vector6 diag;
  diag << 1.0 / (E * I), 1.0 / (E * I), 1.0 / (J * G),
          1.0 / (G * A), 1.0 / (G * A), 1.0 / (E * A);
  return diag.asDiagonal();
}

std::vector<Matrix4> EndPoses(const std::vector<double>& angles_deg, double radius, double z) {
  std::vector<Matrix4> poses;
  poses.reserve(angles_deg.size());
  for (double deg : angles_deg) {
    double rad = deg * M_PI / 180.0;
    Matrix4 pose = Matrix4::Identity();
    pose(0, 3) = radius * std::cos(rad);
    pose(1, 3) = radius * std::sin(rad);
    pose(2, 3) = z;
    poses.push_back(pose);
  }
  return poses;
}

// Mirrors config.py's get_base_poses()/get_tip_poses().
std::vector<Matrix4> ParallelBasePoses() {
  return EndPoses({10, 110, 130, 230, 250, -10}, 0.1, 0.0);
}

std::vector<Matrix4> ParallelTipPoses() {
  return EndPoses({50, 70, 170, 190, 290, 310}, 0.1, /*platform_z_offset=*/-0.1);
}

ParallelRobotSolverConfig BuildParallelConfig() {
  return ParallelRobotSolverConfig(
      /*nodes_per_rod=*/15,
      ParallelRodKInv(),
      /*sigma_strain_rot=*/0.025, /*sigma_strain_pos=*/0.0025,
      /*sigma_small_force=*/1.0e-3, /*sigma_small_moment=*/1.0e-3,
      ParallelBasePoses(), ParallelTipPoses(),
      /*sigma_end_pose_pos=*/1.0e-4, /*sigma_end_pose_rot=*/1.0e-3);
}

}  // namespace

TEST(ParallelRobotModel, rod_lengths_jacobian_matches_numerical_zero_wrench) {
  ParallelRobotSolverConfig config = BuildParallelConfig();
  Vector rod_lengths0 = 0.6 * Vector::Ones(kNumRods);
  double sigma_rod_lengths = 1e-6;
  Vector6Gaussian wrench{Vector6::Zero(), 1e-8 * Matrix6::Identity()};

  std::function<Pose3(const Vector&)> h = [&](const Vector& rod_lengths) -> Pose3 {
    ParallelRobotSolver solver(config);  // fresh instance -- no warm-start bias
    auto sol = solver.solve(rod_lengths, sigma_rod_lengths, wrench);
    return Pose3(sol.marginals.platform_pose.mean);
  };

  Matrix J_numeric = numericalDerivative11<Pose3, Vector, kNumRods>(h, rod_lengths0, 1e-5);

  ParallelRobotSolver solver(config);
  auto sol0 = solver.solve(rod_lengths0, sigma_rod_lengths, wrench);
  Matrix J = sol0.marginals.rod_lengths_jacobian;

  // Looser than the other robot Jacobian tests: get_rod_lengths_jacobian
  // proxies "rod length" with the base pose's own z-translation, which
  // models a longer rod as its base sliding away rather than the rod
  // itself becoming more compliant -- a known, understood ~0.8% gap (see
  // the NOTE on ParallelRobotModel::get_rod_lengths_jacobian), not a test
  // precision issue. This bound exists to catch that gap growing
  // (a real regression), not to demand exact agreement.
  EXPECT(RelativeError(J, J_numeric) < 0.02);
}

TEST(ParallelRobotModel, rod_lengths_jacobian_matches_numerical_loaded_wrench) {
  ParallelRobotSolverConfig config = BuildParallelConfig();
  Vector rod_lengths0 = 0.6 * Vector::Ones(kNumRods);
  double sigma_rod_lengths = 1e-6;

  // A real, nonzero external platform wrench.
  Vector6 loaded_mean = (Vector6() << 0.01, -0.005, 0.008, 0.3, -0.2, 0.4).finished();
  Vector6Gaussian wrench{loaded_mean, 1e-8 * Matrix6::Identity()};

  std::function<Pose3(const Vector&)> h = [&](const Vector& rod_lengths) -> Pose3 {
    ParallelRobotSolver solver(config);  // fresh instance -- no warm-start bias
    auto sol = solver.solve(rod_lengths, sigma_rod_lengths, wrench);
    return Pose3(sol.marginals.platform_pose.mean);
  };

  Matrix J_numeric = numericalDerivative11<Pose3, Vector, kNumRods>(h, rod_lengths0, 1e-5);

  ParallelRobotSolver solver(config);
  auto sol0 = solver.solve(rod_lengths0, sigma_rod_lengths, wrench);
  Matrix J = sol0.marginals.rod_lengths_jacobian;

  // Same known ~0.8% gap as the zero-wrench case above (unaffected by
  // load -- it's about how rod length itself is proxied, not the wrench).
  EXPECT(RelativeError(J, J_numeric) < 0.02);
}

TEST(ParallelRobotModel, tip_wrench_jacobian_matches_numerical_near_zero_wrench) {
  ParallelRobotSolverConfig config = BuildParallelConfig();
  Vector rod_lengths0 = 0.6 * Vector::Ones(kNumRods);
  double sigma_rod_lengths = 1e-6;
  Matrix6 tight_cov = 1e-8 * Matrix6::Identity();

  std::function<Pose3(const Vector6&)> h = [&](const Vector6& wrench_mean) -> Pose3 {
    ParallelRobotSolver solver(config);  // fresh instance -- no warm-start bias
    Vector6Gaussian wrench{wrench_mean, tight_cov};
    auto sol = solver.solve(rod_lengths0, sigma_rod_lengths, wrench);
    return Pose3(sol.marginals.platform_pose.mean);
  };

  Vector6 wrench0 = Vector6::Zero();
  Matrix J_numeric = numericalDerivative11<Pose3, Vector6>(h, wrench0, 1e-5);

  ParallelRobotSolver solver(config);
  Vector6Gaussian wrench{wrench0, tight_cov};
  auto sol0 = solver.solve(rod_lengths0, sigma_rod_lengths, wrench);
  Matrix J = sol0.marginals.tip_wrench_jacobian;

  EXPECT(assert_equal(J_numeric, J, 1e-3));
}

TEST(ParallelRobotModel, tip_wrench_jacobian_matches_numerical_loaded_wrench) {
  ParallelRobotSolverConfig config = BuildParallelConfig();
  Vector rod_lengths0 = 0.6 * Vector::Ones(kNumRods);
  double sigma_rod_lengths = 1e-6;
  Matrix6 tight_cov = 1e-8 * Matrix6::Identity();

  std::function<Pose3(const Vector6&)> h = [&](const Vector6& wrench_mean) -> Pose3 {
    ParallelRobotSolver solver(config);  // fresh instance -- no warm-start bias
    Vector6Gaussian wrench{wrench_mean, tight_cov};
    auto sol = solver.solve(rod_lengths0, sigma_rod_lengths, wrench);
    return Pose3(sol.marginals.platform_pose.mean);
  };

  // A real, nonzero external platform wrench -- checks the Jacobian is
  // still correct away from the near-zero operating point above.
  Vector6 wrench0 = (Vector6() << 0.01, -0.005, 0.008, 0.3, -0.2, 0.4).finished();
  Matrix J_numeric = numericalDerivative11<Pose3, Vector6>(h, wrench0, 1e-5);

  ParallelRobotSolver solver(config);
  Vector6Gaussian wrench{wrench0, tight_cov};
  auto sol0 = solver.solve(rod_lengths0, sigma_rod_lengths, wrench);
  Matrix J = sol0.marginals.tip_wrench_jacobian;

  EXPECT(assert_equal(J_numeric, J, 1e-3));
}
