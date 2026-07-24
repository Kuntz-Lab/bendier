#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/factorTesting.h>

#include "cosserat_rod/BoundaryStressFactor.h"
#include "cosserat_rod/CosseratRodModel.h"
#include "cosserat_rod/CosseratStrainFactor.h"
#include "cosserat_rod/CosseratStressFactor.h"
#include "measurement/ActuationForceMeasFactor.h"
#include "parallel_robot/PlatformWrenchBalanceFactor.h"
#include "parallel_robot/SingleRodBaseFactor.h"
#include "rigid_robot/RigidJointFactor.h"
#include "rigid_robot/RigidJointTorqueFactor.h"
#include "tendon_robot/TendonDiscWrenchFactor.h"
#include "tendon_robot/TendonDisplacementFactor.h"
#include "tendon_robot/TendonRobotSolver.h"
#include "utils/WrenchTransforms.h"

#include <cmath>
#include <functional>

using namespace gtsam;

namespace {

// Some non-trivial, non-identity values for testing
Pose3 pose_a() { return Pose3(Rot3::Rodrigues(0.3, -0.2, 0.5), Point3(0.10, -0.05, 0.20)); }
Pose3 pose_b() { return Pose3(Rot3::Rodrigues(-0.1, 0.4, 0.2), Point3(0.12, 0.03, 0.24)); }
Pose3 pose_c() { return Pose3(Rot3::Rodrigues(0.2, 0.1, -0.3), Point3(-0.08, 0.06, 0.18)); }

Vector6 wrench_1() { Vector6 w; w << 0.05, -0.03, 0.02, 1.0, -0.5, 2.0; return w; }
Vector6 wrench_2() { Vector6 w; w << -0.02, 0.04, -0.01, -0.8, 0.3, 1.5; return w; }

Matrix6 K_inv_1() {
  Vector6 diag;
  diag << 1.0/0.8, 1.0/0.8, 1.0/1.2, 1.0/500.0, 1.0/500.0, 1.0/2000.0;
  return diag.asDiagonal();
}

// Tendon routing holes evenly spaced around a disc of the given radius
std::vector<Point3> holes_at(double radius, double angle_offset, int num_tendons) {
  std::vector<Point3> holes(num_tendons);
  for (int i = 0; i < num_tendons; ++i) {
    double angle = angle_offset + i * (2.0 * M_PI / num_tendons);
    holes[i] = Point3(radius * std::cos(angle), radius * std::sin(angle), 0.0);
  }
  return holes;
}

Vector tensions_of(std::initializer_list<double> vals) {
  Vector t(vals.size());
  int i = 0;
  for (double v : vals) t[i++] = v;
  return t;
}

}

TEST(WrenchTransforms, transform_wrench_translation_jacobians) {
  Vector6 w0 = wrench_1();
  Pose3 p0 = pose_a();
  Pose3 p1 = pose_b();

  Matrix H_w0, H_p0, H_p1;
  transform_wrench_translation(w0, p0, p1, H_w0, H_p0, H_p1);

  std::function<Vector6(const Vector6&, const Pose3&, const Pose3&)> h =
      [](const Vector6& w0_, const Pose3& p0_, const Pose3& p1_) {
        return transform_wrench_translation(w0_, p0_, p1_);
      };

  Matrix numH_w0 = numericalDerivative31<Vector6, Vector6, Pose3, Pose3>(h, w0, p0, p1);
  Matrix numH_p0 = numericalDerivative32<Vector6, Vector6, Pose3, Pose3>(h, w0, p0, p1);
  Matrix numH_p1 = numericalDerivative33<Vector6, Vector6, Pose3, Pose3>(h, w0, p0, p1);

  EXPECT(assert_equal(numH_w0, H_w0, 1e-6));
  EXPECT(assert_equal(numH_p0, H_p0, 1e-6));
  EXPECT(assert_equal(numH_p1, H_p1, 1e-6));

  // Force component must be translation-invariant (no rotation in this transform).
  Vector6 w1 = transform_wrench_translation(w0, p0, p1);
  EXPECT(assert_equal(Vector3(w0.tail<3>()), Vector3(w1.tail<3>()), 1e-9));
}

TEST(WrenchTransforms, spatial_to_body_wrench_jacobians) {
  Vector6 spatial = wrench_1();
  Pose3 pose = pose_a();

  Matrix H_spatial, H_pose;
  spatial_to_body_wrench(spatial, pose, H_spatial, H_pose);

  std::function<Vector6(const Vector6&, const Pose3&)> h =
      [](const Vector6& s, const Pose3& p) { return spatial_to_body_wrench(s, p); };

  Matrix numH_spatial = numericalDerivative21<Vector6, Vector6, Pose3>(h, spatial, pose);
  Matrix numH_pose = numericalDerivative22<Vector6, Vector6, Pose3>(h, spatial, pose);

  EXPECT(assert_equal(numH_spatial, H_spatial, 1e-6));
  EXPECT(assert_equal(numH_pose, H_pose, 1e-6));

  // Round-tripping through the inverse rotation should recover the original wrench.
  Vector6 body = spatial_to_body_wrench(spatial, pose);
  Vector6 roundtrip = body_to_spatial_wrench(body, pose);
  EXPECT(assert_equal(spatial, roundtrip, 1e-9));
}

TEST(WrenchTransforms, body_to_spatial_wrench_jacobians) {
  Vector6 body = wrench_2();
  Pose3 pose = pose_b();

  Matrix H_body, H_pose;
  body_to_spatial_wrench(body, pose, H_body, H_pose);

  std::function<Vector6(const Vector6&, const Pose3&)> h =
      [](const Vector6& b, const Pose3& p) { return body_to_spatial_wrench(b, p); };

  Matrix numH_body = numericalDerivative21<Vector6, Vector6, Pose3>(h, body, pose);
  Matrix numH_pose = numericalDerivative22<Vector6, Vector6, Pose3>(h, body, pose);

  EXPECT(assert_equal(numH_body, H_body, 1e-6));
  EXPECT(assert_equal(numH_pose, H_pose, 1e-6));
}

TEST(CosseratStrainFactor, jacobians_all_magnus_terms) {
  Key p0k = 1, p1k = 2, s0k = 3, s1k = 4;

  Values values;
  values.insert(p0k, pose_a());
  values.insert(p1k, pose_b());
  values.insert(s0k, wrench_1());
  values.insert(s1k, wrench_2());

  // Test all Magnus series term counts from 1 to 4
  for (double ds : {0.001, 0.01, 0.1}) {
    for (int num_magnus_terms = 1; num_magnus_terms <= 4; ++num_magnus_terms) {
      CosseratStrainFactor factor(
          p0k, p1k, s0k, s1k,
          ds, StraightRodNominalStrain(), K_inv_1(),
          noiseModel::Unit::Create(6), num_magnus_terms);

      EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
    }
  }
}

TEST(CosseratStressFactor, jacobians_interior) {
  Key p0k = 1, p1k = 2, s0k = 3, s1k = 4, wk = 5;
  CosseratStressFactor factor(p0k, p1k, s0k, s1k, wk, noiseModel::Unit::Create(6));

  Values values;
  values.insert(p0k, pose_a());
  values.insert(p1k, pose_b());
  values.insert(s0k, wrench_1());
  values.insert(s1k, wrench_2());
  values.insert(wk, wrench_2());

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

TEST(CosseratStressFactor, jacobians_tip) {
  Key p0k = 1, p1k = 2, s0k = 3, s1k = 4;
  CosseratStressFactor factor(p0k, p1k, s0k, s1k, noiseModel::Unit::Create(6));

  Values values;
  values.insert(p0k, pose_a());
  values.insert(p1k, pose_b());
  values.insert(s0k, wrench_1());
  values.insert(s1k, wrench_2());

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

TEST(BoundaryStressFactor, jacobians_tip_and_base) {
  Key sk = 1, wk = 2;

  Values values;
  values.insert(sk, wrench_1());
  values.insert(wk, wrench_2());

  BoundaryStressFactor tip_factor(sk, wk, noiseModel::Unit::Create(6), /*is_base=*/false);
  EXPECT_CORRECT_FACTOR_JACOBIANS(tip_factor, values, 1e-6, 1e-5);

  BoundaryStressFactor base_factor(sk, wk, noiseModel::Unit::Create(6), /*is_base=*/true);
  EXPECT_CORRECT_FACTOR_JACOBIANS(base_factor, values, 1e-6, 1e-5);
}

TEST(SingleRodBaseFactor, jacobians) {
  Key pk = 1, sk = 2;
  Pose3 mount_pose = pose_c();

  SingleRodBaseFactor factor(pk, sk, mount_pose, noiseModel::Unit::Create(6));

  Values values;
  values.insert(pk, pose_a());
  values.insert(sk, wrench_1());

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

TEST(ActuationForceMeasFactor, jacobians) {
  Key wk = 1;
  ActuationForceMeasFactor factor(wk, /*f_z_meas=*/1.7, noiseModel::Unit::Create(1));

  Values values;
  values.insert(wk, wrench_1());

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

TEST(PlatformWrenchBalanceFactor, jacobians) {
  KeyVector s = {1, 2, 3, 4, 5, 6};
  KeyVector p = {11, 12, 13, 14, 15, 16};
  Key pwk = 20, ppk = 21;

  PlatformWrenchBalanceFactor factor(s, p, pwk, ppk, noiseModel::Unit::Create(6));

  Values values;
  Pose3 rod_poses[6] = {pose_a(), pose_b(), pose_c(), pose_a(), pose_b(), pose_c()};
  Vector6 rod_stresses[6] = {
      wrench_1(), wrench_2(), wrench_1(), wrench_2(), wrench_1(), wrench_2()};
  for (int i = 0; i < 6; ++i) {
    values.insert(s[i], rod_stresses[i]);
    values.insert(p[i], rod_poses[i]);
  }
  values.insert(pwk, wrench_2());
  values.insert(ppk, pose_c());

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

TEST(PlatformWrenchBalanceFactor, jacobians_four_rods) {
  KeyVector s = {1, 2, 3, 4};
  KeyVector p = {11, 12, 13, 14};
  Key pwk = 20, ppk = 21;

  PlatformWrenchBalanceFactor factor(s, p, pwk, ppk, noiseModel::Unit::Create(6));

  Values values;
  Pose3 rod_poses[4] = {pose_a(), pose_b(), pose_c(), pose_a()};
  Vector6 rod_stresses[4] = {wrench_1(), wrench_2(), wrench_1(), wrench_2()};
  for (int i = 0; i < 4; ++i) {
    values.insert(s[i], rod_stresses[i]);
    values.insert(p[i], rod_poses[i]);
  }
  values.insert(pwk, wrench_1());
  values.insert(ppk, pose_a());

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

TEST(TendonDiscWrenchFactor, jacobians_interior) {
  Key pose_prev_k = 1, pose_k = 2, pose_next_k = 3;
  Key wrench_k = 4, tensions_k = 5, ext_wrench_k = 6;

  auto holes_prev = holes_at(0.005, 0.0, 4);
  auto holes = holes_at(0.005, 0.2, 4);
  auto holes_next = holes_at(0.005, 0.4, 4);

  TendonDiscWrenchFactor factor(
      pose_prev_k, pose_k, std::optional<Key>(pose_next_k),
      wrench_k, tensions_k, std::optional<Key>(ext_wrench_k),
      holes_prev, holes, holes_next,
      noiseModel::Unit::Create(6));

  Values values;
  values.insert(pose_prev_k, pose_a());
  values.insert(pose_k, pose_b());
  values.insert(pose_next_k, pose_c());
  values.insert(wrench_k, wrench_2());
  values.insert(tensions_k, tensions_of({2.0, 1.5, 0.5, 3.0}));
  values.insert(ext_wrench_k, wrench_1());

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

TEST(TendonDiscWrenchFactor, jacobians_tip) {
  Key pose_prev_k = 1, pose_k = 2;
  Key wrench_k = 3, tensions_k = 4, ext_wrench_k = 5;

  auto holes_prev = holes_at(0.005, 0.0, 4);
  auto holes = holes_at(0.005, 0.2, 4);

  TendonDiscWrenchFactor factor(
      pose_prev_k, pose_k, std::nullopt,
      wrench_k, tensions_k, std::optional<Key>(ext_wrench_k),
      holes_prev, holes, {},
      noiseModel::Unit::Create(6));

  Values values;
  values.insert(pose_prev_k, pose_a());
  values.insert(pose_k, pose_b());
  values.insert(wrench_k, wrench_2());
  values.insert(tensions_k, tensions_of({2.0, 1.5, 0.5, 3.0}));
  values.insert(ext_wrench_k, wrench_1());

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

TEST(TendonDiscWrenchFactor, jacobians_interior_no_external_wrench) {
  Key pose_prev_k = 1, pose_k = 2, pose_next_k = 3;
  Key wrench_k = 4, tensions_k = 5;

  auto holes_prev = holes_at(0.005, 0.0, 4);
  auto holes = holes_at(0.005, 0.2, 4);
  auto holes_next = holes_at(0.005, 0.4, 4);

  TendonDiscWrenchFactor factor(
      pose_prev_k, pose_k, std::optional<Key>(pose_next_k),
      wrench_k, tensions_k, std::nullopt,
      holes_prev, holes, holes_next,
      noiseModel::Unit::Create(6));

  Values values;
  values.insert(pose_prev_k, pose_a());
  values.insert(pose_k, pose_b());
  values.insert(pose_next_k, pose_c());
  values.insert(wrench_k, wrench_2());
  values.insert(tensions_k, tensions_of({2.0, 1.5, 0.5, 3.0}));

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

TEST(TendonDiscWrenchFactor, jacobians_tip_no_external_wrench) {
  Key pose_prev_k = 1, pose_k = 2;
  Key wrench_k = 3, tensions_k = 4;

  auto holes_prev = holes_at(0.005, 0.0, 4);
  auto holes = holes_at(0.005, 0.2, 4);

  TendonDiscWrenchFactor factor(
      pose_prev_k, pose_k, std::nullopt,
      wrench_k, tensions_k, std::nullopt,
      holes_prev, holes, {},
      noiseModel::Unit::Create(6));

  Values values;
  values.insert(pose_prev_k, pose_a());
  values.insert(pose_k, pose_b());
  values.insert(wrench_k, wrench_2());
  values.insert(tensions_k, tensions_of({2.0, 1.5, 0.5, 3.0}));

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

TEST(TendonDiscWrenchFactor, jacobians_interior_three_tendons) {
  Key pose_prev_k = 1, pose_k = 2, pose_next_k = 3;
  Key wrench_k = 4, tensions_k = 5, ext_wrench_k = 6;

  auto holes_prev = holes_at(0.005, 0.0, 3);
  auto holes = holes_at(0.005, 0.2, 3);
  auto holes_next = holes_at(0.005, 0.4, 3);

  TendonDiscWrenchFactor factor(
      pose_prev_k, pose_k, std::optional<Key>(pose_next_k),
      wrench_k, tensions_k, std::optional<Key>(ext_wrench_k),
      holes_prev, holes, holes_next,
      noiseModel::Unit::Create(6));

  Values values;
  values.insert(pose_prev_k, pose_a());
  values.insert(pose_k, pose_b());
  values.insert(pose_next_k, pose_c());
  values.insert(wrench_k, wrench_2());
  values.insert(tensions_k, tensions_of({2.0, 1.5, 0.5}));
  values.insert(ext_wrench_k, wrench_1());

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

TEST(TendonDisplacementFactor, jacobians_two_discs) {
  Key pose0_k = 1, pose1_k = 2, tensions_k = 3, displacements_k = 4;

  auto holes0 = holes_at(0.005, 0.0, 4);
  auto holes1 = holes_at(0.005, 0.2, 4);

  TendonDisplacementFactor factor(
      {pose0_k, pose1_k}, tensions_k, displacements_k,
      {holes0, holes1},
      {0.24, 0.25, 0.23, 0.26},
      1e5,
      noiseModel::Unit::Create(4));

  Values values;
  values.insert(pose0_k, pose_a());
  values.insert(pose1_k, pose_b());
  values.insert(tensions_k, tensions_of({2.0, 1.5, 0.5, 3.0}));
  values.insert(displacements_k, tensions_of({0.001, -0.002, 0.0005, 0.0}));

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

TEST(TendonDisplacementFactor, jacobians_three_discs) {
  Key pose0_k = 1, pose1_k = 2, pose2_k = 3, tensions_k = 4, displacements_k = 5;

  auto holes0 = holes_at(0.005, 0.0, 4);
  auto holes1 = holes_at(0.005, 0.2, 4);
  auto holes2 = holes_at(0.005, 0.4, 4);

  TendonDisplacementFactor factor(
      {pose0_k, pose1_k, pose2_k}, tensions_k, displacements_k,
      {holes0, holes1, holes2},
      {0.24, 0.25, 0.23, 0.26},
      1e5,
      noiseModel::Unit::Create(4));

  Values values;
  values.insert(pose0_k, pose_a());
  values.insert(pose1_k, pose_b());
  values.insert(pose2_k, pose_c());
  values.insert(tensions_k, tensions_of({2.0, 1.5, 0.5, 3.0}));
  values.insert(displacements_k, tensions_of({0.001, -0.002, 0.0005, 0.0}));

  // Exercises Jacobian accumulation on the shared middle disc pose (pose1),
  // which participates in both the [0,1] and [1,2] segments.
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

TEST(TendonDisplacementFactor, jacobians_three_tendons) {
  Key pose0_k = 1, pose1_k = 2, pose2_k = 3, tensions_k = 4, displacements_k = 5;

  auto holes0 = holes_at(0.005, 0.0, 3);
  auto holes1 = holes_at(0.005, 0.2, 3);
  auto holes2 = holes_at(0.005, 0.4, 3);

  TendonDisplacementFactor factor(
      {pose0_k, pose1_k, pose2_k}, tensions_k, displacements_k,
      {holes0, holes1, holes2},
      {0.24, 0.25, 0.23},
      1e5,
      noiseModel::Unit::Create(3));

  Values values;
  values.insert(pose0_k, pose_a());
  values.insert(pose1_k, pose_b());
  values.insert(pose2_k, pose_c());
  values.insert(tensions_k, tensions_of({2.0, 1.5, 0.5}));
  values.insert(displacements_k, tensions_of({0.001, -0.002, 0.0005}));

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

TEST(TendonDisplacementFactor, matches_expected_stretch_formula) {
  // Straight, untwisted poses with identical hole routing at both discs --
  // the geometric term (current length vs. reference length) cancels
  // exactly, since these poses *are* the reference configuration. What's
  // left is a pure, hand-computable elastic-stretch prediction:
  // predicted_i = T_i * dz / EA_i (dz being the reference length here too).
  Key pose0_k = 1, pose1_k = 2, tensions_k = 3, displacements_k = 4;

  const double dz = 0.24;
  auto holes = holes_at(0.005, 0.0, 4);  // same routing angle at both discs

  const std::vector<double> reference_lengths(4, dz);  // == dz, hand-verified below
  const double tendon_stiffness = 1e5;
  const Vector tensions = tensions_of({2.0, 1.5, 0.5, 3.0});

  TendonDisplacementFactor factor(
      {pose0_k, pose1_k}, tensions_k, displacements_k,
      {holes, holes},
      reference_lengths,
      tendon_stiffness,
      noiseModel::Unit::Create(4));

  Values values;
  values.insert(pose0_k, Pose3(Rot3::Identity(), Point3(0, 0, 0)));
  values.insert(pose1_k, Pose3(Rot3::Identity(), Point3(0, 0, dz)));
  values.insert(tensions_k, tensions);

  Vector4 expected_stretch;
  for (int i = 0; i < 4; ++i)
    expected_stretch(i) = tensions(i) * dz / tendon_stiffness;

  // At displacement == the hand-computed prediction, the residual should be
  // (near) exactly zero -- not just self-consistent with its own Jacobian.
  values.insert(displacements_k, Vector(expected_stretch));
  Vector error_at_prediction = factor.unwhitenedError(values);
  EXPECT(assert_equal(Vector(Vector4::Zero()), error_at_prediction, 1e-9));

  // At displacement == 0, the residual should be exactly -expected_stretch
  // -- confirms the formula isn't trivially zero regardless of input.
  values.update(displacements_k, Vector(Vector4::Zero()));
  Vector error_at_zero = factor.unwhitenedError(values);
  EXPECT(assert_equal(Vector(-expected_stretch), error_at_zero, 1e-9));
}

TEST(TendonDisplacementFactor, matches_expected_geometric_formula_at_zero_tension) {
  // Complementary to matches_expected_stretch_formula: here tension is zero
  // (elastic term vanishes) and pose1 is offset laterally from pose0 with no
  // rotation, so every tendon's hole-to-hole segment is the same easily
  // hand-computed vector (dx, 0, dz) regardless of routing radius/angle
  // (the routing offset is identical on both discs and cancels in the
  // difference) -- giving a Pythagorean l_geom = sqrt(dx^2 + dz^2), and a
  // hand-verifiable predicted displacement = L_ref - l_geom.
  Key pose0_k = 1, pose1_k = 2, tensions_k = 3, displacements_k = 4;

  const double dz = 0.24, dx = 0.01;
  auto holes = holes_at(0.005, 0.0, 4);

  const std::vector<double> reference_lengths(4, dz);  // straight-line reference
  const double tendon_stiffness = 1e5;

  TendonDisplacementFactor factor(
      {pose0_k, pose1_k}, tensions_k, displacements_k,
      {holes, holes},
      reference_lengths,
      tendon_stiffness,
      noiseModel::Unit::Create(4));

  Values values;
  values.insert(pose0_k, Pose3(Rot3::Identity(), Point3(0, 0, 0)));
  values.insert(pose1_k, Pose3(Rot3::Identity(), Point3(dx, 0, dz)));
  values.insert(tensions_k, Vector(Vector4::Zero()));

  double l_geom = std::sqrt(dx * dx + dz * dz);
  Vector4 expected_geometric_term = Vector4::Constant(dz - l_geom);

  values.insert(displacements_k, Vector(expected_geometric_term));
  Vector error = factor.unwhitenedError(values);
  EXPECT(assert_equal(Vector(Vector4::Zero()), error, 1e-9));
}

TEST(TendonDisplacementFactor, matches_simple_stretch_model_at_realistic_curvature) {
  // Directly checks the physical quantity the user described: the tendon's
  // current geometric span l_geom is itself the physically stretched
  // length (it's under tension T right now), so the amount of *natural*
  // (unstretched) cable that spool has fed out to reach it is
  // l_geom / (1 + T/EA) -- the standard Hookean stretch relation solved
  // for natural length. Predicted displacement is then reference length
  // minus that natural length used: L_ref - l_geom/(1 + T/EA).
  //
  // This is the "exact" (l_geom-based) formula; the factor itself uses
  // L_ref (not l_geom) in the elastic term as a deliberate simplification
  // (see the header comment). At a realistic small curvature (l_geom close
  // to L_ref) the two agree to second order in (T/EA)*(L_ref - l_geom),
  // which this test confirms is genuinely tiny rather than just assumed.
  Key pose0_k = 1, pose1_k = 2, tensions_k = 3, displacements_k = 4;

  auto holes = holes_at(0.005, 0.0, 1);  // single tendon keeps this simple

  const double dz = 0.24, dx = 0.002;  // small lateral offset -> realistic curvature
  const double EA = 1e5;
  const double tension = 4.0;

  Pose3 pose0(Rot3::Identity(), Point3(0, 0, 0));
  Pose3 pose1(Rot3::Identity(), Point3(dx, 0, dz));

  const double l_geom = std::sqrt(dx * dx + dz * dz);  // current physical (stretched) span
  const double L_ref = dz;                             // straight-line reference length

  TendonDisplacementFactor factor(
      {pose0_k, pose1_k}, tensions_k, displacements_k,
      {holes, holes},
      {L_ref}, EA,
      noiseModel::Unit::Create(1));

  double natural_length_used = l_geom / (1.0 + tension / EA);
  double displacement_from_simple_stretch_model = L_ref - natural_length_used;

  Values values;
  values.insert(pose0_k, pose0);
  values.insert(pose1_k, pose1);
  values.insert(tensions_k, tensions_of({tension}));
  values.insert(displacements_k, tensions_of({displacement_from_simple_stretch_model}));

  Vector error = factor.unwhitenedError(values);
  EXPECT(std::abs(error(0)) < 1e-7);
}

TEST(RigidJointFactor, jacobians_revolute) {
  Key pose_parent_k = 1, pose_child_k = 2, offset_k = 3, joint_vec_k = 4;

  RigidJointFactor factor(
      pose_parent_k, pose_child_k, offset_k, joint_vec_k,
      /*joint_idx=*/1, Vector3(0, 0, 1), JointType::Revolute,
      noiseModel::Unit::Create(6));

  Values values;
  values.insert(pose_parent_k, pose_a());
  values.insert(pose_child_k, pose_b());
  values.insert(offset_k, pose_c());
  values.insert(joint_vec_k, tensions_of({0.3, -0.6, 0.9}));

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

TEST(RigidJointFactor, jacobians_prismatic) {
  Key pose_parent_k = 1, pose_child_k = 2, offset_k = 3, joint_vec_k = 4;

  RigidJointFactor factor(
      pose_parent_k, pose_child_k, offset_k, joint_vec_k,
      /*joint_idx=*/0, Vector3(0, 0, 1), JointType::Prismatic,
      noiseModel::Unit::Create(6));

  Values values;
  values.insert(pose_parent_k, pose_a());
  values.insert(pose_child_k, pose_b());
  values.insert(offset_k, pose_c());
  values.insert(joint_vec_k, tensions_of({0.05, -0.6, 0.9}));

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

TEST(RigidJointTorqueFactor, jacobians_revolute) {
  Key pose_tip_k = 1, pose_child_k = 2, wrench_k = 3;

  RigidJointTorqueFactor factor(
      pose_tip_k, pose_child_k, wrench_k, Vector3(0, 0, 1), JointType::Revolute,
      /*torque_meas=*/0.35, noiseModel::Unit::Create(1));

  Values values;
  values.insert(pose_tip_k, pose_a());
  values.insert(pose_child_k, pose_b());
  values.insert(wrench_k, wrench_1());

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

TEST(RigidJointTorqueFactor, jacobians_prismatic) {
  Key pose_tip_k = 1, pose_child_k = 2, wrench_k = 3;

  RigidJointTorqueFactor factor(
      pose_tip_k, pose_child_k, wrench_k, Vector3(0, 1, 0), JointType::Prismatic,
      /*torque_meas=*/-1.1, noiseModel::Unit::Create(1));

  Values values;
  values.insert(pose_tip_k, pose_b());
  values.insert(pose_child_k, pose_c());
  values.insert(wrench_k, wrench_2());

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-5);
}

// Independent physics check (not just Jacobians): a pure force with no
// moment, applied at the tip, transported back to a joint's own location by
// the factor's internal transform_wrench_translation call and projected onto
// the joint's world-frame axis, should match the textbook "torque = axis .
// (r x F)" lever-arm formula -- computed here by hand with Eigen's cross(),
// entirely independent of the factor's own internals.
TEST(RigidJointTorqueFactor, matches_textbook_cross_product_torque) {
  Vector3 force(2.0, -1.5, 0.7);
  Vector3 axis(0, 0, 1);  // in the joint's local frame

  Point3 tip_point(0.6, 0.05, 0.55);
  Pose3 pose_tip(Rot3::Identity(), tip_point);  // orientation irrelevant to transport

  Pose3 pose_joint(Rot3::Rodrigues(0.1, -0.2, 0.05), Point3(0.3, -0.1, 0.4));

  Vector6 tip_wrench;
  tip_wrench << Vector3::Zero(), force;  // pure force, no moment, at the tip

  Key pose_tip_k = 1, pose_joint_k = 2, wrench_k = 3;
  RigidJointTorqueFactor factor(
      pose_tip_k, pose_joint_k, wrench_k, axis, JointType::Revolute,
      /*torque_meas=*/0.0, noiseModel::Unit::Create(1));

  Values values;
  values.insert(pose_tip_k, pose_tip);
  values.insert(pose_joint_k, pose_joint);
  values.insert(wrench_k, tip_wrench);

  double tau_from_factor = factor.unwhitenedError(values)(0);  // torque_meas is 0, so error == predicted torque

  Vector3 r = tip_point - pose_joint.translation();
  Vector3 axis_world = pose_joint.rotation().rotate(axis);
  double tau_expected = axis_world.dot(r.cross(force));

  EXPECT(std::abs(tau_from_factor - tau_expected) < 1e-9);
}

// init_tendon_disc_config is private and only reachable through a real
// solve (TendonRobotMarginals::tendon_config is populated from it), so this
// exercises the model's own hole/disc-index computation end to end, rather
// than the TendonDisplacementFactor tests above, which all hand-supply
// hole_locations/reference_lengths and never touch this code path.
TEST(TendonRobotModel, disc_geometry_matches_hand_computed) {
  const double rod_length = 0.24;
  const int num_discs = 3;
  const int num_between_nodes = 1;
  const double routing_radius = 0.01;

  TendonRoutingInput routing;
  routing.routing_radius = routing_radius;
  routing.params = {
      RoutingFunctionParams{/*angle_offset=*/0.0, /*total_angle=*/0.0},   // straight, theta = 0 everywhere
      RoutingFunctionParams{/*angle_offset=*/0.0, /*total_angle=*/M_PI},  // spans 0 -> pi along the rod
  };

  TendonRobotSolverConfig config(
      rod_length, num_discs, num_between_nodes,
      K_inv_1(),
      /*sigma_strain_rot=*/0.1, /*sigma_strain_pos=*/0.01,
      /*sigma_small_force=*/1e-4, /*sigma_small_moment=*/1e-5,
      /*sigma_base_pose_pos=*/1e-4, /*sigma_base_pose_rot=*/1e-3,
      routing);

  TendonRobotSolver solver(config);
  VectorXGaussian tensions{Vector::Zero(2), 1e-3 * Matrix::Identity(2, 2)};
  auto solution = solver.solve(tensions);

  const TendonConfig& tendon_config = solution.marginals.tendon_config;

  // num_nodes = 5 (3 discs + 2 between each pair); discs at normalized
  // arclength 0, 0.5, 1 land exactly on pose indices 0, 2, 4.
  std::vector<int> expected_disc_pose_idx = {0, 2, 4};
  EXPECT(tendon_config.disc_pose_idx == expected_disc_pose_idx);

  for (int disc_idx = 0; disc_idx < num_discs; ++disc_idx) {
    double s = static_cast<double>(disc_idx) / (num_discs - 1);
    double theta1 = s * M_PI;

    Point3 expected_hole0(routing_radius, 0.0, 0.0);
    Point3 expected_hole1(routing_radius * std::cos(theta1), routing_radius * std::sin(theta1), 0.0);

    EXPECT(assert_equal(expected_hole0, tendon_config.hole_locations[disc_idx][0], 1e-12));
    EXPECT(assert_equal(expected_hole1, tendon_config.hole_locations[disc_idx][1], 1e-12));
  }
}

// Verifies compute_reference_lengths (also private, also only reachable via
// a real solve) is consistent with the model's own hole geometry: at zero
// tension and zero external wrench, the rod settles at the same straight,
// untwisted configuration that reference-length computation assumes, so
// TendonDisplacementFactor's geometric term should exactly cancel the
// reference length and predicted displacement should land at ~0 -- the
// specific check suggested alongside the original TODO.
TEST(TendonRobotModel, zero_tension_straight_rod_gives_near_zero_displacement) {
  TendonRoutingInput routing;
  routing.routing_radius = 0.01;
  routing.params = {
      RoutingFunctionParams{0.0, 0.0},
      RoutingFunctionParams{M_PI, 0.0},
      RoutingFunctionParams{M_PI / 2, M_PI},  // one helical tendon, so this isn't trivially symmetric
  };

  TendonRobotSolverConfig config(
      /*rod_length=*/0.24, /*num_discs=*/3, /*num_between_nodes=*/1,
      K_inv_1(),
      /*sigma_strain_rot=*/0.1, /*sigma_strain_pos=*/0.01,
      /*sigma_small_force=*/1e-6, /*sigma_small_moment=*/1e-7,
      /*sigma_base_pose_pos=*/1e-6, /*sigma_base_pose_rot=*/1e-6,
      routing);

  TendonRobotSolver solver(config);
  VectorXGaussian tensions{Vector::Zero(3), 1e-8 * Matrix::Identity(3, 3)};
  auto solution = solver.solve(tensions);  // no tip_wrench -> defaults to a small near-zero prior

  EXPECT(solution.marginals.displacements.mean.norm() < 1e-4);
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
