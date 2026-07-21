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
#include "parallel_robot/PlatformWrenchBalanceFactor.h"
#include "parallel_robot/SingleRodBaseFactor.h"
#include "rigid_robot/RigidJointFactor.h"
#include "rigid_robot/RigidJointTorqueFactor.h"
#include "tendon_robot/TendonDiscWrenchFactor.h"
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

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
