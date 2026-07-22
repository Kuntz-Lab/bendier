import numpy as np

# Kuka LBR iiwa 7 R800 -- 7-DOF (redundant for a 6-DOF task pose, so it has a
# nontrivial null space), simple all-revolute serial chain with no gripper
# attached. URDF + meshes pulled automatically by robot_descriptions from the
# open-source facebookresearch/differentiable-robot-model repo.
ROBOT_DESCRIPTION = "iiwa7_description"

# A comfortable, non-singular "ready" configuration for the 7 joints.
HOME_JOINT_ANGLES = np.array([0.0, 0.5, 0.0, -1.2, 0.0, 1.0, 0.0])

# Default calibration uncertainty for each joint's realized offset --
# assembly tolerance / backlash / structural compliance, i.e. how far the
# joint's actual transform can wander from its nominal URDF origin. See
# RigidJointSpec.offset_calibration.
DEFAULT_OFFSET_SIGMA_ROT = 1e-4  # rad
DEFAULT_OFFSET_SIGMA_POS = 1e-4  # m

# The base link is treated as a fixed, well-surveyed anchor -- tight
# calibration relative to the per-joint offsets above.
DEFAULT_BASE_SIGMA_ROT = 1e-6
DEFAULT_BASE_SIGMA_POS = 1e-6

# iiwa7_description's kinematic chain ends at iiwa_link_7, but the actual
# tool/flange frame (iiwa_link_ee) is a further fixed offset past that --
# small, but enough to matter for where "the tip" visually is. Calibration
# uncertainty here is a bit looser than the base's since it's further out
# and would in practice depend on whatever tool is actually mounted.
DEFAULT_TIP_OFFSET_SIGMA_ROT = 1e-4
DEFAULT_TIP_OFFSET_SIGMA_POS = 1e-4


def load_urdf():
    from robot_descriptions.loaders.yourdfpy import load_robot_description
    return load_robot_description(ROBOT_DESCRIPTION)


def home_config(joint_names):
    return HOME_JOINT_ANGLES[:len(joint_names)]


def _rot_pos_cov(sigma_rot, sigma_pos):
    return np.diag([sigma_rot**2] * 3 + [sigma_pos**2] * 3)


def build_joint_specs(
        urdf, sigma_offset_rot=DEFAULT_OFFSET_SIGMA_ROT, sigma_offset_pos=DEFAULT_OFFSET_SIGMA_POS):
    """One RigidJointSpec per actuated joint, in URDF joint order -- the
    offset calibration's mean is the joint's nominal <origin> transform,
    read straight out of the URDF.
    """
    import bendier

    offset_cov = _rot_pos_cov(sigma_offset_rot, sigma_offset_pos)

    specs = []
    for joint in urdf.actuated_joints:
        origin = joint.origin if joint.origin is not None else np.eye(4)
        axis = np.array(joint.axis, dtype=float) if joint.axis is not None else np.array([0.0, 0.0, 1.0])
        joint_type = (
            bendier.JointType.REVOLUTE if joint.type == "revolute" else bendier.JointType.PRISMATIC)
        offset_calibration = bendier.Pose3Gaussian(origin, offset_cov)
        specs.append(bendier.RigidJointSpec(offset_calibration, axis, joint_type))

    return specs


def build_base_calibration(sigma_rot=DEFAULT_BASE_SIGMA_ROT, sigma_pos=DEFAULT_BASE_SIGMA_POS):
    import bendier
    return bendier.Pose3Gaussian(np.eye(4), _rot_pos_cov(sigma_rot, sigma_pos))


def build_tip_offset_calibration(
        urdf, sigma_rot=DEFAULT_TIP_OFFSET_SIGMA_ROT, sigma_pos=DEFAULT_TIP_OFFSET_SIGMA_POS):
    """Nominal transform from the last actuated link out to the true tool
    tip, read straight off the URDF's own fixed joint(s) -- not via
    yourdfpy's FK/kinematics engine (get_transform), just the raw parsed
    <origin> data, same as build_joint_specs does for the actuated joints.
    """
    import bendier

    last_link = urdf.actuated_joints[-1].child
    offset = np.eye(4)
    while True:
        next_joint = next(
            (j for j in urdf.joint_map.values() if j.parent == last_link and j.type == "fixed"),
            None)
        if next_joint is None:
            break
        origin = next_joint.origin if next_joint.origin is not None else np.eye(4)
        offset = offset @ origin
        last_link = next_joint.child

    return bendier.Pose3Gaussian(offset, _rot_pos_cov(sigma_rot, sigma_pos))
