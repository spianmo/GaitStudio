import numpy as np

MHip, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, Spine, Neck, Head, Site, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist = range(17)


def get_joint_angles(pose_3d, joint_idx):
    import vg
    xs = pose_3d[:, joint_idx[1]] - pose_3d[:, joint_idx[0]]
    if len(joint_idx) == 3:
        ys = pose_3d[:, joint_idx[1]] - pose_3d[:, joint_idx[2]]
    elif len(joint_idx) == 4:
        ys = pose_3d[:, joint_idx[3]] - pose_3d[:, joint_idx[2]]

    return vg.angle(xs, ys)


def calc_common_angles(pose_3d, clinical=False):
    MHip, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, Spine1, Neck, \
    Head, Site, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist = range(17)

    angles = {}
    angles['RKnee'] = get_joint_angles(pose_3d, [RHip, RKnee, RAnkle])  # y
    angles['LKnee'] = get_joint_angles(pose_3d, [LHip, LKnee, LAnkle])  # y

    if clinical:
        angles = {k: 180 - v for k, v in angles.items()}

    # angles['RHip'] = get_joint_angles(pose_3d, [Spine, MHip, RKnee, RHip])
    # angles['LHip'] = get_joint_angles(pose_3d, [Spine, MHip, LKnee, LHip])
    return angles
