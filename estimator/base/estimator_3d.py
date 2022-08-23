import abc


class Estimator3D(object):
    """Base class of 3D human pose estimator."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def estimate(self, keypoints, meta):
        """
        Args:
            keypoints: Array of 2d joint positions with confidence (n_joints, 3)
            or without confidence (n_joints, 3)
        Return:
            3d_keypoints: Array of keypoints projected to 3d space position 
            (n_joints, 3)
        """
        pass
