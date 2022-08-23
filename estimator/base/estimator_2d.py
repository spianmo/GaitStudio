import abc


class Estimator2D(object):
    """Base class of 2D human pose estimator."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def estimate(self, video):
        """
        Args:
            video: Array of images (N, BGR, H, W)
        Return:
            keypoints: Array of 2d-keypoint position with confidence levels 
            (n_joints, 3)
            meta: VideoPose3D-compatible metadata object
        """
        pass
