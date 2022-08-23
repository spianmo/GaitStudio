import numpy as np

import mediapipe as mp

from estimator.base.estimator_2d import Estimator2D
from estimator.utils.skeleton_helper import mediapipe2openpose, mediapipe2coco

from estimator.utils.data_utils import suggest_metadata


class MediaPipe_Estimator2D(Estimator2D):
    """2D human pose estimator using MediaPipe

    Methods
    -------
    estimate(video)
        estimate the 2D pose coordinates in the given video file
    """

    BATCH_SIZE = 64

    def __init__(self, out_format='mediapipe', model_complexity=1, return_3D=False):
        """
        Parameters
        ----------
        out_format : str
            output pose topology used; can be 'mediapipe', 'coco' or 'openpose'

        model_complexity : int , optional
            complexity of the used MediaPipe Pose model (0-2) (default=1)

        return_3D : bool , optiona
            return estimated keypoints directly as 3D using the depth estimate
            from MediaPipe Pose (default=False)

        Raises
        ------
        ValueError
            If an unknown pose output_format is specified
        """
        if out_format not in ['mediapipe', 'coco', 'openpose']:
            raise ValueError('Unknown pose topolgy')
        self.out_format = out_format
        self.mp_pose = mp.solutions.pose
        self.model_complexity = model_complexity

    def _image_coordinates(self, X, w, h):
        """Reverse camera frame normalization"""

        assert X.shape[-1] == 2
        return X * [w, h]

    def estimate(self, video):
        """Estimate the 2D pose coordinates in the given video file

        Parameter
        ---------
        video : Video
            Video file packed into the data.Video class for convenience

        Returns
        -------
        dict
            2D coordinates like {'video': {'custom': [np.ndarray]}}

        dict
            metadata as used in VideoPose3D

        """

        with self.mp_pose.Pose(
                static_image_mode=False,
                # model_complexity=self.model_complexity,
                smooth_landmarks=True, ) as pose:

            pose_2d = []
            for frame in video:
                result = pose.process(frame)
                if result.pose_landmarks is not None:
                    pose_2d.append([[p.x, p.y] for p in result.pose_landmarks.landmark])
                else:
                    pose_2d.append([[0, 0] for _ in range(33)])
            pose_2d = np.vstack(pose_2d).reshape(-1, 33, 2)

            if self.out_format == 'coco':
                pose_2d = mediapipe2coco(pose_2d)
            elif self.out_format == 'openpose':
                pose_2d = mediapipe2openpose(pose_2d)

            pose_2d = self._image_coordinates(pose_2d, *video.size)

            # create VideoPose3D-compatible metadata and keypoint structure
            metadata = suggest_metadata(self.out_format)
            video_name = 'video'
            video_meta = {'w': video.size[0], 'h': video.size[1], 'fps': video.fps}
            metadata['video_metadata'] = {video_name: video_meta}
            keypoints = {video_name: {'custom': [pose_2d]}}

        return keypoints, metadata
