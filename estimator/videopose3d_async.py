from pathlib import Path
from urllib import request
from urllib.error import URLError
import yaml

import numpy as np
import torch
from scipy.interpolate import interp1d

from estimator.videoposelib.model import TemporalModel

from estimator.base.estimator_3d import Estimator3D
from estimator.utils.skeleton_helper import OpenPoseSkeleton, CocoSkeleton, mediapipe2coco
from estimator.videoposelib import camera


class VideoPose3DAsync():
    """3D human pose estimator using VideoPose3D

    Methods
    -------
    estimate(keypoints, meta)
        use the given 2D keypoints and the metadata file to 
        estimate 3D keypoints
    """

    CFG_FILE = "model/configs/videopose.yaml"
    CFG_FILE_OP = "model/configs/videopose_op.yaml"

    CKPT_FILE = 'model/checkpoints/pretrained_h36m_detectron_coco.bin'
    CKPT_FILE_OP = 'model/checkpoints/pretrained_video2bvh.pth'

    def __init__(self, openpose=False, use_hfr=True, normalized_skeleton=False):
        """
        Parameters
        ----------
        openpose : bool, optional
            Use openpose format instead of MS coco format as 2D input keypoints
            (default is False)
        use_hfr : bool, optional
            Upsample 2D data to 50FPS if necessary (default is True)
        normalized_skeleton : bool, optional
            Normalize the 2D skeleton so that its femur length is approx.
            the same as in the Human3.6m dataset
        """

        if openpose:
            if not Path(self.CKPT_FILE_OP).exists():
                self._download_openpose_weights()
            ckpt = self.CKPT_FILE_OP

            with Path(self.CFG_FILE_OP).open("r") as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)

            self.in_skeleton = OpenPoseSkeleton()

        else:
            if not Path(self.CKPT_FILE).exists():
                self._download_original_weights()
            ckpt = self.CKPT_FILE

            with Path(self.CFG_FILE).open("r") as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)

            self.in_skeleton = CocoSkeleton()

        self.use_hfr = use_hfr
        self.normalized_skeleton = normalized_skeleton
        self.model = self._create_model(cfg, ckpt)
        self.causal = cfg['MODEL']['causal']

    def _download_original_weights(self):
        """Download the original pretrained weigts by FB"""

        weight_url = "https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin"
        try:
            url_request = request.urlopen(weight_url)
            path = Path(self.CKPT_FILE)
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(url_request.read())
        except URLError:
            print("Could not download weight file. Please check your internet \
                connection and proxy settings")

    def _download_openpose_weights(self):
        """
        Download the  pretrained weigts for OpenPose format by KevinLLT
        (https://github.com/KevinLTT/video2bvh)
        """

        openpose_weights_gid = '1lfTWNqnqIvsf2h959Ole7t8-j86fO1xU',
        try:
            from google_drive_downloader import GoogleDriveDownloader as gdd
            gdd.download_file_from_google_drive(openpose_weights_gid, self.CKPT_FILE_OP)
        except ImportError as error:
            print('GoogleDriveDownloader has to be installed for automatic download' \
                  'You can download the weights manually under: https://drive.google.com/file/d/1lfTWNqnqIvsf2h959Ole7t8/view?usp=sharing')

    def _create_model(self, cfg, ckpt_file):
        """
        Create the VideoPose3D model using the specified config yaml
        and load the pretrained weights

        Parameters
        ----------
        cfg : str
            Path of the config yaml for the model used

        ckpt_file : str
            Path of the stored pretrained weights
        """

        # specify models hyperparameters - loaded from config yaml
        model_params = cfg['MODEL']
        filter_widths = model_params['filter_widths']  # [3,3,3,3,3]
        dropout = model_params['dropout']  # 0.25
        channels = model_params['channels']  # 1024
        causal = model_params['causal']  # False

        n_joints_in = cfg['IN_FORMAT']['num_joints']
        n_joints_out = cfg['OUT_FORMAT']['num_joints']

        # create model and load checkpoint
        model_pos = TemporalModel(n_joints_in, 2, n_joints_out, filter_widths,
                                  causal, dropout, channels)

        checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        if 'pretrained_h36m_detectron_coco.bin' in ckpt_file:
            model_pos.load_state_dict(checkpoint['model_pos'])
        elif 'pretrained_video2bvh.pth' in ckpt_file:
            pretrained_dict = checkpoint['model_state']
            model_dict = model_pos.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict
            }
            model_dict.update(pretrained_dict)
            model_pos.load_state_dict(model_dict)
        else:
            model_pos.load_state_dict(checkpoint)
        model_pos.eval()  # Important for dropout!

        # push to gpu
        if torch.cuda.is_available():
            model_pos = model_pos.cuda()
        model_pos.eval()

        return model_pos

    def _post_process(self, pose_3d):
        """
        Helper method to transform the given 3D coordinates back to world
        coordiantes and rebase the height

        Parameter
        ---------
        pose_3d : np.ndarray
            the estimated 3D pose coordinates

        Returns
        -------
        np.ndarray
            transformed 3D coordinates

        """

        pose_3d = np.ascontiguousarray(pose_3d)
        # transform to world coordinates
        rot = np.array([0.1407056450843811, -0.1500701755285263,
                        -0.755240797996521, 0.6223280429840088], dtype='float32')
        pose_3d = camera.camera_to_world(pose_3d, R=rot, t=0)
        # We don't have the trajectory, but at least we can rebase the height
        pose_3d[:, :, 2] -= np.min(pose_3d[:, :, 2])
        return pose_3d

    def _normalize_skeleton(self, pose_2d):
        """
        Helper method to normalize the 2D skeleton, so that its
        femur length is approx. the same as in Human3.6m subjects

        Parameter
        ---------
        pose_2d : np.ndarray
            the given 2D pose coordinates for estimation
        """

        joint_id = self.in_skeleton.keypoint2index

        # avg femur length in H36m training set
        femur_mean, femur_std = 0.21372303, 0.04855966

        # Normalize to hip-knee distance
        femur_len_r = np.linalg.norm(
            pose_2d[:, joint_id['RHip']] - pose_2d[:, joint_id['RKnee']],
            axis=-1)
        femur_len_l = np.linalg.norm(
            pose_2d[:, joint_id['LHip']] - pose_2d[:, joint_id['LKnee']],
            axis=-1)
        femur_len = (femur_len_r + femur_len_l) / 2.

        # calc scale factor of the datasets femur length to h36m's
        femur_scale = femur_len.mean() / femur_mean

        return pose_2d / femur_scale

    def image_coordinates(self, X, w, h):
        """Reverse camera frame normalization"""

        assert X.shape[-1] == 2
        return X * [w, h]

    def estimate(self, keypoints, fps, w, h):

        keypoints = np.vstack(keypoints).reshape(-1, 33, 2)

        keypoints = mediapipe2coco(keypoints)

        keypoints = self.image_coordinates(keypoints, w, h)

        pad = (self.model.receptive_field() - 1) // 2  # Padding on each side
        causal_shift = pad if self.causal else 0

        if self.use_hfr and fps < 50:
            # interpolate to 50fps
            pose_2d = keypoints
            new_frames = int(50 / fps * len(pose_2d))  # number of frames in 50fps
            old_t = np.linspace(0, 1, len(pose_2d))
            new_t = np.linspace(0, 1, new_frames)
            kps = np.zeros([new_frames, *pose_2d.shape[1:]])
            for i in range(pose_2d.shape[1]):
                for j in range(pose_2d.shape[2]):
                    kps[:, i, j] = interp1d(old_t, pose_2d[:, i, j], kind='cubic')(new_t)
        else:
            # use original fps
            kps = keypoints.copy()

        # Normalize camera frames to image size
        kps = camera.normalize_screen_coordinates(kps, w, h)

        if self.normalized_skeleton:
            kps = self._normalize_skeleton(kps)

        # Pad keypoints with edge mode
        kps = np.expand_dims(np.pad(kps, ((pad + causal_shift, pad - causal_shift),
                                          (0, 0), (0, 0)), 'edge'), axis=0)

        # Run model
        with torch.no_grad():
            kps = torch.from_numpy(kps.astype('float32'))
            if torch.cuda.is_available():
                kps = kps.cuda()
            predicted_3d_pos = self.model(kps).squeeze(0).detach().cpu().numpy()

            predictions = self._post_process(predicted_3d_pos)

        return predictions
