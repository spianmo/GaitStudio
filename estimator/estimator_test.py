from pathlib import Path

from estimator.mediapipe_estimator import MediaPipe_Estimator2D
from estimator.utils.angle_helper import calc_common_angles
from estimator.utils.data_utils import suggest_metadata
from estimator.video import Video
from estimator.videopose3d import VideoPose3D

# map cam id to list index (from VideoPose3D)
cam_map = {
    '54138969': 0,
    '55011271': 1,
    '58860488': 2,
    '60457274': 3,
}

positions_2d = {}
positions_3d = {}
angles_3d = {}

in_dir = Path('/content/input')

# get appropiate meta data for MS coco pose topology
metadata = suggest_metadata('coco')

if __name__ == '__main__':

    # Iterate over all subjects
    for subj_dir in in_dir.iterdir():
        subject = subj_dir.name
        positions_2d[subject] = {}
        positions_3d[subject] = {}
        angles_3d[subject] = {}

        # Iterate over all activities
        for act_dir in subj_dir.iterdir():
            activity = act_dir.name
            positions_2d[subject][activity] = [None] * len(cam_map)
            positions_3d[subject][activity] = [None] * len(cam_map)
            angles_3d[subject][activity] = [None] * len(cam_map)

            # every file is by one camera
            for video_files in act_dir.iterdir():
                cam = video_files.stem.split('.')[1]
                video = Video(video_files)

                # estimate 2D and 3D keypoints using the HPE pipeline
                estimator_2d = MediaPipe_Estimator2D(out_format='coco')
                estimator_3d = VideoPose3D()

                kpts, meta = estimator_2d.estimate(video)
                pose_3d = estimator_3d.estimate(kpts, meta)['video']
                angles = calc_common_angles(pose_3d)

                # save data at correct list position
                id = cam_map[cam]
                pose_2d = kpts['video']['custom'][0]
                positions_2d[subject][activity][id] = pose_2d
                positions_3d[subject][activity][id] = pose_3d
                angles_3d[subject][activity][id] = angles
