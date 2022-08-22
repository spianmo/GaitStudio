# -*- coding: utf-8 -*-
"""
Created on  Jan 2022
@author: Sia_Mahmoudi
"""

import math
from typing import List, Optional, Tuple
import dataclasses
import cv2
import time
import mediapipe as mp
import matplotlib.pyplot as plt
from realsense_camera import *
from mpl_toolkits.mplot3d.axes3d import *
from mediapipe.framework.formats import landmark_pb2

# presets
time_pre = time.time()
time_hey = 0
time_Go = 0
flag_time = 0
wrist_GO = 0
wrist_hey = 0

# mediapipe settings for plotting
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_RGB_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)


# mediapipe drawing class
@dataclasses.dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the white color.
    color: Tuple[int, int, int] = WHITE_COLOR
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2


# initializing camera
rs = RealsenseCamera()

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils


# =============================================================================
def _normalize_color(color):
    return tuple(v / 255. for v in color)


# =============================================================================


# specific function for plotting pose based on depth data
# I use mediapipe source code and add depth value replaced with estimated Z.
def plot_landmarkss(landmarks_D,
                    landmark_list: landmark_pb2.NormalizedLandmarkList,
                    connections: Optional[List[Tuple[int, int]]] = None,
                    landmark_drawing_spec: DrawingSpec = DrawingSpec(
                        color=RED_COLOR, thickness=5),
                    connection_drawing_spec: DrawingSpec = DrawingSpec(
                        color=BLACK_COLOR, thickness=5),
                    elevation: int = 10,
                    azimuth: int = 10):
    # =============================================================================
    """
    Plot the landmarks and the connections in matplotlib 3d based on real depth value
    landmarks_D : depth data extracted from depth camera'
    landmark_list : A normalized landmark list proto message to be plotted.
    connections: A list of landmark index tuples that specifies how landmarks to
        be connected.
    landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
        drawing settings such as color and line thickness.
    connection_drawing_spec: A DrawingSpec object that specifies the
        connections' drawing settings such as color and line thickness.
    elevation: The elevation from which to view the plot.
      azimuth: the azimuth angle to rotate the plot.
    Raises:
      ValueError: If any connetions contain invalid landmark index.
  """
    # =============================================================================

    if not landmark_list:
        return
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elevation, azim=azimuth)
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                 landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        if idx < len(landmarks_D):
            # print(idx,len(landmarks_D))
            ax.scatter3D(
                xs=[landmarks_D[idx]],
                ys=[landmark.x],
                zs=[-landmark.y],
                color=_normalize_color(landmark_drawing_spec.color[::-1]),
                linewidth=landmark_drawing_spec.thickness)
            plotted_landmarks[idx] = (landmarks_D[idx], landmark.x, -landmark.y)
    # print(landmark.x)
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx], plotted_landmarks[end_idx]]

                ax.plot3D(
                    xs=[landmark_pair[0][0], landmark_pair[1][0]],
                    ys=[landmark_pair[0][1], landmark_pair[1][1]],
                    zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    color=_normalize_color(connection_drawing_spec.color[::-1]),
                    linewidth=connection_drawing_spec.thickness)
    plt.show()


# =============================================================================


# =============================================================================
# Using mediapipe for detecting pose of humans
def detectPose(image, depth_frame, pose, display=True):
    """
    draw the landmarks and the connections on RGB image.
    image : RGB image recorded by camera
    depth_frame :Using intelrealsense d435i depth camera for extracting depth frame.
    pose: Setting up the Pose function using mediapipe.
    display: if it's true just Display the original input image and the resultant image also
            Also Plot the Pose landmarks in 3D using estimated z value.
  """
    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Retrieve the height and width of the input image.
    height, width, _ = image.shape

    # Initialize a list to store the detected landmarks.
    landmarks_3D = []
    landmarks_2D = []

    # Check if any landmarks are detected.
    if results.pose_landmarks:

        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)

        # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        # print(results.pose_world_landmarks.landmark)
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            # Append the landmark into the list.
            landmarks_3D.append((int(landmark.x * width), int(landmark.y * height),
                                 (landmark.z * width)))
            landmarks_2D.append((int(landmark.x * width), int(landmark.y * height)))
        landmarks_D = []
        if len(landmarks_2D) >= 1:
            for i in range(len(landmarks_2D)):
                if 0 < landmarks_2D[i][0] < 1280 and 0 < landmarks_2D[i][1] < 720:
                    landmarks_D.append(int(depth_frame[landmarks_2D[i][1], landmarks_2D[i][0]]))

    # =============================================================================
    #             if (int(landmark.x* width ) and landmarks_2D[i][0] <= 720:
    #                 Depth_landmarks_3D.append((int(landmark.x* width ), int(landmark.y* height ),
    #                                        int(depth_frame[landmarks_2D[i][1], landmarks_2D[i][0]])))
    #             print(Depth_landmarks_3D)
    # =============================================================================
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:

        # Display the original input image and the resultant image.
        plt.figure(figsize=[22, 22])
        plt.subplot(121);
        plt.imshow(image[:, :, ::-1]);
        plt.title("Original Image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

        plot_landmarkss(landmarks_D, results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)


    # Otherwise
    else:
        # Return the output image and the found landmarks.
        return output_image, landmarks_3D, landmarks_2D
        # return mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)


def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.
    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360

    # Return the calculated angle.
    return angle


def calculate_landmarks_D(joints, depth_frame):
    landmarks_D = []
    # display depth of each landmarks
    if len(joints) >= 1:
        for i in range(len(joints)):
            if 0 < joints[i][0] < 1280 and 0 < joints[i][1] < 720:
                landmarks_D.append(int(depth_frame[joints[i][1], joints[i][0]]))


def classifyPose(landmarks, output_image, joints, depth_frame, display=False):
    '''
    This function classifies  poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        joints: The list of x, y and z of the landkmarks.
        depth_frame: depth_frame extracted from depth camera.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.
    '''
    global time_hey, time_Go, flag_time, wrist_GO, wrist_hey
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)

    # Calculate the required angles.
    # ----------------------------------------------------------------------------------------------------------------

    # Get the angle between the left shoulder, elbow and wrist points.
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    # Get the angle between the left elbow, shoulder and hip points.
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    if 0 < joints[mp_pose.PoseLandmark.LEFT_ELBOW.value][0] < 1280 and 0 < \
            joints[mp_pose.PoseLandmark.LEFT_ELBOW.value][1] < 720 and 0 < \
            joints[mp_pose.PoseLandmark.LEFT_WRIST.value][0] < 1280 and 0 < \
            joints[mp_pose.PoseLandmark.LEFT_WRIST.value][1] < 720:

        LEFT_ELBOW_depth = depth_frame[
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][1], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][0]]
        LEFT_WRIST_depth = depth_frame[
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][1], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][0]]
        # print(abs(int(LEFT_ELBOW_depth)-int(LEFT_WRIST_depth)))

        # =============================================================================
        #     # Get the angle between the right shoulder, elbow and wrist points.
        #     right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        #                                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        #                                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        #
        # =============================================================================

        # =============================================================================
        #     # Get the angle between the right hip, shoulder and elbow points.
        #     right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        #                                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        #                                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        #     #print(right_shoulder_angle)
        # =============================================================================

        # As for both of them, both arms should be straight and shoulders should be at the specific angle.
        # Check if the both arms are straight.
        if left_elbow_angle > 60 and left_elbow_angle < 130 and left_shoulder_angle > 75 and left_shoulder_angle < 110:

            if 50 > (abs(int(LEFT_ELBOW_depth) - int(LEFT_WRIST_depth))) > 0:
                label = 'Hey'
                time_hey = round(((time.time()) - time_pre), 2)
                wrist_hey = LEFT_WRIST_depth
            # print("Time hey: ",time_hey)
            # Check if shoulders are at the required angle.
            # if left_shoulder_angle > 75 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:

            # Check if it is the warrior II pose.
            # ----------------------------------------------------------------------------------------------------------------
            elif 400 > (abs(int(LEFT_ELBOW_depth) - int(LEFT_WRIST_depth))) > 180:

                label = "GO"
                wrist_GO = LEFT_WRIST_depth
                time_Go = round(((time.time()) - time_pre), 2)
                # Specify the label of the pose that is Warrior II pose.
            else:
                label = '****READY****'

                # Check if the pose is classified successfully
            if label != 'Unknown Pose':
                # Update the color (to green) with which the label will be written on the image.
                color = (0, 255, 0)

                # Write the label on the output image.
            cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            # if D_time!=0 :
            #    cv2.putText(output_image, D_time, (20, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            # Check if the resultant image is specified to be displayed.

            # print(time_hey, time_Go)
            # print("*****************")
            # print(wrist_hey,wrist_GO)
            if flag_time != time_hey:
                if time_hey != 0 and time_Go != 0 and (time_Go > time_hey) and (time_Go - time_hey < 4):
                    print("*")
                    print("*")
                    print('**Order is detected*********')
                    # if ((wrist_GO-wrist_hey)/(time_Go-time_hey))<10:
                    if (time_Go - time_hey) < 1.5:
                        Velocity = round(((wrist_hey - wrist_GO) / ((time_Go - time_hey) * 1000)), 2)
                        # print("displacement",wrist_GO-wrist_hey)
                        print("Go Fast")
                        print("Velocity: ", Velocity)
                        # print('wrist_hey',wrist_hey )

                        cv2.putText(output_image, "GO FAST", (1050, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                        cv2.putText(output_image, f"velocity:{Velocity} m/s", (1000, 105), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (255, 0, 0), 2)

                    if (time_Go - time_hey) > 1.5:
                        Velocity = round(((wrist_hey - wrist_GO) / ((time_Go - time_hey) * 1000)), 2)
                        print("Go Slow")
                        print("Velocity: ", Velocity)
                        cv2.putText(output_image, "GO SLOW", (1050, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                        cv2.putText(output_image, f"velocity:{Velocity} m/s", (1000, 105), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (255, 0, 0), 2)

                    flag_time = time_hey
    if display:

        # Display the resultant image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

    else:

        # Return the output image and the classified label.
        return output_image, time_hey, time_Go, label


while True:

    ret, bgr_frame, depth_frame = rs.get_frame_stream()

    bgr_frame = cv2.flip(bgr_frame, 1)
    depth_frame = cv2.flip(depth_frame, 1)

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = bgr_frame.shape

    # print(depth_colormap_dim,"***",color_colormap_dim)

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(bgr_frame, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                         interpolation=cv2.INTER_AREA)

        # Perform Pose landmark detection.
        bgr_frame, landmarks, joints = detectPose(resized_color_image, depth_frame, pose, display=False)
    else:
        bgr_frame, landmarks, joints = detectPose(bgr_frame, depth_frame, pose, display=False)

    if landmarks:

        # Perform the Pose Classification.
        frame, time_hey, time_Go, _ = classifyPose(landmarks, bgr_frame, joints, depth_frame, display=False)


    else:
        continue

    landmarks_D = []
    # display depth of each landmarks
    if len(joints) >= 1:
        for i in range(len(joints)):
            if 0 < joints[i][0] < 1280 and 0 < joints[i][1] < 720:
                cv2.putText(bgr_frame, str(depth_frame[joints[i][1], joints[i][0]]), joints[i],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                landmarks_D.append(int(depth_frame[joints[i][1], joints[i][0]]))
    cv2.imshow("bgr frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
