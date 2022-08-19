from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
import matplotlib.pyplot as plt
import numpy as np
import sensormotion as sm


def calculateVelocity(point1: list, point2: list, time: float) -> Tuple[float, float, float]:
    velocity_x = (point2[0] - point1[0]) / time
    velocity_y = (point2[1] - point1[1]) / time
    velocity_z = (point2[2] - point1[2]) / time
    return velocity_x, velocity_y, velocity_z


def calculateAcceleration(point1: list, point2: list, time: float) -> List[float]:
    acceleration = [velocity / time for velocity in calculateVelocity(point1, point2, time)]
    return acceleration


def calculateAccelerationList(point_list: list, time: float) -> List[List]:
    acceleration_list = []
    for i in range(len(point_list) - 1):
        acceleration_list.append(calculateAcceleration(point_list[i], point_list[i + 1], time))
    return acceleration_list


def calculateAccelerationListFrame(point_list: list, frames: int) -> Tuple[ndarray, ndarray, ndarray]:
    accelerations = calculateAccelerationList(point_list, 1 / frames)
    return np.array([acceleration[0] for acceleration in accelerations]), np.array(
        [acceleration[1] for acceleration in accelerations]), np.array(
        [acceleration[2] for acceleration in accelerations])


def sensormotionDemo(pts_cam: ndarray, fps: int):
    accelerations_x, accelerations_y, accelerations_z = calculateAccelerationListFrame(
        [keypoints[26] for keypoints in pts_cam.tolist()], fps)

    sampling_rate = fps  # number of samples per second

    frames_time = np.array([frame_index * 1000 / fps for frame_index in range(len(pts_cam) - 1)])

    sm.plot.plot_signal(frames_time,
                        [{'data': accelerations_x, 'label': 'Medio-lateral (ML) - side to side', 'line_width': 0.5},
                         {'data': accelerations_y, 'label': 'Vertical (VT) - up down', 'line_width': 0.5},
                         {'data': accelerations_z, 'label': 'Antero-posterior (AP) - forwards backwards',
                          'line_width': 0.5}],
                        subplots=True, fig_size=(10, 7))

    _ = sm.signal.fft(accelerations_y, sampling_rate, plot=True)

    sm.plot.plot_filter_response(10, sampling_rate, 'low', filter_order=4)

    b, a = sm.signal.build_filter(10, sampling_rate, 'low', filter_order=4)

    # Filter signals
    x_f = sm.signal.filter_signal(b, a, accelerations_x)  # ML medio-lateral
    y_f = sm.signal.filter_signal(b, a, accelerations_y)  # VT vertical
    z_f = sm.signal.filter_signal(b, a, accelerations_z)  # AP antero-posterior

    # Create plots with overlaid filtered signals (in red)
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 9))

    ax[0].set_title('Medio-lateral (ML) - side to side')
    ax[0].plot(frames_time, accelerations_x, linewidth=0.3, color='k')
    ax[0].plot(frames_time, x_f, linewidth=0.8, color='r')

    ax[1].set_title('Vertical (VT) - up down')
    ax[1].plot(frames_time, accelerations_y, linewidth=0.3, color='k')
    ax[1].plot(frames_time, y_f, linewidth=0.9, color='r')

    ax[2].set_title('Antero-posterior (AP) - forwards backwards')
    ax[2].plot(frames_time, accelerations_z, linewidth=0.3, color='k')
    ax[2].plot(frames_time, z_f, linewidth=0.9, color='r')

    fig.subplots_adjust(hspace=.5)

    peak_times, peak_values = sm.peak.find_peaks(frames_time, y_f, peak_type='valley', min_val=0.6, min_dist=10,
                                                 plot=True)

    step_count = sm.gait.step_count(peak_times)
    cadence = sm.gait.cadence(frames_time, peak_times)
    step_time, step_time_sd, step_time_cov = sm.gait.step_time(peak_times)

    print(' - Number of steps: {}'.format(step_count))
    print(' - Cadence: {:.2f} steps/min'.format(cadence))
    print(' - Mean step time: {:.2f}ms'.format(step_time))
    print(' - Step time variability (standard deviation): {:.2f}'.format(step_time_sd))
    print(' - Step time variability (coefficient of variation): {:.2f}'.format(step_time_cov))

    ac, ac_lags = sm.signal.xcorr(y_f, y_f, scale='unbiased', plot=True)

    ac_peak_times, ac_peak_values = sm.peak.find_peaks(ac_lags, ac, peak_type='peak', min_val=0.1, min_dist=30,
                                                       plot=True)

    step_reg, stride_reg = sm.gait.step_regularity(ac_peak_values)
    step_sym = sm.gait.step_symmetry(ac_peak_values)

    print(' - Step regularity: {:.4f}'.format(step_reg))
    print(' - Stride regularity: {:.4f}'.format(stride_reg))
    print(' - Step symmetry: {:.4f}'.format(step_sym))

    x_counts = sm.pa.convert_counts(accelerations_x, frames_time, time_scale='ms', epoch=10, rectify='full',
                                    integrate='simpson', plot=True)
    y_counts = sm.pa.convert_counts(accelerations_y, frames_time, time_scale='ms', epoch=10, rectify='full',
                                    integrate='simpson', plot=True)
    z_counts = sm.pa.convert_counts(accelerations_z, frames_time, time_scale='ms', epoch=10, rectify='full',
                                    integrate='simpson', plot=True)

    vm = sm.signal.vector_magnitude(x_counts, y_counts, z_counts)

    categories, time_spent = sm.pa.cut_points(vm, set_name='butte_preschoolers', n_axis=3, plot=True)

    print('Categories: {}\n'.format(categories))
    print('Time spent:')
    print(time_spent)
