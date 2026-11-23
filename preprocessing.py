
import h5py
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import scipy.signal as sp
from scipy.signal import butter, butter, filtfilt, savgol_filter
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation as R
import sys
import os


class ECGProcessor:
    """ Class that handles ECG processing and heart rate detection
    """

    # Class variables
    fs = 80
    duration = 0
    peaklocations = []
    signal = []
    valid = []
    t_valid = []
    size = ''

    def __init__(self):
        """ Class constructor - class needs no arguments or variables to be set
        """

    def _process_ecg(self, ecg_data):
        """
        Function that runs the class: detects whether the ECG is valid and
        extracts beat locations.

        Parameters
        ----------
        ecg_data : numpy array
            array containing the ECG data.

        Returns
        -------
        None.

        """

        signal = self._filter_ecg(ecg_data)
        self.signal = signal
        self.valid = self._peakcounter(signal)
        self.t_valid = [x + 0.5 for x in np.where(self.valid == False)[0]]
        self.peaklocations = self._pantompkinsdetect(self.signal)

    def _moving_average(self, input, N):
        """
        Function that computes a moving average of length N over the array
        input. The first N values are set to 0.

        Parameters
        ----------
        input : numpy array
            input data.
        N : int
            Moving average window length.

        Returns
        -------
        output : numpy array
            processed array.

        """
        output = np.zeros(len(input))

        for i in range(0, len(input)):
            if i < N:
                output[i] = 0
            else:
                output[i] = sum(input[i - N:i]) / N

        return output

    def _filter_ecg(self, ecg):
        """
        Function that handles ECG filtering. Steps:
            - third-order bandpass filter 15 Hz to 30 Hz
            - difference
            - square
            - moving average over 25 points
            - peaks with an amplitude over 500 are cut off

        Parameters
        ----------
        ecg : numpy array
            ECG data.

        Returns
        -------
        ecgcutoff : numpy array
            Processed ECG data.

        """
        sos = sp.iirfilter(3, [15 / (self.fs / 2), 30 / (self.fs / 2)],
                           btype='bandpass', ftype='butter', output='sos')
        ecgfilt = sp.sosfiltfilt(sos, ecg)
        ecgdiff = np.diff(ecgfilt, prepend=0)
        ecgsquare = ecgdiff ** 2
        ecgmovavg = self._moving_average(ecgsquare, 25)
        ecgcutoff = ecgmovavg  # [min(50000, x) for x in ecgmovavg]

        return ecgcutoff

    def _peakcheck(self, signal, NPKI):
        """ Helper function for Pan-Tompkins detector. Checks whether sample is
        a peak """
        return (signal[0] < signal[1] and signal[2] < signal[1]
                and signal[1] > NPKI / 4)

    def _threshold_I1check(self, signal, threshold, i, lastpeak, mean_rr):
        """ Helper function for Pan-Tompkins detector. Checks whether a peak is
        a heartbeat """
        return (signal > threshold and i > lastpeak +
                max(self.fs * 0.30, mean_rr * 0.2))

    def _threshold_I2check(self, signal, threshold, older_peak, missedpeak,
                           mean_rr):
        """ Helper function for Pan-Tompkins detector. Checks whether a peak is
        a heartbeat when backsearching for missed peaks. """
        return (signal > threshold and older_peak - missedpeak >
                max(self.fs * 0.3, mean_rr * 0.2))

    def _pantompkinsdetect(self, signal):
        """
        Heartbeat detector using a Pan-Tompkins algorithm.

        Parameters
        ----------
        signal : numpy array
            Preprocessed ECG signal.

        Returns
        -------
        outindices : list of ints
            array containing the indices of heartbeats in the ECG signal.

        """
        # detection parameters
        threshold_init = 0
        SPKI = [threshold_init]
        SPKIindices = [0]
        NPKI = [threshold_init / 4]
        NPKIindices = [0]
        # instead of 0, take initial threshold a bit higher
        threshold_I1 = [threshold_init * 0.5]
        thresholdindices = [0]
        threshold_I2 = [0]
        peaks = []
        peaksindices = []
        R_peaks = [0, 0]
        R_peaksindices = [0]
        noise_peaks = []
        lastpeak = -self.fs / 4  # slight offset to prevent missing labels
        lastnoisepeak = -self.fs / 10
        rr = [0]
        mean_rr = 0
        rr_missed = 2 * self.fs
        missedpeaks = [0]

        # Pan-Tompkins peak finding logic
        for i in range(2, len(signal)):
            if self._peakcheck(signal[i - 2:i + 1], NPKI[-1]):
                peak = i - 1
                peaksindices.append(i - 1)
                peaks.append(signal[i - 1])

                if self._threshold_I1check(signal[peak], threshold_I1[-1], i,
                                           lastpeak, mean_rr):
                    R_peaks.append(signal[peak])
                    R_peaksindices.append(peak)

                    SPKI.append(0.125 * signal[peak] + 0.875 * SPKI[-1])
                    SPKIindices.append(peak)
                    rr.append(np.diff(R_peaksindices[-2:])[0])
                    lastpeak = i
                    threshold_I1.append(NPKI[-1] + 0.25 * (SPKI[-1] - NPKI[-1]))
                    # only for missed peaks detection
                    threshold_I2.append(0.5 * threshold_I1[-1])
                    thresholdindices.append(peak)

                    if len(R_peaksindices) > 4:
                        mean_rr = np.mean(rr[-5:])
                        rr_missed = int(1.4 * mean_rr)
                        # slight modification to exclude low-level noise and
                        # prevent double signal peaks
                else:
                    if i - R_peaksindices[-1] > rr_missed:
                        peaks = [x for x in peaksindices if x > i - rr_missed
                                 + self.fs * 0.3 and x <= i]
                        for older_peak in peaks:
                            if self._threshold_I2check(signal[older_peak],
                                                       threshold_I2[-1],
                                                       older_peak,
                                                       missedpeaks[-1],
                                                       mean_rr):
                                missedpeaks.append(older_peak)
                                SPKI.append(
                                    0.75 * signal[older_peak] + 0.25 * SPKI[-1])
                                SPKIindices.append(peak)
                                NPKI.append(min(NPKI[-1], SPKI[-1]))
                                NPKIindices.append(peak)
                                rr.append(np.diff(R_peaksindices[-2:])[0])
                                # only for missed peaks detection
                                threshold_I1.append(
                                    NPKI[-1] + 0.25 * (SPKI[-1] - NPKI[-1]))
                                threshold_I2.append(0.5 * threshold_I1[-1])
                                thresholdindices.append(older_peak)
                                # if none found
                            elif i - missedpeaks[-1] > mean_rr * 1.0:
                                threshold_I1.append(threshold_I1[-1])
                                threshold_I2.append(0.5 * threshold_I2[-1])
                                thresholdindices.append(older_peak)

                    elif i > lastpeak + self.fs / 10:
                        lastnoisepeak = i
                        noise_peaks.append(peak)
                        NPKI.append(
                            0.125 * signal[noise_peaks[-1]] + 0.875 * NPKI[-1])
                        NPKIindices.append(i)
                        threshold_I1.append(
                            NPKI[-1] + 0.25 * (SPKI[-1] - NPKI[-1]))
                        # only for missed peaks detection
                        threshold_I2.append(0.5 * threshold_I1[-1])
                        thresholdindices.append(i)

        # first value not correct, needed to get an initial value
        R_peaksindices.pop(0)
        rr.pop(0)

        outindices = R_peaksindices
        outindices = [x / self.fs for x in outindices]

        return outindices

    def _peakcounter(self, signal):
        """
        Function that counts the amount of peaks in one second of ECG signal.
        If there are more than five peaks, the signal is deemed noisy and the
        detected heart beats invalid.

        Parameters
        ----------
        signal : numpy array
            Input signal, probably raw ECG.

        Returns
        -------
        valid : list of booleans
            List that contains for every second of data boolean stating whether
            the segment is noisy or not.

        """
        valid = []

        if self.size == 's':
            height = 10
        else:
            height = 100

        for i in range(0, len(signal), self.fs):
            peaks, properties = sp.find_peaks(signal[i:i + self.fs],
                                              height=height, prominence=30)
            numpeaks = len(peaks)

            if numpeaks > 5:
                valid.append(False)
            else:
                valid.append(True)

        return valid

    def calc_hr(self, ecg_data, size):
        """
        Public function that takes an ECG signal and returns the current heart
        rate and the time of all heartbeats in that signal.

        Parameters
        ----------
        ecg_data : numpy array
            Raw ECG data.
        size : char
            Size of BioHarness strap used.

        Returns
        -------
        hr : list of floats
            Average heart rate corresponding to the time in t_hr at the same
            index.
        t_hr : list of floats
            List containing the timestamps of all heartbeats.

        """
        self.size = size
        self._process_ecg(ecg_data)
        valid_peaks = [x for x in self.peaklocations if
                       all(abs(x - y) > 0.5 for y in self.t_valid)]

        # calculate RR distances
        rr = np.diff(valid_peaks)

        # calculate BPM values
        length = len(rr)
        hr = np.zeros(length)
        n = 8
        rr_sub = []

        for i in range(0, length):
            if i > (n - 1):
                rr_prev = 1 / (hr[i - 1] / 60)
                rr_temp = rr[i - n:i]
                rr_sub = []

                for value in rr_temp:
                    if value < 1.5 * rr_prev:
                        rr_sub.append(value)

                if len(rr_sub) > 0:
                    hr[i] = 60 * 1 / (sum(rr_sub) / len(rr_sub))
                else:
                    hr[i] = 60 * 1 / rr_prev

            else:
                hr[i] = 60 * 1 / rr[i]

        t_hr = valid_peaks[1:]

        return (hr, t_hr)

participant_ID = 5

filepath = r"C:\Users\ZhangY5\OneDrive - University of Twente\PhD\Experiment\Remco data\Official Experiment\Participant {}".format(participant_ID)

def calc_hr_from_raw_ecg(filename):

    # Load file
    h5 = h5py.File(filename,'r')

    # Extract data
    h5_group = h5.get('CC:86:EC:AB:26:BA')
    h5_sub_group = h5_group.get("raw")
    h5_data = h5_sub_group.get("channel_1")
    ecg = np.array([item for sublist in h5_data for item in sublist])
    sampling_rate = h5_group.attrs.get("sampling rate")
    start_time = h5_group.attrs.get('time')
    signal_length = len(ecg)

    # Calculate HR
    ecgprocessor = ECGProcessor()
    hr, t_hr = ecgprocessor.calc_hr(ecg, 'm')
    hr = hr[1:]
    t_hr = t_hr[1:]

    # Convert starting time to correct format
    start_time = datetime.datetime.strptime(start_time, "%H:%M:%S.%f").time()

    return t_hr, hr, sampling_rate, start_time, signal_length

def interpolate_hr(HR_ts, HR, sampling_rate, signal_length):
    duration = signal_length / sampling_rate
    time_axis = np.arange(0, 0 + duration, 1 / sampling_rate)
    interp_hr = interp1d(HR_ts, HR, kind='linear', fill_value='extrapolate')
    return time_axis, interp_hr(time_axis)

def smooth_data(data, window_length):
    return savgol_filter(data, window_length=window_length, polyorder=1)


filename = filepath + r"\CardioBan\ADL.h5"

t_hr, hr, sampling_rate_hr, start_time_hr, signal_length_hr = calc_hr_from_raw_ecg(filename)
t_hr, hr = interpolate_hr(t_hr, hr, sampling_rate_hr, signal_length_hr)

plt.figure(1)
plt.plot(t_hr, hr, label="hr")
plt.show()

hr = smooth_data(hr, 60*sampling_rate_hr)

plt.figure(1)
plt.plot(t_hr, hr, label="hr")
plt.show()


def read_k5_data(filename):
    # Load file
    k5_df = pd.read_excel(filename)

    # Extract data
    time_axis = k5_df.loc[2:, 't']
    time_axis = np.array([((t.hour * 60 + t.minute) * 60 + t.second) for t in time_axis])
    vo2 = k5_df.loc[2:, 'VO2']
    vco2 = k5_df.loc[2:, 'VCO2']
    start_time = k5_df.iloc[0, 4]
    markers = k5_df.loc[2:, 'Marker'].values.astype(str)

    return time_axis, vo2, vco2, start_time, markers


def find_non_nan_markers(k5_markers, k5_time_axis):
    # Find indices where k5_markers is not NaN or another string
    non_nan_indices = np.where(k5_markers != 'nan')[0]

    # Extract timestamps corresponding to non-NaN markers
    non_nan_timestamps = k5_time_axis[non_nan_indices]

    # Extract marker names corresponding to non-NaN markers
    non_nan_marker_names = k5_markers[non_nan_indices]

    return non_nan_timestamps, non_nan_marker_names


def interpolate_k5(k5_ts, k5_vo2, k5_vco2, sampling_frequency):
    time_axis = np.arange(k5_ts[0], k5_ts[-1], 1 / sampling_frequency)
    interp_k5_vo2 = interp1d(k5_ts, k5_vo2, kind='linear', fill_value='extrapolate')
    interp_k5_vco2 = interp1d(k5_ts, k5_vco2, kind='linear', fill_value='extrapolate')

    return time_axis, interp_k5_vo2(time_axis), interp_k5_vco2(time_axis)


def smooth_data(data, window_length):
    return savgol_filter(data, window_length=window_length, polyorder=1)

filename = filepath+"\K5\ADL.xlsx"

t_k5, vo2, vco2, start_time_k5, markers_k5 = read_k5_data(filename)
marker_timestamps, marker_names = find_non_nan_markers(markers_k5, t_k5)
t_k5, vo2_k5, vco2_k5 = interpolate_k5(t_k5, vo2, vco2, sampling_rate_hr)

plt.figure(1)
plt.plot(t_k5, vo2_k5, label="VO2")
plt.plot(t_k5, vco2_k5, label="VCO2")
plt.show()

vo2_k5 = smooth_data(vo2_k5, 60*sampling_rate_hr)
vco2_k5 = smooth_data(vco2_k5, 60*sampling_rate_hr)

plt.figure(1)
plt.plot(t_k5, vo2_k5, label="VO2")
plt.plot(t_k5, vco2_k5, label="VCO2")
plt.show()


def read_imu_data(filename):
    """
    Read IMU data from a CSV file.

    Parameters:
    filename (str): The path to the CSV file containing IMU data.

    Returns:
    tuple: Time axis, acceleration in x, y, z directions, and start time.
    """
    # Load file
    imu_df = pd.read_csv(filename, on_bad_lines='skip', header=0)
    acceleration, gyrations = remove_gravity(imu_df)

    # Sampling frequency
    fs = 30
    time_axis = np.arange(0, acceleration.shape[0]) / fs

    # Extract start time from filename
    start_time = datetime.datetime.strptime(filename[-10:-4], "%H%M%S").time()
    # return time_axis[1:], acceleration[1:, 0], acceleration[1:, 1], acceleration[1:, 2], start_time
    return (time_axis[1:], acceleration[1:, 0], acceleration[1:, 1], acceleration[1:, 2],
            gyrations[1:, 0], gyrations[1:, 1], gyrations[1:, 2], start_time)


def remove_gravity(data):
    """
    Convert sensor data to global frame.

    Parameters:
    data (DataFrame): Sensor data with quaternion or Euler angles and acceleration.

    Returns:
    ndarray: Acceleration data in the global frame.
    """
    # Check for quaternion data and convert to rotation matrix
    if 'Quat_X' in data.columns:
        # 修复：确保四元数列是数值类型
        quat_cols = ['Quat_X', 'Quat_Y', 'Quat_Z', 'Quat_W']

        # 将字符串转换为数值，非数值转换为NaN
        for col in quat_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # 现在提取数值
        quat = data.loc[:, quat_cols].values

        # 检查是否有NaN值
        nan_mask = np.isnan(quat).any(axis=1)
        if np.any(nan_mask):
            print(f"警告：找到 {np.sum(nan_mask)} 行包含NaN的四元数，将使用单位四元数替换")
            # 用单位四元数替换NaN行
            quat[nan_mask] = [0, 0, 0, 1]

        # 计算四元数范数
        norms = np.sqrt(np.sum(quat ** 2, axis=1))

        # 找到范数为零或无效的四元数
        invalid_mask = (norms == 0) | np.isnan(norms) | np.isinf(norms)
        valid_mask = ~invalid_mask

        if np.any(invalid_mask):
            print(f"警告：找到 {np.sum(invalid_mask)} 个无效四元数，将使用单位四元数替换")

            # 用单位四元数 [0, 0, 0, 1] 替换无效值
            quat[invalid_mask] = [0, 0, 0, 1]

            # 重新计算范数
            norms = np.sqrt(np.sum(quat ** 2, axis=1))

        # 归一化四元数
        quat = quat / norms[:, np.newaxis]

        r = R.from_quat(quat)

    # Check for Euler angles and convert to rotation matrix
    elif 'Euler_X' in data.columns:
        # 同样修复欧拉角数据
        euler_cols = ['Euler_X', 'Euler_Y', 'Euler_Z']
        for col in euler_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        euler = data.loc[:, ['Euler_Z', 'Euler_Y', 'Euler_X']]
        r = R.from_euler('ZYX', euler, degrees=True)
    else:
        # 如果没有四元数或欧拉角数据，返回原始数据
        print("警告：没有找到四元数或欧拉角数据，跳过重力去除")
        return data.loc[:, ['Acc_X', 'Acc_Y', 'Acc_Z']].astype('float32').values, np.zeros((len(data), 3))

    # 同样修复加速度和陀螺仪数据
    acc_cols = ['Acc_X', 'Acc_Y', 'Acc_Z']
    gyr_cols = ['Gyr_X', 'Gyr_Y', 'Gyr_Z']

    for col in acc_cols + gyr_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Extract acceleration data
    acceleration = data.loc[:, acc_cols].astype('float32').values

    # 检查加速度数据中的NaN
    if np.isnan(acceleration).any():
        print(f"警告：加速度数据中有NaN，使用0填充")
        acceleration = np.nan_to_num(acceleration)

    # Extract gyroscope data
    gyrations = data.loc[:, gyr_cols].astype('float32').values
    gyrations = np.deg2rad(gyrations)  # Convert °/s → rad/s

    # 检查陀螺仪数据中的NaN
    if np.isnan(gyrations).any():
        print(f"警告：陀螺仪数据中有NaN，使用0填充")
        gyrations = np.nan_to_num(gyrations)

    # Subtract gravity to get true acceleration
    gravity = np.array([0, 0, 9.81])
    gravity_global = r.inv().apply(gravity)
    acceleration -= gravity_global

    return acceleration, gyrations


def butter_lowpass_filter(data, cut, fs, order=8):
    nyquist = 0.5 * fs
    cut = cut / nyquist
    b, a = butter(order, cut, btype='low', analog=False, output='ba')
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def butter_highpass_filter(data, cut, fs, order=8):
    nyquist = 0.5 * fs
    cut = cut / nyquist
    b, a = butter(order, cut, btype='high', analog=False, output='ba')
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def calculate_velocity(acceleration, time_axis):
    velocity = cumulative_trapezoid(acceleration, x=time_axis, initial=0.0)
    window_size = 5
    rolling_window = (np.convolve(acceleration.T, np.ones(window_size), 'same'))
    velocity[rolling_window == 0] = 0

    return velocity


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global min of dmin-chunks of locals min
    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
    # global max of dmax-chunks of locals max
    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]

    return lmin, lmax


def process_imu_data(filepath):
    """
    Process IMU data from a CSV file to calculate velocity and displacement.

    Parameters:
    filepath (str): The path to the CSV file containing IMU data.

    Returns:
    tuple: A tuple containing time axis, acceleration, velocity, displacement, and start time.
    """
    # Define filter parameters
    sampling_rate = 30  # Sampling rate in Hz

    # Read IMU data from the file
    # t_imu, acc_x, acc_y, acc_z, =start_time_imu = read_imu_data(filepath)

    # Ramon's implementation
    t_imu, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, start_time_imu = read_imu_data(filepath)

    # Apply lowpass filter to accelerometer data
    filtered_x = butter_lowpass_filter(acc_x, 5, sampling_rate)
    filtered_y = butter_lowpass_filter(acc_y, 5, sampling_rate)
    filtered_z = butter_lowpass_filter(acc_z, 5, sampling_rate)

    # Ramon
    filtered_gyr_x = butter_lowpass_filter(gyr_x, 10, sampling_rate)
    filtered_gyr_y = butter_lowpass_filter(gyr_y, 10, sampling_rate)
    filtered_gyr_z = butter_lowpass_filter(gyr_z, 10, sampling_rate)

    # Apply highpass filter to accelerometer data
    filtered_x = butter_highpass_filter(filtered_x, 0.5, sampling_rate)
    filtered_y = butter_highpass_filter(filtered_y, 0.5, sampling_rate)
    filtered_z = butter_highpass_filter(filtered_z, 0.5, sampling_rate)

    # Ramon
    filtered_gyr_x = butter_highpass_filter(filtered_gyr_x, 0.5, sampling_rate)
    filtered_gyr_y = butter_highpass_filter(filtered_gyr_y, 0.5, sampling_rate)
    filtered_gyr_z = butter_highpass_filter(filtered_gyr_z, 0.5, sampling_rate)

    # Estimate linear velocity for each direction
    linear_velocity_x = calculate_velocity(filtered_x, t_imu)
    linear_velocity_y = calculate_velocity(filtered_y, t_imu)
    linear_velocity_z = calculate_velocity(filtered_z, t_imu)

    # Calculate total velocity
    total_velocity = np.sqrt(np.square(linear_velocity_x) +
                             np.square(linear_velocity_y) +
                             np.square(linear_velocity_z))

    # Ramon
    total_gyration = np.sqrt(np.square(filtered_gyr_x) +
                             np.square(filtered_gyr_y) +
                             np.square(filtered_gyr_z))

    # Create linear envelope of velocity
    lmin, lmax = hl_envelopes_idx(total_velocity, dmin=1, dmax=30)
    interp_velocity = interp1d(t_imu[lmax], total_velocity[lmax],
                               kind='linear', fill_value='extrapolate')
    total_velocity = interp_velocity(t_imu)

    # Calculate acceleration
    #total_acceleration = np.diff(total_velocity) / sampling_rate
    total_acceleration = abs(np.sqrt(np.square(acc_x) +
                             np.square(acc_y) +
                             np.square(acc_z)))
    return t_imu, total_velocity, total_acceleration, total_gyration, start_time_imu

IMU_folder_path = filepath + "\IMU"
IMU_filenames = filenames = os.listdir(IMU_folder_path)
IMU_dict = {}

for filename in IMU_filenames:
    dot_number = filename[:5].replace("_", "")
    IMU_dict[dot_number] = process_imu_data(IMU_folder_path+"/"+filename)


def convert_to_unix(time_axis, start_time):
    """
    Convert a time axis and start time to Unix timestamps.

    Parameters:
    time_axis (np.array): The time axis array.
    start_time (datetime.time): The start time.

    Returns:
    np.array: Array of Unix timestamps.
    """
    # Get the current date
    current_date = datetime.datetime.now().date()

    # Convert start time to a datetime object
    start_time = datetime.datetime.combine(current_date, start_time)

    # Convert resampled time axis to datetime format
    resampled_datetime = [start_time + datetime.timedelta(milliseconds=int(t * 1000)) for t in time_axis]

    # Convert datetime objects to Unix timestamps
    unix_timestamps = np.array([(t - datetime.datetime(1970, 1, 1)).total_seconds() for t in resampled_datetime])

    return unix_timestamps


def find_common_indices(signal_times, marker_names):
    """
    Find common indices for overlapping time ranges in signal times.

    Parameters:
    signal_times (list): List of signal times for different measurements.
    marker_names (list): List of marker names.

    Returns:
    tuple: Indices for cutting signals and start/end times.
    """
    # Extract individual signal times
    t_hr_unix = signal_times[0]
    t_k5_unix = signal_times[1]
    t_imu_unix = signal_times[2]
    t_markers = signal_times[3]

    # Determine the common start time
    if "Start" in marker_names:
        start_idx = np.where(marker_names == "Start")[0][0]
        start_time = t_markers[start_idx]
    else:
        start_time = min(t_markers)

    # Determine the common end time
    if marker_names[-1] == "Stop":
        end_time = max(t_markers)
    else:
        end_time = min([times[-1] for times in signal_times[0:3]])

    # Find indices corresponding to the common time range
    HR_start_index = np.argmax(np.array(t_hr_unix) >= start_time)
    HR_end_index = np.argmax(np.array(t_hr_unix) >= end_time)
    k5_start_index = np.argmax(np.array(t_k5_unix) >= start_time)
    k5_end_index = np.argmax(np.array(t_k5_unix) >= end_time)
    imu_start_index = np.argmax(np.array(t_imu_unix) >= start_time)
    imu_end_index = np.argmax(np.array(t_imu_unix) >= end_time)

    return [HR_start_index, HR_end_index, k5_start_index, k5_end_index, imu_start_index, imu_end_index], [start_time,
                                                                                                          end_time]


def cut_signals(imu_dict, hr, vo2, vco2, signal_times, cut_indices, start_end):
    """
    Cut signals to the common time range and adjust times.

    Parameters:
    imu_dict (dict): Dictionary of IMU data.
    hr (np.array): Heart rate data.
    vo2 (np.array): VO2 data.
    vco2 (np.array): VCO2 data.
    signal_times (list): List of signal times for different measurements.
    cut_indices (list): Indices for cutting signals.
    start_end (list): Start and end times for the common range.

    Returns:
    tuple: Cut and adjusted signals and times.
    """
    t_hr = signal_times[0]
    t_k5 = signal_times[1]
    t_imu = signal_times[2]
    t_marker = signal_times[3]

    # Cut signals to the common time range
    hr = hr[cut_indices[0]:cut_indices[1]]
    vo2 = vo2[cut_indices[2]:cut_indices[3]]
    vco2 = vco2[cut_indices[2]:cut_indices[3]]
    t_marker = t_marker[t_marker < start_end[1]]
    t_hr = t_hr[cut_indices[0]:cut_indices[1]]
    t_k5 = t_k5[cut_indices[2]:cut_indices[3]]
    t_imu = t_imu[cut_indices[4]:cut_indices[5]] - start_end[0]

    # Cut and adjust IMU signals
    cut_imu_dict = {}
    # for key, (_, velocity, acceleration, start_time_imu) in imu_dict.items():
    #     cut_imu_dict[key] = [t_imu, velocity[cut_indices[4]:cut_indices[5]], acceleration[cut_indices[4]:cut_indices[5]], start_time_imu]

    # Ramon
    for key, (_, velocity, acceleration, gyration, start_time_imu) in imu_dict.items():
        cut_imu_dict[key] = [t_imu, velocity[cut_indices[4]:cut_indices[5]],
                             acceleration[cut_indices[4]:cut_indices[5]],
                             gyration[cut_indices[4]:cut_indices[5]], start_time_imu]

    # Adjust times relative to start time
    t_hr -= start_end[0]
    t_k5 -= start_end[0]
    t_marker -= start_end[0]

    return hr, t_hr, vo2, vco2, t_k5, cut_imu_dict, t_marker


# Convert various time axes to Unix timestamps
t_k5_unix = convert_to_unix(t_k5, start_time_k5)
t_hr_unix = convert_to_unix(t_hr, start_time_hr)
# t_imu_unix = convert_to_unix(IMU_dict['DOT1'][0], IMU_dict['DOT1'][3])
# Ramon
t_imu_unix = convert_to_unix(IMU_dict['DOT1'][0], IMU_dict['DOT1'][-1])
t_marker_unix = convert_to_unix(marker_timestamps, start_time_k5)

# Find common indices and time range for cutting the signals
cut_indices, start_end = find_common_indices([t_hr_unix, t_k5_unix, t_imu_unix, t_marker_unix], marker_names)

# Cut and adjust the signals based on the common time range
hr, t_hr, vo2, vco2, t_k5, IMU_dict, t_marker = cut_signals(IMU_dict, hr, vo2_k5, vco2_k5,
                                                            [t_hr_unix, t_k5_unix, t_imu_unix, t_marker_unix],
                                                            cut_indices, start_end)


def resample_signal(time_axis, signal, new_sampling_frequency, max_time):
    """
    Resample a signal to a new sampling frequency.

    Parameters:
    time_axis (np.array): Original time axis.
    signal (np.array): Original signal.
    new_sampling_frequency (float): New sampling frequency.
    max_time (float): Maximum time value for the new time axis.

    Returns:
    tuple: New time axis and resampled signal.
    """
    # If the signal has columns (dataframe), use the first column
    if hasattr(signal, 'columns'):
        signal = signal[0]

    # Ensure signal length matches time axis length
    if len(signal) < len(time_axis):
        diff = len(time_axis) - len(signal)
        signal = np.append(signal, np.zeros((diff,)))
    if len(signal) > len(time_axis):
        diff = len(signal) - len(time_axis)
        signal = signal[:-diff]

    # Create the new time axis
    new_time_axis = np.arange(0, max_time, 1 / new_sampling_frequency)

    # Interpolate signal values onto the new time axis
    interp_func = interp1d(time_axis, signal, kind='slinear', fill_value="extrapolate")
    resampled_signal = interp_func(new_time_axis)

    return new_time_axis, resampled_signal


def unix_to_seconds(unix_timestamps):
    """
    Convert Unix timestamps to seconds since the Unix epoch.

    Parameters:
    unix_timestamps (np.array): Array of Unix timestamps.

    Returns:
    np.array: Array of seconds since the Unix epoch.
    """
    seconds = unix_timestamps - np.datetime64('1970-01-01T00:00:00')
    return seconds.astype('int64')


def plot_signals(t_hr, hr, t_k5, vo2_k5, vco2_k5, imu_dict, t_marker, marker_names):
    """
    Plot and export signals.

    Parameters:
    t_hr (np.array): Time axis for heart rate.
    hr (np.array): Heart rate signal.
    t_k5 (np.array): Time axis for k5 data.
    vo2_k5 (np.array): VO2 signal from k5.
    vco2_k5 (np.array): VCO2 signal from k5.
    imu_dict (dict): Dictionary containing IMU data.
    t_marker (np.array): Marker time axis.
    marker_names (list): List of marker names.

    Returns:
    dict: Exported data including resampled signals and markers.
    """
    # Dictionary to store exported data
    exported_data = {}

    # Convert VO2 and VCO2 to per second
    vo2_k5 /= 60
    vco2_k5 /= 60

    # Determine maximum time for resampling
    max_time = max(t_hr)

    # Resample the signals to 1 Hz
    t_hr, hr = resample_signal(t_hr, hr, 1, max_time)
    _, vo2_k5 = resample_signal(t_k5, vo2_k5, 1, max_time)
    t_k5, vco2_k5 = resample_signal(t_k5, vco2_k5, 1, max_time)

    # Resample IMU signals
    for key, (t_imu, velocity, acceleration, gyration, start_time_imu) in imu_dict.items():
        _, velocity = resample_signal(t_imu, velocity, 1, max_time)
        # t_imu, acceleration = resample_signal(t_imu, acceleration, 1, max_time)
        _, acceleration = resample_signal(t_imu, acceleration, 1, max_time)
        t_imu, gyration = resample_signal(t_imu, gyration, 1, max_time)
        imu_dict[key] = {
            't_imu': t_imu,
            'velocity': velocity,
            'acceleration': acceleration,
            # Ramon
            'gyration': gyration,
            'start_time_imu': start_time_imu
        }
        # Export the velocity and acceleration data
        exported_data['velocity_' + key] = velocity
        exported_data['acceleration_' + key] = acceleration
        exported_data['gyration_' + key] = gyration

    # Create subplots with shared x-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    # Plot HR signal
    ax1.plot(t_hr, hr, label='HR', color='tab:blue')
    ax1.set_ylabel('HR', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot VO2 and VCO2 signals
    ax2.plot(t_k5, vo2_k5, label='VO2', color='tab:red', linestyle='dashdot')
    ax2.plot(t_k5, vco2_k5, label='VCO2', color='tab:orange', linestyle='dashdot')
    ax2.set_ylabel('Flow (mL/second)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Plot IMU signals
    ax3.plot(t_imu, velocity, label='IMU', color='tab:green', linestyle='dashdot')
    ax3.set_ylabel('Flow (mL/second)', color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')

    # Plot markers
    for timestamp, marker_name in zip(t_marker, marker_names):
        ax1.axvline(x=timestamp, color='black', linestyle='--')
        ax1.text(timestamp, ax1.get_ylim()[1] * 0.75, marker_name, rotation=90, verticalalignment='top',
                 horizontalalignment='left')

    # Add labels and legend
    ax1.set_xlabel('Time (seconds)')
    plt.title('HR, VO2 and VCO2')
    fig.legend(loc="upper right")
    plt.savefig("hr_vo2_vco2.svg")
    plt.show()

    # Create a dictionary to map timestamps to marker names
    marker_dict = {int(timestamp): name for timestamp, name in zip(t_marker, marker_names)}

    # Resample the marker names to align with the new time axis
    resampled_marker_names = [marker_dict.get(int(t), '') for t in t_hr]

    # Export the data
    exported_data['t'] = t_hr
    exported_data['hr'] = hr
    exported_data['vo2'] = vo2_k5
    exported_data['vco2'] = vco2_k5
    exported_data['markers'] = resampled_marker_names

    return exported_data

resampled_data = plot_signals(t_hr, hr, t_k5, vo2, vco2, IMU_dict, t_marker, marker_names)

ramon_filepath = r"C:\Users\ZhangY5\OneDrive - University of Twente\PhD\Experiment\Remco data\Yaowen\Data\Participant {}".format(participant_ID)
pd.DataFrame(resampled_data).to_excel(ramon_filepath+"/ramon_windowed_data.xlsx")