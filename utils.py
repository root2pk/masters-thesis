import librosa
import librosa.display
import soundfile as sf
from scipy.signal import savgol_filter, cheby2, filtfilt, find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import convolve1d
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.special import erf
import numpy as np
import IPython.display as ipd
import mir_eval

import os
# Set logging level for tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import essentia
# # Set logging level for essentia
essentia.log.warningActive = False               # deactivate the warning level
essentia.log.infoActive = False 
from essentia.standard import TonicIndianArtMusic, PredominantPitchMelodia, EqualLoudness

def ms_to_samples(ms, sr):
    """
    Function to convert milliseconds to samples

    Parameters:
    ms : float
        Time in milliseconds
    sr : int
        Sampling rate
    
    Returns:   
    samples : int
        Time in samples
    """
    samples = int(ms * sr / 1000)

    return samples

def process_filename(filename):
    """
    Function to process the filename

    Parameters:
    filename : str
        Filename containing the raga, piece, instrument and section information

    Returns:
    raga : str
        Raga name
    piece : str
        Piece name
    instrument : str
        Instrument name
    section : str
        Section name
    """
    raga = filename.split("/")[0]
    piece = filename.split("/")[1]
    track_info = filename.split("/")[2].split(".")[1]
    instrument = track_info.split("-")[1]
    section = track_info.split("-")[2] if len(track_info.split("-")) > 2 else ""

    return raga, piece, instrument, section

def merge_intervals(intervals):
    """
    Function to merge overlapping intervals

    Parameters:
    intervals : list
        List of tuples containing the start and end times of the intervals
    
    Returns:
    merged : list
        List of tuples containing the merged intervals
    """
    # Step 1: Sort the intervals based on the starting point
    intervals.sort(key=lambda x: x[0])
    
    merged = []
    for interval in intervals:
        # Step 2 & 3: If merged list is empty or current interval does not overlap
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # There is an overlap, merge the current interval with the last interval in merged
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
    
    return merged

def display_audio(audio, sr):
    ipd.display(ipd.Audio(audio, rate=sr))

def identify_tonic(audio, sr, **kwargs):
    """
    Function to identify the tonic frequency of the audio signal, using TonicIndianArtMusic from Essentia

    Parameters:
    audio : np.ndarray
        Numpy array containing the audio signal
    sr : int
        Sampling rate of the audio signal
    **kwargs : dict
        Additional keyword arguments for TonicIndianArtMusic

    Returns:
    tonic : float
        Tonic frequency in Hz
    """
    # Extract the tonic frequency
    tonic_extractor = TonicIndianArtMusic(sampleRate = sr, **kwargs)
    tonic = tonic_extractor(audio)

    return tonic

def melodia(audio, sr, **kwargs):
    """
    Function to extract pitch values using PredominantPitchMelodia

    Parameters:
    audio : np.ndarray
        Numpy array containing the audio signal
    sr : int
        Sampling rate of the audio signal
    **kwargs : dict
        Additional keyword arguments for PredominantPitchMelodia

    Returns:
    pitch_values : np.ndarray
        Numpy array containing the pitch values
    pitch_confidence : np.ndarray
        Numpy array containing the pitch confidence values
    """
    # It is recommended to apply equal-loudness filter for PredominantPitchMelodia.
    audio = EqualLoudness(sampleRate = sr)(audio)

    # Extract the pitch curve
    # PitchMelodia takes the entire audio signal as input (no frame-wise processing is required).
    pitch_extractor = PredominantPitchMelodia(**kwargs, sampleRate=sr)
    pitch_values, pitch_confidence = pitch_extractor(audio)

    return pitch_values, pitch_confidence

def sonify_pitch_contour(time, pitch, sr=16000, display = False):
    """
    Sonify the pitch contour using a simple sine wave generator

    Parameters:
    time : np.ndarray
        Numpy array containing the time values
    pitch : np.ndarray
        Numpy array containing the pitch values
    sr : int
        Sampling rate, default is 16000
    display : bool
        Display the audio signal using IPython.display.Audio, default is False

    Returns:    
    y : np.ndarray
        Numpy array containing the audio signal
    """
    # Start time at 0
    time = time - time[0]
    # Audio length
    length_in_samples = int(np.ceil(time[-1]) * sr)

    # Convert pitch contour to audio signal
    y = mir_eval.sonify.pitch_contour(time, pitch, fs=sr, length = length_in_samples)
    
    # Display audio
    if display:
        display_audio(y, sr)

    # Return audio signal
    return y

def save_audio(filepath, y, sr):
    # Save the audio signal using soundfile
    sf.write(filepath, y, sr)

def load_normalize(audiofile, sr = None):
    """
    Function to load and normalize the audio signal

    Parameters:
    audiofile : str
        Path to the audio file
    sr : int
        Sampling rate, default is None
    
    Returns:
    y : np.ndarray
        Numpy array containing the audio signal
    sr : int
        Sampling rate of the audio signal
    """
    # Open audio with librosa and normalize
    y, sr = librosa.load(audiofile, sr = sr)
    y = y / np.max(np.abs(y))

    return y, sr

def octave_jumps(f0, voiced_segments, window_size = 20, jump_threshold = 700):
    """
    Function to detect octave jumps in the pitch contour
    For a given window size, if the maximum pitch value changes more than the threshold cents in the window, it is considered an octave jump

    Parameters:
    f0 : np.ndarray
        Numpy array containing the pitch values (in cents)
    voiced_segments : list
        List of tuples containing the start and end indices of the voiced segments
    window_size : int
        Window size for detecting octave jumps, default is 20 samples (125 samples per second)
    jump_threshold : int
        Threshold for detecting octave jumps in cents, default is 700 cents

    Returns:
    octave_jumps : list
        List of tuples containing the start and end indices of the octave jumps
    """

    # Initialize the list of octave jumps
    octave_jumps = []

    # Detect octave jumps in the pitch contour
    for seg in voiced_segments:
        # Extract the pitch values for the current segment
        segment = f0[seg[0]:seg[1]+1]

        if len(segment) >  window_size:
            for i in range(0, len(segment) - window_size + 1):
                window = segment[i:i+window_size]
                min_pitch = np.min(window)
                max_pitch = np.max(window) + 1e-6
                if max_pitch - min_pitch > jump_threshold:
                    octave_jumps.append((seg[0] + i, seg[0] + i + window_size - 1))

    return octave_jumps

def correct_octave_jump(f0):
    """
    Corrects octave errors in the pitch contour within specified intervals.

    Parameters:
    f0 (np.ndarray): Array of pitch values in cents.

    Returns:
    np.ndarray: Corrected pitch contour in cents
    """
    # Create a copy of the pitch contour to avoid modifying the original array
    corrected_f0 = f0.copy()
    voiced = voiced_segments(corrected_f0)

    # Calculate the number of octave jumps
    jumps = octave_jumps(corrected_f0, voiced, jump_threshold = 1000)
    merged_jumps = merge_intervals(jumps)
    n_jumps = len(merged_jumps)
    print("Number of octave jumps: ", n_jumps)

    # Initialize the counter for constant jumps
    constant_jump_count = 0

    # Iterate until the number of jumps does not reduce
    while True:
        previous_n_jumps = n_jumps
        
        for (start_idx, end_idx) in merged_jumps:
            f0_median = np.median(corrected_f0[start_idx:end_idx+1])
            for i in range(start_idx, end_idx + 1):
                if i == start_idx:
                    # Check for octave jump at the start of the segment
                    if np.abs(corrected_f0[i] - f0_median) > 700:
                        corrected_f0[i] -= 1200 * np.sign(corrected_f0[i] - f0_median)
                else:
                    # Check for octave jump within the segment
                    if np.abs(corrected_f0[i] - corrected_f0[i-1]) > 700:
                        corrected_f0[i] -= 1200 * np.sign(corrected_f0[i] - corrected_f0[i-1])
        
        jumps = octave_jumps(corrected_f0, voiced, jump_threshold=1000)
        merged_jumps = merge_intervals(jumps)
        n_jumps = len(merged_jumps)
        print(f"Number of octave jumps: {n_jumps}")
        
        # Check if the number of jumps has stayed constant
        if n_jumps == previous_n_jumps:
            constant_jump_count += 1
        else:
            constant_jump_count = 0
        
        # Break the loop if the number of jumps stays constant for 5 iterations
        if constant_jump_count >= 5:
            break

    return corrected_f0

def high_pass(audio, sr, cutoff, order = 4):
    """
    Function to apply high-pass filter to the audio signal, Chebyshev Type II filter

    Parameters:
    audio : np.ndarray
        Numpy array containing the audio signal
    sr : int
        Sampling rate of the audio signal
    cutoff : float
        Cutoff frequency for the high-pass filter in Hz
    order : int
        Order of the Butterworth filter, default is 4
    
    Returns:
    filtered_audio : np.ndarray
        Numpy array containing the high-pass filtered audio signal
    """
    # Apply high-pass filter to the audio signal
    nyquist = 0.5 * sr
    normalize_cutoff = cutoff / nyquist
    b, a = cheby2(order, 40, normalize_cutoff, 'high', analog=True)
    filtered_audio = filtfilt(b, a, audio)

    return filtered_audio

def gaussian_filter(pitch_values, window_size, sigma):
    """
    Function to apply Gaussian filter to the pitch contour using scipy.ndimage.gaussian_filter1d

    Parameters:
    pitch_values : np.ndarray
        Numpy array containing the pitch values
    window_size : int
        Size of the window for Gaussian filter in samples
    sigma : float
        Standard deviation of the Gaussian filter in samples
    """
    # Create the Gaussian kernel
    x = np.linspace(-int(window_size // 2), int(window_size // 2), window_size)
    gaussian_kernel = np.exp(-0.5 * (x / sigma) ** 2)
    # Normalize the kernel
    gaussian_kernel /= gaussian_kernel.sum()  

    # Apply the Gaussian filter using convolution
    filtered_pitch = convolve1d(pitch_values, gaussian_kernel, mode='reflect')

    return filtered_pitch

def savitsky_golay_filter(pitch_values, window_size, order, peak = False):
    """
    Function to apply Savitsky-Golay filter to the pitch contour using scipy.signal.savgol_filter
    This algorithm considers voiced segments separately and applies the filter to each segment.

    Parameters:
    pitch_values : np.ndarray
        Numpy array containing the pitch values
    window_size : int
        Size of the window for Savitsky-Golay filter in samples
    order : int
        Order of the polynomial for Savitsky-Golay filter
    peak : bool
        Flag to assign peaks to the filtered pitch contour, default is False

    Returns:
    filtered_pitch : np.ndarray
        Numpy array containing the filtered pitch values
    """
    # Ensure window_size is odd and > 1
    if window_size % 2 == 0 or window_size <= 1:
        raise ValueError("window_size must be odd and greater than 1")

    # Identify voiced segments
    voiced_seg = voiced_segments(pitch_values)

    # Initialize the filtered pitch contour
    filtered_pitch = np.zeros_like(pitch_values)

    # Apply Savitsky-Golay filter to each voiced segment
    for seg in voiced_seg:
        # Extract the pitch values for the current segment
        segment = pitch_values[seg[0]:seg[1]+1]
        if len(segment) > window_size:
            # Apply Savitsky-Golay filter to the segment
            filtered_segment = savgol_filter(segment, window_size, order)
            if peak:
                # Detect peaks in the original segment
                peaks, _ = find_peaks(segment)
                minimas = find_minimas(segment, 1200)
                # Assign the peaks to the filtered segment
                filtered_segment[peaks] = segment[peaks]
                filtered_segment[minimas] = segment[minimas]
            # Assign the filtered segment to the filtered pitch contour
            filtered_pitch[seg[0]:seg[1]+1] = filtered_segment
        else:
            filtered_pitch[seg[0]:seg[1]+1] = segment

    return filtered_pitch

def voiced_segments(pitch_values):
    """
    Function to identify voiced segments in the pitch contour

    Parameters:
    pitch_values : np.ndarray
        Numpy array containing the pitch values

    Returns:
    segments : list
        List of tuples containing the start and end indices of the voiced segments
    """
    # Initialize the list of voiced segments
    segments = []

    # Find the indices of voiced segments
    voiced_indices = np.where(pitch_values > 0)[0]

    # Find gaps to identify separate voiced segments
    gaps = np.diff(voiced_indices) > 1
    segment_starts = np.insert(voiced_indices[np.where(gaps)[0] + 1], 0, voiced_indices[0])
    segment_ends = np.append(voiced_indices[np.where(gaps)[0]], voiced_indices[-1])

    # Create a list of tuples for the start and end indices of the voiced segments
    segments = list(zip(segment_starts, segment_ends))

    return segments

def hz_to_cents(frequencies, tonic):
    """
    Function to convert frequency values to cent intervals with respect to a tonic frequency

    Parameters:
    f0 : np.ndarray
        Numpy array containing the frequency values
    tonic : float
        Tonic frequency in Hz

    Returns:
    cents : np.ndarray
        Numpy array containing the cent values    
    """
    # Replace non-positive frequencies with NaN to avoid log2 issues
    frequencies = np.where(frequencies > 0, frequencies, np.nan)

    # Convert frequency values to cent intervals
    cents = 1200 * np.log2(frequencies / tonic)

    return cents

def cents_to_hz(cents, tonic):
    """
    Function to convert cent intervals to frequency values with respect to a tonic frequency

    Parameters:
    cents : np.ndarray
        Numpy array containing the cent values
    tonic : float
        Tonic frequency in Hz

    Returns:
    frequencies : np.ndarray
        Numpy array containing the frequency values
    """
    # Convert cent intervals to frequency values
    frequencies = tonic * 2 ** (cents / 1200)

    return frequencies

def wrap_to_octave(cents):
    """
    Function to wrap cent values to the octave (0 to 1200 cents)

    Parameters:
    cents : np.ndarray
        Numpy array containing the cent values

    Returns:
    wrapped_cents : np.ndarray
        Numpy array containing the wrapped cent values
    """
    # Wrap cent values to the octave
    wrapped_cents = np.mod(cents, 1200)

    return wrapped_cents
    
def gaussian(x, a, b, c, d, alpha):
    """
    Gaussian function for curve fitting

    Parameters:
    x : np.ndarray
        Numpy array containing the input values
    a : float
        Amplitude of the Gaussian
    b : float
        Mean of the Gaussian
    c : float
        Standard deviation of the Gaussian
    d : float
        Offset parameter
    alpha : float
        Skewness parameter, default is None

    Returns:
    y : np.ndarray
        Numpy array containing the output values
    """
    # Normal distribution
    norm_dist = a * np.exp(-0.5 * ((x - b) / c) ** 2)
    
    # Skewness adjustment
    if alpha is None:
        skew_adjustment = 1
    else:
        skew_adjustment = skew_adjustment = (1 + erf(alpha * (x - b) / (c * np.sqrt(2))))
    
    # Combine both parts and add offset
    y = norm_dist * skew_adjustment + d

    return y

def fit_histogram(peaks, kde_vals, bin_centers, extent = 50):
    """
    Function to fit a gaussian to the histogram curve around each peak

    Parameters:
    peaks : list
        Indices of the peaks in the histogram curve
    kde_vals : np.ndarray
        Numpy array containing the kernel density estimate values
    bin_centers : np.ndarray
        Numpy array containing the bin centers
    extent: 

    Returns:
    peak_dict : dict
        Dictionary containing the peak positions as keys and the gaussian parameters
    """


    # Now that peak positions are known, take the histogram curve -50 and +50 around each peak and fit a gaussian
    peak_dict = {}

    for peak in peaks:
        # Calculate start and end indices, considering wrap-around
        start = peak - extent
        end = peak + extent
        
        # Handle wrap-around for start index
        if start < 0:
            x_left = bin_centers[start:]  # From start index to the end
            x_right = bin_centers[:end]  # Starting from the beginning
            x = np.concatenate((x_left, x_right))  # Combine both parts
            y_left = kde_vals[start:]
            y_right = kde_vals[:end]
            y = np.concatenate((y_left, y_right))
        elif end > len(bin_centers):
            # Handle wrap-around for end index
            overflow = end - len(bin_centers)
            x_left = bin_centers[start:]  # From start index to the end
            x_right = bin_centers[:overflow]  # Starting from the beginning
            x = np.concatenate((x_left, x_right))  # Combine both parts
            y_left = kde_vals[start:]
            y_right = kde_vals[:overflow]
            y = np.concatenate((y_left, y_right))
        else:
            # No wrap-around needed
            x = bin_centers[start:end]
            y = kde_vals[start:end]
        
        try:
            # Fit a gaussian to the curve
            initial_guess = [np.max(y), bin_centers[peak], 1, 1, 1]
            popt, _ = curve_fit(gaussian, x, y, p0=initial_guess)
            peak_dict[bin_centers[peak]] = popt
        except Exception as e:
            print(f"Error fitting peak at index {peak}: {e}")
            continue

    return peak_dict

def median_confidences(conf, voiced_segments):
    """
    Function to calculate the median confidence values for each voiced segment

    Parameters:
    conf : np.ndarray
        Numpy array containing the confidence values
    voiced_segments : list
        List of tuples containing the start and end indices of the voiced segments
    
    Returns:
    median_conf : list
        List containing the median confidence values for each voiced segment
    """
    median_conf = []
    for segment in voiced_segments:
        median_conf.append(np.median(conf[segment[0]:segment[1]]))

    return median_conf  

def find_minimas(array, height):
    """
    Function to find the local minimas in an array

    Parameters:
    array : np.ndarray
        Numpy array containing the input values
    height : float
        Minimum height of the local minimas

    Returns:
    minimas : np.ndarray
        Numpy array containing the indices of the local minimas
    """
    # Find the local minimas in the array
    minimas = find_peaks(-array, height = -height)[0]

    return minimas

####################### Statistical Analysis #######################

def t_test(kde_1, kde_2, **kwargs):
    """
    Function to perform a t-test between two kernel density estimates

    Parameters:
    kde_1 : np.ndarray
        Numpy array containing the kernel density estimate values for the first distribution
    kde_2 : np.ndarray
        Numpy array containing the kernel density estimate values for the second distribution
    **kwargs : dict
        Additional keyword arguments for scipy.stats.ttest_ind

    Returns:
    t_stat : float
        T-statistic value
    p_value : float
        P-value
    """
    # Check if the variance is equal, else set equal_var = False
    if not stats.levene(kde_1, kde_2)[1] < 0.05:
        kwargs['equal_var'] = True
        print("Variance is equal, using equal variance t-test")
    else:
        print("Variance is not equal, using Welch's t-test")
        kwargs['equal_var'] = False

    # Perform a t-test between the two distributions
    t_stat, p_value = stats.ttest_ind(kde_1, kde_2, **kwargs)

    return t_stat, p_value

def mann_whitney_u(kde_1, kde_2, **kwargs):
    """
    Function to perform a Mann-Whitney U test between two kernel density estimates

    Parameters:
    kde_1 : np.ndarray
        Numpy array containing the kernel density estimate values for the first distribution
    kde_2 : np.ndarray
        Numpy array containing the kernel density estimate values for the second distribution
    **kwargs : dict
        Additional keyword arguments for scipy.stats.mannwhitneyu

    Returns:
    u_stat : float
        U-statistic value
    p_value : float
        P-value
    """
    # Perform a Mann-Whitney U test between the two distributions
    u_stat, p_value = stats.mannwhitneyu(kde_1, kde_2, **kwargs)

    return u_stat, p_value

def cross_entropy(p, q):
    """
    Function to calculate the cross-entropy between two probability distributions

    Parameters:
    p : np.ndarray
        Numpy array containing the first probability distribution
    q : np.ndarray
        Numpy array containing the second probability distribution

    Returns:
    cross_entropy : float
        Cross-entropy value
    """
    # Normalize the distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Calculate the cross-entropy between the two distributions
    cross_entropy = -np.sum(p * np.log(q + 1e-6))

    return cross_entropy