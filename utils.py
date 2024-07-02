import librosa
import librosa.display
import soundfile as sf
from scipy.signal.windows import get_window
from scipy.signal import medfilt, savgol_filter, cheby2, filtfilt, find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
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

def parse_pitch_file(pitch_file, confidence = False):
    """
    Function to parse the pitch file

    Parameters:
    pitch_file : str
        Path to the pitch file
    confidence : bool
        Flag to read pitch confidence values, default is False

    Returns:
    pitch_times: np.ndarray
        Numpy array containing the time values
    pitch_values: np.ndarray
        Numpy array containing the pitch values
    pitch_confidence: np.ndarray
        Numpy array containing the pitch confidence values
    """
    # Read the pitch file
    pitch_data = np.loadtxt(pitch_file)

    # Extract pitch times, values and confidence
    pitch_times = pitch_data[:, 0]
    pitch_values = pitch_data[:, 1]
    if confidence:
        pitch_confidence = pitch_data[:, 2]
    else:
        pitch_confidence = None

    return pitch_times, pitch_values, pitch_confidence
    """
    Function to plot the spetrogram of the audio signal and pitch contour
    Plot spectrogram with librosa
    """
    times = librosa.times_like(f0, sr=sr, hop_length=128)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    fig, ax = plt.subplots(figsize=(20,10))
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax, sr = sr)
    ax.set(title='Fundamental frequency estimation')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')

def plot_pitch(pitch_values, pitch_confidence, audio, sr, threshold = 0, t1 = None, t2 = None):
    """
    Function to plot pitch values and confidence

    Parameters:
    pitch_values : np.ndarray
        Numpy array containing the pitch values
    pitch_confidence : np.ndarray
        Numpy array containing the pitch confidence values
    audio : np.ndarray
        Numpy array containing the audio signal
    sr : int
        Sampling rate of the audio signal
    threshold : float
        Threshold for pitch confidence values
    t1 : float
        Start time for plotting (in seconds)
    t2 : float
        End time for plotting (in seconds)

    Returns:
    None
    """
    pitch_times = np.linspace(0.0,len(audio)/sr,len(pitch_values))

    # Clean up pitch values and confidences
    pitch_values[pitch_confidence < threshold] = 0
    pitch_values = np.where(pitch_values==0, np.nan, pitch_values)
    pitch_confidence = np.where(pitch_confidence==0, np.nan, pitch_confidence)

    # Plot pitch values and confidence in subplots
    plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.plot(pitch_times, pitch_values)
    plt.title('Estimated Pitch')
    plt.ylabel('Frequency (Hz)')
    if t1 is not None and t2 is not None:
        plt.xlim(t1, t2)
    plt.subplot(2,1,2)
    plt.scatter(pitch_times, pitch_confidence, s = 1, color='red')
    plt.title('Pitch Confidence')
    plt.ylabel('Confidence')
    plt.xlabel('Time (s)')
    if t1 is not None and t2 is not None:
        plt.xlim(t1, t2)
    plt.tight_layout()
    plt.show()

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

def pyin(audio, sr, fmin = 180, fmax = 2000):
    f0, _, voiced_probs = librosa.pyin(audio, fmin=fmin, fmax=fmax, sr=sr)
    
    return f0, voiced_probs

def pitch_histogram(pitch_values, bins = 100, title = None):
    
    # Convert 0 values to NaN
    pitch_values = np.where(pitch_values==0, np.nan, pitch_values)

    # Count how many values are not NaN
    count = np.count_nonzero(~np.isnan(pitch_values))
    
    # Plot the histogram of pitch values
    plt.figure(figsize=(20,10))
    plt.hist(pitch_values, bins = bins, color='cyan')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()

    return count

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
    # Open audio with librosa and normalize
    y, sr = librosa.load(audiofile, sr = sr)
    y = y / np.max(np.abs(y))

    return y, sr

def octave_jumps(f0, voiced_segments, window_size = 20):
    """
    Function to detect octave jumps in the pitch contour

    Parameters:
    f0 : np.ndarray
        Numpy array containing the pitch values
    voiced_segments : list
        List of tuples containing the start and end indices of the voiced segments
    window_size : int
        Window size for detecting octave jumps, default is 20 samples (125 samples per second)

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
                if max_pitch /min_pitch  >= 2:
                    octave_jumps.append((seg[0] + i, seg[0] + i + window_size - 1))


    return octave_jumps

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
    b, a = cheby2(order, 40, normalize_cutoff, 'high', analog=False)
    filtered_audio = filtfilt(b, a, audio)

    return filtered_audio

def median_filter(pitch_values, window_size):
    """
    Function to apply median filter to the pitch contour using scipy.signal.medfilt

    Parameters:
    pitch_values : np.ndarray
        Numpy array containing the pitch values
    window_size : int
        Size of the window for median filter in samples
    """
    # Apply median filter to the pitch contour
    filtered_pitch = medfilt(pitch_values, window_size)

    return filtered_pitch

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

def savitsky_golay_filter(pitch_values, window_size, order):
    """
    Function to apply Savitsky-Golay filter to the pitch contour using scipy.signal.savgol_filter
    This algorithm considers voiced segments separately and applies the filter to each segment.
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
            # Detect peaks in the original segment
            peaks, _ = find_peaks(segment)
            # Assign the peaks to the filtered segment
            filtered_segment[peaks] = segment[peaks]
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
    # Ignore zero values, as they correspond to unvoiced segments
    frequencies = frequencies[frequencies != 0]
    # Convert frequency values to cent intervals
    frequencies = np.where(frequencies == 0, np.nan, frequencies)
    cents = 1200 * np.log2(frequencies / tonic)

    return cents

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

def interval_histogram(cents, bins = 100, title = None, tuning = 'EQ'):
    """
    Function to plot the histogram of cent values

    Parameters:
    cents : np.ndarray
        Numpy array containing the cent values
    bins : int
        Number of bins for the histogram
    title : str
        Title of the histogram plot
    tuning : str
        Tuning system, 'EQ' for equal temperament, 'JI' for just intonation, default is 'EQ'
    """
    # Array containing semitone cent locations
    EQ = np.arange(0, 1300, 100)
    JI = np.array([0, 111, 203, 315, 386, 498, 582, 702, 814, 884, 996, 1088, 1200])
    # Plot the histogram of cent values
    plt.figure(figsize=(20,10))
    plt.hist(cents, bins = bins, color='cyan')
    plt.xlabel('Cents')
    plt.ylabel('Count')
    plt.title(title)

    # Add vertical lines for each semitone location
    if tuning == 'EQ':
        for semitone in EQ:
            plt.axvline(semitone, color='red', linestyle='--', alpha = 0.3)
    elif tuning == 'JI':
        for semitone in JI:
            plt.axvline(semitone, color='red', linestyle='--', alpha = 0.3)
    plt.show()

def gaussian(x, a, b, c):
    """
    Gaussian function for curve fitting

    Parameters:
    x : np.ndarray
        Numpy array containing the x values
    a : float
        Peak height
    b : float
        Peak position
    c : float
        Peak width
    
    Returns:
    y : np.ndarray
        Numpy array containing the Gaussian function values
    """
    y = a * np.exp(-0.5 * ((x - b) / c) ** 2)

    return y

def fit_histogram(cents, bins = 1200, exp_positions = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]):
    """
    Function to fit gaussian peaks on a histogram at given expected positions

    Parameters:
    cents : np.ndarray
        Numpy array containing the cent values (Must be wrapped to the octave 0 to 1200)
    bins : int
        Number of bins for the histogram, default is 1200
    exp_positions : list
        List of expected peak positions in cents, default is [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]

    Returns:
    peak_positions : np.ndarray
        Numpy array containing the peak positions
    peak_heights : np.ndarray
        Numpy array containing the peak heights
    peak_widths : np.ndarray
        Numpy array containing the peak widths
    """
    # Create histogram
    hist, bin_edges = np.histogram(cents, bins = bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    peaks, _ = find_peaks(hist)

    # Fit Gaussian peaks
    peak_positions = []
    peak_heights = []
    peak_widths = []
    for pos in exp_positions:
        # Fit Gaussian peaks
        p0 = [np.max(hist), pos, 50]
        coeff, _ = curve_fit(gaussian, bin_centers, hist, p0 = p0)
        peak_positions.append(coeff[1])
        peak_heights.append(coeff[0])
        peak_widths.append(coeff[2])

    return np.array(peak_positions), np.array(peak_heights), np.array(peak_widths)