from core import to_viterbi_cents, to_local_average_cents, to_weird_viterbi_cents, output_path
import numpy as np
import os


## CREPE functions ##

def get_activation_matrix(activation_file):
    """
    Get the activation matrix from a file containing time, pitch and confidence values

    Parameters:
    activation_file : str
        The file path to the input activation file. Format: .npy file containing the activation matrix
    
    Returns:
    activation : np.ndarray
        Numpy array containing the activation matrix
    """    
    activation = np.load(activation_file)
    return activation

def compute_activation_matrix


def compute_pitch_tracks(activation_file, viterbi=False, step_size=10):
    """
    Compute pitch tracks from the activation matrix, using various viterbi algorithms

    Parameters:
    activation_file : str
        The file path to the input activation file. Format: .npy file containing the activation matrix
    viterbi : 'weird' or bool
        If 'weird', use the 'weird' viterbi algorithm. If True, use the viterbi algorithm. If False, use the local average algorithm.
    step_size : int
        The step size in milliseconds

    Returns:
    time : np.ndarray
        Numpy array containing the time values
    frequency : np.ndarray
        Numpy array containing the frequency values
    confidence : np.ndarray
        Numpy array containing the confidence values
    """
    # Load activation matrix
    activation = get_activation_matrix(activation_file)

    # Compute 
    if viterbi == "weird":
        path, cents = to_weird_viterbi_cents(activation)
        confidence = np.array([activation[i, path[i]] for i in range(len(activation))])
    elif viterbi:
        # NEW!! CONFIDENCE IS NO MORE THE MAX ACTIVATION! CORRECTED TO BE CALCULATED ALONG THE PATH!
        path, cents = to_viterbi_cents(activation)
        confidence = np.array([activation[i, path[i]] for i in range(len(activation))])
    else:
        cents = to_local_average_cents(activation)
        confidence = activation.max(axis=1)

    frequency = 10 * 2 ** (cents / 1200)
    frequency[np.isnan(frequency)] = 0

    time = np.arange(confidence.shape[0]) * step_size / 1000.0

    return time, frequency, confidence

def save_pitch_tracks_from_activation(activation_file, output_path, viterbi=False, step_size=10, verbose=True):
    """
    Compute pitch tracks from the activation matrix, using various viterbi algorithms, and save them

    Parameters:
    activation_file : str
        The file path to the input activation file. Format: .npy file containing the activation matrix
    output_path : str
        The path to the directory where the pitch tracks will be saved
    viterbi : 'weird' or bool
        If 'weird', use the 'weird' viterbi algorithm. If True, use the viterbi algorithm. If False, use the local average algorithm.
    step_size : int
        The step size in milliseconds
    verbose : bool
        If True, print a message when the pitch tracks are saved

    Returns:
    None

    Note:
    The pitch tracks are saved as a csv file with columns time, frequency and confidence
    """
    time, frequency, confidence = compute_pitch_tracks(activation_file, viterbi, step_size)

    # file name, remove everything after the first .
    f0_file = os.path.basename(activation_file)
    f0_file = os.path.split(f0_file)[1].split('.')[0]
    f0_file = os.path.join(output_path, f0_file + f"_viterbi={viterbi}.csv")

    f0_data = np.vstack([time, frequency, confidence]).transpose()
    np.savetxt(f0_file, f0_data, fmt=['%.3f', '%.3f', '%.6f'], delimiter=',',
               header='time,frequency,confidence', comments='')
    if verbose:
        print("CREPE: Saved the estimated frequencies and confidence values "
              "at {}".format(f0_file))  
