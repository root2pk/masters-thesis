# Intonation analysis of Carnatic violin performances in Thodi and Kalyani - Masters Thesis
**Masters Thesis, Sound and Music Computing - UPF Barcelona 2024**

Supplementary code files for the thesis project

## Installation

To install the required dependencies in Python, run:

```sh
pip install -r requirements.txt
```

## File information

1. Pitch extraction and processing are handled in the notebook `extract_process_pitch.ipynb`.
2. he notebook `histograms.ipynb` is used to compute histograms and KDEs (Kernel Density Estimates).
3. Statistical comparison tests on the complete KDEs are conducted using `comparison_metrics.ipynb` and `comparison_graphs.ipynb`.
4. Swara wise metrics are calculated using `swara_metrics.ipynb`.     
5. `utils.py` contains utility functions to accompany the notebooks.
  
## Dataset

This analysis makes use of selected tracks from the [Saraga Carnatic dataset](https://mtg.github.io/saraga/). This dataset comprises 249 recordings, around 53 hours of music, along with multi-tracks, annotations, and metadata.

## Models

### Essentia Melodia model
This model utilizes the Melodia predominant pitch extraction and the `TonicIndianArtMusic` algorithms from Essentia to compute pitch contours and tonic, respectively,

[J. Salamon and E. Gomez, "Melody Extraction From Polyphonic Music Signals Using Pitch Contour Characteristics," in IEEE Transactions on Audio, Speech, and Language Processing, vol. 20, no. 6, pp. 1759-1770, Aug. 2012](https://ieeexplore.ieee.org/document/6155601)
[J. Salamon, S. Gulati, and X. Serra, “A Multipitch Approach to Tonic Identification in Indian Classical Music,” in International Society for Music Information Retrieval Conference (ISMIR’12), 2012](https://repositori.upf.edu/handle/10230/22735)
