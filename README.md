# Adaptive Matrix Profile Indexing for Irregular Multivariate Time Series


## Summary

- [Description](#description)
    - [Goal](#goal) — Adaptive indexing using Matrix Profile on irregular, multivariate time series.
        - [Data Preprocessing](#1-data-preprocessing) — Normalize signals and handle uneven sampling with interpolation or time-aware distances (Soft-DTW, TWED).
        - [Adaptive Window Strategy](#2-adaptive-window-strategy) — Variable-length windowing based on signal or time-based heuristics.
        - [Matrix Profile Integration](#3-matrix-profile-integration) — Use STUMPY or MatrixProfile to compute motifs and profiles in each adaptive window.
        - [Indexing and Retrieval](#4-indexing-and-retrieval) — Store subsequence embeddings using FAISS for fast similarity and anomaly queries.
        - [Anomaly Detection](#5-anomaly-detection) — Detect and rank anomalies using matrix profile values.
    - [Evaluation](#evaluation) — Benchmarked on NAB and lab datasets with metrics like F1, AUC, latency, and robustness.
    - [Deliverables](#deliverables) — Python package, notebooks, benchmarks, and a technical report.
    - [References](#references) — Key literature on Matrix Profiles, Soft-DTW, and adaptive anomaly detection.
- [Installation](#installation) — Setup instructions for using the package.

- [Technologies](#Technologies) - Understand What dependances are used in this package  
    - [Matrix Profile](#matrix-profile-motif--anomaly-discovery)
    - [Similarity Measures & Loss Function](#similarity-measures--loss-functions)
    - [Indexing, Embedding & Retrieval](#indexing-embedding--retrieval)
    - [Clustering & Anomaly Segmentation (Optional)](#clustering--anomaly-segmentation-optional)
    - [Pipeline Integration](#pipeline-integration)

- [Identification Process — Without Reference Motifs (Unsupervised with Matrix Profile)](#identification-process--without-reference-motifs-unsupervised-with-matrix-profile)

- [Identification Process — With Reference Motifs (FAISS + CID-DTW)](#identification-process--with-reference-motifs-faiss--cid-dtw)
## Description 
### Goal 

Develop an adaptive indexing approach that leverages the matrix profile on unevenly sampled, multivariate time series, using variable window lengths tailored to local sampling density and signal behavior.

#### 1. Data Preprocessing

- Handle irregular sampling via:
    - Interpolation (linear, spline, etc.).
    - Time-aware distance functions (Time Warp Edit Distance (TWED) (https://github.com/pfmarteau/TWED) or Soft-DTW (https://github.com/mblondel/soft-dtw)).
- Normalize per channel (z-score or min-max).

#### 2. Adaptive Window Strategy

- Investigate window sizing policies:

    - Signal-based: entropy, variance, or gradient magnitude.
    - Time-based: inter-sample time gap threshold.
- Implement a method to dynamically define windows over each segment of data per channel or jointly.

#### 3. Matrix Profile Integration

- Use stumpy or matrixprofile library to compute:

    - Matrix profiles within each window.
    - Multivariate extensions like mSTAMP or mSTOMP for multidimensional series.
- Adapt the algorithm to use non-uniform windows (a key novelty here).

#### 4. Indexing and Retrieval
- Store subsequence embeddings (from matrix profile distances or motifs) in a fast index (e.g. FAISS [https://github.com/facebookresearch/faiss]).
- Allow fast retrieval of nearest neighbors or anomalies based on profile values.
#### 5. Anomaly Detection
- Anomalies = subsequences with high profile values (i.e., no close neighbor).
- Score and rank anomalies across windows and channels.
 - (Optional) Fuse anomalies across variables using joint profile metrics.

### Evaluation

- Datasets:
    - Numenta NAB [irregular IoT-style data] (https://github.com/numenta/NAB)
    - Custom data of the Signal Processing Lab
- Metrics:
    - Anomaly detection (precision, recall, F1, AUC)
    - Index/query latency
    - Profile computation time vs. standard mSTAMP
    - Robustness to sampling irregularity and noise
### Deliverables

- A modular Python package with:
    - Preprocessing pipeline
    - Adaptive windowing
    - Matrix profile computation (single and multivariate)
    - Index-based anomaly scoring
- Jupyter notebooks with reproducible experiments
- Benchmark results and short technical report or blog
### References

- [1] Time Warp Edit Distance (https://arxiv.org/abs/0802.3522)
- [2] Soft-DTW: a Differentiable Loss Function for Time-Series (https://arxiv.org/abs/1703.01541)
- [3] A Complexity-Invariant Distance Measure for Time Series (https://www.cs.ucr.edu/%7Eeamonn/Complexity-Invariant%20Distance%20Measure.pdf)
- [4] Utilizing an adaptive window rolling median methodology for time series anomaly detection (https://www.sciencedirect.com/science/article/pii/S1877050922023328?ref=cra_js_challenge&fr=RR-1)
- [5] An adaptive sliding window for anomaly detection of time series in wireless sensor networks (https://link.springer.com/article/10.1007/s11276-021-02852-3)
- [6] Anomaly detection in time series: a comprehensive evaluation (https://dl.acm.org/doi/abs/10.14778/3538598.3538602)
- [7] Anomaly Detection for IoT Time-Series Data: A Survey (https://ieeexplore.ieee.org/abstract/document/8926446)
- [8] Adaptive sliding window normalization (https://www.sciencedirect.com/science/article/pii/S030643792400173X)
- [9] Anomaly Detection Using Causal Sliding Windows (https://ieeexplore.ieee.org/abstract/document/7109108)
- [10] Comparing Threshold Selection Methods for Network Anomaly Detection (https://ieeexplore.ieee.org/abstract/document/10659855)
- [11] DAMP: accurate time series anomaly detection on trillions of datapoints and ultra-fast arriving data streams (https://link.springer.com/article/10.1007/s10618-022-00911-7)
- [12] Discovering Multi-Dimensional Time Series Anomalies with K of N Anomaly Detection (https://epubs.siam.org/doi/epdf/10.1137/1.9781611977653.ch77)
- [13] Introduction to Matrix Profiles (https://medium.com/data-science/introduction-to-matrix-profiles-5568f3375d90)
- [14] Matrix Profile Tutorial (https://www.cs.ucr.edu/~eamonn/Matrix_Profile_Tutorial_Part2.pdf)
- [15] C22MP: Fusing catch22 and the Matrix Profile to Produce an Efficient and Interpretable Anomaly Detector (https://www.dropbox.com/scl/fi/3vs0zsh4tw63qrn46uyf9/C22MP_ICDM.pdf?rlkey=dyux24kqpagh3i38iw6obiomq&e=1&dl=0)
- [16] Matrix Profile for Anomaly Detection on Multidimensional Time Series (https://arxiv.org/abs/2409.09298)
 
## Installation

### 1. From PyPI (recommended once the package is published)

```bash
pip install mon_package
```
### 2. From GitHub (for the latest version or pre publication

```bash
pip install git+https://github.com/your-username/ampiimts.git
```

### 3. Local installation (for development or local modifications)

```bash
git clone https://github.com/your-username/ampiimts.git

cd ampiimts

#Standart local installation
pip install .

#Development installation (editable version with test for contribution)
pip install -e .[test]
```

## 🛠️ Technologies

### Matrix Profile (Motif & Anomaly Discovery)

- **STUMPY**: Core library for efficient matrix profile computation on univariate and multivariate time series (`aamp`, `mammp`).
- **matrixprofile (UCR)**: Alternative implementation from academia.
- Used to detect:
  - **Motifs** (frequent repeated patterns)
  - **Discords** (outliers or rare patterns)
  - **Segment boundaries**
- Supports adaptive and irregular windowing strategies via custom wrapping.

### Similarity Measures & Loss Functions

#### DTW-Based Methods

- **DTW (Dynamic Time Warping)**: Classic alignment method to measure similarity between time series with temporal misalignments.
- **Soft-DTW**: Differentiable DTW variant usable as a loss function in learning frameworks.

#### Complexity Correction

- **CID (Complexity-Invariant Distance)**: Normalization term applied to any distance (DTW, Euclidean, etc.) that penalizes shape complexity differences.
- Enhances robustness when comparing complex vs. smooth patterns.

### Indexing, Embedding & Retrieval

#### FAISS (Facebook AI Similarity Search)

- High-performance library for **fast nearest-neighbor vector search**.
- Used in this project to:
  - **Index and compress vector representations** of discovered or reference-based motifs.
  - **Query similarity or anomaly** without needing raw signal or reference CSVs.
  - Enable **real-time recognition and anomaly detection** based on learned memory.

### Clustering & Anomaly Segmentation (Optional)

- Embeddings can be **clustered** (e.g., KMeans, HDBSCAN) to group similar patterns or discover new ones.
- FAISS query results with **high distances** or **out-of-cluster** embeddings are treated as anomalies.
- Adaptive thresholds can be tuned per motif group or learned dynamically.

### Pipeline Integration

- Supports both **reference-free discovery** (pure Matrix Profile) and **reference-driven recognition** (FAISS + DTW/CID).
- Includes:
  - Preprocessing of irregular multivariate data.
  - Adaptive window sizing.
  - Matrix profile generation.
  - Embedding and FAISS indexing.
  - Real-time motif classification and anomaly detection.

#### Identification Process — Without Reference Motifs (Unsupervised with Matrix Profile)

- Input: New time series (CSV or streaming or others)
  - [1] Preprocessing
    - Handle missing values / interpolate
    - Normalization per channel
  - [2] Motif & Anomaly Discovery (STUMPY)
    - Compute Matrix Profile
    - Extract:
      - Motifs (low profile values)
      - Discords (high profile values)
  - [3] Vectorization 
    - Use CID-DTW or MPdist vs other internal subsequences
    - Create internal representation of patterns
  - [4] Clustering 
    - Group similar motifs using clustering (e.g., HDBSCAN)
    - Identify rare/isolated subsequences as anomalies
  - [5] Internal Indexing 
    - Index self-learned motif embeddings in FAISS
    - Enable re-query or future identification without manual reference

 #### Identification Process — With Reference Motifs (FAISS + CID-DTW)

- Input: New time series (CSV or streaming) or others)
  - [1] Preprocessing
    - Handle missing values / interpolate
    - Normalize (z-score, min-max)
  - [2] Subsequence Detection
    - Sliding window or STUMPY (Matrix Profile)
    - Extract candidate subsequences
  - [3] Vectorization (CID-DTW or CID-Soft_DTW)
    - Compare each subsequence to reference motifs
    - Generate fixed-length vectors
  - [4] FAISS Similarity Search
    - Query vector against FAISS index
    - Retrieve top-k closest known motifs
  - [5] Matching / Anomaly Decision
    - If distance ≤ threshold → Recognized Motif 
    - If distance > threshold → Anomaly
