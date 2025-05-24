# Adaptive Matrix Profile Indexing for Irregular Multivariate Time Series


## Summary

- [Installation](#installation) — Setup instructions for using the package.

- [Technologies](#Technologies) - Understand What dependances are used in this package  
    - [Matrix Profile](#matrix-profile-motif--anomaly-discovery)
    - [Similarity Measures & Loss Function](#similarity-measures--loss-functions)
    - [Indexing, Embedding & Retrieval](#indexing-embedding--retrieval)
    - [Clustering & Anomaly Segmentation (Optional)](#clustering--anomaly-segmentation-optional)
    - [Pipeline Integration](#pipeline-integration)

- [Identification Process — Without Reference Motifs (Unsupervised with Matrix Profile)](#identification-process--without-reference-motifs-unsupervised-with-matrix-profile)

- [Identification Process — With Reference Motifs (FAISS + CID-DTW)](#identification-process--with-reference-motifs-faiss--cid-dtw)

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
