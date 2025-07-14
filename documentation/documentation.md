
# AMPIIMTS: Adaptive Matrix Profile Indexing for Irregular Multivariate Time Series


## Introduction and Purpose of the Package


AMPIIMTS (Adaptive Matrix Profile Indexing for Irregular Multivariate Time Series) is a Python toolkit designed to detect recurring patterns (motifs) and anomalies (discords) in time series data, even when the data is irregularly sampled and multivariate. The matrix profile technique underpins this package – a recent data structure (introduced in 2016) that enables fast, exact similarity search in time series. Simply put, a motif is a repeated subsequence pattern in a time series, whereas a discord is an anomalous subsequence that is highly dissimilar from all others. By leveraging matrix profiles, AMPIIMTS can automatically spot such motifs and discords across multiple sensor streams.


The primary goal of AMPIIMTS is to facilitate motif discovery and anomaly detection on complex datasets like environmental sensor readings, where sampling intervals may be uneven and multiple variables interact. It provides a high-level pipeline to ingest raw time-series data (e.g. CSV files with timestamps), preprocess and normalize the data, compute the matrix profile, and return the locations of noteworthy patterns or anomalies. In essence, AMPIIMTS streamlines a sophisticated time-series analysis workflow into a few function calls, making it easier for practitioners to analyze irregular multivariate data. This document gives an overview of how the package works, its internal modules, and how the results (motifs and anomalies) can be interpreted considering real-world events.


## Features and Analysis Workflow


AMPIIMTS Capabilities: The package offers a range of features to handle irregular time series and extract insights:


### Data Preprocessing

It interpolates and aligns irregularly-sampled time series to a common timeline and applies normalization (including ASWN – Adaptive Sliding Window Normalization – with trend removal) to stabilize variance	. This ensures each variable’s baseline and scale are adjusted, making patterns more comparable.


### Automatic Window Selection

AMPIIMTS can automatically determine an appropriate subsequence length (window size) for pattern detection. It does so by clustering the data or using built-in heuristics if no window size is provided by the user. (Internally, a FAISS nearest-neighbor index is used to help choose a window length that captures representative patterns.)

### Clustering of Dimensions

For highly multivariate data (many sensor channels), the package can group related variables via hierarchical correlation clustering. By clustering sensors that behave similarly, the analysis can focus on a smaller number of groups, which improves motif/discord detection in complex systems (or the user can analyze all variables together as one set).


### Matrix Profile Computation

The core analysis uses the matrix profile algorithm (via the STUMPY library) to efficiently find similar subsequences and anomalies. For univariate series, it computes the classic matrix profile (using STUMPY’s stump routine); for multivariate series, it computes a multi-dimensional matrix profile using mstump. This yields a profile of distance values for each time window, where low values indicate a repeating pattern and high values indicate an anomaly.

### Motif and Discord Discovery

Given the matrix profile, AMPIIMTS automatically extracts the top recurring patterns (motifs) and anomalies (discords). Motifs correspond to the smallest distances in the profile (subsequences that closely match each other), and discords correspond to the largest distances (subsequences unlike any others). By default, discords are identified as the top few percent of highest-distance subsequences, while motifs are the frequent patterns with many close neighbors. Users can configure how many motifs to return and the sensitivity for discord detection.

### Iterative Refinement (Smart Interpolation)

A unique feature is “smart” gap-filling: after an initial anomaly detection pass, the package can optionally re-interpolate missing data using information from similar sensors and recompute the matrix profile. This two-pass approach (when enabled) improves analysis by using the discovered patterns to better impute missing values, thus refining the subsequent anomaly detection to focus on true anomalies rather than data gaps.

### Visualization Utilities

AMPIIMTS includes plotting functions to visualize results. It can overlay motifs and discords on the time-series plots and produce heatmaps of the matrix profile. For example, motifs can be highlighted as shaded regions (on the variables where they occur) and discords marked with red vertical line, helping users quickly see where anomalies happened in time and which variables were involved.


## Workflow Overview: The typical usage of AMPIIMTS follows a pipeline of steps from raw data to results:


### Data Ingestion

The user provides either a path to a CSV folder, CSV files or a pandas DataFrame (or a list of DataFrames) containing the time series. Each DataFrame must have a timestamp index or column. The main interface ampiimts() will read all CSV files in a folder if a directory path is given (ignoring files that cannot be parsed) and load each as a DataFrame.


### Preprocessing & Alignment

All input time series are synchronized to a common time grid. Each series is first interpolated individually to fill small gaps (up to a certain limit) and converted to a uniform time frequency	. If multiple DataFrames are provided, the library finds a median sampling interval and aligns all series on that grid, truncating or extending as needed. Series with incompatible frequencies or insufficient overlap may be dropped to ensure a coherent dataset.


### Dimensionality Reduction (optional)

If clustering is enabled (cluster=True), the package performs hierarchical clustering on the variables to group similar ones. This uses the correlation matrix of the time series: variables that are highly correlated are clustered together; under the assumption they capture related phenomena. Only the top k clusters (by average correlation) are retained, and each cluster (up to a set group size) is analyzed separately. This reduces noise and complexity by isolating unrelated sensors. If clustering is off, all variables are treated as one multivariate set.


### Window Size Determination

A crucial parameter is the subsequence length m (window size) for analysis. If the user has not specified a window, AMPIIMTS will choose one automatically. It may do so by trying to identify a characteristic scale in the data – for example, using a clustering of subsequences or a nearest-neighbor search on random segments (leveraging FAISS for efficiency) to find a length that yields low-distance matches. The chosen m is stored in the data attributes for reference. (In time series contexts, a well-chosen window might correspond to a seasonal cycle length or the expected duration of patterns of interest.). The window size used for normalization and matrix profile.


### Normalization

The aligned data is then normalized column-by-column using a sliding window normalization (ASWN) technique. In essence, this removes local mean and scale trends: each subsequence of length m is z-normalized (mean 0, standard deviation 1) unless its variance is extremely low. A small blending factor (α) can be applied to retain part of the longer-term trend if desired. This step ensures that sensors with different units or ranges become comparable and that anomalies are not masked by large-scale trends.


### Matrix Profile Computation

With preprocessed (and possibly clustered) data and a window size m, the next step is to compute the matrix profile. AMPIIMTS calls STUMPY’s algorithms under the hood. For a single DataFrame or a cluster of variables, it uses to compute a multi-dimensional matrix profile across all columns. If a cluster contains only one variable (or if the data is univariate), it uses for a univariate profile. The result is an array of profile values (distances) for each starting timestamp of the window, indicating how similar that subsequence is to its nearest neighbor in the time series. Low values mean that subsequence has a close match (hence part of a motif pair), while high values mean the subsequence is distant from all others, making it an anomaly candidate. The computation is optimized (relying on Numba and parallelism via STUMPY) so even large data can be processed efficiently.


### Pattern & Anomaly Extraction

Once the matrix profile is obtained, the package identifies motifs and discords. The top motifs are typically found by looking for the globally lowest profile values and then finding all occurrences of that pattern within some distance threshold. Discords are identified by taking the highest values in the profile. AMPIIMTS uses a percentage threshold (e.g. top 4% by default) to select discord candidates. It also applies some logic to avoid trivial anomalies – for instance, excluding windows that overlap with missing data or are too close to each other (to ensure distinct anomalies). The output is organized into a result dictionary for each data group, containing the list of discord indices (timestamps) and a summary of motif patterns (including where they occur and which variables are involved).


### Smart Re-Interpolation (optional)

If the option is enabled, AMPIIMTS performs an iterative refinement. After the first pass of anomaly detection, it examines the matrix profile results to identify which sensors are most “stable” (i.e. those without major anomalies). It then uses the information from those stable signals to interpolate missing values in other sensors via similarity matching. In other words, if a certain sensor had gaps, the algorithm finds a few other sensors with similar behavior (via their matrix profile) and uses their data to fill in the gap in a statistically informed way. After this cross-sensor interpolation, the matrix profile is recomputed on the filled data to yield refined motifs and discords	. This two-step process can reveal anomalies that were initially obscured by missing data.


### Visualization (if enabled)

Finally, if the pipeline generates summary plots. One plot overlays the original (interpolated) time series data with highlights for motifs and discords  . Repeated patterns might be shaded with distinct colors, and each anomaly point is marked (e.g. a red line at the time of a discord). Another figure may show a heatmap of the matrix profile across time and dimensions– in this heatmap, cooler colors (lower values) indicate motifs and warmer colors (high values) indicate anomalies. These visual aids help interpret the results in context. After plotting, the ampiimts() function returns a tuple: (interpolated_data, normalized_data, result_dict) for further use in analysis or reporting	.


Through this workflow, AMPIIMTS automates the end-to-end process from raw irregular data to insightful pattern discovery. The combination of advanced techniques (hierarchical clustering, matrix profiles, MDL-based motif selection, etc.) is encapsulated behind a simple interface, enabling analysts to focus on interpreting the patterns and anomalies found.


## Package Structure and Key Modules


The AMPIIMTS package is organized into several Python modules, each responsible for a part of the computation. Below is an overview of the key modules and the functions they provide.


### ampiimts.py


This is the main entry point of the package. It defines the **ampiimts()** function, which coordinates the entire workflow. This function accepts either a directory path containing CSV files or a list of pandas DataFrames, and performs all preprocessing, analysis, and visualization steps. Internally, it delegates the main logic to **process()**, which chains together data preprocessing, matrix profile computation, optional smart interpolation, and result visualization. If enabled, the smart interpolation loop refines missing data points by cross-sensor similarity before re-running the matrix profile. Finally, the pipeline returns interpolated data, normalized data, and a dictionary of discovered motifs and discords.


### pre_processed.py


This module handles the preparation and cleaning of raw time series. Its main functions include :


- **synchronize_on_common_grid()**: Aligns multiple time series to a shared time index using median sampling frequency and resamples all signals accordingly.


- **interpolate()**: Fills small gaps in individual series via time interpolation and removes constant-value plateaus beyond a given threshold.


- **remove_linear_columns()**: Drops any columns with near-linear behavior based on a high R² fit, which could distort motif/discord detection.


- **normalization()** (and internal **aswn_with_trend()**): Applies sliding window normalization to each column, optionally blending with long-term trend using a weighted alpha parameter.


- **cluster_dimensions()**: Groups columns with similar dynamics into clusters using hierarchical correlation-based clustering. This step is optional but beneficial in high-dimensional datasets.


- **pre_processed()**: Serves as the orchestrator of the above. It applies synchronization, interpolation, normalization, and optionally clustering, returning cleaned and pre-structured data ready for matrix profile analysis.


### matrix_profile.py


This module manages the computation of matrix profiles, dispatching to univariate or multivariate methods depending on the input:


- **matrix_profile()**: Determines whether to treat input data as a single DataFrame or a list of clustered DataFrames. If needed, it runs jobs in parallel to compute each matrix profile efficiently.


- **matrix_profile_process()**: Computes the profile and calls the appropriate motif/discord discovery method depending on dimensionality.


The function supports parameters such as max_motifs, discord_top_pct, and mode toggles (motif-only or discord-only), passing these on to the pattern discovery functions. The output is standardized: a dictionary of indices, subspaces, and distance values associated with patterns.


### motif_pattern.py


 This is where the pattern discovery logic lives, leveraging the STUMPY library and custom logic:


  - **discover_patterns_stumpy_mixed()**: For univariate time series. Computes matrix profile using stump, identifies motifs as repeating subsequences with small profile distances, and extracts discords as subsequences with high distances.


  - **discover_patterns_mstump_mixed()**: For multivariate time series. Computes multi-dimensional profiles using mstump, applies dimension reduction via MDL (Minimum Description Length) where needed, and discovers motif patterns and anomalous subsequences across dimensions.


  - **exclude_discords()**: Refines the discord selection to avoid trivial overlaps and ensures the anomalies returned are both significant and temporally distinct.


 This module encapsulates the logic for identifying what makes a subsequence interesting—either by how often it repeats (motif) or by how different it is from all others (discord).


### plotting.py


 This module provides visualization support:


- **plot_all_patterns_and_discords()**: Generates plots for each signal (or cluster) with annotations showing the location of motifs and discords.


- **plot_multidim_patterns_and_discords()**: Handles multi-dimensional plotting. Highlights repeated motifs across relevant dimensions with semi-transparent overlays and marks discord timestamps with vertical lines.


Additional utilities allow heatmap visualization of the matrix profile across dimensions and time, providing insight into where the algorithm detected significant activity.


This modular design ensures flexibility, clarity, and ease of extension. Each module reflects a logical part of the AMPIIMTS pipeline, enabling users to engage with the package at a high level (via ampiimts()) or customize individual stages for advanced use cases.

# 

In summary, each module of AMPIIMTS corresponds to a logical stage of the analysis (preprocessing, core computation, pattern extraction, visualization). The README of the project serves as a base reference and starting point, summarizing these features and how to use the main pipeline	. By understanding the role of each script, one can appreciate how the package implements an end-to-end solution: reading raw irregular time series, cleaning and normalizing them, automatically finding a suitable window, computing matrix profiles to discover motifs and anomalies, and finally outputting results with optional visual context.



## Analysis of Unidimensional vs. Multidimensional Notebooks


The repository includes Jupyter notebooks demonstrating the package on real data, with two main analysis approaches: unidimensional (analyzing each variable separately) and multidimensional (analyzing variables jointly via clustering). Both notebooks start from the same underlying dataset (historical sensor measurements in Beijing), but they apply the AMPIIMTS pipeline in different ways. Below is a concise explanation of each approach and its outcomes:


### Unidimensional Analysis with Auto-selected Window Size via define_m (Single-Variable Patterns)


In this unidimensional analysis, the AMPIIMTS pipeline was applied to a single time series—synthetic daily data simulating periodic behavior with a single irregular jump anomaly. The `define_m` function automatically selected an appropriate window size (~1 day), capturing the natural daily cycle of the signal. The matrix profile was computed for the entire series, revealing repeating motifs (highlighted in green) that correspond to the normal daily pattern, and a sharp discord (highlighted in red) occurring around April 11, 2014. This discord coincides with an artificial “jump” inserted into the data, simulating a sudden deviation from normal behavior.


Because the matrix profile measures the similarity of subsequences, this method clearly identifies the anomalous jump as the most dissimilar window compared to all others. The motif overlay confirms consistent daily structure, reinforcing the interpretability of the detection. This example illustrates how AMPIIMTS, even in univariate mode, effectively highlights local anomalies without prior knowledge, using only the structure of the data itself.


### Multivariate Matrix Profile by clustering  on sensor air pollution data – Without Smart Interpolation (window_size = 1 month)


The matrix profile without smart interpolation shows several distinct anomalies (highlighted in red) around May–July 2015 and again in mid-2016. These periods correspond to documented meteorological extremes in Beijing. Notably, between late June and July 2015, heavy rains affected the city and northern China, disrupting air circulation and increasing pollutant concentrations. The algorithm also flags anomalies in May 2014, which aligns with local reports of a sharp drop in air quality linked to high ozone and humidity levels. This baseline analysis reveals that even without smart gap-filling, the algorithm successfully detects real environmental disruptions across sensor groups.


### Multivariate Matrix Profile by clustering  on sensor air pollution data – With Smart Interpolation (window_size = 1 month)


After enabling smart interpolation, anomalies become more pronounced and better isolated in time. Gaps previously caused by missing data are smoothed using similar sensor behavior, and as a result, the anomaly detection focuses more precisely on periods of genuine multivariate instability. The main discord spikes in mid-2015 remain, reinforcing their validity, but new anomalies emerge clearly in August 2013 and May 2016. The spike in August 2013 may be tied to a regional heatwave and accompanying air pollution, as supported by historical air quality bulletins. This refined detection demonstrates the added value of the iterative re-interpolation step.


### Multivariate Matrix Profile by clustering  on sensor air pollution data – Most Stable Sensor Only (window_size = 1 month)


Focusing only on the most stable sensor cluster (likely O₃ and DEWP), the matrix profile continues to highlight strong anomalies in the summer of 2015 and spring of 2016. The May–June 2015 period remains critical: this interval corresponds to a dense photochemical smog episode triggered by heat and industrial emissions. The clear detection of this known environmental stress through only one stable group validates the robustness of the algorithm even in reduced subspace settings. The 2016 anomaly aligns with several consecutive sand-dust alerts issued that spring, which impacted ozone dispersion and dew point stability.


### Multivariate Matrix Profile by clustering  on sensor air pollution data – Most Stable Sensor Only / Auto-selected Window Size via define_m


(window_size (Auto selected) =56 days 06:00:00)


In this configuration, the normalization / matrix profile was computed using a window size automatically determined by the define_m heuristic, which selects the most representative subsequence length based on data self-similarity. This choice enables the algorithm to adapt to the natural temporal scale of recurring structures in the dataset. The result is a precise identification of a dense cluster of discords between May and July 2015, matching a prolonged ozone and temperature anomaly in Beijing. During this period, the city experienced an extended photochemical smog event driven by stagnant air masses and high solar radiation. The peak matrix profile values suggest that subsequences within this window had no comparable patterns elsewhere in the dataset, confirming the uniqueness of this environmental event. The clarity of detection, without needing manual tuning, underscores the relevance of dynamic window selection for unsupervised pattern discovery.


The analyses conducted using the AMPIIMTS framework illustrate the power of matrix profile-based methods for detecting both repetitive patterns and anomalies in complex, multivariate time series. By leveraging clustering in the preprocessing phase, the package effectively groups sensors with similar dynamics—allowing the algorithm to focus on coherent variable subsets and improving the clarity of detected motifs and discords. This dimensionality reduction step plays a crucial role in isolating the most informative signals and avoiding noise from unrelated measurements.


The results—whether from unidimensional, full multivariate, or reduced cluster-based analyses—consistently highlight known environmental events such as smog episodes, extreme rainfall, or heatwaves in Beijing’s meteorological history. The incorporation of smart interpolation enhances the temporal precision of anomaly detection by compensating for missing data, while the use of define_m for automatic window selection ensures adaptive and context-aware profiling without prior tuning.

# 
Altogether, these findings validate AMPIIMTS as a reliable, modular, and explainable tool for time series motif discovery and anomaly detection. Its ability to integrate interpolation, clustering, dimensionality reduction, and matrix profiling makes it highly suited for environmental monitoring and any domain requiring interpretable unsupervised time series analysis.


## References:


Marrs, T. (2019). Introduction to Matrix Profiles: A Novel Data Structure for Mining Time Series. Medium.


Renouard, G. (2023). AMPIIMTS - Adaptive Matrix Profile Indexing for Irregular Multivariate Time Series [README documentation]. GitHub.


World Meteorological Centre Beijing (2021). North China saw extreme rainfall on 11–12 July 2021. WMC-BJ Bulletin.


Global Times (2021). Beijing poised to experience 2021’s heaviest rainfall with strong winds. Published 11 July 2021.


Davidson, H. (2021). Beijing skies turn orange as sandstorm and pollution send readings off the scale. The Guardian. Published 15 March 2021.


Reuters (2023). Beijing records hottest June day since records began as heatwave hits China. Cited by The Guardian, June 2023.


STUMPY Documentation (n.d.). STUMPY Basics Tutorial: Motifs and Discords. stumpy.docs.


SenX (2022). Matrix Profile of a Time Series to Discover Patterns. SenX Tech Blog.


UCR Time Series Classification Archive (2021). Motif Discovery and Discords in Time Series. University of California, Riverside.
