# Reproduction Guide

## Origin-Destination Trajectory Prediction with Deep Learning and Markov Models

For detailed information, including the data and the data usage for each code file, please refer to `Reproductuction Guide.pdf` in the repository.

This document provides step-by-step instructions for reproducing the data processing workflow, route prediction results, evaluation metrics, statistical tests, and figures reported in the manuscript.

All file paths are relative paths.

---

## 1. Software Environment

The codes were executed in a Linux-based Conda environment.

### Main environment

* Operating system: Linux
* Python version: 3.10.13

### Main Python packages

```text
numpy==1.23.5, pandas==2.3.2, geopandas==0.14.4, networkx==3.4, shapely==2.0.1,
matplotlib==3.9.1, seaborn==0.13.2, scipy==1.15.2, statsmodels==0.14.5, osmnx==1.9.3,
contextily==1.6.2, torch==2.4.1, torch-geometric==2.6.1, torch-scatter==2.1.2,
torch-sparse==0.6.18, torch-cluster==1.6.3, tqdm==4.67.1
```

---

## 2. Repository Structure

This repository is divided into two main folders: `For_test` and `Figure_and_Real`.

The `For_test` folder contains the codes and simulated data required to demonstrate the overall workflow of the study. Because the original AI-Hub data cannot be redistributed, transferred to third parties, or exported outside Korea without prior approval, this repository does not include the original AI-Hub data or direct subsets of it. Instead, the `For_test/data/simulated_raw_data/` folder provides simulated data that follow the same folder structure, file naming convention, and column schema required by the code. For a detailed description of the original data source and download procedure, please refer to `Data Download Instructions.md`. The purpose of this folder is to allow reviewers to execute the complete analytical procedure, including data preprocessing, route prediction, and accuracy calculation, using the simulated data.

The `Figure_and_Real` folder contains the codes used to reproduce the figures and statistical results reported in the manuscript. The input files used in this folder are based on the final outputs generated from the full original dataset used in the actual study. Therefore, while `For_test` demonstrates the full reproducible workflow using the simulated dataset, `Figure_and_Real` provides the final result files needed to reproduce the figures and statistics reported in the manuscript.

The overall repository structure is as follows.

```text
jeju_code_git_anonymous/
├── Data Download Instructions.md              # Instructions for downloading the original AI-Hub dataset and file keys
├── README.md                                  # Overview of the repository and reproducibility package
├── Reproduction Guide.pdf                     # Step-by-step guide for reproducing workflows, figures, tables, and metrics
│
├── For_test/                                  # Executable workflow using simulated data
│ ├── Code7_LSTM/                              # Supporting scripts and files for LSTM model training
│ ├── LSTM_weight/                             # Pretrained LSTM weights used in the For_test workflow
│ ├── data/                                    # Input and intermediate data for the For_test workflow
│ │ ├── jeju_shp/                              # Jeju boundary shapefile
│ │ ├── new_hexagraph/                         # Hexagon grid and hexagon-based road network files
│ │ ├── road_network/                          # Jeju road network graph files
│ │ ├── simulated_processed_inputs/            # Intermediate input files generated from simulated raw data
│ │ └── simulated_raw_data/Jeju_data/          # Simulated raw-style data following the AI-Hub folder and column structure
│ ├── expected_findings/                       # Expected outputs generated from the simulated-data workflow
│ │ ├── code5_shortcut_prediction_route/       # Expected shortest-path prediction outputs
│ │ ├── code6_prediction_markov/               # Expected Basic Markov prediction outputs
│ │ ├── code6.5_prediction_markov_2/           # Expected Conditional Markov prediction outputs
│ │ ├── code7_weight_for_test/                 # Expected example LSTM training outputs
│ │ ├── code8_bash_result/                     # Expected SLURM/bash execution logs for LSTM prediction
│ │ ├── code8_prediction_LSTM/                 # Expected raw LSTM prediction outputs
│ │ ├── code9_refine_LSTM/                     # Expected topology-refined LSTM prediction outputs
│ │ └── code10_accuracy/                       # Expected accuracy files and evaluation results
│ ├── code1_hexa_network.ipynb                 # Construct and visualize the hexagon road network
│ ├── code2_change_gps_to_hexa.ipynb           # Filter GPS points in Jeju and convert trajectories to hexagon sequences
│ ├── code3_create_traveler_feature.ipynb      # Generate traveler-level feature tables
│ ├── code4_create_od_route_segments.ipynb     # Generate connected routes and split them into OD route segments
│ ├── code5_predict_shortest.ipynb             # Predict OD routes using the shortest path baseline
│ ├── code6_predict_markov.ipynb               # Predict OD routes using the Basic Markov model
│ ├── code6.5_predict_attribute_markov.ipynb   # Predict OD routes using the Conditional Markov model
│ ├── code7_train_LSTM_example.ipynb           # Demonstrate LSTM model training using simulated data
│ ├── code8_predict_LSTM.py                    # Generate LSTM route predictions using pretrained weights
│ ├── code8_predict_LSTM.sh                    # SLURM/bash script for running LSTM prediction
│ ├── code9_refine_LSTM.ipynb                  # Refine LSTM predictions using graph topology
│ └── code10_accuracy.ipynb                    # Calculate Jaccard Similarity and Levenshtein Distance
│
└── Figure_and_Real/                           # Reproduction of manuscript figures and statistical results
    ├── data/                                  # Input data for reproducing manuscript figures and statistics
    │ ├── GPS_trajectory_for_figure/           # Trajectory datasets used for route visualization figures
    │ │ ├── Figure_1/                          # Input data for Figure 1
    │ │ ├── Figure_6/                          # Input data for Figure 6
    │ │ ├── Figure_9/                          # Input data for Figure 9
    │ │ ├── Figure_10/                         # Input data for Figure 10
    │ │ └── Figure_11/                         # Input data for Figure 11
    │ ├── accuracy/                            # Accuracy result files generated from the full-data workflow
    │ ├── jeju_shp/                            # Jeju boundary shapefile used for visualization
    │ ├── new_hexagraph/                       # Hexagon road network files used for visualization
    │ └── trajectory/                          # Full-data trajectory and prediction files used for evaluation
    ├── Figure/                                # Reproduced manuscript figures
    ├── drawing_fig_1_6_9_10_11.ipynb          # Generate Figures 1, 6, 9, 10, and 11
    └── evaluation_and_significance_fig_7_8.ipynb # Generate Figures 7 and 8 and perform statistical tests
```

## 3. Data Index

Please refer to `Reproductuction Guide.pdf`.

---

## 4. Code and Data

Please refer to `Reproductuction Guide.pdf`.

---

## 5. Overall Workflow

The full workflow consists of the following stages:

1. Construct and show the hexagon road network. (`For_test/code1`)
2. Filter raw GPS points located within Jeju. (`For_test/code2`)
3. Convert GPS trajectories to hexagon cell sequences. (`For_test/code2`)
4. Create traveler-level property tables. (`For_test/code3`)
5. Generate connected shortcut routes and split them into OD segments. (`For_test/code4`)
6. Predict routes using four models:

   * Shortest Path (`For_test/code5`)
   * Basic Markov (`For_test/code6`)
   * Conditional Markov (`For_test/code6.5`)
   * LSTM (`For_test/code7`, `code8`)
7. Refine LSTM predictions using graph topology. (`For_test/code9`)
8. Compute Jaccard Similarity and Levenshtein Distance. (`For_test/code10`, `Figure_and_Real/Evaluation_and_significance.ipynb`)
9. Conduct statistical significance tests. (`Figure_and_Real/Evaluation_and_significance.ipynb`)
10. Generate manuscript figures. (`Figure_and_Real/drawing_picture.ipynb`)

---

## 6. Recommended Execution Order

Run the files in the following order:

1. `For_test/code1_hexa_network.ipynb`
2. `For_test/code2_change_gps_to_hexa.ipynb`
3. `For_test/code3_create_traveler_feature.ipynb`
4. `For_test/code4_create_od_route_segments.ipynb`
5. `For_test/code5_predict_shortest.ipynb`
6. `For_test/code6_predict_markov.ipynb`
7. `For_test/code6.5_predict_attribute_markov.ipynb`
8. `For_test/code7_train_LSTM_example.ipynb`
9. `For_test/code8_predict_LSTM.py`
10. `For_test/code9_refine_LSTM.ipynb`
11. `For_test/code10_accuracy.ipynb`
12. `Figure_and_Real/Evaluation_and_significance.ipynb`
13. `Figure_and_Real/drawing_picture.ipynb`

---

## 7. Description of Each Code File

For further details regarding the implementation and workflow, please refer to the comments provided at the beginning of each code file in the repository.

---

### 7.1 `For_test/code1_hexa_network.ipynb`

**Input data**

The Jeju road network graph and the base hexagon grid.

**Output data**

The hexagon road network used throughout the study.

**Description**

This notebook shows the hexagon-based road network used in the study. The original road network is integrated with the hexagon grid to create a graph structure that preserves road connectivity while representing trajectories as hexagon sequences. The resulting network serves as the spatial framework for trajectory conversion, route prediction, route refinement, and visualization.

---

### 7.2 `For_test/code2_change_gps_to_hexa.ipynb`

**Input data**

Raw GPS trajectories, the Jeju boundary, and the hexagon road network.

**Output data**

Jeju-filtered GPS trajectories and hexagon-converted trajectory sequences.

**Description**

This notebook preprocesses the original GPS trajectory data. GPS points located outside Jeju are removed, and the remaining trajectories are converted into sequences of hexagon cells using the hexagon road network. This conversion produces the network-based trajectory representation used in all subsequent analyses.

---

### 7.3 `For_test/code3_create_traveler_feature.ipynb`

**Input data**

Hexagon-converted GPS trajectories and traveler-related survey tables.

**Output data**

Traveler property tables.

**Description**

This notebook generates traveler-level feature tables by combining processed trajectory data with traveler survey information. The resulting attributes are used to characterize travelers and are later incorporated into the Conditional Markov and LSTM-based prediction models.

---

### 7.4 `For_test/code4_create_od_route_segments.ipynb`

**Input data**

Hexagon-converted GPS trajectories, traveler property tables, and the hexagon road network.

**Output data**

Connected route sequences, OD route segment files, and traveler-level OD feature tables.

**Description**

This notebook generates topologically connected routes and splits them into OD route segments. Because raw trajectory sequences may contain disconnected transitions, route connectivity is first corrected using the hexagon road network. The resulting routes are then divided into OD segments according to the specified split size (k=8, k=12, and k=16). These OD segment files serve as the primary inputs for all route prediction models used in the study.

---

### 7.5 `For_test/code5_predict_shortest.ipynb`

**Input data**

The hexagon road network and validation OD route segments.

**Output data**

Shortest path route prediction files.

**Description**

This notebook generates route predictions using the shortest path algorithm. For each OD segment, the shortest path between the origin and destination is calculated on the hexagon road network. The resulting routes serve as a baseline model for comparison with the data-driven prediction approaches (Markov, Conditional Markov, and LSTM).

---

### 7.6 `For_test/code6_predict_markov.ipynb`

**Input data**

Training route sequences, validation OD route segments, and the hexagon road network.

**Output data**

Basic Markov route prediction files.

**Description**

This notebook implements the Basic Markov model. Transition probabilities are estimated from the training route sequences, and validation routes are predicted using these learned transition patterns. The model captures common movement behaviors observed in the training trajectories.

---

### 7.7 `For_test/code6.5_predict_attribute_markov.ipynb`

**Input data**

Traveler property tables, training route sequences, validation OD route segments, and the hexagon road network.

**Output data**

Conditional Markov route prediction files.

**Description**

This notebook implements the Conditional Markov model. In contrast to the Basic Markov model, traveler attributes are incorporated into the transition estimation process. This allows route predictions to reflect both observed movement patterns and traveler characteristics.

---

### 7.8 `For_test/code7_train_LSTM_example.ipynb`

**Input data**

The hexagon road network, OD route segment files, and traveler-level OD feature tables.

**Output data**

Example trained LSTM model weights. The model weights generated by `code7_train_LSTM_example.ipynb` are provided only as an example.

**Description**

This notebook demonstrates the training procedure of the LSTM-based route prediction model using the reduced dataset provided in the repository. The model learns route patterns from OD segments, traveler information, and graph-based spatial context. The notebook is included to illustrate the complete model training workflow.

In the shared repository, this notebook is provided as an example implementation in `.ipynb` format and demonstrates the training procedure only for the k=8 setting. In the actual study, the models for k=8, 12, and 16 were trained separately using `.py` and `.sh` scripts in a SLURM-based computing environment. Therefore, this notebook is intended to explain the training workflow step by step, while the trained model weights used in the study are provided separately.

---

### 7.9 `For_test/code8_predict_LSTM.py`

**Input data**

The hexagon road network, validation OD route segments, traveler-level OD feature tables, and trained LSTM model weights. The prediction scripts use the trained model weights from the actual study stored in `./weight/`, rather than the example weights stored in `./weight_for_test/`.

**Output data**

LSTM route prediction files.

**Description**

This script generates route predictions using the trained LSTM model. The model predicts routes for each OD segment while considering traveler attributes and graph structure information. Adjacency-constrained beam search is used during decoding to ensure that predicted routes follow the connectivity of the hexagon road network. Segment-level predictions are subsequently merged into complete route sequences.

---

### 7.10 `For_test/code8_predict_LSTM.sh`

**Input data**

The same inputs required by the LSTM prediction script.

**Output data**

LSTM route prediction files and execution logs.

**Description**

This shell script is provided for running the LSTM prediction process on a Linux server using SLURM. It specifies the computing resources required for execution and calls the Python prediction script automatically.

In the actual study, both the training and validation procedures corresponding to the example workflow shown in `code7_train_LSTM_example.ipynb` were conducted using similar `.py` and `.sh` files in a SLURM-based computing environment.

---

### 7.11 `For_test/code9_refine_LSTM.ipynb`

**Input data**

The hexagon road network and raw LSTM prediction files.

**Output data**

Refined LSTM route prediction files.

**Description**

This notebook refines the raw LSTM prediction results using the topology of the hexagon road network. Any disconnected transitions that remain after prediction are corrected to ensure that all predicted routes form valid connected paths on the network.

---

### 7.12 `For_test/code10_accuracy.ipynb`

**Input data**

Ground-truth validation routes and route predictions generated by the Shortest Path, Basic Markov, Conditional Markov, and LSTM models.

**Output data**

Accuracy result files and evaluation statistics.

**Description**

This notebook evaluates route prediction performance by comparing predicted routes with observed validation trajectories. Jaccard Similarity and Levenshtein Distance are calculated for each model and each split size. The resulting accuracy files are later used to reproduce the performance tables, statistical tests, and evaluation figures reported in the manuscript.

---

### 7.13 `Figure_and_Real/Evaluation_and_significance.ipynb`

**Input data**

Route prediction results and accuracy files generated from the full original dataset used in the actual study, rather than the reduced dataset provided in the `For_test` folder.

**Output data**

Performance figures and statistical significance test results.

**Description**

This notebook reproduces the quantitative evaluation results reported in the manuscript. Unlike the `For_test` workflow, which uses a reduced dataset for demonstration purposes, this notebook uses the final route prediction results generated from the full original dataset used in the actual study. It generates performance comparison figures and performs Friedman tests, pairwise Wilcoxon signed-rank tests, and Holm correction procedures to assess whether differences among prediction models are statistically significant.

---

### 7.14 `Figure_and_Real/drawing_picture.ipynb`

**Input data**

The hexagon road network and figure-specific trajectory datasets generated from the full dataset.

**Output data**

Figures reported in the manuscript.

**Description**

This notebook reproduces the route visualization figures presented in the manuscript. Although each figure uses a separate dataset prepared specifically for visualization, all of these datasets were generated from the full original dataset used in the actual study. The generation procedures are identical to those demonstrated in the `For_test` workflow, but were applied to the complete dataset rather than the reduced sample dataset.

* For Figure 1, the visualization is based on the ground-truth route dataset (`merged_routes_by_12.csv`) generated during the accuracy evaluation procedure in `For_test/code10_accuracy.ipynb`.
* For Figure 6, the visualization uses route predictions generated by the Shortest Path, Basic Markov, Conditional Markov, and LSTM models (`For_test/code5~code9`), together with the ground-truth route dataset generated in `For_test/code10_accuracy.ipynb`. A representative traveler was selected and visualized for comparison.
* For Figure 9, the visualization is based on OD route segments generated in `For_test/code4_create_od_route_segments.ipynb`.
* For Figures 10 and 11, the visualizations use route predictions generated by the four prediction models (`For_test/code5~code9`) together with route segments generated in `For_test/code4_create_od_route_segments.ipynb`. Representative examples were selected and visualized for comparison.

Therefore, while the datasets stored in the `Figure_and_Real` folder are provided specifically for reproducing the manuscript figures, all of them originate from the same workflow described in the `For_test` folder and were generated using the full original dataset employed in the study.


## 8. Description of Each Reported Finding in the Manuscript

This section provides item-specific reproduction instructions for the tables and figures reported in the manuscript. For each reported finding, we describe the required input data, the relevant code steps, the output files, and the relationship between the simulated-data workflow in `For_test` and the full-data results provided in `Figure_and_Real`.

---

### 8.1 Table 1. Data Description

#### Input data

The variables summarized in Table 1 are derived from the simulated raw-style AI-Hub data and the preprocessed trajectory files.

Required input files are:

- Hexagon-based GPS trajectory files:
- `./For_test/data/simulated_processed_inputs/Training/gps_hexa/`
- `./For_test/data/simulated_processed_inputs/Validation/gps_hexa/`

- Traveler master data:
- `./For_test/data/simulated_raw_data/Jeju_data/Training/tn_traveller_master_여행객 Master_H.csv`
- `./For_test/data/simulated_raw_data/Jeju_data/Validation/tn_traveller_master_여행객 Master_H.csv`

- Companion information data:
- `./For_test/data/simulated_raw_data/Jeju_data/Training/tn_companion_info_동반자정보_H.csv`
- `./For_test/data/simulated_raw_data/Jeju_data/Validation/tn_companion_info_동반자정보_H.csv`

- Travel information data:
- `./For_test/data/simulated_raw_data/Jeju_data/Training/tn_travel_여행_H.csv`
- `./For_test/data/simulated_raw_data/Jeju_data/Validation/tn_travel_여행_H.csv`

- Movement history data:
- `./For_test/data/simulated_raw_data/Jeju_data/Training/tn_move_his_이동내역_H.csv`
- `./For_test/data/simulated_raw_data/Jeju_data/Validation/tn_move_his_이동내역_H.csv`

Optional lodge and activity history files are loaded for inspection but are not used in the final feature table.

#### Code step

1. Run `For_test/code3_create_traveler_feature.ipynb`.

This notebook selects and preprocesses the traveler, companion, travel, movement, and trajectory-related variables used in the model input feature table.

#### Output data

- `./For_test/data/simulated_processed_inputs/Training/properties_for_traveler.csv`
- `./For_test/data/simulated_processed_inputs/Validation/properties_for_traveler.csv`

#### Description

Table 1 is a data description table. It summarizes the input variables used in the route prediction models, including tourist information, companion information, tour information, and trajectory data. The table does not report model performance results. Instead, it documents the variable groups, feature names, and data types or encoding schemes used in the analytical workflow.

---

### 8.2 Figure 1. Hexagon-based Trajectory Representation

#### Input data

Figure 1 uses the following spatial and trajectory data:

- Base hexagon grid:
- `./Figure_and_Real/data/new_hexagraph/jeju_hexa_for_use.shp`

- Hexagon road network:
- `./Figure_and_Real/data/new_hexagraph/hexa_network_with_road.shp`

- Merged trajectory data:
- `./Figure_and_Real/data/GPS_trajectory_for_figure/Figure_1/merged_routes_by_{k}.csv`

#### Code step

1. Run `Figure_and_Real/drawing_fig_1_6_9_10_11.ipynb`.
2. Execute the section corresponding to Figure 1.

The file `merged_routes_by_{k}.csv` was generated by applying the same route-merging and accuracy-preparation workflow demonstrated in `For_test/code10_accuracy.ipynb` to the full original dataset. Because the original AI-Hub data cannot be redistributed, the full raw dataset is not included in this repository. Instead, the final derived trajectory file required to reproduce Figure 1 is provided in `Figure_and_Real`.

#### Output data

- `./Figure_and_Real/Figure/Figure_1.png`

#### Description

Figure 1 visualizes the hexagon-based trajectory representation used in the study. It shows the hexagon spatial framework and a representative observed trajectory after conversion into hexagon-based route sequences. The purpose of the figure is to illustrate how raw movement trajectories are represented in the hexagon road network used for route prediction.

### 8.3 Figure 6. Route Prediction Comparison for a Representative Trajectory

#### Input data

Figure 6 uses the following spatial and trajectory data:

* Hexagon road network:

  * `./Figure_and_Real/data/new_hexagraph/hexa_network_with_road.shp`

* Figure-specific trajectory and prediction data:

  * `./Figure_and_Real/data/GPS_trajectory_for_figure/Figure_6/`

The files in `Figure_6/` include the observed GPS-based route and the corresponding route prediction results generated by the four prediction methods: Shortest Path, Basic Markov, Conditional Markov, and LSTM.

#### Code step

1. Run the full route prediction workflow demonstrated in `For_test`:

   * `For_test/code2_change_gps_to_hexa.ipynb`
   * `For_test/code3_create_traveler_feature.ipynb`
   * `For_test/code4_create_od_route_segments.ipynb`
   * `For_test/code5_predict_shortest.ipynb`
   * `For_test/code6_predict_markov.ipynb`
   * `For_test/code6.5_predict_attribute_markov.ipynb`
   * `For_test/code8_predict_LSTM.py`
   * `For_test/code9_refine_LSTM.ipynb`
   * `For_test/code10_accuracy.ipynb`

2. The same workflow was applied to the full original dataset to generate the derived route files used for Figure 6.

3. Run `Figure_and_Real/drawing_fig_1_6_9_10_11.ipynb`.

4. Execute the section corresponding to Figure 6.

Because the original AI-Hub data cannot be redistributed, the raw full dataset is not included in this repository. Instead, the final derived route files required to reproduce Figure 6 are provided in `Figure_and_Real/data/GPS_trajectory_for_figure/Figure_6/`.

#### Output data

* `./Figure_and_Real/Figure/Figure_6.png`

#### Description

Figure 6 compares the observed trajectory with the routes predicted by the four models: Shortest Path, Basic Markov, Conditional Markov, and LSTM. The purpose of the figure is to visually compare how each model reconstructs a representative movement trajectory on the hexagon road network.

---

### 8.4 Figure 7. Jaccard Similarity Comparison

#### Input data

Figure 7 uses the accuracy result files generated from the route prediction outputs.

Required input files are:

* Accuracy result files:

  * `./Figure_and_Real/data/accuracy/accuracy_ver_last_8.csv`
  * `./Figure_and_Real/data/accuracy/accuracy_ver_last_12.csv`
  * `./Figure_and_Real/data/accuracy/accuracy_ver_last_16.csv`

These files contain Jaccard Similarity and Levenshtein Distance values calculated for the four prediction methods and three OD split sizes.

#### Code step

1. Generate OD route segments using:

   * `For_test/code4_create_od_route_segments.ipynb`

2. Generate route predictions using:

   * `For_test/code5_predict_shortest.ipynb`
   * `For_test/code6_predict_markov.ipynb`
   * `For_test/code6.5_predict_attribute_markov.ipynb`
   * `For_test/code8_predict_LSTM.py`

3. Refine LSTM prediction results using:

   * `For_test/code9_refine_LSTM.ipynb`

4. Calculate accuracy metrics using:

   * `For_test/code10_accuracy.ipynb`

5. The same evaluation workflow was applied to the full original dataset to generate the accuracy files used for Figure 7.

6. Run `Figure_and_Real/evaluation_and_significance_fig_7_8.ipynb`.

7. Execute the section corresponding to Figure 7.

#### Output data

* `./Figure_and_Real/Figure/Figure_7.png`

#### Description

Figure 7 reports the Jaccard Similarity results for the route prediction models. It compares the overlap between observed and predicted route sequences across the Shortest Path, Basic Markov, Conditional Markov, and LSTM models. The figure also compares results across OD split sizes, including `k=8`, `k=12`, and `k=16`.

---

### 8.5 Figure 8. Levenshtein Distance Comparison

#### Input data

Figure 8 uses the same accuracy result files as Figure 7.

Required input files are:

* Accuracy result files:

  * `./Figure_and_Real/data/accuracy/accuracy_ver_last_8.csv`
  * `./Figure_and_Real/data/accuracy/accuracy_ver_last_12.csv`
  * `./Figure_and_Real/data/accuracy/accuracy_ver_last_16.csv`

These files are generated by comparing observed validation routes with the routes predicted by the four models.

#### Code step

1. Generate OD route segments using:

   * `For_test/code4_create_od_route_segments.ipynb`

2. Generate route predictions using:

   * `For_test/code5_predict_shortest.ipynb`
   * `For_test/code6_predict_markov.ipynb`
   * `For_test/code6.5_predict_attribute_markov.ipynb`
   * `For_test/code8_predict_LSTM.py`

3. Refine LSTM prediction results using:

   * `For_test/code9_refine_LSTM.ipynb`

4. Calculate accuracy metrics using:

   * `For_test/code10_accuracy.ipynb`

5. The same evaluation workflow was applied to the full original dataset to generate the accuracy files used for Figure 8.

6. Run `Figure_and_Real/evaluation_and_significance_fig_7_8.ipynb`.

7. Execute the section corresponding to Figure 8.

#### Output data

* `./Figure_and_Real/Figure/Figure_8.png`

#### Description

Figure 8 reports the Levenshtein Distance results for the route prediction models. It evaluates the sequence-level difference between observed and predicted routes. Lower Levenshtein Distance values indicate that the predicted route sequence is closer to the observed route sequence. The figure compares the Shortest Path, Basic Markov, Conditional Markov, and LSTM models across the OD split sizes `k=8`, `k=12`, and `k=16`.

---

### 8.6 Figure 9. OD Route Segment Examples

#### Input data

Figure 9 uses OD route segment data generated from the connected route generation and OD splitting workflow.

Required input files are:

* Hexagon road network:

  * `./Figure_and_Real/data/new_hexagraph/hexa_network_with_road.shp`

* Figure-specific OD route segment data:

  * `./Figure_and_Real/data/GPS_trajectory_for_figure/Figure_9/`

The `Figure_9/` folder contains representative OD segment examples prepared for visualization.

#### Code step

1. Run the preprocessing and OD route segmentation workflow demonstrated in `For_test`:

   * `For_test/code2_change_gps_to_hexa.ipynb`
   * `For_test/code3_create_traveler_feature.ipynb`
   * `For_test/code4_create_od_route_segments.ipynb`

2. The same OD route segmentation workflow was applied to the full original dataset to generate the figure-specific OD segment files used for Figure 9.

3. Run `Figure_and_Real/drawing_fig_1_6_9_10_11.ipynb`.

4. Execute the section corresponding to Figure 9.

#### Output data

* `./Figure_and_Real/Figure/Figure_9.png`

#### Description

Figure 9 visualizes representative OD route segment examples. The purpose of the figure is to show how continuous trajectory sequences are converted into connected route sequences and divided into OD segments. These OD route segments are the basic prediction units used by the Shortest Path, Basic Markov, Conditional Markov, and LSTM models.

---

### 8.7 Figure 10. Route Prediction Examples for Selected Cases

#### Input data

Figure 10 uses the following spatial and trajectory data:

* Hexagon road network:

  * `./Figure_and_Real/data/new_hexagraph/hexa_network_with_road.shp`

* Figure-specific trajectory and prediction data:

  * `./Figure_and_Real/data/GPS_trajectory_for_figure/Figure_10/`

The `Figure_10/` folder contains selected observed trajectories and their corresponding predicted routes from the four models.

#### Code step

1. Run the full route prediction workflow demonstrated in `For_test`:

   * `For_test/code2_change_gps_to_hexa.ipynb`
   * `For_test/code3_create_traveler_feature.ipynb`
   * `For_test/code4_create_od_route_segments.ipynb`
   * `For_test/code5_predict_shortest.ipynb`
   * `For_test/code6_predict_markov.ipynb`
   * `For_test/code6.5_predict_attribute_markov.ipynb`
   * `For_test/code8_predict_LSTM.py`
   * `For_test/code9_refine_LSTM.ipynb`
   * `For_test/code10_accuracy.ipynb`

2. The same workflow was applied to the full original dataset to generate the route prediction files used for Figure 10.

3. Run `Figure_and_Real/drawing_fig_1_6_9_10_11.ipynb`.

4. Execute the section corresponding to Figure 10.

#### Output data

* `./Figure_and_Real/Figure/Figure_10.png`

#### Description

Figure 10 presents selected route prediction examples. It compares observed routes with the predicted routes generated by the Shortest Path, Basic Markov, Conditional Markov, and LSTM models. The figure is intended to illustrate qualitative differences among the prediction methods in representative trajectory cases.

---

### 8.8 Figure 11. Route Prediction Examples for Additional Selected Cases

#### Input data

Figure 11 uses the following spatial and trajectory data:

* Hexagon road network:

  * `./Figure_and_Real/data/new_hexagraph/hexa_network_with_road.shp`

* Figure-specific trajectory and prediction data:

  * `./Figure_and_Real/data/GPS_trajectory_for_figure/Figure_11/`

The `Figure_11/` folder contains selected observed trajectories and the corresponding prediction results generated by the four route prediction models.

#### Code step

1. Run the full route prediction workflow demonstrated in `For_test`:

   * `For_test/code2_change_gps_to_hexa.ipynb`
   * `For_test/code3_create_traveler_feature.ipynb`
   * `For_test/code4_create_od_route_segments.ipynb`
   * `For_test/code5_predict_shortest.ipynb`
   * `For_test/code6_predict_markov.ipynb`
   * `For_test/code6.5_predict_attribute_markov.ipynb`
   * `For_test/code8_predict_LSTM.py`
   * `For_test/code9_refine_LSTM.ipynb`
   * `For_test/code10_accuracy.ipynb`

2. The same workflow was applied to the full original dataset to generate the route prediction files used for Figure 11.

3. Run `Figure_and_Real/drawing_fig_1_6_9_10_11.ipynb`.

4. Execute the section corresponding to Figure 11.

Because the original AI-Hub data cannot be redistributed, the raw full dataset is not included in this repository. Instead, the derived input files required to reproduce Figure 11 are provided in `Figure_and_Real/data/GPS_trajectory_for_figure/Figure_11/`.

#### Output data

* `./Figure_and_Real/Figure/Figure_11.png`

#### Description

Figure 11 provides additional qualitative route prediction examples. It compares the observed trajectory with the routes predicted by the Shortest Path, Basic Markov, Conditional Markov, and LSTM models. This figure supplements the route comparison results by showing additional representative cases and illustrating how the models behave across different route patterns.
