[![DOI](https://doi.org/10.1088/1748-0221/16/12/P12035)

# Reconstruction of 3D Shower Structures for Neutrino Experiments

### Authors: [[V. Belavin](https://gitlab.com/SchattenGenie)], [E. Trofimova] (etrofimova@hse.ru), [A. Ustyuzhanin]

## Overview

This directory contains code necessary to run the Electromagnetic Showers (EM) Reconstruction algorithm that is devided into the following parts:
1) Graph Construction;
2) Edge Classification;
3) Showers Clusterization;
4) Parameters Reconstruction.

### Experimental Data

X, Y, Z coordinates and the direction of the EM Showers base-tracks. 

The showers are generated using FairShip framework. 

### Results

The algorithm detects ~ 87% of Showers. The mean energy resolution is 0.101. 


## Running the code

### 1. Data preprocessing

Firstly one has to compile Cython extension for graph calculation:

```
cd data && python setup_opera_distance_metric.py build_ext --inplace
```
Then one could run preprocessing of EM showers in pytorch-geometric graph format:

1) *1 shower per brick* case
```
cd data && python generator_updated.py --df_file ./showers_18k.csv \
--r $r --output_file ./1_shower.pt --num_showers_in_brick 1 && \
python preprocess_dataset.py --input_datafile ./1_shower.pt \
--output_datafile ./1_preprocessed.pt
```
2) *random number of showers per brick* case
```
cd data && python generator_rand.py --df_file ./showers_18k.csv \
--r $r --output_file ./rand_showers.pt && \
python preprocess_dataset.py --input_datafile ./rand_showers.pt \
--output_datafile ./rand_preprocessed.pt
```
The datafile can be accessed by <a href="https://doi.org/10.5281/zenodo.5570901"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.5570901.svg" alt="DOI"></a>.

### 2. Graph Neural Network training edge classifier

Next step is to train classier network, that is going to discriminated edges between those that connect nodes from the same class and those which belogs to different classes.

```
python training_classifier.py --datafile ../data/rand_preprocessed.pt --epochs 1000 --learning_rate 0.001 \
--num_layers_emulsion 5 --num_layers_edge_conv 3 \
--hidden_dim 32 --output_dim 32 --graph_embedder GraphNN_KNN_v2 --edge_classifier EdgeClassifier_v1 \
--project_name em_showers_network_training --workspace YOUR_WORKSPACE_COMET_ML  --outer_optimization true \
--use_scheduler false
```

```Comet.ml```: https://www.comet.ml/ketrint/em-network-training-cv10/view/new/experiments 

### 3. Clustering of EM showers

Using networks weights from previous step we can perform clustering end estimate quality:

```
python clustering.py --datafile ../data/rand_preprocessed.pt \
--num_layers_edge_conv 3 --num_layers_emulsion 5 --threshold 0.2 --min_samples_core 4 \
--hidden_dim 32 --output_dim 32 --graph_embedder GraphNN_KNN_v2 --edge_classifier EdgeClassifier_v1 \
--project_name clustering --workspace YOUR_WORKSPACE_COMET_ML --vanilla_hdbscan False \
--graph_embedder_weights ../data/graph_embedder_rand_preprocessed_45959c8a2e3a4ec69eb31abfb5ad5f54.pt \
--edge_classifier_weights ../data/edge_classifier_rand_preprocessed_45959c8a2e3a4ec69eb31abfb5ad5f54.pt \
--energy_file ./E_pred.npy --energy_true_file ./E_true.npy --z_file ./data_new/z.npy
```
```Comet.ml```

*Clustering parameters choice*: https://www.comet.ml/ketrint/param-choice-10kcv/view/new/experiments

*Clustering on test data*: https://www.comet.ml/ketrint/test-10kcv/view/new/experiments

*Clustering on validation data*: https://www.comet.ml/ketrint/val-10kcv/view/new/experiments




