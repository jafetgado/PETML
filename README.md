# PETML
### Predicting PET-hydrolase activity from protein sequence with semi-supervised learning


This repository contains all scripts and data for experiments, design, and analyses
in building the deep-learning package, DeepPETase.




## File/directory structure
-------------------

`module/`: contains lower-level python scripts including `utils.py` and `models.py`

`experiment/`: contains scripts and data for experimental design of DeepPETase. 
These are organized in sequential steps of the experimental process as described below.

    1. hmmsearch: Retrieve PETase homologs from sequence databases by searching with HMM.
    2. preprocessing: Preprocess sequence data and prepare alignments for deep learning.
    3. hyperparameter_tuning: Optimize hyperparameters for the model (VAE & top model).
    4. training: Train the model using optimized hyperparameters.
    5. finetuning: Fine-tune the model end to end with lower learning rate.

For each step, python scripts are in `experiment/python_scripts`, bash scripts to utilize 
HPC resources are in `experiment/bash_scripts`, the data used or generated are in 
`experiment/data`, and stdout & stderr files are written to `experiment/stdout`.

Perfomance evaluation data in `experiment/data/training/performance` and 
`experiment/data/finetuning/performance` are presented as average values over 3-fold
cross-validation (column 0) and standard deviation (column 1).
    
Sequence data and activity measurements for 379 PETases from 23 studies are in 
`experiment/data/labels/`.





