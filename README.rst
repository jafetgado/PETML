**PETML**
===============

PETML is a machine learning method for predicting PET hydrolase activity of putative sequences.
The predictive method combines supervised learning, using a dataset of 514 activity measurements, 
and "unsupervised" learning, using profile hidden Markov models (pHMM) and Blosum similarity scores, 
to rank the putative enzyme sequences. 


Usage 
-------------

.. code:: shell-session

    git clone https://github.com/jafetgado/PETML.git
    cd PETML
    conda env create -f ./env.yml -p ./env
    conda activate ./env
    python ./petml/run.py --seqfile "./example/sequences.fasta" --outdir ./example 
..



Citation
----------
If you find PETML useful, please cite:

Norton-Baker B, Komp E, Gado JE, et al, 2025. "Machine learning-guided identification of PET hydrolases from natural diversity".
