# smncode2022
Replication package for paper submitted for peer review titled "Statement-based Memory for Neural Source Code Summarization"


Step 1 : Download datasets
Java Dataset can be downloaded from:

Python Datasets can be downloaded from:

Please place the datasets in the immediate directory above the working directory made by this repository

Step 2 : Train Model
To train our optimal configuration : 
```
time python3 train.py --model-type=smn --gpu=0 --data=../javastmt/q90 --epochs=20 --memory-network-input=positional-encoding --max-sent-len 30 --max-sent-cnt 70
```
To train the "EOS embedding" configuration change the --memory-network-input parameter=eos-embedding". Other configurations are separate models. Refer to model.py for --model-type names.

Step 3 : Ensemble predictions

Step 4 : Metrics

Step 5 : Difference set

Step 6 : Standalone predictions
