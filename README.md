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
An outdir directory is automatically created to save the models. Custom outdir directory can be defined using --outdir parameter.
Step 3 : Ensemble predictions

```
python3 predict_ensemble.py outdir/models/model1file.h5 outdir/models/model2file.h5 --data=../javastmt/q90
```
Step 4 : Metrics
Several metric files such as meteor.p exist. An example of how to use them is given below
```
python3 meteor.py outdir/predictions/predict-modelfile.txt --data=../javastmt/output
python3 meteordiff.py outdir/predictions/predict-modelfile1.txt outdir/predictions/predict-modelfile2.txt --data=../javastmt/ --not-diffonly 
```
Meteordiff.py with a --not-diffonly tag gives statistical significance between predicts of model1 vs model2. T-test values assume first model is better than the second so you may see very high p-value if you feed the lower scoring model first.

Step 5 : Difference set
```
python3 meteordiff.py outdir/predictions/predict-modelfile1.txt outdir/predictions/predict-modelfile2.txt --data=../javastmt/
```
Computes features of the difference set
while
```
python3 meteorbetter.py outdir/predictions/predict-modelfile1.txt outdir/predictions/predict-modelfile2.txt --data=../javastmt/
```
Computes the "improoved" set and its features

Step 6 : Standalone predictions
```
python3 predict.py outdir/models/model1file.h5 --data=../javastmt/q90
```
generates predictions for single models and all our configurations (use model.py to guide names of models)
