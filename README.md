The [report/thesis](FYP_Report.pdf) is included in this repository. It contains an introduction, technical background, design and analysis of the model, testing and evaluation, and conclusions.


## Unconstrained PPORank Readme:

## PPORank usage
The implementation of PPORank is in "main.py":
(cpu version) :
The following is an exmaple for GDSC dataset when training with 16 actors, projection dimension of 100 without normalize y 
more details can be found in arguments.py

```
python main.py   --num_processes 16  --nlayers_deep 2 --Data GDSC_ALL --analysis FULL  --algo ppo --f 100  --normalize_y 
```
model logs dir : ./logs

model saved dir: ./Saved

model prediction saved dir: ./results

The clean data could be found in 

[Data Sharing](https://drive.google.com/drive/folders/1-YcEcRP6IObhT8ojes9L29Z54P-japjJ?usp=sharing)

## Scripts for runing the experiments
Download and preprocess the GDSC dataset:
```
python ./preprocess/load_dataset.py load_GDSC.txt
```
Split the data for training and testing, create folds for cross-validation and Pretrain the MF layers' weight
```
python prepare.py config.yaml

```
Runing the experiments on ppo (with config file "./configs/configS_base.yaml"):

```
python results.py > results_ppo.txt

```
PPO experiment with TCGA cohort

```
 python load_TCGA.py
 
 python results_TCGA.py ./TCGA/TCGA_BRCA.npz 
```


## Constrained Readme:

#### Dependencies in requirements.txt

### Usage:

The preprocessed data containing just the drugs with severity scores is in the GitHub already under GDSC_ALL and is split into train/test.

To run, similar to PPORank, but I've added more flags, specifically related to constraints

Need --constrained=True  --target_pen=X

For lagrange loss: --lagrange_loss=True --lagrange_lambda=X --lambda_lr=X

For quadratic loss: --quadratic_loss=True --rho=X

For augmented lagrange loss: --augmented_lagrange_loss=True --lagrange_lambda=X --lambda_lr=X --rho=X

Example: python main.py --epochs=700 --lr=1.5e-4 --lr_sched_mult=15 --constrained=True --lagrange_loss=True --lagrange_lambda=0 --lambda_lr=5e-5 --target_pen=3.6
