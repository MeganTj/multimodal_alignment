# Understanding the Emergence of Multimodal Representation Alignment

This is the official codebase for the paper Understanding the Emergence of Multimodal Representation Alignment.

# Installation

After cloning the directory, initialize the submodule.

```
git submodule init; git submodule update
```

The repo is tested with Python=3.10.15 and PyTorch=2.5.0. A new environment can be created via:
```
conda create -n align python=3.10.15
conda activate align
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

Within the conda environment, dependencies for the vision-language experiments can be installed. 
```
cd platonic-rep-unique
pip install -r requirements.txt
```

We additionally provide experiment hyperparameters in `hyp` folder. To use them, move `hyp` to `experiments/hyp`.


# Synthetic Dataset

Download our synthetic dataset [here](https://drive.google.com/drive/folders/1pRMMlKKrWsomkbvGoopKImx5oIy3m3RI?usp=drive_link) and move the `data` folder under `synthetic/data`.
You can also generate synthetic datasets by running the below:

```
python synthetic/generate_prob_data.py --template-path synthetic/configs/rus_prob_template.yaml
```

To train the unimodal models and compute alignment using gpus 0-3, run the following from the root directory. See `scripts/tune_rus_prob_optuna.sh` for more details. 

```
bash ./scripts/tune_rus_prob_optuna.sh "0,1,2,3" scripts/align_configs/rus_prob.yaml mi=1.159 50 300 2
```

The experiments can be analyzed with the below script.

```
python ./scripts/analyze_rus.py
```

# MultiBench Datasets

Use the following links to download the processed affect datasets from [MultiBench](https://github.com/pliang279/MultiBench): [sarcasm](https://drive.google.com/drive/folders/1JFcX-NF97zu9ZOZGALGU9kp8dwkP7aJ7?usp=drive_link), 
[humor](https://drive.google.com/drive/folders/1Agzm157lciMONHOHemHRSySmjn1ahHX1?usp=drive_link), [mosi](https://drive.google.com/drive/folders/1uEK737LXB9jAlf9kyqRs6B9N6cDncodq?usp=drive_link), 
[mosei](https://drive.google.com/drive/folders/1A_hTmifi824gypelGobgl2M-5Rw9VWHv?usp=drive_link) and move the downloaded datasets to the `datasets` folder. For example, the path to MOSEI should be `datasets/mosei/mosei_senti_data.pkl`. Our AVMNIST dataset can be downloaded [here](https://drive.google.com/drive/folders/17vGI0voQyCTyhqDq3hQhhiwXACV7zXqU?usp=sharing).

An example workflow is as follows. First, run experiments (each bash command will run the same experiment with a different seed) as follows. See `./scripts/tune_real.sh` for more details on the arguments.

```
bash ./scripts/tune_real.sh 0 scripts/align_configs/sarcasm_norm.yaml 50 0 1 "classification" "classification" 2 &&
bash ./scripts/tune_real.sh 0 scripts/align_configs/sarcasm_norm.yaml 50 0 1 "classification" "classification" 22 && 
bash ./scripts/tune_real.sh 0 scripts/align_configs/sarcasm_norm.yaml 50 0 1 "classification" "classification" 42
```

After running experiments with multiple seeds, compute the alignment/performance correlation.

```
python ./scripts/evaluate_multiseed.py --exp-config scripts/align_configs/sarcasm_norm.yaml --modalities 0 1 
```

# Vision-Language

See `platonic-rep-unique` for instructions on setting up and running vision-language experiments.


# Additional Experiments

Experiments for other datasets can be run as follows.

```
bash ./scripts/tune_real.sh 0 scripts/align_configs/mosei_norm.yaml 50 0 1 "regression" "posneg-classification" 2
```

```
bash ./scripts/tune_real.sh 0 scripts/align_configs/mosi_norm.yaml 50 0 1 "regression" "posneg-classification" 2
```

```
bash ./scripts/tune_real.sh 0 scripts/align_configs/humor_norm.yaml 50 0 1 "classification" "classification" 2
```

```
bash ./scripts/tune_real.sh 0 scripts/align_configs/avmnist_mfcc_seq.yaml 100 0 1 "classification" "classification" 2
```
