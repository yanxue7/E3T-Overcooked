# E3T

This work propose an efficient end-to-end trainning approach that combines the mixed partner policy and partner modeling module for zero-shot coordiantion on Overcooked. Our work builds on the codebase of ”On the Utility of Learning about Humans for Human-AI Coordination”, thus we only provide the modified and added source codes in this repository. To run our experiments, please follow the instructions of the original codebase and then merge our files with it.

### Installation

```
conda create -n harl python=3.7
conda activate harl
pip install -r environment.yml
conda install mpi4py
conda install certifi
pip install opencv-python
cd baselines
pip install -e .
cd stable-baselines 
pip install -e .
cd overcooked_ai
pip install -e .
export PYTHONPATH=$PYTHONPATH:/$root path$/human_coordination/
export PYTHONPATH=$PYTHONPATH:/$root path$/human_coordination/human_aware_rl


```

### Train E3T 

```
cd human_aware_rl
sh experiments/ppo_sp_experiments.sh

```

Source code introduction

```
encoder_ppo.py parameters
encoder_ppo2.py training process
encoder_model.py compute loss
encoder_polices.py networks
encoder_runner.py collecting datas
```

```

```

### Evaluate with human proxy

```
cd human_aware_rl
python experiments/encoder_ppo_sp_experiments_lstm.py
```

