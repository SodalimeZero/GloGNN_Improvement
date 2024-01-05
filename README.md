# GloGNN_Improvement

This is the code of GloGNN(ICML2022)'s improvements for course *Advanced Artificial Intelligence (Sun Yat-sen University, Fall 2023)*. 

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:

- torch==1.10.0

- torch-geometric==2.0.2

- networkx==2.3

- scipy==1.5.4

- numpy==1.19.2

- matplotlib==3.1.1

- pandas==1.1.5

You can simply run 

```python
pip install -r requirements.txt
```

## Run pipeline for big-scale datasets
1. Entering the large-scale directory and download 6 big-scale datasets from the repository of [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale). Notice, you should rename the datasets and place them in the right directory.
```python
cd data
```

2. You can run any models implemented in 'models.py'. For examples, you can run our model on 'genius' dataset by the script:
```python
python main.py --dataset genius --sub_dataset None --model mlpnorm_improve
```

For more experiments running details, you can ref the running sh in the 'scripts/' directory.


## Run pipeline for new small-scale datasets
1. Download the new small-scale datasets from the repository of (https://github.com/yandex-research/heterophilous-graphs.git).

2. You can run our model like the script in the below:
```python
python train.py --dataset tolokers --model mlpnorm_improve 
```
For more experiments running details, you can ref the running sh in the 'scripts/' directory.


## Attribution

Parts of this code are based on the following repositories:

- [GloGNN](https://github.com/RecklessRonan/GloGNN.git)
