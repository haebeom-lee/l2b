# Learning to Balance: Bayesian Meta-Learning for Imbalanced and Out-of-distribution Tasks
This is the **TensorFlow implementation** for the paper Learning to Balance: Bayesian Meta-Learning for Imbalanced and Out-of-distribution Tasks (**ICLR 2020**, **oral presentation**) : https://openreview.net/pdf?id=rkeZIJBYvr.

You can reproduce the results of Table 1 in the main paper.

## Abstract
<img align="middle" width="700" src="https://github.com/haebeom-lee/l2b/blob/master/images/concept.png">

While tasks could come with varying the number of instances and classes in realistic settings, the existing meta-learning approaches for few-shot classification assume that number of instances per task and class is fixed. Due to such restriction, they learn to equally utilize the meta-knowledge across all the tasks, even when the number of instances per task and class largely varies. Moreover, they do not consider distributional difference in unseen tasks, on which the meta-knowledge may have less usefulness depending on the task relatedness. To overcome these limitations, we propose a novel meta-learning model that adaptively balances the effect of the meta-learning and task-specific learning within each task. Through the learning of the balancing variables, we can decide whether to obtain a solution by relying on the meta-knowledge or task-specific learning. We formulate this objective into a Bayesian inference framework and tackle it using variational inference. We validate our Bayesian Task-Adaptive Meta-Learning (Bayesian TAML) on two realistic task- and class-imbalanced datasets, on which it significantly outperforms existing meta-learning approaches. Further ablation study confirms the effectiveness of each balancing component and the Bayesian learning framework.

__Contribution of this work__
- We consider a novel problem of meta-learning under a __realistic task distribution__, where the number of instances across classes and tasks could largely vary, or the unseen tasks at meta-test time are largely different from the seen tasks.
- For effective meta-learning with such imbalances, we propose a Bayesian task-adaptive meta-learning (__Bayesian TAML__) framework that can adaptively adjust the effect of the meta-learner and the task-specific learner, differently for each task and class.
- We validate our model on realistic __imbalanced few-shot classification tasks__ with __a varying number of shots per task and class__ and show that it significantly outperforms existing meta-learning models.

__Structure of the posterior inference network__
<img align="middle" width="700" src="https://github.com/haebeom-lee/l2b/blob/master/images/encoder.png">

## Prerequisites
- Python 3.5 (Anaconda)
- Tensorflow 1.12.0
- CUDA 9.0
- cudnn 7.6.5

If you are not familiar with preparing conda environment, please follow the below instructions:
```
$ conda create --name py35 python=3.5
$ conda activate py35
$ pip install --upgrade pip
$ pip install tensorflow-gpu==1.12.0
$ conda install -c anaconda cudatoolkit=9.0
$ conda install -c anaconda cudnn
```

And for data preprocessing,
```
$ pip install tqdm
$ pip install requests
$ pip install Pillow
$ pip install scipy
```

### Data Preparation
Go to the folder of each dataset (i.e. ```data/cifar```, ```data/svhn```, ```data/mimgnet```, or ```data/cub```) and run ```python get_data.py``` there. For example, to download CIFAR-FS dataset and preprocess it,
```
$ cd ./data/cifar
$ python get_data.py
```
It will take some time to download and preprocess each dataset.

## Run
Bash script for running Bayesian TAML model. (You may use only one of the options ```--omega_on```,  ```--gamma_on```, and ```--z_on``` in order to reproduce the ablation studies in the main paper.)

1. __CIFAR / SVHN__ experiment
```
# Meta-training
$ python main.py \
  --gpu_id 0 \
  --savedir "./results/cifar/taml" --id_dataset 'cifar' --ood_dataset 'svhn' \
  --mode 'meta_train' --metabatch 4 --n_steps 5 --way 5 --max_shot 50 --query 15 \
  --n_train_iters 50000 --meta_lr 1e-3 \
  --alpha_on --omega_on --gamma_on --z_on

# Meta-testing
$ python main.py \
  --gpu_id 0 \
  --savedir "./results/cifar/taml" --id_dataset 'cifar' --ood_dataset 'svhn' \
  --mode 'meta_test' --metabatch 4 --n_steps 10 --way 5 --max_shot 50 --query 15 \
  --n_test_episodes 1000 \
  --alpha_on --omega_on --gamma_on --z_on --n_mc_samples 10
```

2. __miniImageNet /CUB__ experiment
```
# Meta-training
$ python main.py \
  --gpu_id 0 \
  --savedir "./results/mimgnet/taml" --id_dataset 'mimgnet' --ood_dataset 'cub' \
  --mode 'meta_train' --metabatch 1 --n_steps 5 --way 5 --max_shot 50 --query 15 \
  --n_train_iters 80000 --meta_lr 1e-4 \
  --alpha_on --omega_on --gamma_on --z_on

# Meta-testing
$ python main.py \
  --gpu_id 0 \
  --savedir "./results/mimgnet/taml" --id_dataset 'mimgnet' --ood_dataset 'cub' \
  --mode 'meta_test' --metabatch 1 --n_steps 10 --way 5 --max_shot 50 --query 15 \
  --n_test_episodes 1000 \
  --alpha_on --omega_on --gamma_on --z_on --n_mc_samples 10
```
- Take a look at the folder ```./runfiles``` for the bash script files of MAML and Meta-SGD models.

## Results
The results in the main paper (average over three independent runs, total 9000 (=3 x 3000) episodes):
|       | CIFAR-FS| SVHN | miniImageNet| CUB |
| ------| ---------------- | ----------------- | ------------------ | ------------------- |
| MAML | 71.55±0.23          | 45.17±0.22             | 66.64±0.22               | 65.77±0.24           |
| Meta-SGD | 72.71±0.21          | 46.45±0.24             | 69.95±0.20               | 65.94±0.22           |
| Bayesian-TAML | __75.15±0.20__         | __51.87±0.23__             | __71.46±0.19__               | __71.71±0.21__           |

The results from running this repo (average over single run, total 1000 episodes):

|       | CIFAR-FS| SVHN | miniImageNet| CUB |
| ------| ---------------- | ----------------- | ------------------ | ------------------- |
| MAML | 72.23±0.67         | 47.19±0.63             | 66.95±0.71              | 66.82±0.73           |
| Meta-SGD | 72.93±0.66          | 47.63±0.73             | 68.04±0.67      | 66.45±0.63           |
| Bayesian-TAML | __74.97±0.62__          | __52.25±0.68__             | __71.27±0.59__
| __72.89±0.62__           |

### Balancing Variables
While running the code, you can see the tendency of the balancing variables every 1000 iterations. Below shows the example tendency of __gamma__ for each layer over 10 randomly sampled tasks. As you can see, gamma increases as the task size (N) gets larger.
```
*** Gamma for task imbalance ***
              conv1 conv2 conv3 conv4 dense
task 1: N= 57 0.772 0.775 0.523 0.627 9.438
task 6: N= 75 0.807 0.917 0.834 0.851 4.897
task 3: N= 88 0.785 0.797 0.562 0.658 8.516
task 7: N=112 0.815 0.932 0.895 0.882 4.591
task 8: N=115 0.829 1.001 1.120 1.010 3.469
task 9: N=141 0.831 0.988 1.091 0.990 3.654
task 5: N=142 0.831 0.992 1.104 0.997 3.602
task 0: N=149 0.827 0.961 0.999 0.939 4.094
task 2: N=185 0.853 1.071 1.435 1.162 2.672
task 4: N=245 0.853 1.073 1.443 1.166 2.656
```

Also, below shows the example tendency of __omega__ for each class. The left part shows the number of instances per class (C1, C2, ..., C5), and the right part shows the actual omega value for each class. As you can see, tail (or smaller) class is more emphasized with bigger omega, and vice versa.
```
*** Omega for class imbalance ***
         C1  C2  C3  C4  C5        C1    C2    C3    C4    C5
task 1:   1   4   6  17  29 --> 0.444 0.273 0.198 0.056 0.029
task 6:  15  15  15  15  15 --> 0.200 0.200 0.200 0.200 0.200
task 3:   1   4  17  31  35 --> 0.539 0.332 0.068 0.033 0.028
task 7:  13  13  18  20  48 --> 0.292 0.292 0.195 0.166 0.055
task 8:  23  23  23  23  23 --> 0.200 0.200 0.200 0.200 0.200
task 9:  14  20  26  38  43 --> 0.384 0.236 0.174 0.113 0.094
task 5:  14  22  26  33  47 --> 0.394 0.206 0.178 0.138 0.084
task 0:  13  13  28  47  48 --> 0.361 0.361 0.140 0.071 0.068
task 2:  37  37  37  37  37 --> 0.200 0.200 0.200 0.200 0.200
task 4:  49  49  49  49  49 --> 0.200 0.200 0.200 0.200 0.200
```
We summarize the behavior of them in the main paper as follows:
<img align="middle" width="700" src="https://github.com/haebeom-lee/l2b/blob/master/images/tendency.png">

## Citation
If you found the provided code useful, please cite our work.
```
@inproceedings{
    lee2020l2b,
    title={Learning to Balance: Bayesian Meta-Learning for Imbalanced and Out-of-distribution Tasks},
    author={Hae Beom Lee and Hayeon Lee and Donghyun Na and Saehoon Kim and Minseop Park and Eunho Yang and Sung Ju Hwang},
    booktitle={ICLR},
    year={2020}
}
```

### TODO
- Provide the code for Multi-dataset experiments
