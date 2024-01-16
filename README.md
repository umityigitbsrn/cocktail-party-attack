# cocktail-party-attack

The implementation of the paper "Cocktail Party Attack: Breaking Aggregation-Based Privacy in Federated Learning using Independent Component Analysis". In order to use the functions implemented, please refer directly to the implementation. For the attack implementation, there will be explanations for each arguement in both `cocktail_party_attack/attack.py` and `cocktail_party_attack/hyperparameter_optimization.py`.

## setting the environment

### using conda

By using the below command you can create a conda environment to use this repository

```commandline
conda env create -f environment.yml
```

## pretraining before attack

To train a model and save it, one can use `train_all` function located in `cocktail_party_attack/train.py`. However, before using it the [data loaders should be received](#retrieving-data-loaders).

One can utilize this function by

```python
from cocktail_party_attack import train_all
```

### retrieving data loaders

For now there are functions to retrieve data loaders of datasets MNIST, CIFAR10, CIFAR100, and Tiny ImageNet. Those functions are `load_mnist_dataloaders`, `load_cifar10_dataloaders`, `load_cifar100_dataloaders`, and `load_tiny_imagenet_dataloaders`, respectively. All the functions are located under `cocktail_party_attack/data.py` and one can use those functions using

```python
from cocktail_party_attack import load_tiny_imagenet_dataloaders # function name 
```

## cocktail party attack

By using `CocktailPartyAttack` class defined in `cocktail_party_attack/cpa.py`, you can conduct attack py following the explanations. Also, you can conduct hyperparameter optimization experiments for attack performance depending on either mean PSNR, mean LPIPS or individual (target image) LPIPS metrics.

Again, you can import this class using

```python
from cocktail_party_attack import CocktailPartyAttack
```
