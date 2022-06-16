# Bayesian VAEs for OoD detection

## Installation:
pip3 install -r requirements.txt

### Example Commands

*BBB VAE trained on CIFAR10 and tested on SVHN as OoD*

```python3 evaluate_results.py --result_folder=./SGHMCVAE_SVHN_results```

*SGHMC VAE trained on SVHN and tested on CIFAR10 as OoD*

```python3 evaluate_results.py --result_folder=./BBBVAE_CIFAR10_results```

*SWAG VAE trained on Fashion-MNIST and tested on MNIST as OoD*

```python3 evaluate_results.py --result_folder=./SWAGVAE_FMNIST_results```

