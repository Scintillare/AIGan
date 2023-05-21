# AIGan PyTorch
Unofficial implementation of AIGan from paper [AI-GAN: Attack-Inspired Generation of Adversarial Examples](https://arxiv.org/abs/2002.02196).

Code is based on [mathcbc/advGAN_pytorch](https://github.com/mathcbc/advGAN_pytorch) and [ctargon/AdvGAN-tf](https://github.com/ctargon/AdvGAN-tf) with my modifications.


## set up environment

Using conda:

```shell
conda env create --name aigan --file=conda-requirements.yml
```

Using pip:

```shell
python -m venv ./.env
pip install -r pip-requirements.txt
```

## training the target model

```shell
python3 train_target_model.py
```

## training the AIGAN

```shell
python3 main.py
```

## testing adversarial examples

```shell
python3 test_adversarial_examples.py
```

