# FactorVAE
Pytorch implementation of FactorVAE proposed in Disentangling by Factorising([http://arxiv.org/abs/1802.05983])
<br>

### Dependencies
```
python 3.6.4
pytorch 0.3.1.post2
visdom
```
<br>

### Usage
initialize visdom
```
python -m visdom.server -p 55558
```
you can run codes using sh files
```
e.g.
sh run_celeba.sh
sh run_3dchairs.sh
```
or you can run your own experiments by setting parameters manually
```
e.g.
python main.py --dataset celeba --gamma 6.4 --lr_VAE 1e-4 --lr_D 5e-5 --z_dim 10 ...
```
check training process on the visdom server
```
localhost:55558
```
<br>

### Results
soon
<br>

### Reference
1. Disentangling by Factorising, Kim et al.

[http://arxiv.org/abs/1802.05983]: http://arxiv.org/abs/1802.05983
