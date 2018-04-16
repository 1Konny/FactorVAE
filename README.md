# FactorVAE
Pytorch implementation of FactorVAE proposed in Disentangling by Factorising, Kim et al.([http://arxiv.org/abs/1802.05983])
<br>

### Dependencies
```
python 3.6.4
pytorch 0.3.1.post2
visdom
```
<br>

### Datasets
1. 3D Chairs Dataset([Aubry et al.])([click to download])
2. CelebA Dataset([website])

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

### Results - 3D Chairs Dataset
each row represents each dimension of latent vector z(i.e. z_j, j=1, ..., 10)

#### Latent Space Traverse 1(representation from true image 1)
![3dchairs_traverse_img1](misc/3dchairs_traverse_img1.jpg)
#### Latent Space Traverse 2(representation from true image 2)
![3dchairs_traverse_img2](misc/3dchairs_traverse_img2.jpg)
#### Latent Space Traverse 3(representation from zero vector)
![3dchairs_traverse_zero_vector](misc/3dchairs_traverse_zero_vector.jpg)
#### Latent Space Traverse 4(representation from normal distribution)
![3dchairs_traverse_normal_dist](misc/3dchairs_traverse_normal_dist.jpg)

<br>

### Reference
1. Disentangling by Factorising, Kim et al.([http://arxiv.org/abs/1802.05983])


[http://arxiv.org/abs/1802.05983]: http://arxiv.org/abs/1802.05983
[Aubry et al.]: http://www.di.ens.fr/~josef/publications/aubry14.pdf
[click to download]: https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar
