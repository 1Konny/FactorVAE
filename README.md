# FactorVAE
Pytorch implementation of FactorVAE proposed in Disentangling by Factorising, Kim et al.([http://arxiv.org/abs/1802.05983])
<br>

### Dependencies
```
python 3.6.4
pytorch 0.4.0 (or check pytorch-0.3.1 branch for pytorch 0.3.1)
visdom
tqdm
```
<br>

### Datasets
1. 2D Shapes(dsprites) Dataset
```
sh scripts/prepare_data.sh dsprites
```
2. 3D Chairs Dataset
```
sh scripts/prepare_data.sh 3DChairs
```
3. CelebA Dataset([download])
```
# first download img_align_celeba.zip and put in data directory like below
└── data
    └── img_align_celeba.zip

# then run scrip file
sh scripts/prepare_data.sh CelebA
```

then data directory structure will be like below<br>
```
.
└── data
    └── dsprites-dataset
        └── dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
    ├── 3DChairs
        └── images
            ├── 1_xxx.png
            ├── 2_xxx.png
            ├── ...
    ├── CelebA
        └── img_align_celeba
            ├── 000001.jpg
            ├── 000002.jpg
            ├── ...
            └── 202599.jpg
    └── ...
```
NOTE: I recommend to preprocess image files(e.g. resizing) BEFORE training and avoid preprocessing on-the-fly.
<br>

### Usage
initialize visdom
```
python -m visdom.server
```
you can reproduce results below as follows
```
e.g.
sh scripts/run_celeba.sh $RUN_NAME
sh scripts/run_dsprites_gamma6p4.sh $RUN_NAME
sh scripts/run_dsprites_gamma10.sh $RUN_NAME
sh scripts/run_3dchairs.sh $RUN_NAME
```
or you can run your own experiments by setting parameters manually
```
e.g.
python main.py --name run_celeba --dataset celeba --gamma 6.4 --lr_VAE 1e-4 --lr_D 5e-5 --z_dim 10 ...
```
check training process on the visdom server
```
localhost:8097
```
<br>

### Results - 2D Shapes(dsprites) Dataset
#### Reconstruction($\gamma$=6.4)
<p align="center">
<img src=misc/2DShapes_reconstruction_gamma6p4_700000.jpg>
</p>

#### Latent Space Traverse($\gamma$=6.4)
<p align="center">
<img src=misc/2DShapes_fixed_ellipse_gamma6p4_700000.gif>
<img src=misc/2DShapes_fixed_square_gamma6p4_700000.gif>
<img src=misc/2DShapes_fixed_heart_gamma6p4_700000.gif>
<img src=misc/2DShapes_random_img_gamma6p4_700000.gif>
</p>
<br>

#### Latent Space Traverse($\gamma$=10)
<p align="center">
<img src=misc/2DShapes_fixed_ellipse_gamma10_1000000.gif>
<img src=misc/2DShapes_fixed_square_gamma10_1000000.gif>
<img src=misc/2DShapes_fixed_heart_gamma10_1000000.gif>
<img src=misc/2DShapes_random_img_gamma10_1000000.gif>
</p>

### Results - CelebA Dataset
#### Reconstruction
<p align="center">
<img src=misc/CelebA_reconstruction_850000.jpg>
</p>

#### Latent Space Traverse
<p align="center">
<img src=misc/CelebA_traverse_850000.png>
<img src=misc/CelebA_fixed_1_850000.gif>
<img src=misc/CelebA_fixed_2_850000.gif>
<img src=misc/CelebA_fixed_3_850000.gif>
<img src=misc/CelebA_fixed_4_850000.gif>
</p>
<br>


### Results - 3D Chairs Dataset
#### Reconstruction
<p align="center">
<img src=misc/3DChairs_reconstruction_1000000.jpg>
</p>

#### Latent Space Traverse
<p align="center">
<img src=misc/3DChairs_traverse_1000000.png>
</p>
<br>

### Reference
1. Disentangling by Factorising, Kim et al.([http://arxiv.org/abs/1802.05983])


[http://arxiv.org/abs/1802.05983]: http://arxiv.org/abs/1802.05983
[download]: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
