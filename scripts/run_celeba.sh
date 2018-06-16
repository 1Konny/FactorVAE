#! /bin/sh

python main.py --dataset celeba --num_workers 4 --batch_size 64 \
               --output_save True --viz_on True \
               --viz_ll_iter 1000 --viz_la_iter 5000 \
               --viz_ra_iter 10000 --viz_ta_iter 10000 \
               --ckpt_save_iter 10000 --max_iter 1e6 \
               --lr_VAE 1e-4 --beta1_VAE 0.9 --beta2_VAE 0.999 \
               --lr_D 1e-5 --beta1_D 0.5 --beta2_D 0.9 \
               --name $1 --z_dim 10 --gamma 6.4 --ckpt_load last
