"""solver.py"""

import time
import os

import visdom
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid
from torchvision import transforms

from utils import cuda, mkdirs
from ops import original_vae_loss, permute_dims
from model import FactorVAE_2D, FactorVAE_3D, Discriminator
from dataset import return_data


class Solver(object):
    def __init__(self, args):
        # Misc
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.name = args.name
        self.max_iter = args.max_iter
        self.global_iter = 0

        # Data
        self.dset_dir = args.dset_dir
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)

        # Networks & Optimizers
        self.z_dim = args.z_dim
        self.gamma = args.gamma

        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE

        self.lr_D = args.lr_D
        self.beta1_D = args.beta1_D
        self.beta2_D = args.beta2_D

        if args.dataset == 'dsprites':
            self.VAE = cuda(FactorVAE_2D(self.z_dim), self.use_cuda)
        else:
            self.VAE = cuda(FactorVAE_3D(self.z_dim), self.use_cuda)
        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))

        self.D = cuda(Discriminator(self.z_dim), self.use_cuda)
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr_D,
                                  betas=(self.beta1_D, self.beta2_D))

        self.nets = [self.VAE, self.D]

        # Visdom
        self.viz_on = args.viz_on
        self.win_id = dict(D_z='win_D_z', recon='win_recon', kld='win_kld', acc='win_acc')
        if self.viz_on:
            self.viz_port = args.viz_port
            self.viz = visdom.Visdom(port=self.viz_port)
            if not self.viz.win_exists(env=self.name+'/lines', win=self.win_id['D_z']):
                self.viz_init()

        # Checkpoint
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
        mkdirs(self.ckpt_dir)

        self.load_ckpt = args.load_ckpt
        if self.load_ckpt:
            self.load_checkpoint()

    def train(self):
        self.net_mode(train=True)

        ones = cuda(torch.ones(self.batch_size), self.use_cuda)
        ones = Variable(ones)
        zeros = cuda(torch.zeros(self.batch_size), self.use_cuda)
        zeros = Variable(zeros)

        out = False
        while not out:
            start = time.time()
            curve_data = []
            for x_vae, x_disc in self.data_loader:
                self.global_iter += 1

                x_vae = Variable(cuda(x_vae, self.use_cuda))
                x_recon, mu, logvar, z = self.VAE(x_vae)
                vae_recon_loss, vae_kld = original_vae_loss(x_vae, x_recon, mu, logvar)

                D_z = self.D(z)
                #vae_tc_1 = F.binary_cross_entropy_with_logits(D_z, ones, -self.gamma)
                #vae_tc_2 = F.binary_cross_entropy_with_logits(D_z, zeros, self.gamma)
                #vae_tc_loss = vae_tc_1 + vae_tc_2
                vae_tc_loss = D_z.mean()*self.gamma

                vae_loss = vae_recon_loss + vae_kld + vae_tc_loss

                self.optim_VAE.zero_grad()
                vae_loss.backward(retain_graph=True)
                self.optim_VAE.step()

                x_disc = Variable(cuda(x_disc, self.use_cuda))
                z_prime = self.VAE(x_disc, no_dec=True)
                z_perm = permute_dims(z_prime)
                D_z_perm = self.D(z_perm.detach())
                D_tc_loss = F.binary_cross_entropy_with_logits(
                    torch.cat([D_z, D_z_perm]), torch.cat([ones, zeros]))

                self.optim_D.zero_grad()
                D_tc_loss.backward()
                self.optim_D.step()

                if self.global_iter%1 == 0:
                    soft_D_z = F.sigmoid(D_z)
                    soft_D_z_perm = F.sigmoid(D_z_perm)
                    disc_acc = ((soft_D_z >= 0.5).sum() + (soft_D_z_perm < 0.5).sum()).float()
                    disc_acc /= 2*self.batch_size
                    if self.viz_on:
                        curve_data.append(torch.Tensor([self.global_iter,
                                                        disc_acc.data[0],
                                                        vae_recon_loss.data[0],
                                                        vae_kld.data[0],
                                                        soft_D_z.mean().data[0],
                                                        soft_D_z_perm.mean().data[0]]))

                if self.global_iter%5 == 0:
                    self.save_checkpoint()
                    print('[{}] vae_recon_loss:{:.3f} vae_kld:{:.3f} vae_tc_loss:{:.3f} D_tc_loss:{:.3f}'.format(
                        self.global_iter, vae_recon_loss.data[0], vae_kld.data[0], vae_tc_loss.data[0], D_tc_loss.data[0]))
                    if self.viz_on:
                        self.visualize(dict(image=[x_vae, x_recon], curve=curve_data))
                        curve_data = []

                if self.global_iter%10 == 0:
                    if self.viz_on:
                        self.traverse()

                if self.global_iter >= self.max_iter:
                    out = True
                    break

            end = time.time()
            print('[time elapsed] {:.2f}s/epoch'.format(end-start))
        print("[Training Finished]")

    def visualize(self, data):
        x_vae, x_recon = data['image']
        curve_data = data['curve']

        sample_x = make_grid(x_vae.data.cpu(), normalize=False)
        sample_x_recon = make_grid(F.sigmoid(x_recon).data.cpu(), normalize=False)
        samples = torch.stack([sample_x, sample_x_recon], dim=0)
        self.viz.images(samples, opts=dict(title=str(self.global_iter)))

        curve_data = torch.stack(curve_data, dim=0)
        curve_x = curve_data[:, 0]
        curve_acc = curve_data[:, 1]
        curve_recon = curve_data[:, 2]
        curve_kld = curve_data[:, 3]
        curve_D_z = curve_data[:, 4:]

        self.viz.line(X=curve_x,
                      Y=curve_D_z,
                      env=self.name+'/lines',
                      win=self.win_id['D_z'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='D(.)',
                        legend=['D(z)', 'D(z_perm)']))
        self.viz.line(X=curve_x,
                      Y=curve_recon,
                      env=self.name+'/lines',
                      win=self.win_id['recon'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='reconstruction loss',))
        self.viz.line(X=curve_x,
                      Y=curve_acc,
                      env=self.name+'/lines',
                      win=self.win_id['acc'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='discriminator accuracy',))
        self.viz.line(X=curve_x,
                      Y=curve_kld,
                      env=self.name+'/lines',
                      win=self.win_id['kld'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='kl divergence',))

    def traverse(self):
        decoder = self.VAE.decode
        encoder = self.VAE.encode
        interpolation = torch.arange(-6, 6.1, 1)

        fixed_idx = 0

        fixed_img, random_img = self.data_loader.dataset.__getitem__(fixed_idx)
        fixed_img = Variable(cuda(fixed_img, self.use_cuda), volatile=True).unsqueeze(0)
        fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

        random_img = Variable(cuda(random_img, self.use_cuda), volatile=True).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        zero_z = Variable(cuda(torch.zeros(1, self.z_dim, 1, 1), self.use_cuda), volatile=True)
        random_z = Variable(cuda(torch.rand(1, self.z_dim, 1, 1), self.use_cuda), volatile=True)

        Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z, 'zero_z':zero_z}
        for key in Z.keys():
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z))
                    samples.append(sample)
            samples = torch.cat(samples, dim=0).data.cpu()
            title = '{}_row_traverse(iter:{})'.format(key, self.global_iter)
            self.viz.images(samples,
                            env=self.name+'/traverse',
                            opts=dict(title=title), nrow=len(interpolation))

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

    def viz_init(self):
        zero_init = torch.zeros([1])
        self.viz.line(X=zero_init,
                      Y=torch.stack([zero_init, zero_init], -1),
                      env=self.name+'/lines',
                      win=self.win_id['D_z'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='D(.)',
                        legend=['D(z)', 'D(z_perm)']))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'/lines',
                      win=self.win_id['recon'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='reconstruction loss',))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'/lines',
                      win=self.win_id['acc'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='discriminator accuracy',))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'/lines',
                      win=self.win_id['kld'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='kl divergence',))

    def save_checkpoint(self, ckptname='last', verbose=True):
        model_states = {'D':self.D.state_dict(),
                        'VAE':self.VAE.state_dict()}
        optim_states = {'optim_D':self.optim_D.state_dict(),
                        'optim_VAE':self.optim_VAE.state_dict()}
        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states}

        filepath = os.path.join(self.ckpt_dir, ckptname)
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        if verbose:
            print("=> saved checkpoint '{}' (iter {})".format(filepath, self.global_iter))

    def load_checkpoint(self, ckptname='last', verbose=True):
        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.global_iter = checkpoint['iter']
            self.VAE.load_state_dict(checkpoint['model_states']['VAE'])
            self.D.load_state_dict(checkpoint['model_states']['D'])
            self.optim_VAE.load_state_dict(checkpoint['optim_states']['optim_VAE'])
            self.optim_D.load_state_dict(checkpoint['optim_states']['optim_D'])

            if verbose:
                print("=> loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter))
        else:
            if verbose:
                print("=> no checkpoint found at '{}'".format(filepath))
