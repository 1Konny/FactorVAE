"""solver.py"""

import time
from pathlib import Path

import visdom
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid
from torchvision import transforms

from utils import cuda
from model import FactorVAE_2D, FactorVAE_3D, Discriminator
from dataset import return_data


def original_vae_loss(x, x_recon, mu, logvar):
    batch_size = x.size(0)
    if batch_size == 0:
        recon_loss = 0
        kl_divergence = 0
    else:
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
        kl_divergence = -0.5*(1 + logvar - mu**2 - logvar.exp()).sum(1).mean()

    return recon_loss, kl_divergence


def permute_dims(z):
    assert z.dim() == 2

    B, d = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B)
        if z.is_cuda:
            perm = perm.cuda()

        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)


class Solver(object):
    def __init__(self, args):

        # Misc
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.global_iter = 0

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
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        if self.viz_on:
            self.viz = visdom.Visdom(env=self.viz_name, port=self.viz_port)
            self.viz_curves = visdom.Visdom(env=self.viz_name+'/train_curves', port=self.viz_port)
            self.win_D_z = None
            self.win_recon = None
            self.win_kld = None
            self.win_acc = None

        # Checkpoint
        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.viz_name)
        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.load_ckpt = args.load_ckpt
        if self.load_ckpt:
            self.load_checkpoint()

        # Data
        self.dset_dir = args.dset_dir
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.data_loader = return_data(args)

    def traverse(self):
        decoder = self.VAE.decode
        encoder = self.VAE.encode
        interpolation = torch.arange(-6, 6.1, 1)
        viz = visdom.Visdom(env=self.viz_name+'/traverse', port=self.viz_port)

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
            viz.images(samples, opts=dict(title=title), nrow=len(interpolation))

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


                if self.global_iter%1000 == 0:
                    soft_D_z = F.sigmoid(D_z)
                    soft_D_z_perm = F.sigmoid(D_z_perm)
                    disc_acc = ((soft_D_z >= 0.5).sum() + (soft_D_z_perm < 0.5).sum()).float()
                    disc_acc /= 2*self.batch_size
                    curve_data.append(torch.Tensor([self.global_iter,
                                                    disc_acc.data[0],
                                                    vae_recon_loss.data[0],
                                                    vae_kld.data[0],
                                                    soft_D_z.mean().data[0],
                                                    soft_D_z_perm.mean().data[0]]))

                if self.global_iter%5000 == 0:
                    self.save_checkpoint()
                    self.visualize(dict(image=[x_vae, x_recon], curve=curve_data))
                    print('[{}] vae_recon_loss:{:.3f} vae_kld:{:.3f} vae_tc_loss:{:.3f} D_tc_loss:{:.3f}'.format(
                        self.global_iter, vae_recon_loss.data[0], vae_kld.data[0], vae_tc_loss.data[0], D_tc_loss.data[0]))
                    curve_data = []

                if self.global_iter%100000 == 0:
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

        if self.win_D_z is None:
            self.win_D_z = self.viz_curves.line(
                                        X=curve_x,
                                        Y=curve_D_z,
                                        opts=dict(
                                            xlabel='iteration',
                                            ylabel='D(.)',
                                            legend=['D(z)', 'D(z_perm)']))
        else:
            self.win_D_z = self.viz_curves.line(
                                        X=curve_x,
                                        Y=curve_D_z,
                                        win=self.win_D_z,
                                        update='append',
                                        opts=dict(
                                            xlabel='iteration',
                                            ylabel='D(.)',
                                            legend=['D(z)', 'D(z_perm)']))

        if self.win_recon is None:
            self.win_recon = self.viz_curves.line(
                                        X=curve_x,
                                        Y=curve_recon,
                                        opts=dict(
                                            xlabel='iteration',
                                            ylabel='reconsturction loss',))
        else:
            self.win_recon = self.viz_curves.line(
                                        X=curve_x,
                                        Y=curve_recon,
                                        win=self.win_recon,
                                        update='append',
                                        opts=dict(
                                            xlabel='iteration',
                                            ylabel='reconsturction loss',))

        if self.win_acc is None:
            self.win_acc = self.viz_curves.line(
                                        X=curve_x,
                                        Y=curve_acc,
                                        opts=dict(
                                            xlabel='iteration',
                                            ylabel='discriminator accuracy',))
        else:
            self.win_acc = self.viz_curves.line(
                                        X=curve_x,
                                        Y=curve_acc,
                                        win=self.win_acc,
                                        update='append',
                                        opts=dict(
                                            xlabel='iteration',
                                            ylabel='discriminator accuracy',))

        if self.win_kld is None:
            self.win_kld = self.viz_curves.line(
                                        X=curve_x,
                                        Y=curve_kld,
                                        opts=dict(
                                            xlabel='iteration',
                                            ylabel='vae kl divergence',))
        else:
            self.win_kld = self.viz_curves.line(
                                        X=curve_x,
                                        Y=curve_kld,
                                        win=self.win_kld,
                                        update='append',
                                        opts=dict(
                                            xlabel='iteration',
                                            ylabel='vae kl divergence',))

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

    def save_checkpoint(self, filename='ckpt.tar', silent=True):
        model_states = {'D':self.D.state_dict(),
                        'VAE':self.VAE.state_dict()}
        optim_states = {'optim_D':self.optim_D.state_dict(),
                        'optim_VAE':self.optim_VAE.state_dict()}
        win_states = {'D_z':self.win_D_z,
                      'recon':self.win_recon,
                      'kld':self.win_kld,
                      'acc':self.win_acc}
        states = {'iter':self.global_iter,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = self.ckpt_dir.joinpath(filename)
        torch.save(states, file_path.open('wb+'))
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename='ckpt.tar'):
        file_path = self.ckpt_dir.joinpath(filename)
        if file_path.is_file():
            checkpoint = torch.load(file_path.open('rb'))
            self.global_iter = checkpoint['iter']
            self.win_D_z = checkpoint['win_states']['D_z']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_acc = checkpoint['win_states']['acc']
            self.VAE.load_state_dict(checkpoint['model_states']['VAE'])
            self.D.load_state_dict(checkpoint['model_states']['D'])
            self.optim_VAE.load_state_dict(checkpoint['optim_states']['optim_VAE'])
            self.optim_D.load_state_dict(checkpoint['optim_states']['optim_D'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
