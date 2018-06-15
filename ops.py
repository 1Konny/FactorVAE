import torch
import torch.nn.functional as F

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
