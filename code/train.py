from tqdm import trange
from torch.nn import functional as F
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from utils import *
import torchvision.utils as vutils

from os import path




def train(data_loader, generator, discriminator, d_opt, g_opt, stage_idx, z, args):

    generator.train()
    discriminator.train()

    epochs = 2000
    decay_lr = 1600

    train_it = iter(data_loader)
    origin = next(train_it)

    if torch.cuda.is_available():
        for z_idx in range(len(z)):
            z[z_idx] = z[z_idx].cuda(0, non_blocking=True)
        origin = origin.cuda(0, non_blocking = True)
    
    x_in = F.interpolate(origin, (args.size_list[stage_idx], args.size_list[stage_idx]), mode = 'bilinear', align_corners = True)
    vutils.save_image(x_in.detach().cpu(), path.join(args.res_dir, "ORGINAL_{}.png".format(stage_idx)), nrow = 1, normalize = True)

    x_in_list = [x_in]
    for idx in range(1, stage_idx + 1):
        x_in_list.append(F.interpolate(origin, (args.size_list[idx], args.size_list[idx]), mode = 'bilinear', align_corners = True))

    tqdm_train = trange(0, epochs, initial = 0, total = epochs)

 
    d_losses = AverageMeter()
    g_losses = AverageMeter()
    for i in tqdm_train:
        if i == decay_lr:
            for params in d_opt.param_groups:
                params['lr'] *= 0.1

            for params in g_opt.param_groups:
                params['lr'] *= 0.1
            print("Generator and Discriminator's learning rate updated")

        # update Generator's weights
        for _ in range(3):
            g_opt.zero_grad()

            out = generator(z)            

            g_mse = F.mse_loss(out[-1], x_in)

            sqrt_rmse = [1.0]
            # calc rmse for every scale (except stage 0)
            for idx in range(1, stage_idx + 1):
                sqrt_rmse.append(torch.sqrt(F.mse_loss(out[idx], x_in_list[idx])))

            # 각 scale의 sqrt_rmse의 값을 랜덤 값에 곱해준 리스트 생성

            z_list = [F.pad(sqrt_rmse[z_idx] * torch.randn(args.batch_size, 3, args.size_list[z_idx],
                                               args.size_list[z_idx]).cuda(args.gpu, non_blocking=True),
                            [5, 5, 5, 5], value=0) for z_idx in range(stage_idx + 1)]
            
            x_fake_list = generator(z_list)
            g_fake_logit = discriminator(x_fake_list[-1])
            if torch.cuda.is_available():
                ones = torch.ones_like(g_fake_logit).cuda(0)
            else:
                ones = torch.ones_like(g_fake_logit)

            if args.gantype == 'wgangp':
                # wgan gp
                g_fake = -torch.mean(g_fake_logit, (2, 3))
                g_loss = g_fake + 10.0 * g_mse
            elif args.gantype == 'zerogp':
                # zero centered GP
                g_fake = F.binary_cross_entropy_with_logits(g_fake_logit, ones, reduction='none').mean()
                g_loss = g_fake + 100.0 * g_mse

            g_loss.backward()
            g_opt.step()
            g_losses.update(g_loss.item(), x_in.size(0))

        # Update Discriminator's weights
        for _ in range(3):
            x_in.requires_grad = True

            d_opt.zero_grad()
            x_fake_list = generator(z_list)

            d_fake_logit = discriminator(x_fake_list[-1].detach())
            d_real_logit = discriminator(x_in)

            if torch.cuda.is_available():
                ones = torch.ones_like(d_real_logit).cuda(0)
                zeros = torch.zeros_like(d_fake_logit).cuda(0)

            if args.gantype == 'wgangp':
                # wgan gp
                d_fake = torch.mean(d_fake_logit, (2, 3))
                d_real = -torch.mean(d_real_logit, (2, 3))
                d_gp = compute_grad_gp_wgan(discriminator, x_in, x_fake_list[-1], 0)
                d_loss = d_real + d_fake + 0.1 * d_gp

            elif args.gantype == 'zerogp':
                # zero centered GP
                d_fake = F.binary_cross_entropy_with_logits(d_fake_logit, zeros, reduction='none').mean()
                d_real = F.binary_cross_entropy_with_logits(d_real_logit, ones, reduction='none').mean()
                d_gp = compute_grad_gp(torch.mean(d_real_logit, (2, 3)), x_in)
                d_loss = d_real + d_fake + 10.0 * d_gp

            d_loss.backward()
            d_opt.step()
            d_losses.update(d_loss.item(), x_in.size(0))

        tqdm_train.set_description(f'Stage: [{stage_idx}/{args.num_scale}] Avg Loss: D[{d_losses.avg : .3f}] G[{g_losses.avg : .3f}] RMSE[{sqrt_rmse[-1] : .3f}]')