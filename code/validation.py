import torch
import torchvision.utils as vutils
import os
from torch.nn import functional as F

def validation(data_loader, generator, discriminator, stage_idx, z, args):
    discriminator.eval()
    generator.eval()

    val_iter = iter(data_loader)
    origin = next(val_iter)

    if torch.cuda.is_available():
        origin = origin.cuda(0, non_blocking = True)
    x_in = F.interpolate(origin, (args.size_list[stage_idx], args.size_list[stage_idx]), mode='bilinear', align_corners=True)
    vutils.save_image(x_in.detach().cpu(), os.path.join(args.res_dir, 'ORG_{}.png'.format(stage_idx)),
                      nrow=1, normalize=True)
    x_in_list = [x_in]
    for xidx in range(1, stage_idx + 1):
        x_tmp = F.interpolate(origin, (args.size_list[xidx], args.size_list[xidx]), mode='bilinear', align_corners=True)
        x_in_list.append(x_tmp)

    for z_idx in range(len(z)):
        z[z_idx] = z[z_idx].cuda(0, non_blocking=True)

    with torch.no_grad():
        out = generator(z)

        # calculate rmse for each scale
        rmse_list = [1.0]
        for rmseidx in range(1, stage_idx + 1):
            rmse = torch.sqrt(F.mse_loss(out[rmseidx], x_in_list[rmseidx]))
            if args.validation:
                rmse /= 100.0
            rmse_list.append(rmse)
        if len(rmse_list) > 1:
            rmse_list[-1] = 0.0
        if args.validation:
            vutils.save_image(out[-1].detach().cpu(), os.path.join(args.res_dir, 'validation_REC_{}.png'.format(stage_idx)),
                              nrow=1, normalize=True)
        else:
            vutils.save_image(out[-1].detach().cpu(), os.path.join(args.res_dir, 'REC_{}.png'.format(stage_idx)),
                              nrow=1, normalize=True)

        for k in range(50):
            z_list = [F.pad(rmse_list[z_idx] * torch.randn(1, 3, args.size_list[z_idx],
                                               args.size_list[z_idx]).cuda(0, non_blocking=True),
                            [5, 5, 5, 5], value=0) for z_idx in range(stage_idx + 1)]
            x_fake_list = generator(z_list)
            if args.validation:
                vutils.save_image(x_fake_list[-1].detach().cpu(), os.path.join(args.res_dir, 'validation_GEN_{}_{}.png'.format(stage_idx, k)),
                                  nrow=1, normalize=True)
            else:
                vutils.save_image(x_fake_list[-1].detach().cpu(), os.path.join(args.res_dir, 'GEN_{}_{}.png'.format(stage_idx, k)),
                                  nrow=1, normalize=True)

