import argparse
from datetime import datetime
import os
import numpy as np

import torch
from torch import autograd
from torch.nn import functional as F
import torch.nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from models.discriminator import Discriminator
from models.generator import Generator

from dataset import *

from train import *
from validation import *
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Simultaneous Training')
parser.add_argument('--data_dir', default='./data/', help='path to dataset')
parser.add_argument('--gantype', default = 'zerogp')
parser.add_argument("--batch_size", default = 1)
parser.add_argument("--gpu", default = 0)
parser.add_argument("--validation", default = 0, help = "evaluate model on validation set")
parser.add_argument("--load_model", default = None, type = str, help = "path of pretrained model")
parser.add_argument("--img_size_min", default = 25, type = int)
parser.add_argument("--img_size_max", default = 250, type = int)


def main():
    args = parser.parse_args()
    if args.load_model is not None:
        this_time_model = args.load_model
    else:
        this_time_model = f'SinGAN_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{args.gantype}'
    if os.path.isdir('./logs') is False:
        os.makedirs('./logs')
    if os.path.isdir('./results') is False:
        os.makedirs('./results')
    if args.load_model is None:
        os.makedirs(os.path.join('./logs', this_time_model))      
    if os.path.isdir(os.path.join('./results', this_time_model)) is False:
        os.makedirs(os.path.join('./results', this_time_model)) 

    args.log_dir = os.path.join('./logs', this_time_model)
    args.res_dir = os.path.join('./results', this_time_model)
    
    
    # datasets
    train_dataset, _ = get_dataset(args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1,
                                               shuffle = False, num_workers = 8,
                                               pin_memory= True)

    # models
    scale_factor = 4/3
    min_max_ratio = args.img_size_max / args.img_size_min
    args.num_scale = int(np.round(np.log(min_max_ratio)/np.log(scale_factor)))
    args.size_list = [int(args.img_size_min * scale_factor ** i) for i in range(args.num_scale + 1)]

    discriminator = Discriminator()
    generator = Generator(25, args.num_scale, scale_factor)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        discriminator = discriminator.cuda(0)
        generator = generator.cuda(0)

    # optimizers
    dis_opt = torch.optim.Adam(discriminator.discriminators[0].parameters(), 5e-4, (0.5, 0.999))
    gen_opt = torch.optim.Adam(generator.generators[0].parameters(), 5e-4, (0.5, 0.999))

    # load pretrained model
    args.stage = 0
    if args.load_model is not None:
        check_load = open(os.path.join(args.log_dir, "checkpoint.txt"), 'r')
        to_restore = check_load.readlines()[-1].strip()
        load_file = os.path.join(args.log_dir, to_restore)
        if os.path.isfile(load_file):
            print("=> loading checkpoint '{}'".format(load_file))
            checkpoint = torch.load(load_file, map_location='cpu')
            for _ in range(int(checkpoint['stage'])):
                generator.progress()
                discriminator.progress()
            networks = [discriminator, generator]
           
            torch.cuda.set_device(args.gpu)
            networks = [x.cuda(args.gpu) for x in networks]

            discriminator, generator, = networks
            
            args.stage = checkpoint['stage']
            print("stage: ",args.stage)
            discriminator.load_state_dict(checkpoint['D_state_dict'])
            generator.load_state_dict(checkpoint['G_state_dict'])
            dis_opt.load_state_dict(checkpoint['d_optimizer'])
            gen_opt.load_state_dict(checkpoint['g_optimizer'])
            print("=> loaded checkpoint '{}' (stage {})"
                  .format(load_file, checkpoint['stage']))
        else:
            print("=> no checkpoint found at '{}'".format(args.log_dir))

    # Training
    fixed_latents = [F.pad(torch.randn(args.batch_size, 3, args.size_list[0], args.size_list[0]), [5,5,5,5], value = 0)]
    zero_latents = [F.pad(torch.zeros(args.batch_size, 3, args.size_list[idx], args.size_list[idx]), [5,5,5,5], value = 0) for idx in range(1, args.num_scale + 1)]
    fixed_latents = fixed_latents + zero_latents
    
    if args.validation:
        validation(train_loader, generator, discriminator, args.stage, fixed_latents, args)
        return
    else:        
        for stage_idx in range(args.stage, args.num_scale + 1):
            
            train(train_loader, generator, discriminator, dis_opt, gen_opt, stage_idx, fixed_latents, args)
            validation(train_loader, generator, discriminator, stage_idx, fixed_latents, args)
            discriminator.progress()
            generator.progress()
            if torch.cuda.is_available():
                discriminator = discriminator.cuda(0)
                generator = generator.cuda(0)
                
            # Update the networks at finest scale
            for net_idx in range(generator.current_scale):
                for param in generator.generators[net_idx].parameters():
                    param.requires_grad = False
                for param in discriminator.discriminators[net_idx].parameters():
                    param.requires_grad = False

            dis_opt = torch.optim.Adam(discriminator.discriminators[discriminator.current_scale].parameters(),
                                        5e-4, (0.5, 0.999))
            gen_opt = torch.optim.Adam(generator.generators[generator.current_scale].parameters(),
                                        5e-4, (0.5, 0.999))


            if stage_idx == 0:            
                check_list = open(os.path.join(args.log_dir, "checkpoint.txt"), "a+")

            save_checkpoint({
                'stage': stage_idx + 1,
                'D_state_dict': discriminator.state_dict(),
                'G_state_dict': generator.state_dict(),
                'd_optimizer': dis_opt.state_dict(),
                'g_optimizer': gen_opt.state_dict()
            }, check_list, args.log_dir, stage_idx + 1)

            if stage_idx == args.num_scale:
                check_list.close()


        
if __name__ == '__main__':
    main()