from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F
import torch

class Generator(nn.Module):
    def __init__(self, img_size_min, num_scale, scalefactor = 4/3):
        super(Generator, self).__init__()
        self.hidden = 32
        self.current_scale = 0
        self.img_size_min = img_size_min
        self.scalefactor = scalefactor
        self.num_scale = num_scale

        self.size_list = [int(self.img_size_min * (self.scalefactor ** i)) for i in range(num_scale + 1)]
        print(f"size_list : {self.size_list}")

        self.generators = nn.ModuleList()

        temp_gene = nn.ModuleList()

        temp_gene.append(nn.Sequential(nn.Conv2d(3, self.hidden, 3, 1),
                                             nn.BatchNorm2d(self.hidden),
                                             nn.LeakyReLU(0.2)))
        for _ in range(3):
            temp_gene.append(nn.Sequential(nn.Conv2d(self.hidden, self.hidden, 3, 1),
                                                 nn.BatchNorm2d(self.hidden),
                                                 nn.LeakyReLU(0.2)))
        temp_gene.append(nn.Sequential(nn.Conv2d(self.hidden, 3, 3, 1),
                                             nn.Tanh()))
        
        temp_gene = nn.Sequential(*temp_gene)

        self.generators.append(temp_gene)

    def forward(self, z, img = None):
        ret = []
        out = None
        if img != None:
            out = img
        else:
            out = self.generators[0](z[0])
        ret.append(out)
        for i in range(1, self.current_scale + 1):
            out = F.interpolate(out, (self.size_list[i], self.size_list[i]), mode = 'bilinear', align_corners = True)
            prev = out
            out = F.pad(out, [5,5,5,5], value = 0)
            out += z[i]
            out = self.generators[i](out) + prev
            ret.append(out)
            
        return ret

    def progress(self):
        self.current_scale += 1

        if self.current_scale % 4 == 0:
            self.hidden *= 2
        temp_gene = nn.ModuleList()

        temp_gene.append(nn.Sequential(nn.Conv2d(3, self.hidden, 3, 1),
                                             nn.BatchNorm2d(self.hidden),
                                             nn.LeakyReLU(0.2)))
        for _ in range(3):
            temp_gene.append(nn.Sequential(nn.Conv2d(self.hidden, self.hidden, 3, 1),
                                                 nn.BatchNorm2d(self.hidden),
                                                 nn.LeakyReLU(0.2)))
        temp_gene.append(nn.Sequential(nn.Conv2d(self.hidden, 3, 3, 1),
                                             nn.Tanh()))
        
        temp_gene = nn.Sequential(*temp_gene)
        

        if self.current_scale % 4 != 0:
            # continue start learning from prev generator's parameters
            temp_gene.load_state_dict(self.generators[-1].state_dict())

        self.generators.append(temp_gene)
        
