from torch import nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.hidden = 32
        self.current_scale = 0

        self.discriminators = nn.ModuleList()

        temp_disc = nn.ModuleList()

        temp_disc.append(nn.Sequential(nn.Conv2d(3, self.hidden, 3, 1, 1),
                                       nn.LeakyReLU(0.2)))
        for _ in range(3):
            temp_disc.append(nn.Sequential(nn.Conv2d(self.hidden, self.hidden, 3, 1, 1),
                                           nn.BatchNorm2d(self.hidden),
                                           nn.LeakyReLU(0.2)))

        temp_disc.append(nn.Sequential(nn.Conv2d(self.hidden, 1, 3, 1, 1)))
        
        temp_disc = nn.Sequential(*temp_disc)

        self.discriminators.append(temp_disc)

    def forward(self, x):
        out = self.discriminators[self.current_scale](x)
        return out

    def progress(self):
        self.current_scale += 1
        if self.current_scale % 4 == 0:
            self.hidden *= 2

        temp_disc = nn.ModuleList()
        temp_disc.append(nn.Sequential(nn.Conv2d(3, self.hidden, 3, 1, 1),
                                       nn.LeakyReLU(0.2)))
        for _ in range(3):
            temp_disc.append(nn.Sequential(nn.Conv2d(self.hidden, self.hidden, 3, 1, 1),
                                           nn.BatchNorm2d(self.hidden),
                                           nn.LeakyReLU(0.2)))
        temp_disc.append(nn.Sequential(nn.Conv2d(self.hidden, 1, 3, 1, 1)))
        
        temp_disc = nn.Sequential(*temp_disc)

        if self.current_scale % 4 != 0:
            # continue start learning from prev discriminator's parameters
            temp_disc.load_state_dict(self.discriminators[-1].state_dict())

        self.discriminators.append(temp_disc)
        print("PROGRESSION DONE")