import torch.nn as nn
import torch.nn.functional as F
import torch 

class EBM(nn.Module):
    def __init__(self, dim=2):
        super(EBM, self).__init__()
        # The normalizing constant logZ(θ)        
        self.z = nn.Parameter(torch.tensor([1000.0], requires_grad=True))  # Init logZ to 0; adjust if needed
        self.f = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 1),  
        )
        self.apply(self.weights_init)
        
    def forward(self, x):
        p = self.f(x) 
        return p
    
    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            # Xavier uniforme su pesi
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, -1000.0)

