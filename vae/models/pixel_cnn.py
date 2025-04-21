import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    """
    Weight initialization for convolutional layers.
    Uses Xavier (Glorot) uniform for Conv weights, zeros for bias.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)   # Initialises weights for Conv using Xavier uniform
            if m.bias is not None:                   # If bias exists (for 1x1 or other convs)
                m.bias.data.fill_(0)                 # Set bias to zero
        except AttributeError:
            print("Skipping initialization of ", classname)

class GatedActivation(nn.Module):
    """
    Gated activation for PixelCNN (splits channel, applies tanh/sigmoid).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)      # Split channels into two halves
        return torch.tanh(x) * torch.sigmoid(y)   # Output is elementwise multiplication of tanh and sigmoid of splits


"""
Masked convolutional block for PixelCNN with vertical and horizontal stacks.
Includes gating and embedding for class conditioning.
"""
class MaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=40):
        super().__init__()
        self.mask_type = mask_type
        self.residual = residual
        self.class_cond_projection = nn.Linear(n_classes, 2 * dim)
        kernel_shp = (kernel // 2 + 1, kernel)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(dim, dim * 2, kernel_shp, 1, padding_shp)
        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)
        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(dim, dim * 2, kernel_shp, 1, padding_shp)
        self.horiz_resid = nn.Conv2d(dim, dim, 1)
        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1] = 0
        self.horiz_stack.weight.data[:, :, :, -1] = 0

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()
        h_proj = self.class_cond_projection(h).unsqueeze(-1).unsqueeze(-1)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert + h_proj)
        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)
        out = self.gate(v2h + h_horiz + h_proj)
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)
        return out_v, out_h
    

"""
Main PixelCNN network for grayscale or discrete-valued images.
"""
class PixelCNN(nn.Module):
    def __init__(self, input_dim=512, dim=64, n_layers=8, n_classes=40):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, dim)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True
            self.layers.append(MaskedConv2d(mask_type, dim, kernel, residual, n_classes))
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )
        
    def forward(self, x, label):
        shp = x.size() + (-1, )
        x = self.embedding(x.view(-1)).view(shp)
        x = x.permute(0, 3, 1, 2)
        x_v, x_h = x, x
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)
        return self.output_conv(x_h)
    
    def generate(self, label, shape=(8, 8), batch_size=1):
        param = next(self.parameters())
        x = torch.zeros((batch_size, *shape), dtype=torch.int64, device=param.device)
        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x