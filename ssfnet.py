import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout))),
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)
            x = mlp(x)
        return x


NUM_CLASS = 16


class SSFTTnet(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASS, num_tokens=4, dim=64,
                 depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(SSFTTnet, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8 * 28, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wV)
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()
        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, mask=None):
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        wa = rearrange(self.token_wA, 'b h w -> b w h')
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')
        A = A.softmax(dim=-1)
        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)
        return x


if __name__ == '__main__':
    model = SSFTTnet()
    model.eval()
    print(model)
    input = torch.randn(64, 1, 30, 13, 13)
    y = model(input)
    print(y.size())
