from typing import Optional, List

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from itertools import cycle, zip_longest

from torch.nn import Module, ModuleList

from einops import rearrange, reduce, pack, unpack

from local_attention import LocalMHA
from local_attention.transformer import FeedForward, DynamicPositionBias

from gateloop_transformer import SimpleGateLoopLayer as GateLoop

def exists(val):
    return val is not None

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

class SqueezeExcite(Module):
    def __init__(self, dim, reduction_factor = 4, dim_minimum = 8):
        super().__init__()
        dim_inner = max(dim_minimum, dim // reduction_factor)
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim_inner, 1),
            nn.SiLU(),
            nn.Conv1d(dim_inner, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        seq, device = x.shape[-2], x.device

        # cumulative mean - since it is autoregressive

        cum_sum = x.cumsum(dim = -2)
        denom = torch.arange(1, seq + 1, device = device).float()
        cum_mean = cum_sum / rearrange(denom, 'n -> n 1')

        # glu gate

        gate = self.net(cum_mean)

        return x * gate

class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class ChannelTranspose(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = rearrange(x, 'b c n -> b n c')
        out = self.fn(x, **kwargs) + x
        return rearrange(out, 'b n c -> b c n')

class CausalConv1d(Module):
    def __init__(self, chan_in, chan_out, kernel_size, pad_mode = 'reflect', **kwargs):
        super().__init__()
        kernel_size = kernel_size
        dilation = kwargs.get('dilation', 1)
        stride = kwargs.get('stride', 1)
        self.pad_mode = pad_mode
        self.causal_padding = dilation * (kernel_size - 1) + (1 - stride)

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.causal_padding, 0), mode = self.pad_mode)
        return self.conv(x)

class CausalConvTranspose1d(Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride, **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.padding = kernel_size - 1
        self.conv = nn.ConvTranspose1d(chan_in, chan_out, kernel_size, stride, **kwargs)

    def forward(self, x):
        n = x.shape[-1]

        out = self.conv(x)
        out = out[..., :(n * self.upsample_factor)]

        return out
    
class FiLM(Module):
    def __init__(self, dim, dim_cond):
        super().__init__()
        self.to_cond = nn.Linear(dim_cond, dim * 2)

    def forward(self, x, cond):
        gamma, beta = self.to_cond(cond).chunk(2, dim = -1)
        return x * gamma.unsqueeze(-1) + beta.unsqueeze(-1)

def ResidualUnit(chan_in, chan_out, dilation, kernel_size = 7, squeeze_excite = False, pad_mode = 'reflect'):
    return Residual(Sequential(
        CausalConv1d(chan_in, chan_out, kernel_size, dilation = dilation, pad_mode = pad_mode),
        nn.ELU(),
        CausalConv1d(chan_out, chan_out, 1, pad_mode = pad_mode),
        nn.ELU(),
        SqueezeExcite(chan_out) if squeeze_excite else None
    ))

class EncoderBlock(Module):
    def __init__(self, chan_in, chan_out, stride, cycle_dilations = (1, 3, 9), squeeze_excite = False, pad_mode = 'reflect') -> None:
        super().__init__()

        it = cycle(cycle_dilations)
        residual_unit = partial(ResidualUnit, squeeze_excite = squeeze_excite, pad_mode = pad_mode)

        self.layers = nn.Sequential(
            residual_unit(chan_in, chan_in, next(it)),
            residual_unit(chan_in, chan_in, next(it)),
            residual_unit(chan_in, chan_in, next(it)),
            CausalConv1d(chan_in, chan_out, 2 * stride, stride = stride)
        )

    def forward(self, x):
        return self.layers(x)

class FiLMDecoderBlock(Module):
    def __init__(self, chan_in, chan_out, stride, cond_channels, cycle_dilations = (1, 3, 9), squeeze_excite = False, pad_mode = 'reflect'):
        super().__init__()
        
        even_stride = (stride % 2 == 0)
        residual_unit = partial(ResidualUnit, squeeze_excite = squeeze_excite, pad_mode = pad_mode)

        it = cycle(cycle_dilations)

        self.upsample = CausalConvTranspose1d(chan_in, chan_out, 2 * stride, stride = stride)

        self.films = ModuleList([
            FiLM(chan_out, cond_channels) if cond_channels > 0 else nn.Identity(),
            FiLM(chan_out, cond_channels) if cond_channels > 0 else nn.Identity(),
            FiLM(chan_out, cond_channels) if cond_channels > 0 else nn.Identity()
        ])

        self.residual_units = ModuleList([
            residual_unit(chan_out, chan_out, next(it)),
            residual_unit(chan_out, chan_out, next(it)),
            residual_unit(chan_out, chan_out, next(it))
        ])

    def forward(self, x, cond):
        x = self.upsample(x)
        for film, res in zip(self.films, self.residual_units):
            x = film(x, cond)
            x = res(x)

        return x

class LocalTransformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        window_size,
        dynamic_pos_bias = False,
        **kwargs
    ):
        super().__init__()
        self.window_size = window_size
        self.layers = ModuleList([])

        self.pos_bias = None
        if dynamic_pos_bias:
            self.pos_bias = DynamicPositionBias(dim = dim // 2, heads = heads)

        for _ in range(depth):
            self.layers.append(ModuleList([
                LocalMHA(
                    dim = dim,
                    heads = heads,
                    qk_rmsnorm = True,
                    window_size = window_size,
                    use_rotary_pos_emb = not dynamic_pos_bias,
                    gate_values_per_head = True,
                    use_xpos = True,
                    **kwargs
                ),
                FeedForward(dim = dim)
            ]))

    def forward(self, x):
        w = self.window_size

        attn_bias = self.pos_bias(w, w * 2) if exists(self.pos_bias) else None

        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x
            x = ff(x) + x

        return x
    

class SpeakerEncoder(Module):
    def __init__(
        self,
        *,
        channels = 32,
        strides = (2, 4, 5, 8),
        channel_mults = (2, 4, 8, 16),
        embedding_dim = 64,
        input_channels = 1,
        cycle_dilations = (1, 3, 9),
        use_gate_loop_layers = False,
        squeeze_excite = False,
        pad_mode = 'reflect',
    ):
        super().__init__()
        
        self.encoder = Encoder(
            channels = channels,
            strides = strides,
            channel_mults = channel_mults,
            embedding_dim = embedding_dim,
            input_channels = input_channels,
            cycle_dilations = cycle_dilations,
            use_gate_loop_layers = use_gate_loop_layers,
            squeeze_excite = squeeze_excite,
            pad_mode = pad_mode
        )
        self.learnable_query = nn.Parameter(torch.randn((1, 1, embedding_dim)))

    def forward(self, x, mask = None):
        """speaker encoder

        Args:
            x (torch.FloatTensor): [B, 1, T]
            mask (torch.FloatTensor, optional): mask for attention. Defaults to None.

        Returns:
            spkemb (torch.FloatTensor): [B, C]
        """
        emb = self.encoder(x) # [B, C, N]

        # learnable pooling
        B, d_k, _ = emb.shape
        query = self.learnable_query.expand(B, -1, -1) # [B, 1, C]
        key = emb # [B, C, N]
        value = emb.transpose(1, 2) # [B, N, C]

        score = torch.matmul(query, key) # [B, 1, N]
        score = score / (d_k ** 0.5)
        if exists(mask):
            score.masked_fill_(mask==0, -1e9)
        probs = F.softmax(score, dim=-1) # [B, 1, N]
        out = torch.matmul(probs, value) # [B, 1, C]
        out = out.squeeze(1) # [B, C]

        return out


class Encoder(Module):
    def __init__(
        self,
        *,
        channels = 64,
        strides = (2, 4, 5, 8),
        channel_mults = (2, 4, 8, 16),
        embedding_dim = 64,
        input_channels = 1,
        cycle_dilations = (1, 3, 9),
        use_gate_loop_layers = False,
        squeeze_excite = False,
        pad_mode = 'reflect',
    ):
        super().__init__()
        layer_channels = tuple(map(lambda t: t * channels, channel_mults))
        layer_channels = (channels, *layer_channels)
        chan_in_out_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        encoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(chan_in_out_pairs, strides):
            encoder_blocks.append(EncoderBlock(chan_in, chan_out, layer_stride, cycle_dilations, squeeze_excite, pad_mode))

            if use_gate_loop_layers:
                encoder_blocks.append(Residual(ChannelTranspose(GateLoop(chan_out, use_heinsen = False))))

        self.encoder = nn.Sequential(
            CausalConv1d(input_channels, channels, 7, pad_mode = pad_mode),
            *encoder_blocks,
            CausalConv1d(layer_channels[-1], embedding_dim, 3, pad_mode = pad_mode)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(Module):
    def __init__(
        self,
        *,
        channels = 40,
        strides = (2, 4, 5, 8),
        channel_mults = (2, 4, 8, 16),
        embedding_dim = 64,
        input_channels = 1,
        cond_channels = 64,
        cycle_dilations = (1, 3, 9),
        use_gate_loop_layers = False,
        squeeze_excite = False,
        pad_mode = 'reflect',
        use_local_attn = True,
        attn_window_size = 128,
        attn_dim_head = 64,
        attn_heads = 8,
        attn_depth = 1,
        attn_xpos_scale_base = None,
        attn_dynamic_pos_bias = False,
    ) -> None:
        super().__init__()
        layer_channels = tuple(map(lambda t: t * channels, channel_mults))
        layer_channels = (channels, *layer_channels)
        chan_in_out_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        attn_kwargs = dict(
            dim = embedding_dim,
            dim_head = attn_dim_head,
            heads = attn_heads,
            depth = attn_depth,
            window_size = attn_window_size,
            xpos_scale_base = attn_xpos_scale_base,
            dynamic_pos_bias = attn_dynamic_pos_bias,
            prenorm = True,
            causal = True
        )

        self.decoder_attn = LocalTransformer(**attn_kwargs) if use_local_attn else None
        self.decoder_init = CausalConv1d(embedding_dim, layer_channels[-1], 7, pad_mode = pad_mode)

        decoder_blocks = []
        for ((chan_in, chan_out), layer_stride) in zip(reversed(chan_in_out_pairs), reversed(strides)):
            decoder_blocks.append(FiLMDecoderBlock(chan_out, chan_in, layer_stride, cond_channels, cycle_dilations, squeeze_excite, pad_mode))

            if use_gate_loop_layers:
                decoder_blocks.append(Residual(ChannelTranspose(GateLoop(chan_in))))
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        self.decoder_out = CausalConv1d(channels, input_channels, 7, pad_mode = pad_mode)

    def forward(self, x, cond):
        """StreamVC Decoder

        Args:
            x (_type_): [B, C, N]
            cond (_type_): [B, C, 1]

        Returns:
            _type_: _description_
        """
        if exists(self.decoder_attn):
            x = rearrange(x, 'b c n -> b n c')
            x = self.decoder_attn(x)
            x = rearrange(x, 'b n c -> b c n')

        x = self.decoder_init(x)
        for block in self.decoder_blocks:
            x = block(x, cond)
        out = self.decoder_out(x)
        return out


class StreamVC(Module):
    def __init__(
        self
    ) -> None:
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()
        self.spk_enc = SpeakerEncoder()

    def forward(self, x):
        emb = self.enc(x)
        spk_emb = self.spk_enc(x)
        out = self.dec(emb, spk_emb)
        return out
    
    
if __name__ == "__main__":
    model = StreamVC()
    x = torch.randn(2, 1, 24000)
    out = model(x)
    print(out.shape)