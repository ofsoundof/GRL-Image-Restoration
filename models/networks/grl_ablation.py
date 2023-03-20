"""
Efficient and Explicit Modelling of Image Hierarchies for Image Restoration
Image restoration transformers with global, regional, and local modelling
This is the orginal implementation of GRL. Major difference between grl_ablation.py and grl.py
1) Development phase.
    grl_ablation.py: developed in the early phase.
    grl.py: adapted from grl_ablation.py.
2) buffers and GPU memory efficiency.
    grl_ablation.py: could use shared or unique buffers.
    grl.py: only use the shared buffers
2) Ablation study.
    grl_ablation.py: contains more ablation study.
    grl.py:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn import checkpoint_wrapper
from omegaconf import OmegaConf
from models.common import Upsample, UpsampleOneStep
from models.common.mixed_attn_block import (
    MixAttnTransformerBlock,
)
from models.common.mixed_attn_block_efficient import (
    _get_stripe_info,
    EfficientMixAttnTransformerBlock,
)
from models.common.ops import (
    bchw_to_blc,
    blc_to_bchw,
    calculate_mask,
    calculate_mask_all,
    get_relative_coords_table_all,
    get_relative_position_index_simple,
)
from models.common.swin_v1_block import (
    build_last_conv,
)
from timm.models.layers import to_2tuple, trunc_normal_


class TransformerStage(nn.Module):
    """Transformer stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads_window (list[int]): Number of window attention heads in different layers.
        num_heads_stripe (list[int]): Number of stripe attention heads in different layers.
        stripe_size (list[int]): Stripe size. Default: [8, 8]
        stripe_groups (list[int]): Number of stripe groups. Default: [None, None].
        stripe_shift (bool): whether to shift the stripes. This is used as an ablation study.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qkv_proj_type (str): QKV projection type. Default: linear. Choices: linear, separable_conv.
        anchor_proj_type (str): Anchor projection type. Default: avgpool. Choices: avgpool, maxpool, conv2d, separable_conv, patchmerging.
        anchor_one_stage (bool): Whether to use one operator or multiple progressive operators to reduce feature map resolution. Default: True.
        anchor_window_down_factor (int): The downscale factor used to get the anchors.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        pretrained_window_size (list[int]): pretrained window size. This is actually not used. Default: [0, 0].
        pretrained_stripe_size (list[int]): pretrained stripe size. This is actually not used. Default: [0, 0].
        conv_type: The convolutional block before residual connection.
        init_method: initialization method of the weight parameters used to train large scale models.
            Choices: n, normal -- Swin V1 init method.
                    l, layernorm -- Swin V2 init method. Zero the weight and bias in the post layer normalization layer.
                    r, res_rescale -- EDSR rescale method. Rescale the residual blocks with a scaling factor 0.1
                    w, weight_rescale -- MSRResNet rescale method. Rescale the weight parameter in residual blocks with a scaling factor 0.1
                    t, trunc_normal_ -- nn.Linear, trunc_normal; nn.Conv2d, weight_rescale
        fairscale_checkpoint (bool): Whether to use fairscale checkpoint.
        offload_to_cpu (bool): used by fairscale_checkpoint
        args:
            out_proj_type (str): Type of the output projection in the self-attention modules. Default: linear. Choices: linear, conv2d.
            local_connection (bool): Whether to enable the local modelling module (two convs followed by Channel attention). For GRL base model, this is used.                "local_connection": local_connection,
            euclidean_dist (bool): use Euclidean distance or inner product as the similarity metric. An ablation study.
            double_window (bool): Ablaltion study. Default: False,
            stripe_square (bool): Ablaltion study. Default: False,
            separable_conv_act (bool): Ablaltion study. Default: True,
            use_buffer (bool): Whether to use buffer.  Default: False.
                False: the attention masks, tables, and indices are pre-computed. Huge GPU memory consumption when the window size is large.
                True:
                    use_efficient_buffer=False: buffers are not shared. computed for each layer during forward pass. Slow forward pass.
                    use_efficient_buffer=True: pre-computed and shared buffers. Small GPU memory consumption, fast forward pass. Need to allocate buffers manually.
            use_efficient_buffer (bool): Whether to use efficient buffer. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads_window,
        num_heads_stripe,
        window_size,
        stripe_size,
        stripe_groups,
        stripe_shift,
        mlp_ratio=4.0,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="avgpool",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=[0, 0],
        pretrained_stripe_size=[0, 0],
        conv_type="1conv",
        init_method="",
        fairscale_checkpoint=False,
        offload_to_cpu=False,
        args=None,
    ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.init_method = init_method
        self.use_efficient_buffer = args.use_efficient_buffer
        if self.use_efficient_buffer:
            transformer_block = EfficientMixAttnTransformerBlock
        else:
            transformer_block = MixAttnTransformerBlock

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = transformer_block(
                dim=dim,
                input_resolution=input_resolution,
                num_heads_w=num_heads_window,
                num_heads_s=num_heads_stripe,
                window_size=window_size,
                window_shift=i % 2 == 0,
                stripe_size=stripe_size,
                stripe_groups=stripe_groups,
                stripe_type="H" if i % 2 == 0 else "W",
                stripe_shift=i % 4 in [2, 3] if stripe_shift else False,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type=qkv_proj_type,
                anchor_proj_type=anchor_proj_type,
                anchor_one_stage=anchor_one_stage,
                anchor_window_down_factor=anchor_window_down_factor,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                pretrained_stripe_size=pretrained_stripe_size,
                res_scale=0.1 if init_method == "r" else 1.0,
                args=args,
            )
            if fairscale_checkpoint:
                block = checkpoint_wrapper(block, offload_to_cpu=offload_to_cpu)
            self.blocks.append(block)

        self.conv = build_last_conv(conv_type, dim)

    def _init_weights(self):
        for n, m in self.named_modules():
            if self.init_method == "w":
                if isinstance(m, (nn.Linear, nn.Conv2d)) and n.find("cpb_mlp") < 0:
                    print("nn.Linear and nn.Conv2d weight initilization")
                    m.weight.data *= 0.1
            elif self.init_method == "l":
                if isinstance(m, nn.LayerNorm):
                    print("nn.LayerNorm initialization")
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 0)
            elif self.init_method.find("t") >= 0:
                scale = 0.1 ** (len(self.init_method) - 1) * int(self.init_method[-1])
                if isinstance(m, nn.Linear) and n.find("cpb_mlp") < 0:
                    trunc_normal_(m.weight, std=scale)
                elif isinstance(m, nn.Conv2d):
                    m.weight.data *= 0.1
                print(
                    "Initialization nn.Linear - trunc_normal; nn.Conv2d - weight rescale."
                )
            else:
                raise NotImplementedError(
                    f"Parameter initialization method {self.init_method} not implemented in TransformerStage."
                )

    def forward(self, x, x_size, table_index_mask=None):
        res = x
        if self.use_efficient_buffer:
            for blk in self.blocks:
                res = blk(res, x_size, table_index_mask)
        else:
            for blk in self.blocks:
                res = blk(res, x_size)
        res = bchw_to_blc(self.conv(blc_to_bchw(res, x_size)))

        return res + x

    def flops(self):
        flops = 0
        flops += self.blocks.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9

        return flops


class GRL(nn.Module):
    r"""Image restoration transformer with global, non-local, and local connections
    Args:
        img_size (int | list[int]): Input image size. Default 64
        in_channels (int): Number of input image channels. Default: 3
        out_channels (int): Number of output image channels. Default: None
        embed_dim (int): Patch embedding dimension. Default: 96
        upscale (int): Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range (float): Image range. 1. or 255.
        upsampler (str): The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        depths (list[int]): Depth of each Swin Transformer layer.
        num_heads_window (list[int]): Number of window attention heads in different layers.
        num_heads_stripe (list[int]): Number of stripe attention heads in different layers.
        window_size (int): Window size. Default: 8.
        stripe_size (list[int]): Stripe size. Default: [8, 8]
        stripe_groups (list[int]): Number of stripe groups. Default: [None, None].
        stripe_shift (bool): whether to shift the stripes. This is used as an ablation study.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qkv_proj_type (str): QKV projection type. Default: linear. Choices: linear, separable_conv.
        anchor_proj_type (str): Anchor projection type. Default: avgpool. Choices: avgpool, maxpool, conv2d, separable_conv, patchmerging.
        anchor_one_stage (bool): Whether to use one operator or multiple progressive operators to reduce feature map resolution. Default: True.
        anchor_window_down_factor (int): The downscale factor used to get the anchors.
        out_proj_type (str): Type of the output projection in the self-attention modules. Default: linear. Choices: linear, conv2d.
        local_connection (bool): Whether to enable the local modelling module (two convs followed by Channel attention). For GRL base model, this is used.
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        pretrained_window_size (list[int]): pretrained window size. This is actually not used. Default: [0, 0].
        pretrained_stripe_size (list[int]): pretrained stripe size. This is actually not used. Default: [0, 0].
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        conv_type (str): The convolutional block before residual connection. Default: 1conv. Choices: 1conv, 3conv, 1conv1x1, linear
        init_method: initialization method of the weight parameters used to train large scale models.
            Choices: n, normal -- Swin V1 init method.
                    l, layernorm -- Swin V2 init method. Zero the weight and bias in the post layer normalization layer.
                    r, res_rescale -- EDSR rescale method. Rescale the residual blocks with a scaling factor 0.1
                    w, weight_rescale -- MSRResNet rescale method. Rescale the weight parameter in residual blocks with a scaling factor 0.1
                    t, trunc_normal_ -- nn.Linear, trunc_normal; nn.Conv2d, weight_rescale
        fairscale_checkpoint (bool): Whether to use fairscale checkpoint.
        offload_to_cpu (bool): used by fairscale_checkpoint
        double_window (bool): Ablaltion study. Default: False,
        stripe_square (bool): Ablaltion study. Default: False,
        separable_conv_act (bool): Ablaltion study. Default: True,
        use_buffer (bool): Whether to use buffer.  Default: False.
            False: the attention masks, tables, and indices are pre-computed. Huge GPU memory consumption when the window size is large.
            True:
                use_efficient_buffer=False: buffers are not shared. computed for each layer during forward pass. Slow forward pass.
                use_efficient_buffer=True: pre-computed and shared buffers. Small GPU memory consumption, fast forward pass. Need to allocate buffers manually.
        use_efficient_buffer (bool): Whether to use efficient buffer. Default: False.
        euclidean_dist (bool): use Euclidean distance or inner product as the similarity metric. An ablation study.
    """

    def __init__(
        self,
        img_size=64,
        in_channels=3,
        out_channels=None,
        embed_dim=96,
        upscale=2,
        img_range=1.0,
        upsampler="",
        depths=[6, 6, 6, 6, 6, 6],
        num_heads_window=[3, 3, 3, 3, 3, 3],
        num_heads_stripe=[3, 3, 3, 3, 3, 3],
        window_size=8,
        stripe_size=[8, 8],  # used for stripe window attention
        stripe_groups=[None, None],
        stripe_shift=False,
        mlp_ratio=4.0,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="separable_conv",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        out_proj_type="linear",
        local_connection=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=[0, 0],
        pretrained_stripe_size=[0, 0],
        conv_type="1conv",
        init_method="n",  # initialization method of the weight parameters used to train large scale models.
        fairscale_checkpoint=False,  # fairscale activation checkpointing
        offload_to_cpu=False,
        double_window=False,
        stripe_square=False,
        separable_conv_act=True,
        use_buffer=True,
        use_efficient_buffer=False,
        euclidean_dist=False,
        **kwargs,
    ):
        super(GRL, self).__init__()
        # Process the input arguments
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_out_feats = 64
        self.embed_dim = embed_dim
        self.upscale = upscale
        self.upsampler = upsampler
        self.img_range = img_range
        if in_channels == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        max_stripe_size = max([0 if s is None else s for s in stripe_size])
        max_stripe_groups = max([0 if s is None else s for s in stripe_groups])
        max_stripe_groups *= anchor_window_down_factor
        self.pad_size = max(window_size, max_stripe_size, max_stripe_groups)
        # if max_stripe_size >= window_size:
        #     self.pad_size *= anchor_window_down_factor
        # if stripe_groups[0] is None and stripe_groups[1] is None:
        #     self.pad_size = max(stripe_size)
        # else:
        #     self.pad_size = window_size
        self.input_resolution = to_2tuple(img_size)
        self.window_size = to_2tuple(window_size)
        self.shift_size = [w // 2 for w in self.window_size]
        self.stripe_size = stripe_size
        self.stripe_groups = stripe_groups
        self.pretrained_window_size = pretrained_window_size
        self.pretrained_stripe_size = pretrained_stripe_size
        self.anchor_window_down_factor = anchor_window_down_factor
        self.use_buffer = use_buffer
        self.use_efficient_buffer = use_efficient_buffer

        # Head of the network. First convolution.
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # Body of the network
        self.norm_start = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # stochastic depth decay rule
        args = OmegaConf.create(
            {
                "double_window": double_window,
                "stripe_square": stripe_square,
                "separable_conv_act": separable_conv_act,
                "out_proj_type": out_proj_type,
                "local_connection": local_connection,
                "use_buffer": use_buffer,
                "use_efficient_buffer": use_efficient_buffer,
                "euclidean_dist": euclidean_dist,
            }
        )
        if self.use_efficient_buffer and self.use_buffer:
            for k, v in self.set_table_index_mask(self.input_resolution).items():
                self.register_buffer(k, v)

        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = TransformerStage(
                dim=embed_dim,
                input_resolution=to_2tuple(img_size),
                depth=depths[i],
                num_heads_window=num_heads_window[i],
                num_heads_stripe=num_heads_stripe[i],
                window_size=to_2tuple(window_size),
                stripe_size=stripe_size,
                stripe_groups=stripe_groups,
                stripe_shift=stripe_shift,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type=qkv_proj_type,
                anchor_proj_type=anchor_proj_type,
                anchor_one_stage=anchor_one_stage,
                anchor_window_down_factor=anchor_window_down_factor,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i]) : sum(depths[: i + 1])
                ],  # no impact on SR results
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                pretrained_stripe_size=pretrained_stripe_size,
                conv_type=conv_type,
                init_method=init_method,
                fairscale_checkpoint=fairscale_checkpoint,
                offload_to_cpu=offload_to_cpu,
                args=args,
            )
            self.layers.append(layer)
        self.norm_end = norm_layer(embed_dim)

        # Tail of the network
        self.conv_after_body = build_last_conv(conv_type, embed_dim)

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_out_feats, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_out_feats)
            self.conv_last = nn.Conv2d(num_out_feats, out_channels, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(
                upscale,
                embed_dim,
                out_channels,
            )
        elif self.upsampler == "nearest+conv":
            # for real-world SR (less artifacts)
            assert self.upscale == 4, "only support x4 now."
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_out_feats, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_up1 = nn.Conv2d(num_out_feats, num_out_feats, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_out_feats, num_out_feats, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_out_feats, num_out_feats, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_out_feats, out_channels, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, out_channels, 3, 1, 1)

        self.apply(self._init_weights)
        if init_method in ["l", "w"] or init_method.find("t") >= 0:
            for layer in self.layers:
                layer._init_weights()

    def set_table_index_mask(self, x_size):
        """
        Two used cases:
        1) At initialization: set the shared buffers.
        2) During forward pass: get the new buffers if the resolution of the input changes
        """
        # ss - stripe_size, sss - stripe_shift_size
        ss, sss = _get_stripe_info(self.stripe_size, self.stripe_groups, True, x_size)
        df = self.anchor_window_down_factor

        table_w = get_relative_coords_table_all(
            self.window_size, self.pretrained_window_size
        )
        table_sh = get_relative_coords_table_all(ss, self.pretrained_stripe_size, df)
        table_sv = get_relative_coords_table_all(
            ss[::-1], self.pretrained_stripe_size, df
        )

        index_w = get_relative_position_index_simple(self.window_size)
        index_sh_a2w = get_relative_position_index_simple(ss, df, False)
        index_sh_w2a = get_relative_position_index_simple(ss, df, True)
        index_sv_a2w = get_relative_position_index_simple(ss[::-1], df, False)
        index_sv_w2a = get_relative_position_index_simple(ss[::-1], df, True)

        mask_w = calculate_mask(x_size, self.window_size, self.shift_size)
        mask_sh_a2w = calculate_mask_all(x_size, ss, sss, df, False)
        mask_sh_w2a = calculate_mask_all(x_size, ss, sss, df, True)
        mask_sv_a2w = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, False)
        mask_sv_w2a = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, True)
        return {
            "table_w": table_w,
            "table_sh": table_sh,
            "table_sv": table_sv,
            "index_w": index_w,
            "index_sh_a2w": index_sh_a2w,
            "index_sh_w2a": index_sh_w2a,
            "index_sv_a2w": index_sv_a2w,
            "index_sv_w2a": index_sv_w2a,
            "mask_w": mask_w,
            "mask_sh_a2w": mask_sh_a2w,
            "mask_sh_w2a": mask_sh_w2a,
            "mask_sv_a2w": mask_sv_a2w,
            "mask_sv_w2a": mask_sv_w2a,
        }

    def get_table_index_mask(self, device=None, input_resolution=None):
        if self.use_buffer and input_resolution == self.input_resolution:
            return {
                "table_w": self.table_w,
                "table_sh": self.table_sh,
                "table_sv": self.table_sv,
                "index_w": self.index_w,
                "index_sh_a2w": self.index_sh_a2w,
                "index_sh_w2a": self.index_sh_w2a,
                "index_sv_a2w": self.index_sv_a2w,
                "index_sv_w2a": self.index_sv_w2a,
                "mask_w": self.mask_w,
                "mask_sh_a2w": self.mask_sh_a2w,
                "mask_sh_w2a": self.mask_sh_w2a,
                "mask_sv_a2w": self.mask_sv_a2w,
                "mask_sv_w2a": self.mask_sv_w2a,
            }
        else:
            table_index_mask = self.set_table_index_mask(input_resolution)
            for k, v in table_index_mask.items():
                table_index_mask[k] = v.to(device)
            return table_index_mask

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Only used to initialize linear layers
            # weight_shape = m.weight.shape
            # if weight_shape[0] > 256 and weight_shape[1] > 256:
            #     std = 0.004
            # else:
            #     std = 0.02
            # print(f"Standard deviation during initialization {std}.")
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.pad_size - h % self.pad_size) % self.pad_size
        mod_pad_w = (self.pad_size - w % self.pad_size) % self.pad_size
        # print("padding size", h, w, self.pad_size, mod_pad_h, mod_pad_w)

        try:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        except BaseException:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant")
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = bchw_to_blc(x)
        x = self.norm_start(x)
        x = self.pos_drop(x)

        if self.use_efficient_buffer:
            table_index_mask = self.get_table_index_mask(x.device, x_size)
        else:
            table_index_mask = None

        for layer in self.layers:
            x = layer(x, x_size, table_index_mask)

        x = self.norm_end(x)  # B L C
        x = blc_to_bchw(x, x_size)

        return x

    def forward(self, x):
        # print(x.shape)
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == "nearest+conv":
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(
                self.conv_up1(
                    torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
                )
            )
            x = self.lrelu(
                self.conv_up2(
                    torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
                )
            )
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            if self.in_channels == self.out_channels:
                x = x + self.conv_last(res)
            else:
                x = self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, : H * self.upscale, : W * self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops

    def convert_checkpoint(self, state_dict):
        for k in list(state_dict.keys()):
            if (
                k.find("relative_coords_table") >= 0
                or k.find("relative_position_index") >= 0
                or k.find("attn_mask") >= 0
                or k.find("model.table_") >= 0
                or k.find("model.index_") >= 0
                or k.find("model.mask_") >= 0
                # or k.find(".upsample.") >= 0
            ):
                state_dict.pop(k)
                print(k)
        return state_dict


if __name__ == "__main__":
    window_size = 8

    # Tiny, 0.33 M
    # model = GRL(
    #     upscale=4,
    #     img_size=64,
    #     window_size=window_size,
    #     depths=[4, 4, 4, 4],
    #     embed_dim=32,
    #     num_heads_window=[2, 2, 2, 2],
    #     num_heads_stripe=[2, 2, 2, 2],
    #     mlp_ratio=2,
    #     qkv_proj_type="linear",
    #     anchor_proj_type="avgpool",
    #     anchor_window_down_factor=2,
    #     out_proj_type="linear",
    #     conv_type="1conv",
    #     upsampler="pixelshuffledirect",
    # )

    # Small, 3.49 M
    # model = GRL(
    #     upscale=4,
    #     img_size=64,
    #     window_size=window_size,
    #     depths=[4, 4, 4, 4],
    #     embed_dim=128,
    #     num_heads_window=[2, 2, 2, 2],
    #     num_heads_stripe=[2, 2, 2, 2],
    #     mlp_ratio=2,
    #     qkv_proj_type="linear",
    #     anchor_proj_type="avgpool",
    #     anchor_window_down_factor=2,
    #     out_proj_type="linear",
    #     conv_type="1conv",
    #     upsampler="pixelshuffle",
    # )

    # Base, 13.84 M
    # model = GRL(
    #     upscale=4,
    #     img_size=64,
    #     window_size=window_size,
    #     depths=[4, 4, 4, 4, 4, 4, 4, 4],
    #     embed_dim=192,
    #     num_heads_window=[4, 4, 4, 4, 4, 4, 4, 4],
    #     num_heads_stripe=[4, 4, 4, 4, 4, 4, 4, 4],
    #     mlp_ratio=2,
    #     qkv_proj_type="linear",
    #     anchor_proj_type="avgpool",
    #     anchor_window_down_factor=2,
    #     out_proj_type="linear",
    #     conv_type="1conv",
    #     upsampler="pixelshuffle",
    # )

    # Large, 24.29 M
    # model = GRL(
    #     upscale=4,
    #     img_size=64,
    #     window_size=window_size,
    #     depths=[8, 8, 8, 8, 8, 8, 8, 8],
    #     embed_dim=192,
    #     num_heads_window=[4, 4, 4, 4, 4, 4, 4, 4],
    #     num_heads_stripe=[4, 4, 4, 4, 4, 4, 4, 4],
    #     mlp_ratio=2,
    #     qkv_proj_type="linear",
    #     anchor_proj_type="avgpool",
    #     anchor_window_down_factor=2,
    #     out_proj_type="linear",
    #     conv_type="1conv",
    #     upsampler="pixelshuffle",
    # )

    # Huge, 47.83 M
    # model = GRL(
    #     upscale=4,
    #     img_size=64,
    #     window_size=window_size,
    #     depths=[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    #     embed_dim=192,
    #     num_heads_window=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    #     num_heads_stripe=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    #     mlp_ratio=2,
    #     qkv_proj_type="linear",
    #     anchor_proj_type="avgpool",
    #     anchor_window_down_factor=2,
    #     out_proj_type="linear",
    #     conv_type="1conv",
    #     upsampler="pixelshuffle",
    # )

    # Giant, MLP4 - 117.16 M, MLP2 - 83.54 M
    # model = GRL(
    #     upscale=4,
    #     img_size=64,
    #     window_size=window_size,
    #     depths=[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    #     embed_dim=256,
    #     num_heads_window=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    #     num_heads_stripe=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    #     mlp_ratio=4,
    #     qkv_proj_type="linear",
    #     anchor_proj_type="avgpool",
    #     anchor_window_down_factor=2,
    #     out_proj_type="linear",
    #     conv_type="1conv",
    #     upsampler="pixelshuffle",
    # )

    # Compare with HAT Large, 43.22 M
    # model = GRL(
    #     upscale=4,
    #     img_size=64,
    #     window_size=window_size,
    #     depths=[12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    #     embed_dim=192,
    #     num_heads_window=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    #     num_heads_stripe=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    #     mlp_ratio=2,
    #     qkv_proj_type="linear",
    #     anchor_proj_type="avgpool",
    #     anchor_window_down_factor=2,
    #     out_proj_type="linear",
    #     conv_type="1conv",
    #     upsampler="pixelshuffle",
    # )

    ####################
    # Final version
    ####################

    # Tiny-final, 0.91M
    # model = GRL(
    #     upscale=4,
    #     img_size=64,
    #     window_size=window_size,
    #     depths=[4, 4, 4, 4],
    #     embed_dim=64,
    #     num_heads_window=[2, 2, 2, 2],
    #     num_heads_stripe=[2, 2, 2, 2],
    #     mlp_ratio=2,
    #     qkv_proj_type="linear",
    #     anchor_proj_type="avgpool",
    #     anchor_window_down_factor=2,
    #     out_proj_type="linear",
    #     conv_type="1conv",
    #     upsampler="pixelshuffledirect",
    # )

    # Small-final, 3.49M
    # model = GRL(
    #     upscale=4,
    #     img_size=64,
    #     window_size=window_size,
    #     depths=[4, 4, 4, 4],
    #     embed_dim=128,
    #     num_heads_window=[2, 2, 2, 2],
    #     num_heads_stripe=[2, 2, 2, 2],
    #     mlp_ratio=2,
    #     qkv_proj_type="linear",
    #     anchor_proj_type="avgpool",
    #     anchor_window_down_factor=2,
    #     out_proj_type="linear",
    #     conv_type="1conv",
    #     upsampler="pixelshuffle",
    # )

    # Large, 20.22 M
    model = GRL(
        upscale=4,
        img_size=64,
        window_size=window_size,
        depths=[4, 4, 8, 8, 8, 4, 4],
        embed_dim=180,
        num_heads_window=[3, 3, 3, 3, 3, 3, 3],
        num_heads_stripe=[3, 3, 3, 3, 3, 3, 3],
        mlp_ratio=2,
        qkv_proj_type="linear",
        anchor_proj_type="avgpool",
        anchor_window_down_factor=2,
        out_proj_type="linear",
        conv_type="1conv",
        upsampler="pixelshuffle",
        local_connection=True,
        use_buffer=False,
        use_efficient_buffer=True,
    )

    print(model)
    # print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 3, 64, 64))
    x = model(x)
    print(x.shape)
    num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params += p.numel()
    print(f"Number of parameters {num_params / 10 ** 6: 0.2f}")
