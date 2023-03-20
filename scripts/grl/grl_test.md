# Test commands for GRL.



```bash
# Set the model zoo directory
MODEL_ZOO=/home/thor/projects/data/LightningIR/model_zoo


########################################################################################################################
## Demosaicking
########################################################################################################################
torchx run -- -j 1x2 -- \
    -m training=False gpus=2 experiment=dm/grl model=grl/grl_small \
    load_state_dict=True pretrained_checkpoint="${MODEL_ZOO}/GRL/dm_grl_small.ckpt"


########################################################################################################################
## Image Denoising
########################################################################################################################

# Set grayscale or color image denoising
VAL=(set12+bsd68+urban100 mcmaster+cbsd68+kodak24+urban100)
C=(1 3)
METRIC=(restorer_gray restorer)
# j=0 for grayscale image denoising
# j=1 for color image denoising
j=0
SIGMA=15

## GRL-Tiny
MODEL=grl_tiny
torchx run -- -j 1x2 -- \
    -m training=False gpus=2 experiment=dn/grl/grl_p256 model=grl/${MODEL} data_module.noise_sigma=${SIGMA} data_module.num_channels=${C[j]} data_module.val.dataset=${VAL[j]} metric=${METRIC[j]} \
    model.anchor_window_down_factor=4 model.fairscale_checkpoint=True \
    load_state_dict=True pretrained_checkpoint="${MODEL_ZOO}/GRL/dn_${MODEL}_c${C[j]}s${SIGMA}.ckpt"

## GRL-Small
MODEL=grl_small
torchx run -- -j 1x2 -- \
    -m training=False gpus=2 experiment=dn/grl/grl_p256 model=grl/${MODEL} data_module.noise_sigma=${SIGMA} data_module.num_channels=${C[j]} data_module.val.dataset=${VAL[j]} metric=${METRIC[j]} \
    model.anchor_window_down_factor=4 model.fairscale_checkpoint=True \
    load_state_dict=True pretrained_checkpoint="${MODEL_ZOO}/GRL/dn_${MODEL}_c${C[j]}s${SIGMA}.ckpt"
    
## GRL-Base
MODEL=grl_base
torchx run -- -j 1x2 -- \
    -m training=False gpus=2 experiment=dn/grl/grl_p256 model=grl/${MODEL} data_module.noise_sigma=${SIGMA} data_module.num_channels=${C[j]} data_module.val.dataset=${VAL[j]} metric=${METRIC[j]} \
    model.anchor_window_down_factor=2 model.window_size=32 model.fairscale_checkpoint=True tile=256 tile_overlap=32 \
    load_state_dict=True pretrained_checkpoint="${MODEL_ZOO}/GRL/dn_${MODEL}_c${C[j]}s${SIGMA}.ckpt"


########################################################################################################################
## Single-Image Super-Resolution
########################################################################################################################

## GRL-Tiny
MODEL=grl_tiny
SCALE=2
torchx run -- -j 1x2 -- \
    -m training=False gpus=2 experiment=sr/grl/grl_p256 model=grl/${MODEL} data_module.scale=${SCALE} data_module.val.dataset=set5 \
    tile=0 tile_overlap=0 model.anchor_window_down_factor=4 model.fairscale_checkpoint=True \
    load_state_dict=True pretrained_checkpoint="${MODEL_ZOO}/GRL/sr_${MODEL}_c3x${SCALE}.ckpt"

## GRL-Small
MODEL=grl_small
SCALE=2
torchx run -- -j 1x2 -- \
    -m training=False gpus=2 experiment=sr/grl/grl_p256 model=grl/${MODEL} data_module.scale=${SCALE} data_module.val.dataset=set5 \
    tile=0 tile_overlap=0 model.anchor_window_down_factor=4 model.fairscale_checkpoint=True \
    load_state_dict=True pretrained_checkpoint="${MODEL_ZOO}/GRL/sr_${MODEL}_c3x${SCALE}.ckpt"

## GRL-Base
MODEL=grl_base
SCALE=2
torchx run -- -j 1x2 -- \
    -m training=False gpus=2 experiment=sr/grl/grl_p256 model=grl/${MODEL} data_module.scale=${SCALE} data_module.val.dataset=set5 \
    tile=0 tile_overlap=0 model.anchor_window_down_factor=2 model.fairscale_checkpoint=True \
    load_state_dict=True pretrained_checkpoint="${MODEL_ZOO}/GRL/sr_${MODEL}_c3x${SCALE}.ckpt"
    

########################################################################################################################
## JPEG image compression artifact removal
########################################################################################################################

VAL=(classic5+live1+bsds500+urban100 live1+bsds500+urban100)
C=(1 3)
METRIC=(restorer_jpeg_gray restorer_jpeg)
MODEL=grl_small
j=0
QUALITY=10

torchx run -- -j 1x2 -- \
    -m training=False gpus=2 experiment=jpeg/grl/grl_p288 model=grl/${MODEL} \
    data_module.quality_factor=${QUALITY} data_module.num_channels=${C[j]} data_module.val.dataset=${VAL[j]} metric=${METRIC[j]} \
    tile=288 tile_overlap=36 model.fairscale_checkpoint=True num_workers=2 \
    load_state_dict=True pretrained_checkpoint="${MODEL_ZOO}/GRL/jpeg_${MODEL}_c${C[j]}q${QUALITY}.ckpt"


########################################################################################################################
## Blind Image SR
########################################################################################################################
torchx run -- -j 1x2 -- \
    -m training=False gpus=2 experiment=bsr/grl model=grl/grl_base_bsr \
    bsr_psnr_checkpoint=null bsr_discriminator_checkpoint=null \
    load_state_dict=True pretrained_checkpoint="${MODEL_ZOO}/GRL/bsr_grl_base.ckpt"


########################################################################################################################
## Defocus Deblurring
########################################################################################################################

# Single pixel
torchx run -- -j 1x2 -- \
    -m training=False gpus=2 experiment=db_defocus/grl_p480 model=grl/grl_base model.fairscale_checkpoint=True \
    load_state_dict=True pretrained_checkpoint="${MODEL_ZOO}/GRL/db_defocus_single_pixel_grl_base.ckpt"
    
# Dual pixel
torchx run -- -j 1x2 -- \
    -m training=False gpus=2 experiment=db_defocus/grl_dual_p480 model=grl/grl_base model.fairscale_checkpoint=True \
    load_state_dict=True pretrained_checkpoint="${MODEL_ZOO}/GRL/db_defocus_dual_pixel_grl_base.ckpt"
    
    
########################################################################################################################
## Motion Deblurring
########################################################################################################################
torchx run -- -j 1x2 -- \
    -m training=False gpus=2 experiment=db_motion/grl_p480 model=grl/grl_base model.fairscale_checkpoint=True tile=0 tile_overlap=0 \
    data_module.train.dataset=gopro data_module.val.dataset=gopro+hide \
    load_state_dict=True pretrained_checkpoint="${MODEL_ZOO}/GRL/db_motion_grl_base_gopro.ckpt"

torchx run -- -j 1x2 -- \
    -m training=False gpus=2 experiment=db_motion/grl_p480 model=grl/grl_base model.fairscale_checkpoint=True tile=0 tile_overlap=0 \
    data_module.train.dataset=realblur-j data_module.val.dataset=realblur-j \
    load_state_dict=True pretrained_checkpoint="${MODEL_ZOO}/GRL/db_motion_grl_base_realblur_j.ckpt"

torchx run -- -j 1x2 -- \
    -m training=False gpus=2 experiment=db_motion/grl_p480 model=grl/grl_base model.fairscale_checkpoint=True tile=0 tile_overlap=0 \
    data_module.train.dataset=realblur-r data_module.val.dataset=realblur-r \
    load_state_dict=True pretrained_checkpoint="${MODEL_ZOO}/GRL/db_motion_grl_base_realblur_r.ckpt"
```