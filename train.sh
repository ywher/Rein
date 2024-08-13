# dg, gta
# python tools/train.py configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py

# dg, city to acdc
# CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --config configs/dinov2_citys2acdc/rein_dinov2_mask2former_512x512_bs1x4.py

# CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --config configs/dinov2_citys2acdc/rein_dinov2_mask2former_1024x1024_bs1x2.py

# CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --config configs/dinov2_acdc/rein_dinov2_mask2former_acdc_1024x1024_bs1x2.py


### dg kyxz
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
--config configs/dinov2/rein_dinov2_mask2former_kyxz_512x512_bs1x4.py

# PORT=12345 CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh configs/dinov2/rein_dinov2_mask2former_1024x1024_bs4x2.py 2

# dg, bev to bev
# bev_2024
# CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --config configs/dinov2/rein_dinov2_mask2former_bev_512x512_bs1x4.py

# bev_20234_1024
# CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --config configs/dinov2/rein_dinov2_mask2former_bev20234_512x512_bs1x4.py

# dg HYRoad to HYRoad
# CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --config configs/dinov2/rein_dinov2_mask2former_HYroad_512x512_bs1x4.py

# dg mapillary to mapillary
# CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --config configs/dinov2/rein_dinov2_mask2former_mapillary_512x512_bs1x4.py