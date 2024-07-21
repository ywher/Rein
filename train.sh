# dg, gta
# python tools/train.py configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py

# dg, city to acdc
# CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --config configs/dinov2_citys2acdc/rein_dinov2_mask2former_512x512_bs1x4.py

# dg, bev to bev
# bev_2024
# CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --config configs/dinov2/rein_dinov2_mask2former_bev_512x512_bs1x4.py

# bev_20234_1024
# CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --config configs/dinov2/rein_dinov2_mask2former_bev20234_512x512_bs1x4.py

# dg HYRoad to HYRoad
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
--config configs/dinov2/rein_dinov2_mask2former_HYroad_512x512_bs1x4.py

# dg mapillary to mapillary
# CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --config configs/dinov2/rein_dinov2_mask2former_mapillary_512x512_bs1x4.py