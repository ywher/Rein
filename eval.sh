# CUDA_VISIBLE_DEVICES=1 python tools/test.py \
# --config configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py \
# --checkpoint checkpoints/dinov2_rein_and_head.pth \
# --backbone dinov2_converted.pth

### city to acdc
# CUDA_VISIBLE_DEVICES=1 python tools/test.py \
# --config configs/dinov2_citys2acdc/rein_dinov2_mask2former_512x512_bs1x4.py \
# --checkpoint work_dirs/rein_dinov2_mask2former_512x512_bs1x4/iter_40000.pth \
# --backbone checkpoints/dinov2_converted.pth \
# --show_dir work_dirs/rein_dinov2_mask2former_512x512_bs1x4/pred \
# --out work_dirs/rein_dinov2_mask2former_512x512_bs1x4/pred_out


### bev2024
CUDA_VISIBLE_DEVICES=1 python tools/test.py \
--config configs/dinov2/rein_dinov2_mask2former_bev_512x512_bs1x4.py \
--checkpoint work_dirs/rein_dinov2_mask2former_bev_512x512_bs1x4/iter_10000.pth \
--backbone checkpoints/dinov2_converted.pth \
--show_dir work_dirs/rein_dinov2_mask2former_bev_512x512_bs1x4/pred \
--out work_dirs/rein_dinov2_mask2former_bev_512x512_bs1x4/pred_out