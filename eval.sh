# CUDA_VISIBLE_DEVICES=1 python tools/val.py \
# --config configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py \
# --checkpoint checkpoints/dinov2_rein_and_head.pth \
# --backbone dinov2_converted.pth

### acdc to acdc
# folder="rein_dinov2_mask2former_acdc_512x512_bs1x4"
# CUDA_VISIBLE_DEVICES=1 python tools/val.py \
# --config configs/dinov2_acdc/${folder}.py \
# --checkpoint work_dirs/${folder}/iter_40000.pth \
# --backbone checkpoints/dinov2_converted.pth \
# --show_dir work_dirs/${folder}/pred \
# --out work_dirs/${folder}/pred_out

### city to acdc
# folder="rein_dinov2_mask2former_1024x1024_bs1x2"
# CUDA_VISIBLE_DEVICES=1 python tools/val.py \
# --config configs/dinov2_citys2acdc/${folder}.py \
# --checkpoint work_dirs/${folder}/iter_40000.pth \
# --backbone checkpoints/dinov2_converted_1024x1024.pth \
# --show_dir work_dirs/${folder}/pred \
# --out work_dirs/${folder}/pred_city_val

### bev2024
# CUDA_VISIBLE_DEVICES=1 python tools/val.py \
# --config configs/dinov2/rein_dinov2_mask2former_bev_512x512_bs1x4.py \
# --checkpoint work_dirs/rein_dinov2_mask2former_bev_512x512_bs1x4/iter_10000.pth \
# --backbone checkpoints/dinov2_converted.pth \
# --show_dir work_dirs/rein_dinov2_mask2former_bev_512x512_bs1x4/pred \
# --out work_dirs/rein_dinov2_mask2former_bev_512x512_bs1x4/pred_out

### bev20234
# exp_folder="rein_dinov2_mask2former_bev20234_512x512_bs1x4_4witers"
# CUDA_VISIBLE_DEVICES=1 python tools/val.py \
# --config configs/dinov2/rein_dinov2_mask2former_bev20234_512x512_bs1x4.py \
# --checkpoint work_dirs/${exp_folder}/iter_40000.pth \
# --backbone checkpoints/dinov2_converted.pth \
# --show_dir work_dirs/${exp_folder}/pred \
# --out work_dirs/${exp_folder}/pred_trainid

### HYRoad
# folder_name="rein_dinov2_mask2former_HYroad_512x512_bs1x4_4witers"
folder_name="rein_dinov2_mask2former_HYRoad_1024x1024_bs1x2"
model_name="iter_40000.pth"
CUDA_VISIBLE_DEVICES=0 python tools/val.py \
--config configs/dinov2/${folder_name}.py \
--checkpoint work_dirs/${folder_name}/${model_name} \
--backbone checkpoints/dinov2_converted_1024x1024.pth \
--show_dir work_dirs/${folder_name}/pred \
--out work_dirs/${folder_name}/pred_trainid