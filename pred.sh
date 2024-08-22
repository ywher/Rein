### cityscapes to acdc
# exp_folder="rein_dinov2_mask2former_1024x1024_bs1x2"
# CUDA_VISIBLE_DEVICES=1 python tools/test.py \
# --config configs/dinov2_citys2acdc/${exp_folder}.py \
# --checkpoint work_dirs/${exp_folder}/iter_40000.pth \
# --backbone checkpoints/dinov2_converted_1024x1024.pth \
# --out work_dirs/${exp_folder}/pred_acdc_test \

# exp_folder="rein_dinov2_mask2former_acdc_1024x1024_bs1x2"
# CUDA_VISIBLE_DEVICES=0 python tools/test.py \
# --config configs/dinov2_acdc/${exp_folder}.py \
# --checkpoint work_dirs/${exp_folder}/iter_40000.pth \
# --backbone checkpoints/dinov2_converted_1024x1024.pth \
# --out work_dirs/${exp_folder}/pred_acdc_test \

### bev20234
# exp_folder="rein_dinov2_mask2former_bev20234_512x512_bs1x4_4witers"
# CUDA_VISIBLE_DEVICES=1 python tools/test.py \
# --config configs/dinov2/rein_dinov2_mask2former_bev20234_512x512_bs1x4.py \
# --checkpoint work_dirs/${exp_folder}/iter_40000.pth \
# --backbone checkpoints/dinov2_converted.pth \
# --out work_dirs/${exp_folder}/pred_1-L \


### HYRoad
# exp_folder="rein_dinov2_mask2former_HYRoad_512x512_bs1x4_4witers"
exp_folder="rein_dinov2_mask2former_HYRoad_1024x1024_bs1x2"
config_path="configs/dinov2/rein_dinov2_mask2former_HYRoad_1024x1024_bs1x2.py"
CUDA_VISIBLE_DEVICES=1 python tools/test.py \
--config ${config_path} \
--checkpoint work_dirs/${exp_folder}/iter_40000.pth \
--backbone checkpoints/dinov2_converted_1024x1024.pth \
--out work_dirs/${exp_folder}/pred_all_trainid \

### 
# CUDA_VISIBLE_DEVICES=0 python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --show-dir ${SHOW_DIR} \
# --opacity 1 --eval-option imgfile_prefix=labelTrainIds to_label_id=False --eval mIoU --save_logits 