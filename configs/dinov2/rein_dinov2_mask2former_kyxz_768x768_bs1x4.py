# dataset config
_base_ = [
    "../_base_/datasets/dg_kyxz_768x768.py",
    "../_base_/default_runtime.py",
    "../_base_/models/rein_6cls_dinov2_mask2former.py"
]
crop_size = (768, 768)
model = dict(
    backbone=dict(
        img_size=(768, 768),
        init_cfg=dict(
            checkpoint="checkpoints/dinov2_converted_768x768.pth",
        ),
    ),
    data_preprocessor=dict(
        size=crop_size,
    ),
    test_cfg=dict(
        crop_size=(768, 768),
        stride=(683, 683),
    ),
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomChoiceResize",
        scales=[int(768 * x * 0.1) for x in range(5, 21)],
        resize_type="ResizeShortestEdge",
        max_size=1536,
    ),
    dict(type="RandomCrop", crop_size={{_base_.crop_size}}, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
train_dataloader = dict(batch_size=4, dataset=dict(pipeline=train_pipeline))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(
        type="AdamW", lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            "norm": dict(decay_mult=0.0),
            "query_embed": embed_multi,
            "level_embed": embed_multi,
            "learnable_tokens": embed_multi,
            "reins.scale": embed_multi,
        },
        norm_decay_mult=0.0,
    ),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=10000),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=0.9,
        begin=10000,
        end=40000,
        by_epoch=False,
    ),
]

# training schedule for 160k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=5000)  # 40000, 10000
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=100, log_metric_by_epoch=False),  # 50
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=5000, max_keep_ckpts=3  # 4000
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
