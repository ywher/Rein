acdc_type = "CityscapesDataset"
acdc_root = "data/acdc/"
acdc_crop_size = (1024, 1024)
acdc_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomChoiceResize",
        scales=[int(540 * x * 0.1) for x in range(5, 21)],
        resize_type="ResizeShortestEdge",
        max_size=2048,
    ),
    dict(type="RandomCrop", crop_size=acdc_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
acdc_val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(960, 540), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
acdc_test_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="Resize", scale=(960, 540), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_acdc = dict(
    type=acdc_type,
    data_root=acdc_root,
    data_prefix=dict(
        img_path="rgb_anon/train",
        seg_map_path="gt/train",
    ),
    img_suffix="_rgb_anon.png",
    seg_map_suffix="_gt_labelTrainIds.png",
    pipeline=acdc_train_pipeline,
)
val_acdc = dict(
    type=acdc_type,
    data_root=acdc_root,
    data_prefix=dict(
        img_path="rgb_anon/val",
        seg_map_path="gt/val",
    ),
    img_suffix="_rgb_anon.png",
    seg_map_suffix="_gt_labelTrainIds.png",
    pipeline=acdc_val_pipeline,
)
test_acdc = dict(
    type=acdc_type,
    data_root=acdc_root,
    data_prefix=dict(
        img_path="rgb_anon/test",
    ),
    img_suffix="_rgb_anon.png",
    pipeline=acdc_test_pipeline,
)

