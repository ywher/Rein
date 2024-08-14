bev_20234_type = "CityscapesDataset"
bev_20234_root = "data/bev_20234_1024/"  # bev_20234_1024
bev_20234_crop_size = (1024, 1024)
bev_20234_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    # dict(type="Resize", scale=(1024, 1024)),  # (1024, 512)
    dict(
        type="RandomChoiceResize",
        scales=[int(1024 * x * 0.1) for x in range(5, 21)],
        resize_type="ResizeShortestEdge",
        max_size=2048,
    ),
    dict(type="RandomCrop", crop_size=bev_20234_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
bev_20234_val_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
bev_20234_test_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_bev_20234 = dict(
    type=bev_20234_type,
    data_root=bev_20234_root,
    data_prefix=dict(
        img_path="image/train",
        seg_map_path="label/train",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=bev_20234_train_pipeline,
)
val_bev_20234 = dict(
    type=bev_20234_type,
    data_root=bev_20234_root,
    data_prefix=dict(
        img_path="image/val",
        seg_map_path="label/val",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=bev_20234_val_pipeline,
)
test_bev_20234 = dict(
    type=bev_20234_type,
    data_root=bev_20234_root,
    data_prefix=dict(
        img_path="1-L/avm",
    ),
    img_suffix=".png",
    pipeline=bev_20234_test_pipeline,
)
