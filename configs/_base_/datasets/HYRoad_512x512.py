HYRoad_type = "CityscapesDataset"
HYRoad_root = "data/HYRoad/"
HYRoad_crop_size = (512, 512)
HYRoad_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    # dict(type="Resize", scale=(512, 512)),  # (1024, 512)
    dict(type="RandomCrop", crop_size=HYRoad_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
HYRoad_test_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_HYRoad = dict(
    type=HYRoad_type,
    data_root=HYRoad_root,
    data_prefix=dict(
        img_path="image/train",
        seg_map_path="label/train",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=HYRoad_train_pipeline,
)
val_HYRoad = dict(
    type=HYRoad_type,
    data_root=HYRoad_root,
    data_prefix=dict(
        img_path="image/val",
        seg_map_path="label/val",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=HYRoad_test_pipeline,
)
