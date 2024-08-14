kyxz_type = "CityscapesDataset"
kyxz_root = "data/kyxz/"
kyxz_crop_size = (768, 768)
kyxz_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    # dict(type="Resize", scale=(512, 512)),  # (1024, 512)
    dict(type="RandomCrop", crop_size=kyxz_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
kyxz_val_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
kyxz_test_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_kyxz = dict(
    type=kyxz_type,
    data_root=kyxz_root,
    data_prefix=dict(
        img_path="image/train",
        seg_map_path="label/train",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=kyxz_train_pipeline,
)
val_kyxz = dict(
    type=kyxz_type,
    data_root=kyxz_root,
    data_prefix=dict(
        img_path="image/val",
        seg_map_path="label/val",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=kyxz_val_pipeline,
)
test_kyxz = dict(
    type=kyxz_type,
    data_root=kyxz_root,
    data_prefix=dict(
        img_path="image/all",
    ),
    img_suffix=".png",
    pipeline=kyxz_test_pipeline,
)
