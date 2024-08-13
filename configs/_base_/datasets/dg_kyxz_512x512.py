_base_ = [
    "./kyxz_512x512.py",
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset={{_base_.train_kyxz}},
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset={{_base_.val_kyxz}},
)
# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset={{_base_.test_kyxz}},
)
val_evaluator = dict(
    type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["bev"]
)
# test_evaluator=val_evaluator
test_evaluator = dict(
    type='DGIoUMetric',
    iou_metrics=['mIoU'],
    format_only=True,
    output_dir='work_dirs/format_results')
