_base_ = [
    '../_base_/models/dinov2_upernet_vit-b14_ln_mln.py',
    '../_base_/datasets/cag.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model_wrapper_cfg=dict(
    type='MMDistributedDataParallel', find_unused_parameters=True)

crop_size = (518, 518)
# dataset settings
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=None,
    std=None,
    bgr_to_rgb=True,
    size=crop_size,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=2)
test_dataloader = val_dataloader
