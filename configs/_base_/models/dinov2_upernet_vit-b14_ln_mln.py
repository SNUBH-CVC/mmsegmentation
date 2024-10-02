# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
# mean, std 같은 값 사용
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],  
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DinoVisionTransformer',
        img_size=518,
        patch_size=14,
        output_cls_token=False,
        out_indices=(2, 5, 8, 11),
        ckpt_path="./pretrain/dinov2_vitb14_pretrain.pth",
    ),
    neck=dict(
        type='MultiLevelNeck',
        in_channels=[768, 768, 768, 768],
        out_channels=768,
        scales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
        ]),  
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
        ]),  
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable
