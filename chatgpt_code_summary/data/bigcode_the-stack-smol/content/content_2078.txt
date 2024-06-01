# model settings
model = dict(
    type='CenterNet',
    pretrained='./pretrain/darknet53.pth',
    backbone=dict(
        type='DarknetV3',
        layers=[1, 2, 8, 8, 4],
        inplanes=[3, 32, 64, 128, 256, 512],
        planes=[32, 64, 128, 256, 512, 1024],
        norm_cfg=dict(type='BN'),
        out_indices=(1, 2, 3, 4),
        frozen_stages=1,
        norm_eval=False),
    neck=dict(type='None'),
    bbox_head=dict(
        type='CXTHead',
        inplanes=(128, 256, 512, 1024),
        head_conv=128,
        wh_conv=64,
        use_deconv=False,
        norm_after_upsample=False,
        hm_head_conv_num=2,
        wh_head_conv_num=2,
        ct_head_conv_num=1,
        fovea_hm=False,
        num_classes=81,
        use_exp_wh=False,
        wh_offset_base=16,
        wh_agnostic=True,
        wh_heatmap=True,
        shortcut_cfg=(1, 2, 3),
        shortcut_attention=(False, False, False),
        norm_cfg=dict(type='BN'),
        norm_wh=False,
        hm_center_ratio=0.27,
        hm_init_value=None,
        giou_weight=5.,
        merge_weight=1.,
        hm_weight=1.,
        ct_weight=1.))
cudnn_benchmark = True
# training and testing settings
train_cfg = dict(
    vis_every_n_iters=100,
    debug=False)
test_cfg = dict(
    score_thr=0.01,
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0004,
                 paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 5,
    step=[9, 11])
checkpoint_config = dict(save_every_n_steps=200, max_to_keep=1, keep_every_n_epochs=9)
bbox_head_hist_config = dict(
    model_type=['ConvModule', 'DeformConvPack'],
    sub_modules=['bbox_head'],
    save_every_n_steps=200)
# yapf:disable
log_config = dict(interval=20)
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'ttf53_whh_3lr_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
