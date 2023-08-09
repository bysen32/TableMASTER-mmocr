_base_ = [
    '../../_base_/default_runtime.py'
]


alphabet_file = './tools/data/alphabet/structure_alphabet_handmade.txt'
alphabet_len = len(open(alphabet_file, 'r').readlines())
max_seq_len = 500

start_end_same = False
label_convertor = dict(
            type='TableMasterConvertor',
            dict_file=alphabet_file,
            max_seq_len=max_seq_len,
            start_end_same=start_end_same,
            with_unknown=True)

if start_end_same:
    PAD = alphabet_len + 2
else:
    PAD = alphabet_len + 3

model = dict(
    type='TABLEMASTER',
    backbone=dict(
        type='TableResNetExtra',
        input_dim=3,
        gcb_config=dict(
            ratio=0.0625,
            headers=1,
            att_scale=False,
            fusion_type="channel_add",
            layers=[False, True, True, True],
        ),
        layers=[1,2,5,3]),
    encoder=dict(
        type='PositionalEncoding',
        d_model=512,
        dropout=0.2,
        max_len=5000),
    decoder=dict(
        type='TableMasterDecoder',
        N=3,
        decoder=dict(
            self_attn=dict(
                headers=8,
                d_model=512,
                dropout=0.),
            src_attn=dict(
                headers=8,
                d_model=512,
                dropout=0.),
            feed_forward=dict(
                d_model=512,
                d_ff=2024,
                dropout=0.),
            size=512,
            dropout=0.),
        d_model=512),
    loss=dict(type='MASTERTFLoss', ignore_index=PAD, reduction='mean'),
    bbox_loss=dict(type='TableL1Loss', reduction='sum'),
    label_convertor=label_convertor,
    max_seq_len=max_seq_len)

albu_train_transforms = [
    dict(type='ImageCompression', p=0.05),

    dict(type='RandomShadow', p=0.05),

    dict(type='OneOf',
         transforms=[
            dict(type='RandomBrightnessContrast', p=1.0),
            dict(type='HueSaturationValue', p=1.0),
            dict(type='ColorJitter', p=1.0),
            dict(type="CLAHE", p=1.0),
            dict(type='FancyPCA', p=1.0),
         ],
         p=0.05),
    
    dict(type='OneOf',
         transforms=[
            dict(type='RGBShift', p=1.0),
            dict(type='ToGray', p=1.0),
            dict(type='RandomGamma', p=1.0),
            dict(type='RandomToneCurve', p=1.0),
         ],
         p=0.05),

    dict(type='OneOf',
        transforms=[
            dict(type='ChannelShuffle', p=1.0),
        ],
        p=0.05),

    dict(type='OneOf',
        transforms=[
            dict(type='AdvancedBlur', p=1.0),
            dict(type='GaussianBlur', p=1.0),
            dict(type='Sharpen', p=1.0),
        ],
        p=0.05),
    
    dict(
        type='OneOf',
        transforms=[
            dict(type='GaussNoise', p=1.0),
            dict(type='ISONoise', p=1.0),
        ],
        p=0.05),

    dict(type='PixelDropout', p=0.05)
]

TRAIN_STATE = True
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomLineMask', p=0.6),
    dict(
        type='TableAspect',
        ratio=(0.8, 1.25),
        p=0.2,
    ),
    dict(
        type='TableRotate',
        p=0.02,
    ),
    dict(
        type='TableResize',
        keep_ratio=True,
        long_size=480),
    dict(
        type='TablePad',
        size=(480, 480),
        pad_val=0,
        return_mask=True,
        mask_ratio=(8, 8),
        train_state=TRAIN_STATE),
    dict(
        type="Albu",
        transforms=albu_train_transforms,),
    dict(type='TableBboxEncode'),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'scale_factor',
            'bbox', 'bbox_masks', 'pad_shape',
        ]),
]

valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TableResize',
        keep_ratio=True,
        long_size=480),
    dict(
        type='TablePad',
        size=(480, 480),
        pad_val=0,
        return_mask=True,
        mask_ratio=(8, 8),
        train_state=TRAIN_STATE),
    dict(type='TableBboxEncode'),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'scale_factor',
            'img_norm_cfg', 'ori_filename',
            'bbox', 'bbox_masks', 'pad_shape',
        ]),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TableResize',
        keep_ratio=True,
        long_size=480),
    dict(
        type='TablePad',
        size=(480, 480),
        pad_val=0,
        return_mask=True,
        mask_ratio=(8, 8),
        train_state=TRAIN_STATE),
    #dict(type='TableBboxEncode'),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'scale_factor',
            'img_norm_cfg', 'ori_filename', 'pad_shape'
        ]),
]

dataset_type = 'TEDSDataset'
train_img_prefix = '/media/ubuntu/Date12/TableStruct/new_data/train_jpg480max/'
train_anno_file1 = '/media/ubuntu/Date12/TableStruct/new_data/tablemaster_wire/10fold8/cell_box_label/StructureLabelAddEmptyBbox_train/'
train1 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix,
    ann_file=train_anno_file1,
    loader=dict(
        type='JsonLoader',
        repeat=1,
        max_seq_len=max_seq_len,
        parser=dict(
            type='TableStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)

valid_img_prefix = '/media/ubuntu/Date12/TableStruct/new_data/train_jpg480max/'
valid_anno_file1 = '/media/ubuntu/Date12/TableStruct/new_data/tablemaster_wire/10fold8/cell_box_label/StructureLabelAddEmptyBbox_valid/'
valid = dict(
    type=dataset_type,
    img_prefix=valid_img_prefix,
    ann_file=valid_anno_file1,
    loader=dict(
        type='JsonLoader',
        repeat=1,
        max_seq_len=max_seq_len,
        parser=dict(
            type='TableStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=valid_pipeline,
    dataset_info='table_master_dataset',
    test_mode=True)

test_img_prefix = '/media/ubuntu/Date12/TableStruct/new_data/train_jpg480max/'
test_anno_file1 = '/media/ubuntu/Date12/TableStruct/new_data/tablemaster_wire/10fold8/cell_box_label/StructureLabelAddEmptyBbox_valid/'
test = dict(
    type=dataset_type,
    img_prefix=test_img_prefix,
    ann_file=test_anno_file1,
    loader=dict(
        type='JsonLoader',
        repeat=1,
        max_seq_len=max_seq_len,
        parser=dict(
            type='TableStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=test_pipeline,
    dataset_info='table_master_dataset',
    test_mode=True)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(type='ConcatDataset', datasets=[train1]),
    val=dict(type='ConcatDataset', datasets=[valid]),
    test=dict(type='ConcatDataset', datasets=[test]))

# optimizer
optimizer = dict(type='Ranger', lr=1e-3)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-5)
total_epochs = 30

# evaluation
evaluation = dict(interval=2, start=24, metric='acc')

# fp16
fp16 = dict(loss_scale='dynamic')

# checkpoint setting
checkpoint_config = dict(interval=2)

# log_config
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook')
    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
load_from = './checkpoints/pubtabnet_epoch_16_0.7767.pth'
# load_from = './checkpoints/10fold0_best.pth'
resume_from = None
# resume_from = './checkpoints/epoch_30_val93.03.pth'
workflow = [('train', 1)]

# if raise find unused_parameters, use this.
# find_unused_parameters = True