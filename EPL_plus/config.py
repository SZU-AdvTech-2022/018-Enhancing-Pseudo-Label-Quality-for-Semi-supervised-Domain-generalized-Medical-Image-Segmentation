
name = 'MMS_deeplab_2%_D_CM'
# hyperparameter
default_config = dict(
    batch_size=8,         # 32
    num_epoch=60,          # 50
    learning_rate=1e-3,            # learning rate of Adam
    weight_decay=0.01,             # weight decay 
    num_workers=8,

    train_name = name,
    model_path = name+'.pt',
    test_vendor = 'D',
    ratio = 0.02,                   # 2%
    CPS_weight = 1,  #3
    gpus= [0,2,3],  #[0, 1, 2, 3]
    ifFast = False,
    Pretrain = True,
    pretrain_file = "/home/jiangsiyao/projects/EPL_SemiDG-master/resnet50_v1c.pth",

    restore = False,
    restore_from = name+'.pt',

    # for cutmix
    cutmix_mask_prop_range = (0.25, 0.5),
    cutmix_boxmask_n_boxes = 3,
    cutmix_boxmask_fixed_aspect_ratio = True,
    cutmix_boxmask_by_size = True,
    cutmix_boxmask_outside_bounds = True,
    cutmix_boxmask_no_invert = True,

    Fourier_aug = True,
    fourier_mode = 'AS',

    # for bn
    bn_eps = 1e-5,
    bn_momentum = 0.1,
)
