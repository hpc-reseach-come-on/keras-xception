from easydict import EasyDict as edict

xception_cfg = edict({
    'num_classes': 1000,
    'train_dir':'/gdata/ImageNet2012/train/',
    'train_file':'/gdata/ImageNet2012/train_label.txt',
    'val_dir':'/userhome/data/imagenet/eval/',
    'val_file':'/gdata/ImageNet2012/validation_label.txt',
    'learning_rate': 0.001,
    'lr_init': 0.045,
    'lr_decay_rate': 0.94,
    'lr_decay_epoch': 2,
    'momentum': 0.9,
    'epoch_size': 250,
    'batch_size': 512,
    'buffer_size': 1000,
    'image_height': 299,
    'image_width': 299,
    'save_checkpoint_steps': 1562,
    'keep_checkpoint_max': 1000,
})