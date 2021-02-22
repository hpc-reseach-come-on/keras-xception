import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from dataset import create_dataset, create_cifar10, limin_create_dataset, limin_create_cifar10, create_random_dataset
from config import xception_cfg as cfg
#from config_cifar10 import cifar10_cfg as cfg
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import os


# os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'

def main():
    train_dataset, val_dataset = create_dataset()
    #print(type(train_dataset))
    
    #train_dataset, val_dataset = create_cifar10()
    #train_dataset, val_dataset = create_random_dataset()
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = Xception(weights=None, classes=cfg.num_classes)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=cfg.lr_init,
                                                                     decay_steps=cfg.lr_decay_epoch * int(
                                                                         1280000 / cfg.batch_size + 1),
                                                                     decay_rate=cfg.lr_decay_rate, staircase=True,
                                                                     name=None)
        # opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(tf.keras.optimizers.SGD(momentum=cfg.momentum,
        # learning_rate=lr_schedule),loss_scale='dynamic')
        model.compile(optimizer=tf.keras.optimizers.SGD(momentum=cfg.momentum, learning_rate=lr_schedule),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        save_options = tf.saved_model.SaveOptions()  # experimental_io_device='/job:localhost')
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='./model/weights.{epoch:02d}.smooth.hdf5',
            options=save_options)

        model_tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir='./log', histogram_freq=0, write_graph=False, write_images=False,
            update_freq=5, profile_batch=2, embeddings_freq=0,
            embeddings_metadata=None)

    model.fit(train_dataset, epochs=cfg.epoch_size, callbacks=[model_tensorboard], validation_data=val_dataset,
                  initial_epoch=0)


if __name__ == "__main__":
    main()