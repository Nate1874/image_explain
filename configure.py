import tensorflow as tf


"""This script defines hyperparameters.
"""

def configure():
    flags = tf.app.flags
    # training
    flags.DEFINE_string('data_dir', '/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/imagenet/',
            'the directory where the input data is stored')
    flags.DEFINE_integer('train_epochs', 1000,
            'the number of epochs to use for training')
    flags.DEFINE_integer('epochs_per_eval', 4,
            'the number of training epochs to run between evaluations')
    flags.DEFINE_integer('batch_size', 20,
            'the number of examples processed in each training batch')
    flags.DEFINE_float('learning_rate_gen', 1e-3, 'learning rate for generator')
    flags.DEFINE_float('learning_rate_dis', 5e-6, 'learning rate for discriminator')
    flags.DEFINE_float('shift_para', 3, 'hyper-parameter for shifting loss')
    flags.DEFINE_float('area_para', 3.5, 'hyper-parameter for area loss rate')
    flags.DEFINE_integer('num_parallel_batches', 15,
            'The number of records that are processed in parallel \
            during input processing. This can be optimized per data set but \
            for generally homogeneous data sets, should be approximately the \
            number of available CPU cores.')

    flags.DEFINE_string('model_dir', '/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/model_new',
            'the directory where the model will be stored')
    flags.DEFINE_string('vgg_path', '/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/vgg_16.ckpt',
            'the directory where the model will be stored')
    # validation / prediction
    flags.DEFINE_integer('checkpoint_num', 912767,
            'which checkpoint is used for validation/prediction')
    flags.DEFINE_string('samledir', '/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/all_files/',
            'the directory where the prediction images is stored')
#     flags.DEFINE_string('samledir', '/home/grads/h/hao.yuan/Dropbox/code_tamu/interpretation/results_vis/',
#             'the directory where the prediction images is stored')
    flags.DEFINE_boolean('validation_in_train', True,
            'if we perform evaluation during training')


    # network
    flags.DEFINE_integer('network_depth', 4, 'the network depth')
    flags.DEFINE_integer('num_classes', 1000, 'the number of classes')
    flags.DEFINE_integer('start_channel_num', 64,
                'number of filters for initial_conv')
    flags.DEFINE_integer('encoder_out_num', 1,
                'number of output channels of the encoder')	
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


conf = configure()
