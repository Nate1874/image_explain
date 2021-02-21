import tensorflow as tf
import os
import sys
import numpy as np
import cv2

from network import Network
from data_input import input_fn
from scipy.misc import imread, imsave
from progress.bar import Bar
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

class Model(object):

    def __init__(self, conf):
        self.conf = conf


    def _model_fn(self, features, labels, mode):
        """Initializes the Model representing the model layers
        and uses that model to build the necessary EstimatorSpecs for
        the `mode` in question. For training, this means building losses,
        the optimizer, and the train op that get passed into the EstimatorSpec.
        For evaluation and prediction, the EstimatorSpec is returned without
        a train op, but with the necessary parameters for the given mode.
        Args:
            features: tensor representing input images
            labels: tensor representing class labels for all input images
            mode: current estimator mode; should be one of
                `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
        Returns:
            ModelFnOps
        """
        
        net = Network(self.conf)
     #   real_labels = tf.identity(labels)
        real_labels = labels
    #    print("labels shape=========", real_labels.dtype)

        gen_logits, mask, img_logits, new_input, new_input_vis, prob, ori_logits = net(features, mode == tf.estimator.ModeKeys.TRAIN)
       # prob = tf.layers.average_pooling2d(prob, 15, 1, padding= 'same')
        labels = tf.argmax(ori_logits, axis=-1)
        labels = tf.cast(labels, tf.int32)
        print("new label shape,", labels.dtype)

        predictions = {
            'classes': tf.argmax(img_logits, axis=-1),
            'probabilities': tf.nn.softmax(img_logits, name='softmax_tensor'),
            'logits': img_logits,
        #    'name': features[-1],
            'mask': mask,
            'raw': features,
            'new_img': new_input,
            'new_input_vis': new_input_vis,
            'prob_map': prob
         #   'true_label': real_labels
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)     
        
        #calculate loss for discriminator
        dis_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            labels=labels, logits=img_logits))

        print('the shape of label is,', labels.get_shape())
        print('the shape of logits is ,', img_logits.get_shape())

    #    label_f = tf.range(0, img_logits.get_shape()[0]) * img_logits.get_shape()[1] + labels
    #    class_sores =  tf.gather(tf.reshape(img_logits, [-1]),label_f)
        class_sores= tf.squeeze(tf.gather_nd(img_logits,tf.stack([tf.range(tf.shape(labels)[0])[...,tf.newaxis], labels[...,tf.newaxis]], axis=2)))
        print(class_sores.get_shape())         


        gradients = tf.gradients(class_sores, new_input)[0]
        print(gradients.get_shape())
    #    gradients = tf.multiply(gradients, new_input)
        print(gradients.get_shape())
        gradients = tf.reduce_max(tf.abs(gradients), axis=-1, keepdims=True)

        gradients_max = tf.contrib.distributions.percentile(gradients,99, axis=(1,2), keep_dims=True)
        gradients_min = tf.reduce_min(gradients, axis=(1,2), keepdims=True)
        print(gradients_max.get_shape())
        print(gradients_min.get_shape())
    #    gradients_norm = tf.clip_by_value(2*tf.divide(tf.subtract(gradients, gradients_min), gradients_max-gradients_min)-1, -1, 1)
        gradients_norm = tf.clip_by_value(tf.divide(tf.subtract(gradients, gradients_min), gradients_max-gradients_min), 0, 1) # 0-1
        print('saliency:', gradients_norm.get_shape())
        print('mask:', mask.get_shape())
     #   threshold = tf.constant(0.2, dtype=tf.float32)


        weights= tf.multiply(tf.abs(0.2-gradients_norm), mask) # 0.2- the gradient in mask
        flags = tf.to_float(tf.greater_equal(gradients_norm, 0.2)) # get the label for cross entropy

        flags = tf.layers.average_pooling2d(flags, 3, 1, padding= 'same') # get smoothed label 

        flags = tf.to_float(tf.greater_equal(flags, 0.5)) 



        gen_loss_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels= flags, logits= gen_logits)

        shift_loss = self.cal_shift_loss(mask)

        area_loss =tf.reduce_mean(mask)
        # Calculate loss, which includes softmax cross entropy and L2 regularization.
        cost = (dis_loss+ self.conf.shift_para *shift_loss + self.conf.area_para *area_loss)

        cost_matrix =tf.divide(100, cost*tf.ones_like(flags))*flags-9*flags + cost*tf.to_float(tf.less_equal(flags, 0.5))

        cost_adjusted = cost_matrix*weights
        print('the weights,', weights.get_shape())
    #    gen_L2 = self.get_gen_l2_loss()
   #     continous_loss = self.cal_shift_loss(prob)*0.5
        gen_loss = tf.reduce_mean(gen_loss_ce*cost_adjusted)# + continous_loss# +gen_L2

        tf.summary.image('original', features,max_outputs=10)
        tf.summary.image('saliency', gradients_norm,max_outputs=10)
        tf.summary.image('labels', flags,max_outputs=10)
        tf.summary.image('mask', mask,max_outputs=10)
        tf.summary.image('new_input', new_input_vis,max_outputs=10)
        
        
        # Create a tensor named cross_entropy for logging purposes.
        tf.identity(gen_loss, name='gen_loss')
        tf.summary.scalar('gen_loss', gen_loss)

        # tf.identity(gen_L2, name='gen_loss_l2')
        # tf.summary.scalar('gen_loss_l2', gen_L2)

        tf.identity(self.conf.shift_para *shift_loss, name='shift_loss')
        tf.summary.scalar('shift_loss', self.conf.shift_para *shift_loss)

        tf.identity(self.conf.area_para *area_loss, name='area_loss')
        tf.summary.scalar('area_loss', self.conf.area_para *area_loss)

        tf.identity(labels, name='pred')
        tf.summary.scalar('pred', labels)

        tf.identity(real_labels, name='raw_label')
        tf.summary.scalar('raw_label', real_labels)

        tf.identity(dis_loss, name='dis_loss')
        tf.summary.scalar('dis_loss', dis_loss)


        # Add weight decay to the loss.
        var_gen = [v for v in tf.global_variables() if ('generator' in v.name)]
        var_dis = [v for v in tf.global_variables() if ('vgg_16' in v.name)]

        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()
            
            # Learning rate.
            # initial_learning_rate = self.conf.learning_rate
            # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
            # boundaries = [int(batches_per_epoch * epoch) for epoch in [150, 200]]
            # vals = [initial_learning_rate * decay for decay in [1, 0.25, 0.25*0.25]]
            # learning_rate = tf.train.piecewise_constant(global_step, boundaries, vals)

            # Create a tensor named learning_rate for logging purposes
            # tf.identity(learning_rate, name='learning_rate')
            # tf.summary.scalar('learning_rate', learning_rate)

            # optimizer = tf.train.MomentumOptimizer(
            # 				learning_rate=learning_rate,
            # 				momentum=self.conf.momentum)
            

            optimizer_gen = tf.train.AdamOptimizer(learning_rate=self.conf.learning_rate_gen)
            optimizer_gen = tf.contrib.estimator.TowerOptimizer(optimizer_gen)
         #   optimizer_dis = tf.train.AdamOptimizer(learning_rate=self.conf.learning_rate_dis)
        #    optimizer_dis = tf.contrib.estimator.TowerOptimizer(optimizer_dis)

            # Batch norm requires update ops to be added as a dependency to train_op
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op_gen = optimizer_gen.minimize(gen_loss, global_step= global_step, var_list= var_gen)
           #     train_op_dis = optimizer_dis.minimize(dis_loss, var_list= var_dis)
            #    train_op = tf.group(train_op_gen, train_op_dis)
                train_op = train_op_gen
        else:
            train_op = None

        print(predictions['classes'])
    #    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
        accuracy = tf.metrics.accuracy(labels, real_labels)
        metrics = {'accuracy': accuracy}

        # Create a tensor named train_accuracy for logging purposes
        tf.identity(accuracy[1], name='train_accuracy')
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=gen_loss+dis_loss,
                train_op=train_op,
                eval_metric_ops=metrics)
        

    def cal_shift_loss(self, mask):
        len_mask = mask.shape[1].value
        mask_right_diff =tf.abs(mask[:,0:len_mask-1,:,:]- mask[:,1:len_mask,:,:])
        mask_right_loss =tf.reduce_mean(mask_right_diff)
        
        mask_up_diff = tf.abs(mask[:,:,0:len_mask-1,:]- mask[:,:,1:len_mask,:])
        mask_up_loss = tf.reduce_mean(mask_up_diff)
        return mask_right_loss+mask_up_loss

    def get_gen_l2_loss(self, l2_lambda=0.0001):
        loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.global_variables() if ('generator' in v.name and 'bias' not in v.name)]) * l2_lambda
        return loss_l2
    
    def train(self):
        # Using the Winograd non-fused algorithms provides a small performance boost.
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
        save_checkpoints_steps = 16000
        distribution = tf.contrib.distribute.MirroredStrategy(
            ["/device:GPU:1", "/device:GPU:2","/device:GPU:3","/device:GPU:4"])

        # config = tf.estimator.RunConfig(save_checkpoints_steps=save_checkpoints_steps,
        #                                 keep_checkpoint_max=0,
        #                                 train_distribute=distribution,
        #                                 eval_distribute=distribution)        

        
        #int(1281167/self.conf.batch_size) 

        run_config = tf.estimator.RunConfig().replace(
                        save_checkpoints_steps=save_checkpoints_steps,
                        keep_checkpoint_max=0)
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=self.conf.vgg_path,
                                        vars_to_warm_start=['.*vgg.*']) 

        classifier = tf.estimator.Estimator(
                        model_fn=tf.contrib.estimator.replicate_model_fn(self._model_fn),
                   #     model_fn = self._model_fn,
                        model_dir=self.conf.model_dir,
                        config=run_config,
                        warm_start_from=ws)

        for l in range(self.conf.train_epochs // self.conf.epochs_per_eval):
            tensors_to_log = {
                'dis_loss': 'dis_loss',
                'shift_loss': 'shift_loss',
                'area_loss': 'area_loss',
                'gen_loss': 'gen_loss',
                'train_accuracy': 'train_accuracy'

            }

            logging_hook = tf.train.LoggingTensorHook(
                                tensors=tensors_to_log, every_n_iter=100)

            print('Starting a training cycle.')

            def input_fn_train():
                return input_fn(
                            is_training =True,
                            data_dir=self.conf.data_dir,
                            batch_size=self.conf.batch_size,
                            num_epochs=self.conf.epochs_per_eval,
                            num_parallel_batches=self.conf.num_parallel_batches)

            classifier.train(input_fn=input_fn_train, hooks=[logging_hook])
            print('One epoch done.')
            if self.conf.validation_in_train != False:
                print('Starting to evaluate.')

                def input_fn_eval():
                    return input_fn(
                                is_training=False,
                                data_dir=self.conf.data_dir,
                                batch_size=self.conf.batch_size,
                                num_epochs=1,
                                num_parallel_batches=self.conf.num_parallel_batches)

                classifier.evaluate(input_fn=input_fn_eval)
                preds= classifier.predict(input_fn= input_fn_eval)
                print('Starting to predict.')

                for i, pred in enumerate(preds):
                    if i == 80:
                        break
                  #  print(pred.shape)                    
                    raw = pred['raw']
                    mask = pred['mask']
                    image_new = pred['new_input_vis']
                    self.save_image(raw, mask, image_new, i, l)
                print('Image Saved.')
    

    def predict(self):
        # Using the Winograd non-fused algorithms provides a small performance boost.
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
     #   save_checkpoints_steps = 16000

        classifier = tf.estimator.Estimator(
                        model_fn=tf.contrib.estimator.replicate_model_fn(self._model_fn),
                   #     model_fn = self._model_fn,
                        model_dir=self.conf.model_dir)

        def input_fn_eval():
            return input_fn(
                        is_training=False,
                        data_dir=self.conf.data_dir,
                        batch_size=self.conf.batch_size,
                        num_epochs=1,
                        num_parallel_batches=self.conf.num_parallel_batches)

        tensors_to_log = {
            'raw_label': 'raw_label'
         #   'pred': 'pred'           
        }
    
        logging_hook = tf.train.LoggingTensorHook(
                            tensors=tensors_to_log, every_n_iter=1)
        checkpoint_file = os.path.join(self.conf.model_dir, 
                            'model.ckpt-'+str(self.conf.checkpoint_num))
  #      classifier.evaluate(input_fn=input_fn_eval, checkpoint_path= checkpoint_file)
    #    classifier.evaluate(input_fn=input_fn_eval, hooks=[logging_hook])
        preds= classifier.predict(input_fn= input_fn_eval, checkpoint_path= checkpoint_file)
        print('Starting to predict.')

        bar = Bar('Processing', max=1281167)
        for i, pred in enumerate(preds):
            # if i == 10:
            #     break
            #  print(pred.shape)
            # if i ==5000:
            #     break                    
            raw = pred['raw']
            prob_map = pred['prob_map']
         #   labels = pred['true_label']
            self.save_image2(raw,prob_map,i)
            bar.next()
         #   mask = pred['mask']
         #   image_new = pred['new_input_vis']
          #  prob_map =pred['prob_map']
          #  self.save_image2(raw, prob_map, i)
          #  self.save_image2(prob_map, i)
        bar.finish()
        print('Image Saved.')


    def saveMask(self, mask, path):
        # mask1 = mask.cpu().data.numpy()[0]
        # # tranpose the image to BGR format
        # mask1 = np.transpose(mask1, (1, 2, 0))
        # # normalize the mask
        # mask1 = (mask1 - np.min(mask1)) / np.max(mask1)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cv2.imwrite(path, np.uint8(255 * heatmap))

 #   def save_image2(self, raw, prob_map, idx):
    def save_image2(self, raw, prob_map, idx, ):
       # print(labels)
        path_raw = self.conf.samledir +str(idx)+'_raw.npy'
        
        path_prob = self.conf.samledir +str(idx)+'_prob.npy'

        np.save(path_raw, raw)
        np.save(path_prob, prob_map)
    #  #   np.save(path,raw)
    #     imsave(path, raw)
    #     path = self.conf.samledir +str(idx)+'_prob_map.png'
    #  #   np.save(path, mask), path
    #   #  path = self.conf.samledir +str(idx)+ '_new_image.png'
    #  #   np.save(path, new_img)
    #     self.saveMask(prob_map[:,:,0], path)
    # #    imsave(path, prob_map[:,:,0])   
    #    path = self.conf.samledir + name_new + '.npz'
   #     np.save(path, prob_map)

    def save_image(self, raw, mask, new_img, idx, epoch):
        path = self.conf.samledir +str(idx)+ '_epoch'+ str(epoch)+'.png'
     #   np.save(path,raw),
        imsave(path, raw)
        path = self.conf.samledir +str(idx)+ '_epoch'+ str(epoch)+'_mask.png'
     #   np.save(path, mask)
        mask = mask.astype(np.uint8)*255
        imsave(path, mask[:,:,0])
        path = self.conf.samledir +str(idx)+ '_epoch'+ str(epoch)+'_new_image.png'
     #   np.save(path, new_img)
        imsave(path, new_img)        
        


    def _mean_image_add(self, image, means=_CHANNEL_MEANS):
        """Subtracts the given means from each image channel.

        For example:
            means = [123.68, 116.779, 103.939]
            image = _mean_image_subtraction(image, means)

        Note that the rank of `image` must be known.

        Args:
            image: a tensor of size [height, width, C].
            means: a C-vector of values to subtract from each channel.
            num_channels: number of color channels in the image that will be distorted.

        Returns:
            the centered image.

        Raises:
            ValueError: If the rank of `image` is unknown, if `image` has a rank other
            than three or if the number of channels in `image` doesn't match the
            number of values in `means`.
        """


        # We have a 1-D tensor of means; convert to 3-D.
        means = tf.expand_dims(tf.expand_dims(means, 0), 0)

        return image + means





