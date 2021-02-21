# Copyright 2017 Ruth Fong. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to compute SaliencyMasks."""
import numpy as np
import tensorflow as tf
import os
from base import SaliencyMask
from progress.bar import Bar
class GradCam(SaliencyMask):
    """A SaliencyMask class that computes saliency masks with Grad-CAM.
  
    https://arxiv.org/abs/1610.02391

    Example usage (based on Examples.ipynb):

    grad_cam = GradCam(graph, sess, y, images, conv_layer = end_points['Mixed_7c'])
    grad_mask_2d = grad_cam.GetMask(im, feed_dict = {neuron_selector: prediction_class}, 
                                    should_resize = False, 
                                    three_dims = False)

    The Grad-CAM paper suggests using the last convolutional layer, which would 
    be 'Mixed_5c' in inception_v2 and 'Mixed_7c' in inception_v3.

    """
    def __init__(self, graph, session, y, x, conv_layer):
        super(GradCam, self).__init__(graph, session, y, x)
        self.conv_layer = conv_layer
        self.gradients_node = tf.gradients(y, conv_layer)[0]

    def GetMask(self, x_value, feed_dict={}, should_resize = True, three_dims = True):
        """
        Returns a Grad-CAM mask.
        
        Modified from https://github.com/Ankush96/grad-cam.tensorflow/blob/master/main.py#L29-L62

        Args:
          x_value: Input value, not batched.
          feed_dict: (Optional) feed dictionary to pass to the session.run call.
          should_resize: boolean that determines whether a low-res Grad-CAM mask should be 
              upsampled to match the size of the input image
          three_dims: boolean that determines whether the grayscale mask should be converted
              into a 3D mask by copying the 2D mask value's into each color channel
            
        """
        feed_dict[self.x] = [x_value]
        (output, grad) = self.session.run([self.conv_layer, self.gradients_node], 
                                               feed_dict=feed_dict)
        output = output[0]
        grad = grad[0]

        weights = np.mean(grad, axis=(0,1))
        grad_cam = np.ones(output.shape[0:2], dtype=np.float32)

        # weighted average
        for i, w in enumerate(weights):
            grad_cam += w * output[:, :, i]

        # pass through relu
        grad_cam = np.maximum(grad_cam, 0)

        # resize heatmap to be the same size as the input
        if should_resize:
            grad_cam = grad_cam / np.max(grad_cam) # values need to be [0,1] to be resized
            with self.graph.as_default():
                grad_cam = np.squeeze(tf.image.resize_bilinear(
                    np.expand_dims(np.expand_dims(grad_cam, 0), 3), 
                    x_value.shape[:2]).eval(session=self.session))

        # convert grayscale to 3-D
        if three_dims:
            grad_cam = np.expand_dims(grad_cam, axis=2)
            grad_cam = np.tile(grad_cam,[1,1,3])

        return grad_cam


import sys

sys.path.append('/mnt/dive/shared/hao.yuan/models/research/slim')

# old_cwd = os.getcwd()
# os.chdir('/mnt/dive/shared/hao.yuan/models/research/slim')
from nets import vgg
from scipy.misc import imread, imsave, imresize
ckpt ='/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/vgg_16.ckpt'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

label_list= np.load('/mnt/dive/shared/hao.yuan/tempspace2/hyuan/label_train.npy')
indexes  =np.load('/mnt/dive/shared/hao.yuan/tempspace2/hyuan/train_idx.npy')
print('Starting ===================')
my_path  = '/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/all_files/'
save_path = '/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/results_train/'


graph=tf.Graph()
with graph.as_default():
    images = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))

    logits,end_points = vgg.vgg_16(images)
 #   print(end_points)
    sess = tf.Session(graph=graph)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt)
    logits = graph.get_tensor_by_name('vgg_16/fc8/squeezed:0')
    neuron_selector = tf.placeholder(tf.int32)
    y = logits[0][neuron_selector]
    prediction = tf.argmax(logits, 1)
    grad_cam = GradCam(graph, sess, y, images, conv_layer = end_points['vgg_16/conv5/conv5_3'])


batch_size= 1
bar = Bar('Processing', max=int(64497/batch_size))
for i in range(64497):
    idx = indexes[i]
 #   print('1111')
    file_path = my_path+str(idx)+'_raw.npy'


#    path = '/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/results_npy/'+str(i)+'_raw.npy'
    im  = np.load(file_path)
    prediction_class = sess.run(prediction, feed_dict = {images: [im]})[0]
    
    grad_mask_2d = grad_cam.GetMask(im, feed_dict = {neuron_selector: prediction_class}, 
                                    should_resize = False, 
                                    three_dims = False)
    grad_mask_2d_l =  imresize(grad_mask_2d, (224, 224), interp='bilinear')
 #   imsave('test1.png', grad_mask_2d_l)
#     np.save('/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/results_others/'+str(i)+'_cam.npy', grad_mask_2d_l)

#  #   imsave('/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/res_all/'+str(i)+'_cam.png', grad_mask_2d_l)
#  #   print(grad_mask_2d.shape)
#     print('Now process image: ', i)

    np.save(save_path+str(idx)+'_cam.npy', grad_mask_2d_l)
   # np.save('/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/results_others/'+str(i)+'_smo_grd.npy', smoothgrad_mask_grayscale)
   # print('Now process image: ', i)
    bar.next()
 #   end = time.time()
 #   print('Time cost is', end-start)
bar.finish()



