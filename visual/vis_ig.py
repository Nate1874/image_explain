import tensorflow as tf
import numpy as np
import PIL.Image
from matplotlib import pylab as P
import pickle
import os
slim=tf.contrib.slim
from scipy.misc import imsave
import sys
from progress.bar import Bar


sys.path.append('/mnt/dive/shared/hao.yuan/models/research/slim')

# old_cwd = os.getcwd()
# os.chdir('/mnt/dive/shared/hao.yuan/models/research/slim')
from nets import vgg
#os.chdir(old_cwd)

import saliency


ckpt ='/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/vgg_16.ckpt'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

graph=tf.Graph()
with graph.as_default():
    images = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))

    logits,_ = vgg.vgg_16(images)
    sess = tf.Session(graph=graph)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt)
    logits = graph.get_tensor_by_name('vgg_16/fc8/squeezed:0')
    neuron_selector = tf.placeholder(tf.int32)
    y = logits[0][neuron_selector]
    prediction = tf.argmax(logits, 1)
    integrated_gradients = saliency.IntegratedGradients(graph, sess, y, images)

label_list= np.load('/mnt/dive/shared/hao.yuan/tempspace2/hyuan/label_train.npy')
indexes  =np.load('/mnt/dive/shared/hao.yuan/tempspace2/hyuan/train_idx.npy')
print('Starting ===================')
my_path  = '/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/all_files/'
save_path = '/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/results_train/'

batch_size= 1
bar = Bar('Processing', max=int(64497/batch_size))
for i in range(64497):
    idx = indexes[i]
 #   print('1111')
    file_path = my_path+str(idx)+'_raw.npy'
   # path = '/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/results_npy/'+str(i)+'_raw.npy'
    im  = np.load(file_path)
    prediction_class = label_list[idx]
    # gradient_saliency = saliency.GradientSaliency(graph, sess, y, images)
    
    # vanilla_mask_3d = gradient_saliency.GetMask(im, feed_dict = {neuron_selector: prediction_class})
    # vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)

    # smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(im, feed_dict = {neuron_selector: prediction_class})
    # smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
    

    # guided_backprop = saliency.GuidedBackprop(graph, sess, y, images)
    # vanilla_guided_backprop_mask_3d = guided_backprop.GetMask(im, feed_dict = {neuron_selector: prediction_class})
    # vanilla_mask_grayscale_bp = saliency.VisualizeImageGrayscale(vanilla_guided_backprop_mask_3d)


    
    baseline = np.zeros(im.shape)
    baseline.fill(-1)
    vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(im, feed_dict = {neuron_selector: prediction_class}, x_steps=25, x_baseline=baseline)
    vanilla_mask_grayscale_ig = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)

    # imsave('/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/res_all/'+str(i)+'_grd.png', vanilla_mask_grayscale)
    # imsave('/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/res_all/'+str(i)+'_smo_grd.png', smoothgrad_mask_grayscale)
    # imsave('/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/res_all/'+str(i)+'_bp.png', vanilla_mask_grayscale_bp)
    # imsave('/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/res_all/'+str(i)+'_ig.png', vanilla_mask_grayscale_ig)

    np.save(save_path+str(idx)+'_ig.npy', vanilla_mask_grayscale_ig)
   # np.save('/mnt/dive/shared/hao.yuan/tempspace2/hyuan/self_interpretation/results_others/'+str(i)+'_smo_grd.npy', smoothgrad_mask_grayscale)
   # print('Now process image: ', i)
    bar.next()
 #   end = time.time()
 #   print('Time cost is', end-start)
bar.finish()



