import tensorflow as tf 
import ops

tf.set_random_seed(2345)
slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

class Network(object):

	def __init__(self, conf):
		self.conf = conf
		self.num_classes = self.conf.num_classes
	#	self.gen_depth = self.conf.gen_depth
		self.channel_axis =3
		self.conv_size = (3, 3)

	def __call__(self, inputs, training):
		"""Add operations to classify a batch of input images.
		Args:
			inputs: A Tensor representing a batch of input images.
			training: A boolean. Set to True to add operations required only when
				training the classifier.
		Returns:
			A logits Tensor with shape [<batch_size>, self.num_classes].
		"""

		return self._build_network(inputs, training)


	################################################################################
	# Composite blocks building the network
	################################################################################
	def _build_network(self, inputs, training):
		"""Build the network.
		"""
	#	images = inputs[0]
		images = inputs
		print('the shape of input is: ', images.get_shape())
		self.logits = self.generator(images, training)
		self.prob = tf.nn.sigmoid(self.logits)

		####this may not useful
		self.prob  = tf.layers.average_pooling2d(self.prob , 15, 1, padding= 'same')

		threshold = tf.constant(0.35, dtype=tf.float32)
		# based on prob, generate a mask and stop the gradient 
		if training == True:
			self.mask = tf.cast(tf.less_equal(tf.random_uniform(tf.shape(self.prob),
					dtype=tf.float32, seed=2356),self.prob),tf.float32)
		else:
			self.mask = tf.cast(tf.greater_equal(self.prob, threshold),tf.float32)
		print('The shape of mask is ', self.mask.get_shape())
		self.mask = tf.stop_gradient(self.mask)	
		self.new_input = tf.multiply(self.mask, images)
		
		print('The shape of new input is', self.new_input.get_shape())
		means = tf.expand_dims(tf.expand_dims(_CHANNEL_MEANS, 0), 0)
		self.new_input_vis = tf.multiply(self.mask, images)-tf.multiply((1-self.mask), means)


		self.img_logits, _ =self.discriminator(self.new_input)
		self.ori_logits, _ = self.discriminator(images)
		
		return self.logits, self.mask, self.img_logits, self.new_input, self.new_input_vis , self.prob, self.ori_logits


	def generator(self, inputs, training):	
		with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
			outputs= inputs
			down_outputs = []
			for layer_index in range(self.conf.network_depth - 1):
				is_first = True if not layer_index else False
				name = 'down%s' % layer_index
				outputs = self.build_down_block(
					outputs, name, down_outputs, training, is_first )	
				print('After the down block, the shape is:', outputs.get_shape())

			outputs= self.build_bottom_block(outputs,'bottom', training)
			print('After bottom block, the shape is', outputs.get_shape())

			for layer_index in range(self.conf.network_depth-2, -1, -1):
				is_final = True if layer_index == 0 else False
				name = 'up%s' % layer_index
				down_inputs = down_outputs[layer_index]
				outputs = self.build_up_block(
					outputs, down_inputs, name, training,is_final )
				print('After the up block, the shape is:', outputs.get_shape())
			
			return outputs	

	def build_down_block(self, inputs, name, down_outputs, training, first=False ):
		out_num = self.conf.start_channel_num if first else 2 * \
			inputs.shape[self.channel_axis].value
		conv1 = ops.conv(inputs, out_num, self.conv_size, 1, 'SAME',
							name+'/conv1', training)
		conv2 = ops.conv(conv1, out_num, self.conv_size, 1, 'SAME',
							name+'/conv2', training)
		down_outputs.append(conv2)
		conv3 = ops.conv(conv2, out_num, self.conv_size, 2, 'SAME',
							name+'/conv3_down', training)
		return conv3

	def build_bottom_block(self, inputs, name, training):
		out_num = inputs.shape[self.channel_axis].value
		conv1 = ops.conv(
			inputs, 2*out_num, self.conv_size, 1, 'SAME',name+'/conv1',training)
		conv2 = ops.conv(
			conv1, out_num, self.conv_size, 1, 'SAME', name+'/conv2',training)
		return conv2

	def build_up_block(self, inputs, down_inputs, name, training, final=False ):
		out_num = inputs.shape[self.channel_axis].value
		conv1 = ops.deconv(
			inputs, out_num, self.conv_size, name+'/deconv1',training)
		conv1 = tf.concat(
			[conv1, down_inputs], self.channel_axis, name=name+'/concat')
		# try add.
		conv2 = ops.conv(
			conv1, out_num, self.conv_size, 1, 'SAME', name+'/conv2', training)

		out_num = self.conf.encoder_out_num if final else out_num/2
		conv3 = ops.conv(
			conv2, out_num, self.conv_size, 1, 'SAME', name+'/conv3', training,
			not final)
		return conv3


	def discriminator(self, inputs,
			num_classes=1000,
			is_training=False,
			dropout_keep_prob=0.5,
			spatial_squeeze=True,
			scope='vgg_16',
			fc_conv_padding='VALID',
			global_pool=False):
		"""Oxford Net VGG 16-Layers version D Example.
		Note: All the fully_connected layers have been transformed to conv2d layers.
				To use in classification mode, resize input to 224x224.
		Args:
			inputs: a tensor of size [batch_size, height, width, channels].
			num_classes: number of predicted classes. If 0 or None, the logits layer is
			omitted and the input features to the logits layer are returned instead.
			is_training: whether or not the model is being trained.
			dropout_keep_prob: the probability that activations are kept in the dropout
			layers during training.
			spatial_squeeze: whether or not should squeeze the spatial dimensions of the
			outputs. Useful to remove unnecessary dimensions for classification.
			scope: Optional scope for the variables.
			fc_conv_padding: the type of padding to use for the fully connected layer
			that is implemented as a convolutional layer. Use 'SAME' padding if you
			are applying the network in a fully convolutional manner and want to
			get a prediction map downsampled by a factor of 32 as an output.
			Otherwise, the output prediction map will be (input / 32) - 6 in case of
			'VALID' padding.
			global_pool: Optional boolean flag. If True, the input to the classification
			layer is avgpooled to size 1x1, for any input size. (This is not part
			of the original VGG architecture.)
		Returns:
			net: the output of the logits layer (if num_classes is a non-zero integer),
			or the input to the logits layer (if num_classes is 0 or None).
			end_points: a dict of tensors with intermediate activations.
		"""

		with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=tf.AUTO_REUSE) as sc:
			end_points_collection = sc.original_name_scope + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.
			with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
								outputs_collections=end_points_collection):
				net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')

				# Use conv2d instead of fully_connected layers.
				net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
				net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
									scope='dropout6')
				net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
				# Convert end_points_collection into a end_point dict.
				end_points = slim.utils.convert_collection_to_dict(end_points_collection)
				if global_pool:
					net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
					end_points['global_pool'] = net
				if num_classes:
					net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
									scope='dropout7')
					net = slim.conv2d(net, num_classes, [1, 1],
									activation_fn=None,
									normalizer_fn=None,
									scope='fc8')
					if spatial_squeeze:
						net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
					end_points[sc.name + '/fc8'] = net

				print('Net: ',net.get_shape())
		#		print('End_point: ',end_points.get_shape())
				return net, end_points
	#	vgg_16.default_image_size = 224