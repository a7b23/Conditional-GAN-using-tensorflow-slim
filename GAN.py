import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from util import *

batch_size = 64
visualize_dim = 196 #number of images to be sampled and stored in the outputDir 
display_step = 100

modelsDir = './model1'
outputDir = './images1'
model_save_step = 5000
save_imageSamples_step = 500 #number of steps after images have to be saved

n_epochs = 50000

inputDataDir = 'MNIST_data/'

if not os.path.exists(modelsDir) :
	os.mkdir(modelsDir)

if not os.path.exists(outputDir) :
	os.mkdir(outputDir)

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def generator(z,y,train = True,scope = "generator") :
	with tf.variable_scope(scope) as sc :
		with slim.arg_scope([slim.conv2d_transpose,slim.fully_connected],
										  biases_initializer=tf.constant_initializer(0.0),
										  activation_fn=None):
			with slim.arg_scope([slim.batch_norm],decay = 0.95,center = True,scale = True,activation_fn = tf.nn.relu,is_training = train) :

				yb = tf.reshape(y,[-1,1,1,10])
				z = tf.concat([z,y],1)
				net = slim.fully_connected(z,1024,scope = "linear_1")
				net = slim.batch_norm(net,scope = "bn_1")
				net = tf.concat([net,y],1)
				net = slim.batch_norm(slim.fully_connected(net,128*7*7,scope= "linear_2"),scope = "bn_2")
				net = tf.reshape(net,[-1,7,7,128])
				net = tf.concat([net,yb*tf.ones([batch_size,7,7,10])],3)
				net = slim.conv2d_transpose(net,64,[5,5],stride = [2,2],padding = 'SAME',scope = 'conv_transpose1')
				net = slim.batch_norm(net,scope = 'bn_3')
				net = tf.concat([net,yb*tf.ones([batch_size,14,14,10])],3)
				net = tf.nn.sigmoid(slim.conv2d_transpose(net,1,[5,5],stride = [2,2],scope = 'conv_transpose_2'))
				return net

def sampler(z,y,train = True,scope = "generator") :
	with tf.variable_scope(scope,reuse = True) as sc :
		with slim.arg_scope([slim.conv2d_transpose,slim.fully_connected],
										  biases_initializer=tf.constant_initializer(0.0),
										  activation_fn=None):
			with slim.arg_scope([slim.batch_norm],decay = 0.95,center = True,scale = True,activation_fn = tf.nn.relu,is_training = train) :

				yb = tf.reshape(y,[-1,1,1,10])
				z = tf.concat([z,y],1)
				net = slim.fully_connected(z,1024,scope = "linear_1")
				net = slim.batch_norm(net,scope = "bn_1")
				net = tf.concat([net,y],1)
				net = slim.batch_norm(slim.fully_connected(net,128*7*7,scope= "linear_2"),scope = "bn_2")
				net = tf.reshape(net,[-1,7,7,128])
				net = tf.concat([net,yb*tf.ones([visualize_dim,7,7,10])],3)
				net = slim.conv2d_transpose(net,64,[5,5],stride = [2,2],padding = 'SAME',scope = 'conv_transpose1')
				net = slim.batch_norm(net,scope = 'bn_3')
				net = tf.concat([net,yb*tf.ones([visualize_dim,14,14,10])],3)
				net = tf.nn.sigmoid(slim.conv2d_transpose(net,1,[5,5],stride = [2,2],scope = 'conv_transpose_2'))
				return net


def discriminator(image,y,train = True,scope = "discriminator",reuse = False) :
	with tf.variable_scope(scope,reuse = reuse) as sc :
		with slim.arg_scope([slim.conv2d,slim.fully_connected],
										  biases_initializer=tf.constant_initializer(0.0),
										  activation_fn=None):
			with slim.arg_scope([slim.batch_norm],decay = 0.95,center = True,scale = True,activation_fn = None,is_training = train) :
				yb = tf.reshape(y,[-1,1,1,10])
				x = tf.concat([image,yb*tf.ones([batch_size,28,28,10])],3)
				net = lrelu(slim.conv2d(x,64,[5,5],stride = [2,2],scope = 'conv_1'))
				net = tf.concat([net,yb*tf.ones([batch_size,14,14,10])],3)
				net = lrelu(slim.batch_norm(slim.conv2d(net,128,[5,5],stride = [2,2],scope = 'conv_2'),scope = 'bn_1'))
				net = slim.flatten(net)
				net = tf.concat([net,y],1)
				net = slim.fully_connected(net,1024,scope = 'linear_1')
				net = lrelu(slim.batch_norm(net,scope = 'bn_2'))
				net = slim.fully_connected(net,1,scope = 'linear_2')

				return net

def build_model(x,z,y) :
	image_gen = generator(z,y)
	real_out = discriminator(x,y)
	fake_out = discriminator(image_gen,y,reuse = True)
	t_vars = tf.trainable_variables()
	gen_vars = [var for var in t_vars if 'generator' in var.name]
	disc_vars = [var for var in t_vars if 'discriminator' in var.name]
	return real_out,fake_out,gen_vars,disc_vars

def getLoss(logit,label) :
	return tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=label)

def getLosses(real_logits,fake_logits) :
	d_real_loss = tf.reduce_mean(getLoss(real_logits,tf.ones_like(real_logits)))
	d_fake_loss = tf.reduce_mean(getLoss(fake_logits,tf.zeros_like(fake_logits)))
	g_loss = tf.reduce_mean(getLoss(fake_logits,tf.ones_like(fake_logits)))
	d_loss = d_real_loss + d_fake_loss
	return d_loss,g_loss



y = tf.placeholder(tf.float32, shape=[None, 10])
z = tf.placeholder(tf.float32, shape=[None, 100])

image_gen = generator(z,y)
fake_logits = discriminator(image_gen,y)

g_loss = tf.reduce_mean(getLoss(fake_logits,tf.ones_like(fake_logits)))

x = tf.placeholder(tf.float32, shape=[None, 28,28,1])

real_logits = discriminator(x,y,reuse = True)


# real_logits,fake_logits,gen_vars,disc_vars = build_model(x,z,y)
t_vars = tf.trainable_variables()
gen_vars = [var for var in t_vars if 'generator' in var.name]
disc_vars = [var for var in t_vars if 'discriminator' in var.name]

g_optim = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(g_loss, var_list=gen_vars)

g_summary = tf.summary.scalar('generator_loss', g_loss)
d_real_loss = tf.reduce_mean(getLoss(real_logits,tf.ones_like(real_logits)))
d_fake_loss = tf.reduce_mean(getLoss(fake_logits,tf.zeros_like(fake_logits)))

d_loss = d_real_loss + d_fake_loss


d_summary = tf.summary.scalar('discriminator_loss', d_loss)

d_optim = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(d_loss, var_list=disc_vars)

sampled = sampler(z,y)

sess = tf.InteractiveSession()
train_writer = tf.summary.FileWriter('/tmp/gan_logs', sess.graph)
sess.run(tf.global_variables_initializer())

mnist = input_data.read_data_sets(inputDataDir, one_hot=True) 



sample_z = np.random.uniform(-1, 1, size=(visualize_dim, 100)).astype(np.float32)
sample_x,sample_y = mnist.train.next_batch(visualize_dim)
sample_x = np.reshape(sample_x,[-1,28,28,1])

saver = tf.train.Saver()

for epoch in range(n_epochs) :
	data_x,data_y = mnist.train.next_batch(batch_size)
	data_x = np.reshape(data_x,[-1,28,28,1])
	z_noise = np.random.uniform(-1, 1, size=(batch_size, 100)).astype(np.float32)

	_,summary = sess.run([d_optim,d_summary],feed_dict = {x:data_x,y:data_y,z:z_noise})
	train_writer.add_summary(summary, epoch)

	_,summary = sess.run([g_optim,g_summary],feed_dict = {y:data_y,z:z_noise})
	train_writer.add_summary(summary, epoch)

	_,summary = sess.run([g_optim,g_summary],feed_dict = {y:data_y,z:z_noise})
	train_writer.add_summary(summary, epoch)

	if (np.mod(epoch,display_step) == 1) :
		print('%d steps reached'%epoch)

	if (np.mod(epoch,save_imageSamples_step) == 1) :
		samples = sess.run(sampled,feed_dict = {y:sample_y,z:sample_z})
		save_images(samples, image_manifold_size(samples.shape[0]), os.path.join(outputDir,'image_'+str(epoch)+'.png'))

	if(np.mod(epoch,model_save_step) == 1) and not epoch == 1:
		saver.save(sess,os.path.join(modelsDir,'gan_model_'+str(epoch)+'.ckpt'))
		



