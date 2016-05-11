
# coding: utf-8

# In[150]:

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
import numpy
import scipy.ndimage
import glob
import os


# In[151]:

def identity(x):
    return x

def leaky_relu(x):
    alpha = 0.1
    out = tf.maximum(alpha*x,x)
    return out

def batch_norm(_X, name):
    with tf.variable_scope(name):
        _, _, _, c = _X.get_shape().as_list()
        mean, var = tf.nn.moments(_X, [0, 1, 2])
        offset = tf.get_variable('offset',
                                 shape=[c],
                                 initializer=tf.constant_initializer(value=0.0))
        scale = tf.get_variable('scale',
                                 shape=[c],
                                 initializer=tf.constant_initializer(value=1.0))
        output = tf.nn.batch_normalization(_X, mean, var, offset, scale, 1e-5)
    return output

def conv(_X, out_dim, name, size=3, gain=numpy.sqrt(2), func=leaky_relu):
    with tf.variable_scope(name):
        in_dim = _X.get_shape().as_list()[-1]
        stddev = gain / numpy.sqrt(size*size*in_dim)
        w_init = tf.random_normal_initializer(stddev=stddev)
        w = tf.get_variable('w', shape=[size, size, in_dim, out_dim], initializer=w_init)
        b_init = tf.constant_initializer(value=0.0)
        b = tf.get_variable('b', shape=[out_dim], initializer=b_init)
        output = func(tf.nn.conv2d(_X, w, strides=[1, 1, 1,1], padding='SAME') + b)
    return output

def generate_samples():
    n_samples = 4123 
    n_epochs = 1000
    n_batches = n_samples/batch_size
    for _ in range(n_epochs):
        sample_ids = numpy.random.permutation(n_samples)
        for i in range(n_batches):
            inds = slice(i*batch_size, (i+1)*batch_size)
            yield sample_ids[inds]
            
images = numpy.load('images.npy')
labels = numpy.load('labels.npy')


# In[109]:

def zoom(image, zoom_factor, order):
    image = scipy.ndimage.interpolation.zoom(image, zoom_factor, mode='nearest', order=order)
    w, h = image.shape
    if zoom_factor > 1.0:
        image = image[(w-width)/2:width+(w-width)/2, (h-height)/2:height+(h-height)/2]
    else:
        width_diff = width - w
        top_diff = width_diff / 2
        bot_diff = width_diff - top_diff
        height_diff = height - h
        left_diff = height_diff / 2
        right_diff = height_diff - left_diff
        image = numpy.pad(image, ((top_diff, bot_diff), (left_diff, right_diff)), 'edge')   
    return image


def augment_sample(image, label):
    # Flipping
    if numpy.random.binomial(1, 0.5):
        image = image[:, ::-1, :]
        label = label[:, ::-1, :]
    
    # Rotating
    r = numpy.random.uniform(-15.0, 15.0)
    for i in range(n_channels):
        image[:,:,i] = scipy.ndimage.interpolation.rotate(image[:,:,i], r, reshape=False, mode='nearest')
    for i in range(n_classes):
        label[:,:,i] = scipy.ndimage.interpolation.rotate(label[:,:,i], 15, reshape=False, mode='nearest', order=0)
        
    # Zooming
    z = numpy.random.uniform(0.9, 1.1)
    for i in range(n_channels):
        image[:,:,i] = zoom(image[:,:,i], z, 3)
    for i in range(n_classes):
        label[:,:,i] = zoom(label[:,:,i], z, 0)
    
    # Shifting
    sv = numpy.random.randint(-20, 20)
    sh = numpy.random.randint(-20, 20)
    for i in range(n_channels):
        image[:,:,i] = scipy.ndimage.interpolation.shift(image[:,:,i], [sv, sh], mode='nearest')
    for i in range(n_classes):
        label[:,:,i] = scipy.ndimage.interpolation.shift(label[:,:,i], [sv, sh], mode='nearest', order=0)
    
    return(image, label)


def generate_batch():
    for samples in generate_samples():
        image_batch = images[samples, :, :, :]
        label_batch = labels[samples, :, :, :]
        for i in range(image_batch.shape[0]):
            image_batch[i], label_batch[i] = augment_sample(image_batch[i], label_batch[i])
        yield(image_batch, label_batch)


# In[110]:

# Parameters
batch_size = 2
display_step = 20

# Network Parameters
width = 216
height = 160
n_channels = 4
n_classes = 5 # total classes (normal, non-normal)

# tf Graph input
x = tf.placeholder(tf.float32, [None, width, height, n_channels])
y = tf.placeholder(tf.float32, [None, width, height, n_classes])
lr = tf.placeholder(tf.float32)
weights = tf.placeholder(tf.float32, [batch_size*width*height])

def layer(_X, name):
    with tf.variable_scope(name):
        c = conv(_X, 64, 'conv')
        bn = batch_norm(c, 'bn')
    return bn
    
def tom_net(_X):
    # Convolution Layer
    nb_layers = 25
    _input = batch_norm(_X, 'input_bn')
    for i in range(nb_layers):
        _input = layer(_input, str(i))

    out = conv(_input, n_classes, 'output', gain=1.0, func=identity)
    return out

# Construct model
pred = tom_net(x)

# Define loss and optimizer
pred_reshape = tf.reshape(pred, [batch_size*width*height, n_classes])
y_reshape = tf.reshape(y, [batch_size*width*height, n_classes])

error = tf.nn.softmax_cross_entropy_with_logits(pred_reshape, y_reshape)
cost = tf.reduce_mean(error * weights)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,3), tf.argmax(y,3))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()


# In[111]:

arg_labels = numpy.argmax(labels, axis=3)
class_weights = numpy.zeros(n_classes)
for i in range(n_classes):
    class_weights[i] = 1/numpy.mean(arg_labels == i)**0.3
class_weights /= numpy.sum(class_weights)


# In[ ]:

# Launch the graph
sess = tf.Session()
sess.run(init)


# In[ ]:

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
model_path = "./models-e2e/model.ckpt"
learning_rate = 0.001

# saver.restore(sess, model_path)

# For each iteration over all data
for step, (image_batch, label_batch) in enumerate(generate_batch()):            
    label_vect = numpy.reshape(numpy.argmax(label_batch, axis=3), [batch_size*width*height])
    weight_vect = class_weights[label_vect]
    # Fit training using batch data
    feed_dict = {x: image_batch, y: label_batch, weights: weight_vect, lr:learning_rate}
    loss, acc, _ = sess.run([cost, accuracy, optimizer], feed_dict=feed_dict)
    if step % display_step == 0:
        print("Step %d, Minibatch Loss=%0.6f , Training Accuracy=%0.5f " 
              % (step, loss, acc))

        # Save the variables to disk.
        saver.save(sess, model_path)
    if step % 2000 == 0:
        learning_rate *= 0.9


