import tensorflow as tf


# Parameters
learning_rate = 0.001
batch_size = 64 * 4
display_step = 20

# Network Parameters
patch_size = 28
n_channels = 4
n_classes = 4 # 1-5
dropout = 0.8 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, patch_size, patch_size, n_channels])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Create AlexNet model
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def alex_net(_X, _weights, _biases, _dropout):
    # Reshape input picture
    # _X = tf.reshape(_X, shape=[-1, 28, 28, 4])

    # Convolution Layer
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    pool1 = max_pool('pool1', conv1, k=2)
    # Apply Normalization
    norm1 = norm('norm1', pool1, lsize=4)
    # Apply Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)

    # Convolution Layer
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    pool2 = max_pool('pool2', conv2, k=2)
    # Apply Normalization
    norm2 = norm('norm2', pool2, lsize=4)
    # Apply Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)

    # Convolution Layer
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    # Max Pooling (down-sampling)
    pool3 = max_pool('pool3', conv3, k=2)
    # Apply Normalization
    norm3 = norm('norm3', pool3, lsize=4)
    # Apply Dropout
    norm3 = tf.nn.dropout(norm3, _dropout)

    # Fully connected layer
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation

    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation

    # Output, class prediction
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 4, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wd1': tf.Variable(tf.random_normal([4*4*256, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = alex_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()


# In[53]:

import SimpleITK as sitk
import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
from collections import Counter

def find_files_in_dir(path, endswith):
    for _file in os.listdir(path):
        if _file.endswith(endswith):
            yield _file
            
def load_images(path, subpaths):
    for sp in subpaths:
        fullpath = os.path.join(path, sp)
        image_files = list(find_files_in_dir(fullpath, 'mha'))
        assert len(image_files) == 1
        image = sitk.ReadImage(os.path.join(fullpath, image_files[0]))
        yield sitk.GetArrayFromImage(image)
        
def merge_images_into_channels(images):
    image = np.zeros((len(images), images.shape[0], images.shape[1]))
    for i, img in images:
        image[i, :, :] = img
    return image

def normalize_images(images):
    out = []
    for image in images:
        mean = np.mean(image)
        std = np.std(image)
        out.append((image - mean) / std)
        print('===>', np.min(image), np.max(image))
        print('===>', mean, std)
        print('===>', np.min(out), np.max(out))
        # out.append(np.uint8(image * 255.0 / np.max(image)))
    return out

def show_images(images, label_image, Z):
    f, ((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(1, 5)
    ax = [ax1, ax2, ax3, ax4]
    for i in range(channels):
        cax = ax[i].imshow(images[i][Z,:,:])
        ax[i].axis('off')
        f.colorbar(cax, ax = ax[i])

    ax5.imshow(label_image[Z,:,:])
    ax5.axis('off')
    
    fig = plt.gcf()
    fig.set_size_inches(23, 4)    
    

def get_patch(images, x, y, z, size):
    patch = np.zeros((size, size, len(images)))
    for idx, image in enumerate(images):
        patch[:, :, idx] = image[z, x-size/2:x+size/2, y-size/2:y+size/2]
    return patch


def get_label(label_image, x, y, z):
    out = np.zeros(label_count)
    out[label_image[z,x,y]] = 1
    return out > 0




def generate_all_patches_from_indices(image_id, images_norm):
    label_count = 5
    indices = np.zeros(label_count)
    label_indices = {}
    
    # Load patched indices
    for label in range(1, label_count):
        filename = '/home/ubuntu/projects/BRATS/data/Patches/patch_indices_%04d_%d.txt' % (image_id, label)
        data = pd.read_csv(filename, header=None)
        data = data.iloc[np.random.permutation(len(data))]
        label_indices[label] = data
    
    # Assuming label 2 has the max number of patches
    maxlen = len(label_indices[2])
    
    # Loop through indices
    X = np.zeros((batch_size, patch_size, patch_size, n_channels))
    Y = np.zeros((batch_size, 4), dtype=bool)
    X_ind = 0
    for i in range(maxlen):
        for label in range(1, label_count):
            (x,y,z) = label_indices[label].iloc[indices[label]]
            patch = get_patch(images_norm, x, y, z, patch_size)
            X[X_ind] = patch
            Y[X_ind, label-1] = True

            X_ind += 1
            indices[label] += 1
            indices[label] %= min(maxlen, len(label_indices[label]))

            if X_ind % batch_size == 0:
                X_ind = 0
                yield (X, Y)


def load_all_files(image_id):
    path = 'data/Images/%04d/' % image_id
    subpaths = ['VSD.Brain.XX.O.MR_Flair', 'VSD.Brain.XX.O.MR_T1', 'VSD.Brain.XX.O.MR_T1c', 'VSD.Brain.XX.O.MR_T2']
    label_path = 'VSD.Brain_3more.XX.XX.OT'

    images = list(load_images(path, subpaths))
    images_norm = normalize_images(images)
    label_image = list(load_images(path, [label_path]))[0]
    return (images_norm, label_image)
    

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
model_path = "./models_1to5/model.ckpt"

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess, model_path)

    # For each iteration over all data
    for _iter in range(100):
        # For each patient
        for image_id in range(1, 15):

            # Load patient images
            (images_norm, label_image) = load_all_files(image_id)

            # Create patch generator
            data_generator = generate_all_patches_from_indices(image_id, images_norm)

            # Keep training until reach there is data
            step = 1
            for (batch_xs, batch_ys) in data_generator:

                # Fit training using batch data
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
                if step % display_step == 0:
                    # Calculate batch accuracy
                    acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    # Calculate batch loss
                    loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})

                    print("Iter %d, Image %d, Step %d, Minibatch Loss=%0.6f , Training Accuracy=%0.5f " 
                          % (_iter, image_id, step*batch_size, loss, acc))

                    # Save the variables to disk.
                    saver.save(sess, model_path)

                    step += 1
                    tf.train.get_checkpoint_state('model.tf')



                    # print "Optimization Finished!"
                    # Calculate accuracy for 256 mnist test images
                    # print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})



