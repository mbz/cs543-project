
# coding: utf-8

# In[78]:

import SimpleITK as sitk
import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
# import matplotlib as plt
from collections import Counter


# In[50]:

def find_files_in_dir(path, endswith):
    for _file in os.listdir(path):
        if _file.endswith(endswith):
            yield _file


# In[51]:

def load_images(path, subpaths):
    for sp in subpaths:
        fullpath = os.path.join(path, sp)
        image_files = list(find_files_in_dir(fullpath, 'mha'))
        assert len(image_files) == 1
        image = sitk.ReadImage(os.path.join(fullpath, image_files[0]))
        yield sitk.GetArrayFromImage(image)


# In[52]:

def merge_images_into_channels(images):
    image = np.zeros((len(images), images.shape[0], images.shape[1]))
    for i, img in images:
        image[i, :, :] = img
    return image


# In[53]:

def normalize_images(images):
    out = []
    for image in images:
        # mean = np.mean(image)
        # std = np.std(image)
        # out.append((image - mean) / std)
        out.append(np.uint8(image * 255.0 / np.max(image)))
    return out


# In[54]:

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


# In[55]:

def get_patch(images, x, y, z, size):
    patches = [image[z, x-size/2:x+size/2, y-size/2:y+size/2] for image in images]
    return patches


# In[56]:

def get_label(label_image, x, y, z):
    out = np.zeros(label_count)
    out[label_image[z,x,y]] = 1
    return out > 0


# In[9]:

path = '/home/ubuntu/projects/BRATS/data/Images/0001/'
subpaths = ['VSD.Brain.XX.O.MR_Flair', 'VSD.Brain.XX.O.MR_T1', 'VSD.Brain.XX.O.MR_T1c', 'VSD.Brain.XX.O.MR_T2']
label_path = 'VSD.Brain_3more.XX.XX.OT'

images = list(load_images(path, subpaths))
images_norm = normalize_images(images)
label_image = list(load_images(path, [label_path]))[0]



# In[11]:

shape = images[0].shape

patch_size = 34
channels = len(images)
label_count = 5

total_count = (shape[1]-patch_size) * (shape[2]-patch_size) * shape[0]
print(total_count)


# In[13]:

def batch_data_generator(images_norm, label_image, batch_size):
    X = np.zeros((batch_size, channels, patch_size, patch_size))
    Y = np.zeros((batch_size, label_count))
    i = 0
    for z in range(shape[0]):
        for x in range(patch_size/2, shape[1]-patch_size/2):
            for y in range(patch_size/2, shape[2]-patch_size/2):
                patch = get_patch(images_norm, x, y, z, patch_size)
                   
                X[i,:,:,:] = patch
                Y[i,:] = get_label(label_image, x, y, z)
                i += 1
                if i % batch_size == 0:
                    i = 0
                    yield (X, Y)


# # Data Generation

# In[14]:

def generate_all_patches(images_norm, label_image):
    i = ii = 0
    with open('data/patches.txt', 'w') as f:
        for z in range(shape[0]):
            for x in range(patch_size/2, shape[1]-patch_size/2):
                for y in range(patch_size/2, shape[2]-patch_size/2):
                    i += 1
                    if i % 100000 == 0:
                        print('%d/%d:%d' % (i, total_count, ii))
                    
                    patch = get_patch(images_norm, x, y, z, patch_size)

                    ratio = float(np.sum(patch)) / patch_size / patch_size / channels
                    if ratio < 0.25:
                        continue

                    label = label_image[z,x,y]
                    
                    ii += 1
                    for c in range(channels):
                        sub_dir = '%08d' % (ii / 1000)
                        filename = '/home/ubuntu/projects/BRATS/data/patches/%d/%s/patch_%d_%d_%d_%d.png' % (c, sub_dir, x, y, z, label)
                        # sp.misc.imsave(filename, patch[c])
                        directory = os.path.dirname(filename)
                        if not os.path.exists(directory):
                            os.makedirs(directory)                        
                        plt.image.imsave(filename, patch[c])
                        f.write('%s\t%d\n' % (filename, label))


# # Stat Generation

# In[62]:

def generate_all_patch_indices(image_id, images_norm, label_image):
    i = 0
    for z in range(shape[0]):
        for x in range(patch_size/2, shape[1]-patch_size/2):
            for y in range(patch_size/2, shape[2]-patch_size/2):
                i += 1
                if i % 100000 == 0:
                    print('%d/%d' % (i, total_count))

                patch = get_patch(images_norm, x, y, z, patch_size)

                ratio = np.mean(patch) / 255.0
                if ratio < 0.1:
                    continue

                label = label_image[z,x,y]
                
                with open('/home/ubuntu/projects/BRATS/data/Patches/patch_indices_%04d_%d.txt' % (image_id, label), 'a') as f:
                    f.write('%d,%d,%d\n' % (x, y, z))


# # In[77]:

# for image_id in range(1,2):
#     print('=' * 20)
#     print(image_id)
#     print('=' * 20)
   
#     path = 'data/Images/%04d/' % image_id
#     subpaths = ['VSD.Brain.XX.O.MR_Flair', 'VSD.Brain.XX.O.MR_T1', 'VSD.Brain.XX.O.MR_T1c', 'VSD.Brain.XX.O.MR_T2']
#     label_path = 'VSD.Brain_3more.XX.XX.OT'

#     images = list(load_images(path, subpaths))
#     images_norm = normalize_images(images)
#     label_image = list(load_images(path, [label_path]))[0]
    
#     c = Counter()
#     c.update(label_image.ravel())
    
#     with open('/home/ubuntu/projects/BRATS/data/Patches/label_count.txt', 'a') as f:
#         for key in range(10):
#             f.write('%d\t' % c[key])        
#         f.write('\n')
    
#     generate_all_patch_indices(image_id, images_norm, label_image)


# In[ ]:




# # Priliminary Results

# In[12]:

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD

def generate_model():
    input_shape = (channels, patch_size, patch_size)

    model = Sequential()

    model.add(Convolution2D(128, 7, 7, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(label_count))
    model.add(Activation('softmax'))

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


# In[ ]:




# In[131]:

def generate_all_patches_from_indices(image_id, images_norm):
    indices = np.zeros(label_count, dtype=int)
    label_indices = {}
    for label in range(label_count):
        filename = '/home/ubuntu/projects/BRATS/data/Patches/patch_indices_%04d_%d.txt' % (image_id, label)
        data = pd.read_csv(filename, header=None)
        data = data.iloc[np.random.permutation(len(data))]
        label_indices[label] = data
    
    batch_len = 128
    maxlen = 10000
    
    X = np.zeros((batch_len*label_count, channels, patch_size, patch_size))
    Y = np.zeros((batch_len*label_count, label_count), dtype=bool)
    X_ind = 0
    for i in range(maxlen):
        for label in range(label_count):
            (x,y,z) = label_indices[label].iloc[indices[label]]
            patch = get_patch(images_norm, x, y, z, patch_size)
            X[X_ind] = patch
            Y[X_ind, label] = True
            
            X_ind += 1
            indices[label] += 1
            indices[label] %= min(maxlen, len(label_indices[label]))
            
            if X_ind % label_count == 0:
                X_ind = 0
                yield (X, Y)


# In[ ]:




# In[118]:

model = generate_model()


# In[134]:

generator = generate_all_patches_from_indices(1, images_norm)


# In[136]:

model.fit_generator(generator, 
                    samples_per_epoch = 128*8, 
                    nb_epoch = 10, 
                    verbose = True, 
                    show_accuracy=True,
                    callbacks=[],
                    validation_data=None,
                    class_weight=None,
                    nb_worker=1)


model.save_weights('model.keras')