#part 2
import glob
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from PIL import Image

from os.path import join
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Activation,MaxPooling2D,BatchNormalization,Activation, Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam , SGD
import seaborn as sbn
from keras.layers import LeakyReLU
from livelossplot import PlotLossesKeras


def load_tfl_data(data_dir, crop_shape=(81, 81)):
 images = np.memmap(join(data_dir, 'data.bin'), mode='r', dtype=np.uint8).reshape([-1] + list(crop_shape) + [3])
 labels = np.memmap(join(data_dir, 'labels.bin'), mode='r', dtype=np.uint8)
 return {'images': images, 'labels': labels}


def viz_my_data(images, labels, predictions=None, num=(5, 5), labels2name={0: 'No TFL', 1: 'Yes TFL'}):
 assert images.shape[0] == labels.shape[0]
 assert predictions is None or predictions.shape[0] == images.shape[0]
 h = 5
 n = num[0] * num[1]
 ax = plt.subplots(num[0], num[1], figsize=(h * num[0], h * num[1]), gridspec_kw={'wspace': 0.05}, squeeze=False,
                   sharex=True, sharey=True)[1]  # .flatten()
 idxs = np.random.randint(0, images.shape[0], n)
 for i, idx in enumerate(idxs):
  ax.flatten()[i].imshow(images[idx])
  title = labels2name[labels[idx]]
  if predictions is not None: title += ' Prediction: {:.2f}'.format(predictions[idx])
  ax.flatten()[i].set_title(title)
 plt.show()

def preper_data_set (dir,dir2,dirData,dirLabels):
 flist = []
 labelled_list = []
 #dir = r"C:\Users\user\OneDrive\Desktop\pic\leftImg8bit\train"  # default_base
 for filename in os.listdir(dir):
  f = os.path.join(dir, filename)
  flist.append(glob.glob(os.path.join(f, '*_leftImg8bit.png')))
 #dir2 = r"C:\Users\user\OneDrive\Desktop\pic\gtFine\train"
 for filename in os.listdir(dir2):
  f = os.path.join(dir2, filename)
  labelled_list.append(glob.glob(os.path.join(f, '*_labelIds.png')))
 data_array = []
 label_array = []
 for city in labelled_list:
  for labelled in city:
   #img = matplotlib._png.read_png_int(labelled)
   img = np.array(Image.open(labelled))
   nineteen_list = np.where(img == 19)
   # print(nineteen_list)
   if len(nineteen_list[0]) > 0:
    flag = True
    while flag:
     y = random.randint(41, 2007)
     x = random.randint(41, 983)
     if img[x][y] != 19:
      flag = False
    no_traffic = (x, y)
    original_image_name = labelled[labelled.rindex('\\') + 1:labelled.rindex("gtFine_labelIds.png")] + "leftImg8bit.png"
    city_ = labelled.split("\\")[6]
    original_img = Image.open(dir + "\\" + city_ + "\\" + original_image_name)
    #pic = original_img.load()
    no_traffic_crop_image = original_img.crop((y - 40, x - 40, y + 41, x + 41))
    #  no_traffic_crop_image.show()
    no_traffic_u = np.array(no_traffic_crop_image, dtype=np.int16)
    no_traffic_u = np.uint8(no_traffic_u)
    traffic_x = random.choice(nineteen_list[1])
    index_traffic_x = np.where(nineteen_list[1] == traffic_x)
    traffic_y = nineteen_list[0][index_traffic_x][0]
    traffic_crop_image = original_img.crop((traffic_x - 40, traffic_y - 40, traffic_x + 41, traffic_y + 41))
    #  traffic_crop_image.show()
    traffic_u = np.array(traffic_crop_image, dtype=np.int16)
    traffic_u = np.uint8(traffic_u)
    data_array.append(traffic_u)
    data_array.append(no_traffic_u)
    one = np.uint8(1)
    zero = np.uint8(0)
    label_array.append(one)
    label_array.append(zero)
#r"C:\Users\user\OneDrive\Desktop\pic\train\data.bin"
 with open(dirData, "wb") as f:
  np.array(data_array).tofile(f)
#r"C:\Users\user\OneDrive\Desktop\pic\train\labels.bin"
 with open(dirLabels, "wb") as g:
  np.array(label_array).tofile(g)
 # pic_file.close()
 # labels_file.close()

#preper_data_set(r"C:\שנה ג סמס ב\בוטקאמפ\leftImg8bit_trainvaltest\leftImg8bit\train",
#                 r"C:\שנה ג סמס ב\בוטקאמפ\gtFine_trainvaltest\gtFine\train",
#                 r"C:\Users\hadas\PycharmProjects\model\train\data.bin",
#                r"C:\Users\hadas\PycharmProjects\model\train\labels.bin")
#preper_data_set(r"C:\שנה ג סמס ב\בוטקאמפ\leftImg8bit_trainvaltest\leftImg8bit\val",
#                 r"C:\שנה ג סמס ב\בוטקאמפ\gtFine_trainvaltest\gtFine\val",
#                r"C:\Users\hadas\PycharmProjects\model\val\data.bin",
#                 r"C:\Users\hadas\PycharmProjects\model\val\labels.bin")
# root = './'  #this is the root for your val and train datasets
data_dir = r'C:\Users\hadas\PycharmProjects\model'
datasets = {
 'val': load_tfl_data(join(data_dir, 'val')),
 'train': load_tfl_data(join(data_dir, 'train')),
}
for k, v in datasets.items():
 print('{} :  {} 0/1 split {:.1f} %'.format(k, v['images'].shape, np.mean(v['labels'] == 1) * 100))

viz_my_data(num=(6, 6), **datasets['train'])


def tfl_model():
 input_shape = (81, 81, 3)

 model = Sequential()

 def conv_bn_relu(filters, **conv_kw):
  model.add(Conv2D(filters, use_bias=False, kernel_initializer='he_normal', **conv_kw))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  #model.add(LeakyReLU(alpha=0.1))

 def dense_bn_relu(units):
  model.add(Dense(units, use_bias=False, kernel_initializer='he_normal'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  #model.add(LeakyReLU(alpha=0.1))

 def spatial_layer(count, filters):
  for i in range(count):
   conv_bn_relu(filters, kernel_size=(3, 3))
  conv_bn_relu(filters, kernel_size=(3, 3), strides=(2, 2))

 conv_bn_relu(32, kernel_size=(3, 3), input_shape=input_shape)
 spatial_layer(1, 32)
 spatial_layer(2, 64)
 spatial_layer(2, 96)

 model.add(Flatten())
 dense_bn_relu(96)
 model.add(Dense(2, activation='softmax'))
 return model



#prepare our model
#prepare our model
m = tfl_model()
m.summary()
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)

#val_loss: 0.3238 - val_accuracy: 0.8890
#sparse_categorical_crossentropy
m.compile(optimizer=SGD(learning_rate=lr_schedule),loss=sparse_categorical_crossentropy,metrics=['accuracy'])

train,val = datasets['train'],datasets['val']
#train it, the model uses the 'train' dataset for learning. We evaluate the "goodness" of the model, by predicting the label of the images in the val dataset.
#es = tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', mode='min', restore_best_weights=True)
#verbose=1, min_delta=0.001
history=m.fit(train['images'],train['labels'],validation_data=(val['images'],val['labels']),epochs = 20)



epochs = history.history
epochs['train_acc'] = epochs['accuracy']
plt.figure(figsize=(10,10))
for k in ['train_acc','val_accuracy']:
    plt.plot(range(len(epochs[k])), epochs[k],label=k)

plt.legend()


predictions = m.predict(val['images'])
sbn.distplot(predictions[:,0])
plt.show()

predicted_label = np.argmax(predictions, axis=-1)
print ('accuracy:', np.mean(predicted_label==val['labels']))

viz_my_data(num=(6,6),predictions=predictions[:,1],**val)