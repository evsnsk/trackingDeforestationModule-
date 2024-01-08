
import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import keras 
import pandas as pd
from datetime import datetime 

from keras.utils import normalize
from keras.metrics import MeanIoU



size_x = 128 
SIZE_Y = 128
n_classes=2 


train_images = []

for directory_path in glob.glob("input/images/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img = cv2.imread(img_path, 1)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)
       
#Convert list to array for machine learning processing        
train_images = np.array(train_images)

#Capture mask/label info as a list
train_masks = [] 
for directory_path in glob.glob("input/label/"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        train_masks.append(mask)
        
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)

###############################################
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder() ##############створити мітку класу
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)

#################################################?????????????
##train_images = np.expand_dims(train_images, axis=3)
##train_images = normalize(train_images, axis=1)
##

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

#Create a subset of data for quick testing
#Picking 10% for testing and remaining for training
from sklearn.model_selection import train_test_split
X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.3, random_state = 0)

#Further split training data t a smaller subset for quick testing of models
X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.3, random_state = 0)

print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 

from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))



test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

######################################################
#Reused parameters in all models

n_classes=2
activation='softmax'
#activation='softplus'
#activation='relu'
LR = 0.0001
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 0.5])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
#total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.FScore()]


########################################################################

BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

# preprocess input
X_train1 = preprocess_input1(X_train)
X_test1 = preprocess_input1(X_test)
#####################################################################
#####Model 1
###Using same backbone for both models
##
##### define model (Change to unet or Linknet based on the need )
model1 = sm.Unet(BACKBONE1, encoder_weights='imagenet', classes=n_classes, activation=activation)

 #compile keras model with defined optimozer, loss and metrics
#model1.compile(optim, total_loss, metrics=metrics)

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

print(model1.summary())

start1 = datetime.now() 

history1=model1.fit(X_train1,
        y_train_cat,
        batch_size=5, 
        epochs=100,
        verbose=1,
        validation_data=(X_test1, y_test_cat))

stop1 = datetime.now()

 ###Execution time of the model 
execution_time_unet = stop1-start1
print("Unet execution time is: ", execution_time_unet)


model1.save('unet_res34_backbone_50epochs.hdf5')

#####convert the history.history dict to a pandas DataFrame:     
hist1_df = pd.DataFrame(history1.history) 
hist1_csv_file = 'history_unet_50epochs.csv'
with open(hist1_csv_file, mode='w') as f:
    hist1_df.to_csv(f)
########################################################




##plot Model 1 Unet
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, label='Training loss (Втрата навчання)', color = 'red')
plt.plot(epochs, val_loss, label='Validation loss (Втрати навчання)', color = 'green')
plt.title('Training and validation loss (Втрати навчання та тестування)')
plt.xlabel('Epochs (Епохи)')
plt.ylabel('Loss (Втрати)')
plt.legend()
plt.show()

acc = history1.history['f1-score']
val_acc = history1.history['val_f1-score']

plt.plot(epochs, acc, color = 'red', label='Training F (Навчання F1)')
plt.plot(epochs, val_acc, color = 'green', label='Validation F1 (Тестування F1)')
plt.title('Training and validation F1 (Тестування та навчання F1')
plt.xlabel('Epochs (Епохи)')
plt.ylabel('F')
plt.legend()
plt.show()
##########
##
##
##
#####Model 2
###Using the same backbone as unet
##
##### define model (Change to unet or Linknet based on the need )
model2 = sm.Linknet(BACKBONE1, encoder_weights='imagenet', classes=n_classes, activation=activation)
##
 # compile keras model with defined optimozer, loss and metrics
model2.compile(optim, total_loss, metrics=metrics)

print(model2.summary())

start2 = datetime.now() 

history2=model2.fit(X_train1, 
        y_train_cat,
        batch_size=5, 
        epochs=100,
        verbose=1,
        validation_data=(X_test1, y_test_cat))

stop2 = datetime.now()

 #Execution time of the model 
execution_time_linknet = stop2-start2
print("Linknet execution time is: ", execution_time_linknet)

model2.save('linknet_res34_backbone_50epochs.hdf5')

 # convert the history.history dict to a pandas DataFrame:     
hist2_df = pd.DataFrame(history2.history) 
hist2_csv_file = 'history_linknet.csv'
with open(hist2_csv_file, mode='w') as f:
    hist2_df.to_csv(f)
## ##########################################################
##
## ###
## #plot the training and validation accuracy and loss at each epoch
loss = history2.history['loss']
val_loss = history2.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, label='Training loss (Втрата навчання)', color = 'red')
plt.plot(epochs, val_loss, label='Validation loss (Втрати навчання)', color = 'green')
plt.title('Training and validation loss (Втрати навчання та тестування)')
plt.xlabel('Epochs (Епохи)')
plt.ylabel('Loss (Втрати)')
plt.legend()
plt.show()

acc = history2.history['f1-score']
val_acc = history2.history['val_f1-score']

plt.plot(epochs, acc, color = 'red', label='Training F (Навчання F1)')
plt.plot(epochs, val_acc, color = 'green', label='Validation F1 (Тестування F1)')
plt.title('Training and validation F1 (Тестування та навчання F1')
plt.xlabel('Epochs (Епохи)')
plt.ylabel('F')
plt.legend()
plt.show()

#####################################################
##
from keras.models import load_model
##
### FOR NOW LET US FOCUS ON A SINGLE MODEL
##
##Set compile=False as we are not loading it for training, only for prediction.
model_unet = load_model('unet_res34_backbone_50epochs.hdf5', compile=False)
model_linknet = load_model('linknet_res34_backbone_50epochs.hdf5', compile=False)

##
##
##from keras.callbacks import Callback,ModelCheckpoint
##from keras.models import Sequential,load_model
##from keras.layers import Dense, Dropout
##from keras.wrappers.scikit_learn import KerasClassifier
##import keras.backend as K
##
##
##def get_f1(y_true, y_pred): #taken from old keras source code
##    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
##    print(true_positives)
##    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
##    print(possible_positives)
##    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
##    print(predicted_positives)
##    precision = true_positives / (predicted_positives + K.epsilon())
##    print(precision)
##    recall = true_positives / (possible_positives + K.epsilon())
##    print(recall)
##    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
##    print(f1_val)
##    return f1_val
##
##
##
##
y_pred_unet=model_unet.predict(X_test1)
y_pred_unet_argmax=np.argmax(y_pred_unet, axis=3)

y_pred_linknet=model_linknet.predict(X_test1)
y_pred_linknet_argmax=np.argmax(y_pred_linknet, axis=3)
##
##from sklearn.metrics import f1_score
##
##
##F1_unet = sklearn.metrics.f1_score(y_test[:,:,:,0], y_pred_unet_argmax, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
##print("Mean F1 - Unet:",F1_unet)
####


###IOU
##y_pred_unet=model_unet.predict(X_test1)
##y_pred_unet_argmax=np.argmax(y_pred_unet, axis=3)
##
##y_pred_linknet=model_linknet.predict(X_test1)
##y_pred_linknet_argmax=np.argmax(y_pred_linknet, axis=3)
##
###Using built in keras function
##from keras.metrics import MeanIoU
##n_classes = 2
##
##IOU_unet = MeanIoU(num_classes=n_classes)  
##IOU_unet.update_state(y_test[:,:,:,0], y_pred_unet_argmax)
##
##IOU_linknet = MeanIoU(num_classes=n_classes)  
##IOU_linknet.update_state(y_test[:,:,:,0], y_pred_linknet_argmax)
##
##print("Mean IoU using Unet =", IOU_unet.result().numpy())
##print("Mean IoU using linknet =", IOU_linknet.result().numpy())


##############################################################


import random
test_img_number = random.randint(0, len(X_test1)-1)
test_img = X_test1[test_img_number]
ground_truth=y_test[test_img_number]
test_img_input=np.expand_dims(test_img, 0)
test_img_input1 = preprocess_input1(test_img_input)

test_pred_unet = model_unet.predict(test_img_input1)
test_prediction_unet = np.argmax(test_pred_unet, axis=3)[0,:,:]

test_pred_linknet = model_linknet.predict(test_img_input1)
test_prediction_linknet = np.argmax(test_pred_linknet, axis=3)[0,:,:]

#plt.figure(figsize=(12, 6))
plt.subplot()
plt.title('Testing Image')
plt.axis('off')
plt.imshow(test_img[:,:,0])
plt.show()

plt.subplot()
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0])
plt.axis('off')
plt.show()


#plt.figure(figsize=(12, 6))
plt.subplot()
plt.title('Unet result')
plt.imshow(test_prediction_unet )
plt.axis('off')
plt.show()

plt.subplot()
plt.title('Linknet result')
plt.imshow(test_prediction_linknet)
plt.axis('off')
plt.show()



