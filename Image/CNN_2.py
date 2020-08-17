import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
%matplotlib inline
np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical #convert to one-hot-encoding
from keras.utils.np_utils import to_categorical #convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
sns.set(style='white',context='notebook',palette='deep')

#load the data
train=pd.read_csv(r"C:\Users\Aakash Krishnan U\Documents\data\MNIST\train.csv")
test=pd.read_csv(r"C:\Users\Aakash Krishnan U\Documents\data\MNIST\test.csv")
Y_train=train['label']
X_train=train.drop(labels=["label"],axis=1)
del train
g=sns.countplot(Y_train)
Y_train.value_counts()

#check the null data
X_train.isnull().any().describe()
test.isnull().any().describe()

#normalise the data
X_train=X_train/255.0
test=test/255.0

#reshape image in 3 dimensions(height=28px,width=28px,canal=1)
X_train=X_train.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)

#encode labels to one hot vectors
Y_train=to_categorical(Y_train,num_classes=10)

#set the random seed
random_seed=2
#split the train and validation set for the fitting
X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.1,random_state=random_seed)
#example
g=plt.imshow(X_train[0][:,:,0])

#set the CNN model
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))
#define the optimizer
optimizer=RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)
#compile the model
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
#set the learning rate annealer
learning_rate_reduction=ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5,min_lr=0.00001)
epochs=1#turn epochs to 30 to get 0.9967 accuracy
batch_size=86

#data augmentation
datagen=ImageDataGenerator(
     featurewise_center=False,#set input mean to 0 over the dataset
     samplewise_center=False,#set each sample mean to 0
     featurewise_std_normalization=False,#divide inputs by std of the dataset
     samplewise_std_normalization=False,#divide each input by its standerd
     zca_whitening=False,#apply zca whitening
     rotation_range=10,#randomly rotate images in the range(degrees o to 180)
     zoom_range=0.1,#randomly zoom image
     width_shift_range=0.1,#randomly shift images horizontaly
     height_shift_range=0.1,#randomly shift images vertically
     horizontal_flip=False,#randomly flip images
     vertical_flip=False,#randomly flip images
 )
datagen.fit(X_train)
#fit the model
history=model.fit_generator(datagen.flow(X_train,Y_train,batch_size=batch_size),
                            epochs=epochs,validation_data=(X_val,Y_val),)

#evaluate the model
#plot the loss and accuracy curves for training and validation
fig,ax=plt.subplots(2,1)
ax[0].plot(history.history['loss'],color='b',label='Training Loss')
ax[0].plot(history.history['val_loss'],color='r',label='validation Loss',axes=ax[0])
legend=ax[0].legend(loc='best',shadow=True)
ax[1].plot(history.history['acc'],color='b',label='Training Accuracy')
ax[1].plot(history.history['val_acc'],color='r',label='Validation Accuracy')
legend=ax[1].legend(loc='best',shadow=True)

#plot the confusion matrix
def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')

predict the values from the validation dataset
Y_pred=model.predict(X_val)
#convert the predictions classes to one hot vector
Y_pred_classes=np.argmax(Y_pred,axis=1)
#convert validation observations to one hot vector
Y_true=np.argmax(Y_val,axis=1)
#compute the confusion matrix
confusion_mtx=confusion_matrix(Y_true,Y_pred_classes)
#plot the confusion matrix
plot_confusion_matrix(confusion_mtx,classes=range(10))
