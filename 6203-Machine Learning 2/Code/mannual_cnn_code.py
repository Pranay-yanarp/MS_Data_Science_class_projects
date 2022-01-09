#### Cotton Disease Prediction
#### ML2 Final Project

### Importing required packages

import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import numpy as np
train_data_path = "train/"
validation_data_path = "val/"
 


training_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')


training_data = training_datagen.flow_from_directory(train_data_path, 
                                      target_size=(300, 300), 
                                      batch_size=32,
                                      class_mode='binary')  
''' 
training_data.class_indices

{'diseased cotton leaf': 0,
 'diseased cotton plant': 1,
 'fresh cotton leaf': 2,
 'fresh cotton plant': 3}

'''

 
# We will need to use ImageGenerator in Validation set as well but only rescaling
# only rescaling
valid_datagen = ImageDataGenerator(rescale=1./255)

# this is a similar generator, for validation data
valid_data = valid_datagen.flow_from_directory(validation_data_path,
                                  target_size=(300,300),
                                  batch_size=32,
                                  class_mode='binary')

model_path = 'v1_Mannual_run_cnn_cotton.h5'
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


#Building cnn model
mannual_cnn_model = keras.models.Sequential([
                                    keras.layers.Conv2D(32, 3, input_shape=[300, 300, 3]),
                                    keras.layers.MaxPooling2D(pool_size=(2,2)),
                                    keras.layers.Conv2D(64, 3),
                                    keras.layers.MaxPooling2D(pool_size=(2,2)),
                                    keras.layers.Conv2D(128, 3),
                                    keras.layers.MaxPooling2D(pool_size=(2,2)),
                                    keras.layers.Conv2D(128, 3),
                                    keras.layers.MaxPooling2D(pool_size=(2,2)),
                                    keras.layers.Conv2D(128, 3),
                                    keras.layers.MaxPooling2D(pool_size=(2,2)),                                    

                                    keras.layers.Dropout(0.3),                                                                        
                                    keras.layers.Flatten(), 
                                    keras.layers.Dense(units=128, activation='relu'), 
                                    keras.layers.Dropout(0.1),                                    
                                    keras.layers.Dense(units=256, activation='relu'),                                    
                                    keras.layers.Dropout(0.2),                                    
                                    keras.layers.Dense(units=4, activation='softmax') 
])



# compile cnn model
mannual_cnn_model.compile(optimizer = Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

mannual_cnn_model.summary()

# train cnn model
history = mannual_cnn_model.fit(training_data, 
                          epochs=100, 
                          verbose=1, 
                          validation_data= valid_data,
                          callbacks=callbacks_list) 


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
 

final_model = tensorflow.keras.models.load_model('v1_Mannual_run_cnn_cotton.h5')

val_preds = final_model.predict(np.array([x for i in range(len(valid_data)) for x in valid_data[i][0]]))
val_preds = np.argmax(val_preds, axis=1)
val_actual = [x for i in range(len(valid_data)) for x in valid_data[i][1]]
#val_actual = np.argmax(val_actual, axis=1)

print("Accuracy:" + str(accuracy_score(val_actual, val_preds)))
print("F1 Score:" + str(f1_score(val_actual, val_preds, average='micro')))
print("Confusion Matrix:\n" + str(confusion_matrix(val_actual, val_preds)))
print("Classification Report:\n" + str(classification_report(val_actual, val_preds)))


#### Reference
## https://github.com/krishnaik06/Cotton-Disease-Prediction-Deep-Learning
## https://www.tensorflow.org/tutorials/images/data_augmentation
## https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
## https://github.com/krishnaik06/Deployment-Deep-Learning-Model