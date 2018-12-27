from tensorflow import python as tf
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)

#traindf = pd.read_csv('data/train_csv_new.csv', sep=',')
#testdf = pd.read_csv('data/test_csv_new.csv', sep=',')

"""
training_set = train_datagen.flow_from_dataframe(dataframe=traindf,
                                                 directory='data/masked_rotated_h_train/small vehicle/',
                                                 x_col='tag_id',
                                                 y_col='sub_class',
                                                 target_size=(30, 75),
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 has_ext=False)

test_set = test_datagen.flow_from_dataframe(dataframe=testdf,
                                            directory='data/masked_rotated_h_test/small vehicle/',
                                            x_col='tag_id',
                                            y_col='sub_class',
                                            target_size=(30, 75),
                                            batch_size=32,
                                            class_mode='categorical',
                                            has_ext=False)"""

training_set = train_datagen.flow_from_directory('data2/masked_rotated_h_train/small vehicle/',
                                                 target_size=(30, 75),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data2/masked_rotated_h_test/small vehicle/',
                                            target_size=(30, 75),
                                            batch_size=32,
                                            class_mode='categorical')

# Initialising
cnn_classifier = tf.keras.models.Sequential()

# 1st conv. layer
cnn_classifier.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(30, 75, 3), activation='relu'))
cnn_classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# 2nd conv. layer
cnn_classifier.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
cnn_classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# 3nd conv. layer
cnn_classifier.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
cnn_classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Flattening
cnn_classifier.add(tf.keras.layers.Flatten())


# Full connection
cnn_classifier.add(tf.keras.layers.Dense(units=256, activation='relu'))
cnn_classifier.add(tf.keras.layers.Dropout(0.5))
cnn_classifier.add(tf.keras.layers.Dense(units=6, activation='sigmoid'))

cnn_classifier.summary()
# Compiling the CNN
cnn_classifier.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

cnn_classifier.fit_generator(training_set,
                             steps_per_epoch=6406,
                             epochs=10,
                             validation_data=test_set,
                             validation_steps=903)

# saving model and weights
cnn_classifier.save_weights('data2/vehicle_classification_weights_dropout.h5')
cnn_classifier.save('data2/vehicle_classification_model_dropout.h5')
