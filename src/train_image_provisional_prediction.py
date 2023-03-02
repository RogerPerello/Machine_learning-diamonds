import pandas as pd
import tensorflow as tf
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v3 import MobileNetV3Large
from keras.applications.mobilenet_v3 import preprocess_input as preprocess_input_mobilenet
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, save_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


processed_images_path = 'src/data/processed/images'
df_images_data= pd.read_csv('src/data/processed/images_data_processed.csv')

# CAUTION: The "generator" gives different images every time
# It is unavoidable, so this model will be slightly different if loaded again
# The final model on "streamlit" won't be altered, since is trained with de data from "fixed_images_dataframe.pkl"
# Is not advisable to train this model without GPU, since it can be rather slow

# Preprocessing
df_images_data = df_images_data[['Id', 'price']]

df_images_data['Id'] = df_images_data['Id'].apply(lambda x: x + '.jpg')

scaler = StandardScaler()

df_images_data['price'] = scaler.fit_transform(df_images_data[['price']])

X_train, X_test, y_train, y_test = train_test_split(df_images_data['Id'], df_images_data.drop(columns='Id'), train_size=0.8, random_state=42)

df_train = pd.concat((X_train, y_train), axis=1)
df_test = pd.concat((X_test, y_test), axis=1)

# Images generation
data_augmentation = ImageDataGenerator(rotation_range=20,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        zoom_range=0.1,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        validation_split=0.3,
                                        fill_mode='constant',
                                        cval=(176+177+181) / 3,
                                        preprocessing_function=preprocess_input_mobilenet
                                        )

train_generator = data_augmentation.flow_from_dataframe(dataframe=df_train,
                                                        directory=processed_images_path,
                                                        target_size=(224, 224),
                                                        class_mode='raw',
                                                        shuffle=False,
                                                        x_col='Id',
                                                        y_col=list(df_images_data.columns[1:]),
                                                        seed=42,
                                                        subset='training',
                                                        batch_size=256
                                                        )

validation_generator = data_augmentation.flow_from_dataframe(dataframe=df_train,
                                                                directory=processed_images_path,
                                                                target_size=(224, 224),
                                                                class_mode='raw',
                                                                shuffle=False,
                                                                x_col='Id',
                                                                y_col=list(df_images_data.columns[1:]),
                                                                seed=42,
                                                                subset='validation',
                                                                batch_size=256
                                                                )

test_generator = data_augmentation.flow_from_dataframe(dataframe=df_test,
                                                        directory=processed_images_path,
                                                        target_size=(224, 224),
                                                        class_mode='raw',
                                                        shuffle=False,
                                                        x_col='Id',
                                                        y_col=list(df_images_data.columns[1:]),
                                                        seed=42,
                                                        batch_size=256
                                                        )

# Architecture
tf.random.set_seed(42)

base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

top_model = base_model.output

top_model = GlobalAveragePooling2D()(top_model)

top_model = Dense(1024, activation='relu')(top_model)

top_model = Dense(512, activation='relu')(top_model)

top_model = Dense(256, activation='relu')(top_model)

top_model = Dense(128, activation='relu')(top_model)

top_model = Dense(64, activation='relu')(top_model)

top_model = Dense(32, activation='relu')(top_model)

top_model = Dense(16, activation='relu')(top_model)

top_model = Dense(8, activation='relu')(top_model)

output_layer = Dense(1, activation='linear')(top_model)

model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Training
print('--- Training started ---')

start_time = time.time()

history = model.fit(train_generator,
                    epochs=100,
                    batch_size=128,
                    validation_data=validation_generator,
                    callbacks=[early_stop, reduce_lr]
                    )

execution_time = time.time() - start_time

print(f'--- Training done in {round(execution_time, 2)} sec/s ---\n')

# Serialization
save_model(model, 'src/models/predict_from_images/price_prediction_images.h5')

print('--- Serialization done ---')
