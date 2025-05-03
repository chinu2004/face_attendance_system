import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import shutil
# ----- STEP 1: AUGMENTATION -----

source_dir = 'faces'  # Original folder with 3 class folders
augmented_dir = 'face_augmented'
img_size = 224
augment_count = 10  # Number of augmented images per original

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

print("📈 Augmenting images...")

os.makedirs(augmented_dir, exist_ok=True)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    save_path = os.path.join(augmented_dir, class_name)
    os.makedirs(save_path, exist_ok=True)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, 0)

        i = 0
        for batch in datagen.flow(img, batch_size=1, save_to_dir=save_path, save_prefix='aug', save_format='jpg'):
            i += 1
            if i >= augment_count:
                break

print("✅ Augmentation completed.")

# ----- STEP 2: TRAINING -----

print("🧠 Training the model...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Split for training and validation
)

train_generator = train_datagen.flow_from_directory(
    augmented_dir,
    target_size=(img_size, img_size),
    batch_size=8,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    augmented_dir,
    target_size=(img_size, img_size),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)

# Load MobileNetV2
base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet')
base_model.trainable = False  # Freeze feature extractor

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss',patience=10, restore_best_weights=True)

# Train the model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop]
)

# Save model
model.save("face_classifier.keras")
print("✅ Model trained and saved as 'face_classifier.keras'.")
try:
    if os.path.exists(augmented_dir):
        shutil.rmtree(augmented_dir)
        print(f"🗑️ Deleted folder '{augmented_dir}' after training.")
except Exception as e:
    print(f"⚠️ Could not delete folder '{augmented_dir}': {e}")

