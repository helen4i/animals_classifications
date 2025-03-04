import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Batch size
batch_size = 16

# Directory paths
train_data_dir = "./anim_dataset/Training Data/Training Data"
validation_data_dir = "./anim_dataset/Validation Data/Validation Data"

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,  
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load dataset from directory. Use 224x224 to match MobileNetV2
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical"
)

validation_generator = train_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),  
    batch_size=batch_size,
    class_mode="categorical"
)

# Mixed Precision for Faster Training
mixed_precision.set_global_policy("mixed_float16")


# Load the MobileNetV2 base model (pretrained on ImageNet)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pretrained layers

# Define the classification model
model = Sequential([
    base_model,
    BatchNormalization(),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(15, activation='softmax')  #15 classes of animals
])

# Compile with a lower learning rate for stability
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(
    train_generator,
    epochs=10,  
    validation_data=validation_generator
)

# Save the model in the Keras format
model.save("animal_classifier_model.keras")