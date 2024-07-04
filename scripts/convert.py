import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, BatchNormalization, Bidirectional
import h5py
import numpy as np

# Function to load data from the h5 file
def load_data_from_h5(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][:]
        return data

# Define the model architecture
def build_model(input_dim, num_classes):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(None, input_dim)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
    return model


input_dim = 51864 
num_classes = 29  

# Load weights from the .h5 file
file_path = '/home/whisper/dataset-d82nldmq9.h5'
data = load_data_from_h5(file_path, 'data/0/batch_0')

model = build_model(input_dim, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the full model including architecture and weights as a TensorFlow SavedModel
saved_model_path = '/home/whisper/saved_model'
model.save(saved_model_path)
print("Model saved successfully as a TensorFlow SavedModel.")

# Convert the SavedModel to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('whisper.tflite', 'wb') as f:
    f.write(tflite_model)
print("Model converted to TensorFlow Lite and saved successfully.")

# Optimize the model using default quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Save the quantized TensorFlow Lite model
with open('whisper_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)
print("Quantized model converted to TensorFlow Lite and saved successfully.")
