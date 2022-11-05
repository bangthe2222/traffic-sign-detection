import tensorflow as tf
keras_model = tf.keras.models.load_model("my_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] 
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open('model_fp16.tflite', 'wb') as f:
  f.write(tflite_model)