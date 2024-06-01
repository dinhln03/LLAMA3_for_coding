import tensorflow as tf
import tensorflow_zero_out 
import numpy as np
import os

# Create a model using low-level tf.* APIs
class ZeroOut(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int32)])
  def __call__(self, x):
    return tensorflow_zero_out.zero_out(x)
model = ZeroOut()
# (ro run your model) result = Squared(5.0) # This prints "25.0"
# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_tf_dir")
concrete_func = model.__call__.get_concrete_function()

# Convert the model.
# Notes that for the versions earlier than TensorFlow 2.7, the
# from_concrete_functions API is able to work when there is only the first
# argument given:
# > converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func],
                                                            )
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)