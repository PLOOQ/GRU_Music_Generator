import os
import time
import numpy as np
import tensorflow as tf
from Data_Retriever import midiString
from tensorflow.keras.layers.experimental import preprocessing
from mido import Message, MidiFile, MidiTrack
 
# Import Data
text = midiString
 
# Number of characters in the imported data
# print('Length of text: {} characters'.format(len(text)))
 
# First 250 characters in text
# print("First 250 characters:")
# print(text[:250])
 
# Number of unique characters in the file
vocab = sorted(set(text))
# print('There are {} unique characters in the text data'.format(len(vocab)))
 
# Create StringLookup layer to convert character into a numeric ID:
ids_from_chars = preprocessing.StringLookup(
   vocabulary=list(vocab))
 
# And layer to invert conversion
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
   vocabulary=ids_from_chars.get_vocabulary(), invert=True)
 
 
# Create a function that inverts the conversion and joins the characters
def text_from_ids(ids):
   return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)
 
 
# Convert the text into a stream of numeric IDs split into tokens.
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
# print("Character IDs: ")
# print(all_ids)
# <tf.Tensor: shape=(1115394,), dtype=int64, numpy=array([20, 49,
#  58, ..., 47, 10,  2])>
 
# Convert to dataset form
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
# for ids in ids_dataset.take(10):
#     print(chars_from_ids(ids).numpy().decode('utf-8'))
# <
# m
# e
# t
# a
 
# m
# e
# s
# s
 
# Sequence length
seq_length = 100
 
# Examples per epoch
examples_per_epoch = len(text)//(seq_length+1)
 
# convert individual characters to sequences of desired size
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
# for seq in sequences.take(1):
#   print(chars_from_ids(seq))
# tf.Tensor(
# [b'<' b'm' b'e' b't' b'a' b' ' b'm' b'e' b's' b's' b'a' b'g' b'e' b' '
#  b't' b'r' b'a' b'c' b'k' b'_' b'n' b'a' b'm' b'e' b' ' b'n' b'a' b'm'
#  b'e' b'=' b"'" b'P' b'i' b'a' b'n' b'o' b'\\' b'x' b'0' b'0' b"'" b' '
#  b't' b'i' b'm' b'e' b'=' b'0' b'>' b'\n' b'<' b'm' b'e' b't' b'a' b' '
#  b'm' b'e' b's' b's' b'a' b'g' b'e' b' ' b't' b'i' b'm' b'e' b'_' b's'
#  b'i' b'g' b'n' b'a' b't' b'u' b'r' b'e' b' ' b'n' b'u' b'm' b'e' b'r'
#  b'a' b't' b'o' b'r' b'=' b'2' b' ' b'd' b'e' b'n' b'o' b'm' b'i' b'n'
#  b'a' b't' b'o'], shape=(101,), dtype=string)
 
# Convert to text for visualization
# for seq in sequences.take(5):
#     print(text_from_ids(seq).numpy())
# b"<meta message track_name name='Piano\\x00' time=0>\n<meta message
# time_signature numerator=2 denominato"
# b"r=8 clocks_per_click=24 notated_32nd_notes_per_beat=8 time=0>\n<meta
# message key_signature key='C' tim"
# b'e=0>\n<meta message set_tempo tempo=500000 time=0>\ncontrol_change
# channel=0 control=121 value=0 time=0'
# b'\nprogram_change channel=0 program=0 time=0\ncontrol_change channel=0
# control=7 value=100 time=0\ncontro'
# b'l_change channel=0 control=10 value=64 time=0\ncontrol_change channel=0
# control=91 value=30 time=0\ncon'
 
 
# Create function to create input/target pairs
def split_input_target(sequence):
   input_text = sequence[:-1]
   target_text = sequence[1:]
   return input_text, target_text
 
 
# Map function
dataset = sequences.map(split_input_target)
 
# for input_example, target_example in dataset.take(1):
#     print("Input :", text_from_ids(input_example).numpy())
#     print("Target:", text_from_ids(target_example).numpy())
# Input : b"<meta message track_name name='Piano\\x00' time=0>\n<meta message
#  time_signature numerator=2 denominat"
# Target: b"meta message track_name name='Piano\\x00' time=0>\n<meta message
# time_signature numerator=2 denominato"
 
# Batch size
BATCH_SIZE = 64
 
# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000
 
# Create dataset
dataset = (
   dataset
   .shuffle(BUFFER_SIZE)
   .batch(BATCH_SIZE, drop_remainder=True)
   .prefetch(tf.data.experimental.AUTOTUNE))
 
# Length of the vocabulary in chars
vocab_size = len(vocab)
 
# The embedding dimension
embedding_dim = 256
 
# Number of RNN units
rnn_units = 1024  # Can probably tinker with rnn units as well
 
 
# Class for first model
class MyModel(tf.keras.Model):
 
   def __init__(self, vocab_size, embedding_dim, rnn_units):
       super().__init__(self)
 
       # Create input layer:
       # A trainable lookup table that will map each character-ID
       # to a vector with embedding_dim dimensions
       self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
 
       # Create GRU layer:
       # A type of RNN with size units=rnn_units (You can
       # also use an LSTM layer here)
       self.gru = tf.keras.layers.GRU(rnn_units,
                                      return_sequences=True,
                                      return_state=True)
 
       # Create output layer:
       # Layer with vocab_size logit outputs.
       # logit is the logarithmic
       # of the odds for each character in the vocabulary
       self.dense = tf.keras.layers.Dense(vocab_size)
 
   # For each character the model looks up the embedding, runs the GRU
   # one timestep with the embedding as input, and applies the dense layer
   # to generate logits predicting the log-likelihood of the next character:
   def call(self, inputs, states=None, return_state=False, training=False):
       x = inputs
       x = self.embedding(x, training=training)
       if states is None:
           states = self.gru.get_initial_state(x)
       x, states = self.gru(x, initial_state=states, training=training)
       x = self.dense(x, training=training)
 
       if return_state:
           return x, states
       else:
           return x
 
 
# Left Off
# Instantiate model
model = MyModel(
   # Be sure the vocabulary size matches the `StringLookup` layers.
   vocab_size=len(ids_from_chars.get_vocabulary()),
   embedding_dim=embedding_dim,
   rnn_units=rnn_units)
 
# Try the model:
for input_example_batch, target_example_batch in dataset.take(1):
   example_batch_predictions = model(input_example_batch)
   # print("(batch_size, sequence_length, vocab_size):")
   # print(example_batch_predictions.shape)
   # (64, 100, 67)  # (batch_size, sequence_length, vocab_size)
 
# Model Summary
# model.summary()
 
# sampled_indices = tf.random.categorical(example_batch_predictions[0],
#                                       num_samples=1)
# sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
 
# This gives us, at each timestep, a prediction of the next character index:
# array([ 7, 57, 47, 49,  4, 11, 33, 22, 23, 45, 26,  4, 16, 40, 53, 32, 27,
#         6, 42, 11, 43, 50, 25, 13, 52, 37,  5,  3, 35, 50, 21, 18, 26, 55,
#        23, 30,  6, 49, 25, 52, 11, 45, 61,  6, 52, 42, 15, 57, 40, 31, 61,
#        18, 52, 18, 57, 15,  8, 17, 24, 34, 58, 57, 34, 50, 64, 53, 23, 52,
#        56, 26,  1, 63, 35, 35, 46, 57, 24, 35, 20, 49, 31, 15, 11, 52, 41,
#        20, 45, 44, 50, 48, 59, 60, 46,  3,  5, 48, 28,  4, 64, 57])
 
 
# Decode these to see the text predicted by this untrained model:
# print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
# print()
# print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())
 
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
 
example_batch_loss = loss(target_example_batch, example_batch_predictions)
mean_loss = example_batch_loss.numpy().mean()
# print("Prediction shape: ", example_batch_predictions.shape,
#      " # (batch_size, sequence_length, vocab_size)")
# print("Mean loss:        ", mean_loss)
 
 
# A newly initialized model shouldn't be too sure of itself, the output
# logits should all have similar magnitudes. To confirm this you can check
# that the exponential of the mean loss is approximately equal to
# the vocabulary size. A much higher loss means the model is sure of its
# wrong answers, and is badly initialized:
 
# print("Mean loss:" + str(tf.exp(mean_loss).numpy()))
# print("Vocabulary Size:" + str(vocab_size))
 
model.compile(optimizer='adam', loss=loss)
 
# Configure checkpoints
# Use a tf.keras.callbacks.ModelCheckpoint to
# ensure that checkpoints are saved during training:
 
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
 
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
   filepath=checkpoint_prefix,
   save_weights_only=True)
 
# # Execute the training
# EPOCHS = 1
# print("\n" + "Training method 1" + "\n")
# history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
 
 
# Each time you call the model you pass in some text and an internal state.
# The model returns a prediction for the next character and its new state.
# Pass the prediction and state back in to continue generating text.
 
 
# The following makes a single step prediction:
class OneStep(tf.keras.Model):
   # temperature parameter to generate more or less random predictions.
   def __init__(self, model, chars_from_ids, ids_from_chars, temperature=0.8):
       super().__init__()
       self.temperature = temperature
       self.model = model
       self.chars_from_ids = chars_from_ids
       self.ids_from_chars = ids_from_chars
 
   # Create a mask to prevent "" or "[UNK]" from being generated.
       skip_ids = self.ids_from_chars(['', '[UNK]'])[:, None]
       sparse_mask = tf.SparseTensor(
           # Put a -inf at each bad index.
           values=[-float('inf')]*len(skip_ids),
           indices=skip_ids,
           # Match the shape to the vocabulary
           dense_shape=[len(ids_from_chars.get_vocabulary())])
       self.prediction_mask = tf.sparse.to_dense(sparse_mask)
 
   @tf.function
   def generate_one_step(self, inputs, states=None):
       # Convert strings to token IDs.
       input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
       input_ids = self.ids_from_chars(input_chars).to_tensor()
 
       # Run the model.
       # predicted_logits.shape is [batch, char, next_char_logits]
       predicted_logits, states = self.model(inputs=input_ids, states=states,
                                             return_state=True)
       # Only use the last prediction.
       predicted_logits = predicted_logits[:, -1, :]
       predicted_logits = predicted_logits/self.temperature
       # Apply the prediction mask:
       # prevent "" or "[UNK]" from being generated.
       predicted_logits = predicted_logits + self.prediction_mask
 
       # Sample the output logits to generate token IDs.
       predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
       predicted_ids = tf.squeeze(predicted_ids, axis=-1)
 
       # Convert from token ids to characters
       predicted_chars = self.chars_from_ids(predicted_ids)
 
       # Return the characters and model state.
       return predicted_chars, states
 
 
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
