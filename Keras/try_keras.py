import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas
import keras_encoder as enc

"""apply on a general / simple neural network"""
"""replace x Train with input column and y train with output column"""

"""bei memory problemen sequence cutten in 100 stücke"""


"""Was bekommt man als result?"""
"""Website code"""
train = pandas.read_csv("./cleanData/cleanData/20D16.tsv", sep= '\t')
test = pandas.read_csv("./cleanData/cleanData/20B12.tsv", sep= '\t')

## Test Encoder Stack
dataset_creator = enc.NMTDataset('seq-seq')
BUFFER_SIZE = 1273
BATCH_SIZE = 32
num_examples = 100
train_dataset, val_dataset, inp_seq, targ_seq = dataset_creator.call_seq(num_examples, BUFFER_SIZE, BATCH_SIZE,train)
test_dataset, val_testdataset, inp_testseq, targ_testseq = dataset_creator.call_seq(num_examples, BUFFER_SIZE, BATCH_SIZE,train)
example_input_batch, example_target_batch = next(iter(train_dataset))
test_input_batch, test_target_batch = next(iter(test_dataset))

vocab_inp_size = 32
vocab_tar_size = 32
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]


embedding_dim = 32
units = 32
steps_per_epoch = num_examples//BATCH_SIZE

## Test Encoder Stack
print(vocab_inp_size)
encoder = enc.Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_h, sample_c = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder h vecotr shape: (batch size, units) {}'.format(sample_h.shape))
print ('Encoder c vector shape: (batch size, units) {}'.format(sample_c.shape))

x_train = sample_output
y_train, some_h, some_c = encoder(example_target_batch, sample_hidden)

x_test, some_h_test, some_c_test = encoder(test_input_batch, sample_hidden)
y_test, some_h_test1, some_c_test1 = encoder(test_target_batch, sample_hidden)


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(32)
])


predictions = model(x_train[:1]).numpy()
print(predictions.shape)

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.CategoricalCrossentropy()

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])





"""
Epoch 1/5
1/1 [==============================] - 0s 452ms/step - loss: 0.2649 - accuracy: 0.0397
Epoch 2/5
1/1 [==============================] - 0s 142ms/step - loss: 0.0553 - accuracy: 0.0426
Epoch 3/5
1/1 [==============================] - 0s 104ms/step - loss: -0.1059 - accuracy: 0.0489
Epoch 4/5
1/1 [==============================] - 0s 97ms/step - loss: -0.2539 - accuracy: 0.0602
Epoch 5/5
1/1 [==============================] - 0s 65ms/step - loss: -0.3691 - accuracy: 0.0679
1/1 - 0s - loss: -4.9407e-01 - accuracy: 0.0867 - 101ms/epoch - 101ms/step


Epoch 1/5
1/1 [==============================] - 1s 501ms/step - loss: 0.0275 - accuracy: 0.0218
Epoch 2/5
1/1 [==============================] - 0s 92ms/step - loss: -0.1303 - accuracy: 0.0470
Epoch 3/5
1/1 [==============================] - 0s 118ms/step - loss: -0.2826 - accuracy: 0.0649
Epoch 4/5
1/1 [==============================] - 0s 71ms/step - loss: -0.3981 - accuracy: 0.0745
Epoch 5/5
1/1 [==============================] - 0s 61ms/step - loss: -0.4831 - accuracy: 0.0769
1/1 - 0s - loss: -5.9351e-01 - accuracy: 0.0930 - 93ms/epoch - 93ms/step

"""