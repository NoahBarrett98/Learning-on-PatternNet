
from helper_modules import util
import functools
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import helper_modules.data_pipeline as dp
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

classes_to_include = [
    'intersection',
    'crosswalk'
]

### initalize loaders ###
train_data = dp.training_data_loader(base_dir="C:/Users/Noah Barrett/Desktop/School/Research 2020/data/PatternNet/PatternNet/images/train_data")
test_data = dp.testing_data_loader(base_dir="C:/Users/Noah Barrett/Desktop/School/Research 2020/data/PatternNet/PatternNet/images/test_data")
### load data ###
train_data.load_data(selected_classes=classes_to_include)
test_data.load_data(selected_classes=classes_to_include)

### Define the CNN model ###

n_filters = 12  # base number of convolutional filters

def make_standard_classifier(n_outputs=1):
    ### clean up definitions ###
    Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu')
    BatchNormalization = tf.keras.layers.BatchNormalization
    Flatten = tf.keras.layers.Flatten
    Dense = functools.partial(tf.keras.layers.Dense, activation='relu')

    model = tf.keras.Sequential([
        Conv2D(filters=1 * n_filters, kernel_size=5, strides=2),
        BatchNormalization(),

        Conv2D(filters=2 * n_filters, kernel_size=5, strides=2),
        BatchNormalization(),

        Conv2D(filters=4 * n_filters, kernel_size=3, strides=2),
        BatchNormalization(),

        Conv2D(filters=6 * n_filters, kernel_size=3, strides=2),
        BatchNormalization(),

        Flatten(),
        Dense(512),
        Dense(n_outputs, activation=None),
    ])


    return model


standard_classifier = make_standard_classifier()


### Train the standard CNN ###

# Training hyperparameters
batch_size = 25
num_epochs = 50
learning_rate = 5e-4

### prep train-data ###
train_data.prepare_for_training(batch_size=batch_size)
test_data.prepare_for_testing()

#using Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate) # define our optimizer
loss_history = util.LossHistory(smoothing_factor=0.99) # to record loss evolution
plotter = util.PeriodicPlotter(sec=2, scale='semilogy')
if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

@tf.function
def standard_train_step(x, y):
    with tf.GradientTape() as tape:
        # feed the images into the model
        logits = standard_classifier(x)

        # Compute the loss, cross entropy with logits
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

    # Backpropagation
    grads = tape.gradient(loss, standard_classifier.trainable_variables)
    optimizer.apply_gradients(zip(grads, standard_classifier.trainable_variables))
    return loss

def standard_train():
    # The training loop!
    for epoch in range(num_epochs):
      for idx in tqdm(range(train_data.get_ds_size()//batch_size)):
        # Grab a batch of training data and propagate through the network
        # avg around 1.5s time
        x, y = train_data.get_train_batch()

        # convert to binary: loader loads as bool array

        # first step avg 1.5s after that avg 0.5s
        loss = standard_train_step(x, y)

        # Record the loss and plot the evolution of the loss as a function of training
        loss_history.append(loss.numpy().mean())
        #plotter.plot(loss_history.get())

    print(loss_history.get())
    """
    Plot
    """
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.plot(loss_history.get())
    plt.show()
    """
    evaluate
    """
    batch_x, batch_y = test_data.get_test_batch(batch_size=500)
    y_pred_standard = tf.round(tf.nn.sigmoid(standard_classifier.predict(batch_x)))
    acc_standard = tf.reduce_mean(tf.cast(tf.equal(batch_y, y_pred_standard), tf.float32))
    print("Standard CNN accuracy on (potentially biased) training set: {:.4f}".format(acc_standard.numpy()))

standard_train()
#images, labels = train_data.get_train_batch()
#util.show_batch(images, labels)
#plt.show()