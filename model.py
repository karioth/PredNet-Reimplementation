from PredNet import *
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import models

# PredNet model definition
class PredNetModel(models.Model):
    ''' Puts together a classical version of the PredNet architecture using the modern modular implementation'''
    def __init__(self, stack_sizes, R_stack_sizes, A_filt_sizes, Ahat_filt_sizes, R_filt_sizes, layer_loss_weights, time_loss_weights,**kwargs):
        super(PredNetModel, self).__init__(**kwargs)
        cells = [
                PredNet_Cell(
                    stack_size=stack_size,
                    R_stack_size=R_stack_size,
                    A_filt_size=A_filt_size,
                    Ahat_filt_size=Ahat_filt_size,
                    R_filt_size=R_filt_size)

                for stack_size, R_stack_size, A_filt_size, Ahat_filt_size, R_filt_size in zip(
                    stack_sizes, R_stack_sizes, A_filt_sizes, Ahat_filt_sizes, R_filt_sizes)] # initialize the cells according to the hyperparameters.

        # self.nb_layers = len(stack_sizes)
        # self.layer_loss_weights = layer_loss_weights # weighting for each layer in final loss.
        # self.time_loss_weights = time_loss_weights # weighting for the timesteps in final loss.

        #PredNet architecture
        self.prednet = PredNet(cell = cells, return_sequences = True) # pass the cells to the PredNet(RNN) class

        #Layers for additional error computations for weighted loss during traning
        self.timeDense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)
        self.flatten =  tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)

        # self.optimizer = tf.keras.optimizers.Adam()
        # self.mae_loss = tf.keras.losses.MeanAbsoluteError()

        self.metric_loss = tf.keras.metrics.Mean(name="loss")

    @tf.function
    def call(self, input, training=False):
        x = self.prednet(input, training=training)
        return x

    @tf.function
    def train_step(self, x, target):
        with tf.GradientTape() as tape:
            all_error = self(x, training = True) #set traning = True to get errors as output

            #apply the additional error computations
            time_error = self.timeDense(all_error)
            flattened = self.flatten(time_error)
            prediction_error = self.dense(flattened)

            loss = self.mae_loss(target, prediction_error) # target is a 0 initialized array reflecting the self-supervided goal of minimizing overall prediction error.

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.metric_loss.update_state(loss)

    def validate(self, val_data):
        # Reset validation metrics
        self.metric_loss.reset_states()
        # Iterate over validation data
        for x_val, target_val in val_data:
          all_error_val = self(x_val, training = True)

          #apply the additional error computations
          time_error = self.timeDense(all_error_val)
          flattened = self.flatten(time_error)
          prediction_error_val = self.dense(flattened)

          val_loss = self.mae_loss(target_val, prediction_error_val)

          self.metric_loss.update_state(val_loss)

        return self.metric_loss.result()
