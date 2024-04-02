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
        self.cells = [
                PredNet_Cell(
                    stack_size=stack_size,
                    R_stack_size=R_stack_size,
                    A_filt_size=A_filt_size,
                    Ahat_filt_size=Ahat_filt_size,
                    R_filt_size=R_filt_size)

                for stack_size, R_stack_size, A_filt_size, Ahat_filt_size, R_filt_size in zip(
                    stack_sizes, R_stack_sizes, A_filt_sizes, Ahat_filt_sizes, R_filt_sizes)] # initialize the cells according to the hyperparameters.

        # self.nb_layers = len(stack_sizes)
        self.layer_loss_weights = layer_loss_weights # weighting for each layer in final loss.
        self.time_loss_weights = time_loss_weights # weighting for the timesteps in final loss.

        #PredNet architecture
        self.prednet = PredNet(cell = self.cells, return_sequences = True) # pass the cells to the PredNet(RNN) class

        #Layers for additional error computations for weighted loss during traning
        self.timeDense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, trainable=False), weights=[self.layer_loss_weights, np.zeros(1)], trainable=False)
        self.flatten =  tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, weights=[self.time_loss_weights, np.zeros(1)], trainable=False)

        # self.optimizer = tf.keras.optimizers.Adam()
        # self.mae_loss = tf.keras.losses.MeanAbsoluteError()

        #self.metric_loss = tf.keras.metrics.Mean(name="loss")

    @tf.function
    def call(self, input, training=False):
        x = self.prednet(input, training=training)
        return x

    @tf.function
    def train_step(self, data):
        x, target = data
        with tf.GradientTape() as tape:
            all_error = self(x, training = True) #set traning = True to get errors as output

            #apply the additional error computations
            time_error = self.timeDense(all_error)
            flattened = self.flatten(time_error)
            prediction_error = self.dense(flattened)

            loss = self.compute_loss(y = target, y_pred = prediction_error) # target is a 0 initialized array reflecting the self-supervided goal of minimizing overall prediction error.

            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(target, prediction_error)
        
        return {m.name: m.result() for m in self.metrics}
    
    # @property
    # def metrics(self):
    #     return [self.metric_loss]
    
    @tf.function
    def test_step(self, data):
        x_val, target_val = data
        all_error_val = self(x_val, training = True)
        #apply the additional error computations
        time_error = self.timeDense(all_error_val)
        flattened = self.flatten(time_error)
        prediction_error_val = self.dense(flattened)

        self.compute_loss(y=target_val, y_pred = prediction_error_val)
       
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def get_config(self):
        cell_configs = [cell.get_config() for cell in self.cells]
        config = super(PredNetModel, self).get_config()
        config.update({
            'cells': cell_configs,  # Store the configurations of the cells
            'layer_loss_weights': self.layer_loss_weights,
            'time_loss_weights': self.time_loss_weights,
        })
        return config

    @classmethod
    def from_config(cls, config):
        cells = [PredNet_Cell.from_config(**cell_config) for cell_config in config['cells']]
        return cls(cells, **config)
        
        
