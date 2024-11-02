import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from utils.transformer import TransformerEncoder, PatchClassEmbedding
from utils.data import one_hot
import datetime    
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from scipy import stats
import time
import sklearn

# TRAINER CLASS 
class Trainer:
    def __init__(self, config):
        self.config = config       
        self.model_size =  self.config['MODEL_SIZE']
        self.n_heads = self.config[self.model_size]['N_HEADS']
        self.n_layers = self.config[self.model_size]['N_LAYERS']
        self.embed_dim = self.config[self.model_size]['EMBED_DIM']
        self.dropout = self.config[self.model_size]['DROPOUT']
        self.mlp_head_size = self.config[self.model_size]['MLP']
        self.d_model = (self.embed_dim * self.n_heads) #ORIGINAL: (64 * self.n_heads)
        self.d_ff = self.d_model #self.d_model * 4
        self.pos_emb = self.config['POS_EMB']
    
    def get_activation_function(self,name):
        if name == 'relu':
            return tf.nn.relu
        elif name == 'leaky-relu':
            return tf.nn.leaky_relu
        elif name == 'selu':
            return tf.nn.selu
        elif name == 'sigmoid':
            return tf.nn.sigmoid
        elif name == 'silu':
            return tf.nn.silu
        elif name == 'swish':
            return tf.nn.swish
        elif name == 'elu':
            return tf.nn.elu
        elif name == 'gelu':
            return tf.nn.gelu
        elif name == 'mish':
            return lambda x: x * tf.math.tanh(tf.math.softplus(x))
        elif name == 'softmax':
            return tf.nn.softmax
        elif name == 'softplus':
            return tf.nn.softplus
        else:
            print(f"Activation function {name} not supported.")
            print('Using GeLU as default')
            return tf.nn.gelu

    def convert_data2tf(self, X_train, Y_train, X_test, Y_test, X_val, Y_val):
        train_len = len(Y_train)
        test_len = len(Y_test)
        val_len = len(Y_val)

        train_steps = np.ceil(float(train_len)/self.config['BATCHSIZE'])
        test_steps = np.ceil(float(test_len)/self.config['BATCHSIZE'])
        val_steps = np.ceil(float(val_len)/self.config['BATCHSIZE'])
        
        print("Value of Train_steps: ",train_steps)
        print("Value of Test_steps: ",test_steps)
        print("Value of Val_steps: ",val_steps)

        ds_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        ds_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        ds_val = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
                    
        ds_train = ds_train.map(lambda x,y : one_hot(x,y,self.config['NUM_CLASSES']), 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(X_train.shape[0])
        
        ds_test = ds_test.map(lambda x,y : one_hot(x,y,self.config['NUM_CLASSES']), 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.cache()

        ds_val = ds_val.map(lambda x,y : one_hot(x,y,self.config['NUM_CLASSES']), 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.cache()

        ds_train = ds_train.batch(self.config['BATCHSIZE'])
        ds_test = ds_test.batch(self.config['BATCHSIZE'])
        ds_val = ds_val.batch(self.config['BATCHSIZE'])
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE) 
        return ds_train,ds_test,ds_val,train_steps,test_steps,val_steps   


    def build_act(self, transformer):
        inputs = tf.keras.layers.Input(shape=(30, 68))
        x = tf.keras.layers.Dense(self.d_model)(inputs)
        x = PatchClassEmbedding(self.d_model, 30, 
                                pos_emb=None)(x)
        x = transformer(x)
        x = tf.keras.layers.Lambda(lambda x: x[:,0,:])(x)
        x = tf.keras.layers.Dense(self.mlp_head_size)(x)
        outputs = tf.keras.layers.Dense(self.config['NUM_CLASSES'])(x)
        return tf.keras.models.Model(inputs, outputs)

    def get_model(self):
        transformer = TransformerEncoder(self.d_model, self.n_heads, self.d_ff, self.dropout, self.get_activation_function(self.config['ACTIVATION_FUNCTION']), self.n_layers)
        self.model = self.build_act(transformer)
        
        if self.config['OPTIMIZER'] == 'AdamW':
            print('Optimizer in Get Model(): ','AdamW')
            optimizer = tfa.optimizers.AdamW(learning_rate=self.config['LR_RATE'],weight_decay=self.config['WD'])
        elif self.config['OPTIMIZER'] == 'LAMB':
            print('Optimizer in Get Model(): ','LAMB')
            optimizer = tfa.optimizers.LAMB(learning_rate=self.config['LR_RATE'])
        elif self.config['OPTIMIZER'] == 'LazyAdam':
            print('Optimizer in Get Model(): ','LazyAdam')
            optimizer = tfa.optimizers.LazyAdam(learning_rate=self.config['LR_RATE'])
        elif self.config['OPTIMIZER'] == 'RAdam':
            print('Optimizer in Get Model(): ','RAdam')
            optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.config['LR_RATE'])
        elif self.config['OPTIMIZER'] == 'SGDW':
            print('Optimizer in Get Model(): ','SGDW')
            optimizer = tfa.optimizers.SGDW(learning_rate=self.config['LR_RATE'], weight_decay=self.config['WD'], momentum=0.9)
        else:
            print(f'Optimizer {self.optimizer_name} not found!')
            print('Using Default AdamW Optimizer')
            optimizer = tfa.optimizers.AdamW(learning_rate=self.config['LR_RATE'],weight_decay=self.config['WD'])

        self.model.compile(optimizer=optimizer,
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
                        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])
        return self.model
    
    def do_training_and_testing(self,ds_train,ds_test,ds_val,train_steps,test_steps,val_steps):
        self.get_model() 
        model_history = self.model.fit(ds_train,
        epochs=self.config['EPOCHS'], initial_epoch=0, verbose=self.config['TRAINING_VERBOSE'],
        steps_per_epoch= int(train_steps),
        validation_data =ds_val,
        validation_steps = val_steps,
        )
        start_time = time.time()
        # Evaluate the model
        _, test_accuracy = self.model.evaluate(ds_test, steps=test_steps)
        end_time = time.time()
        # Compute and print the evaluation time
        evaluation_time = end_time - start_time
            
         #LAST EPOCH TRAINED MODEL CLASS-WISE ACCURACY on TEST SET   
        X, y = tuple(zip(*ds_test))
        y_pred = np.argmax(tf.nn.softmax(self.model.predict(tf.concat(X, axis=0)), axis=-1),axis=1)
        balanced_test_accuracy = sklearn.metrics.balanced_accuracy_score(tf.math.argmax(tf.concat(y, axis=0), axis=1), y_pred)
        
        # Obtain true labels
        y_true = np.argmax(np.vstack(y), axis=1)
        cm = confusion_matrix(y_true, y_pred)
        training_acc_hist = model_history.history['accuracy']
        validation_acc_hist = model_history.history['val_accuracy']
        training_loss_hist = model_history.history['loss']
        validation_loss_hist = model_history.history['val_loss']
        
        return self.model,training_acc_hist,validation_acc_hist,training_loss_hist,validation_loss_hist,test_accuracy,balanced_test_accuracy,evaluation_time,cm
        

    def do_training(self,ds_train,ds_test,ds_val,train_steps,test_steps,val_steps):
        self.get_model()
        if self.config['VAL_FLAG']:           
            model_history = self.model.fit(ds_train,
            epochs=self.config['EPOCHS'], initial_epoch=0, verbose=self.config['TRAINING_VERBOSE'],
            steps_per_epoch= int(train_steps),
            validation_data =ds_val,
            validation_steps = val_steps,
            )
            _, val_accuracy = self.model.evaluate(ds_val, steps=val_steps)
        
        return val_accuracy
        