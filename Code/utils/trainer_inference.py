import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from utils.transformer import TransformerEncoder, PatchClassEmbedding, Patches

class Trainer:
    def __init__(self, config, activation,optimizer_name,trainset_xdata,num_classes):
        self.config = config      
        self.model_size =  self.config['MODEL_SIZE']
        self.n_heads = self.config[self.model_size]['N_HEADS']
        self.n_layers = self.config[self.model_size]['N_LAYERS']
        self.embed_dim = self.config[self.model_size]['EMBED_DIM']
        self.dropout = self.config[self.model_size]['DROPOUT']
        self.mlp_head_size = self.config[self.model_size]['MLP']
        self.activation = activation
        self.optimizer_name = optimizer_name
        self.d_model = 64 * self.n_heads
        self.d_ff = self.d_model * 4
        self.pos_emb = self.config['POS_EMB']
        self.X_train = trainset_xdata
        self.num_classes = num_classes

    def build_act(self, transformer):
        inputs = tf.keras.layers.Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
        x = tf.keras.layers.Dense(self.d_model)(inputs)
        x = PatchClassEmbedding(self.d_model, self.X_train.shape[1], 
                                pos_emb=None)(x)
        x = transformer(x)
        x = tf.keras.layers.Lambda(lambda x: x[:,0,:])(x)
        x = tf.keras.layers.Dense(self.mlp_head_size)(x)
        outputs = tf.keras.layers.Dense(self.num_classes)(x)
        return tf.keras.models.Model(inputs, outputs)

    def get_model(self,lr,wd):
        transformer = TransformerEncoder(self.d_model, self.n_heads, self.d_ff, self.dropout, self.activation, self.n_layers)
        self.model = self.build_act(transformer)
        if self.optimizer_name == 'AdamW':
            print('Optimizer in Get Model(): ','AdamW')
            optimizer = tfa.optimizers.AdamW(learning_rate=lr,weight_decay=wd)
        elif self.optimizer_name == 'LAMB':
            print('Optimizer in Get Model(): ','LAMB')
            optimizer = tfa.optimizers.LAMB(learning_rate=lr)
        elif self.optimizer_name == 'LazyAdam':
            print('Optimizer in Get Model(): ','LazyAdam')
            optimizer = tfa.optimizers.LazyAdam(learning_rate=lr)
        elif self.optimizer_name == 'RAdam':
            print('Optimizer in Get Model(): ','RAdam')
            optimizer = tfa.optimizers.RectifiedAdam(learning_rate=lr)
        elif self.optimizer_name == 'SGDW':
            print('Optimizer in Get Model(): ','SGDW')
            optimizer = tfa.optimizers.SGDW(learning_rate=lr, weight_decay=wd, momentum=0.9)
        else:
            print(f'Optimizer {self.optimizer_name} not found!')
            print('Using Default AdamW Optimizer')
            optimizer = tfa.optimizers.AdamW(learning_rate=lr,weight_decay=wd)

        self.model.compile(optimizer=optimizer,
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
                        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])
        return self.model