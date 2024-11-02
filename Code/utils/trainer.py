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

# TRAINER CLASS 
class Trainer:
    def __init__(self, config,X_train,Y_train,X_test,Y_test,X_val,Y_val):
        self.config = config       
        self.model_size =  self.config['MODEL_SIZE']
        self.n_heads = self.config[self.model_size]['N_HEADS']
        self.n_layers = self.config[self.model_size]['N_LAYERS']
        self.embed_dim = self.config[self.model_size]['EMBED_DIM']
        self.dropout = self.config[self.model_size]['DROPOUT']
        self.mlp_head_size = self.config[self.model_size]['MLP']
        self.d_model = 64 * self.n_heads
        self.d_ff = self.d_model * 4
        self.pos_emb = self.config['POS_EMB']
        self.X_train = X_train
        self.y_train = Y_train
        self.X_test = X_test
        self.y_test = Y_test
        self.X_val = X_val
        self.y_val = Y_val

    
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

    def batch_prefetch_tfdata(self):
        self.ds_train = self.ds_train.batch(self.config['BATCHSIZE'])
        self.ds_test = self.ds_test.batch(self.config['BATCHSIZE'])
        self.ds_val = self.ds_val.batch(self.config['BATCHSIZE'])
        self.ds_train = self.ds_train.prefetch(tf.data.experimental.AUTOTUNE)
        self.ds_test = self.ds_test.prefetch(tf.data.experimental.AUTOTUNE)
        self.ds_val = self.ds_val.prefetch(tf.data.experimental.AUTOTUNE)        

    def convert_data2tf(self):
        self.train_len = len(self.y_train)
        self.test_len = len(self.y_test)
        self.val_len = len(self.y_val)

        self.train_steps = np.ceil(float(self.train_len)/self.config['BATCHSIZE'])
        self.test_steps = np.ceil(float(self.test_len)/self.config['BATCHSIZE'])
        self.val_steps = np.ceil(float(self.val_len)/self.config['BATCHSIZE'])
        
        print("Value of Train_steps: ",self.train_steps)
        print("Value of Test_steps: ",self.test_steps)
        print("Value of Val_steps: ",self.val_steps)

        self.ds_train = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        self.ds_test = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))
        self.ds_val = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
                    
        self.ds_train = self.ds_train.map(lambda x,y : one_hot(x,y,self.config['NUM_CLASSES']), 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_train = self.ds_train.cache()
        self.ds_train = self.ds_train.shuffle(self.X_train.shape[0])
        
        self.ds_test = self.ds_test.map(lambda x,y : one_hot(x,y,self.config['NUM_CLASSES']), 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_test = self.ds_test.cache()

        self.ds_val = self.ds_val.map(lambda x,y : one_hot(x,y,self.config['NUM_CLASSES']), 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_val = self.ds_val.cache()

        self.batch_prefetch_tfdata()

    def build_act(self, transformer):
        inputs = tf.keras.layers.Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
        x = tf.keras.layers.Dense(self.d_model)(inputs)
        x = PatchClassEmbedding(self.d_model, self.X_train.shape[1], 
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

    
    def plot_confusion_matrix(self, y_true, y_pred,test_accuracy):
        cm = confusion_matrix(y_true, y_pred)
        class_accuracy = np.diag(cm) / cm.sum(axis=1)

        #Plot Confusion Matrix
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.config['CLASSES'], yticklabels=self.config['CLASSES'])
        plt.xlabel("Predicted", fontweight='bold')
        plt.ylabel("True", fontweight='bold')
        plt.title("Confusion Matrix", fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')
        
        # Plot Bar Plot of Class-wise Accuracies
        plt.figure(figsize=(10, 8))
        sns.barplot(x=self.config['CLASSES'], y=class_accuracy, color="blue")
        for i in range(len(class_accuracy)):
            plt.text(i, class_accuracy[i] + 0.005, f'{class_accuracy[i]:.2f}', color='red', ha='center', fontweight='bold')
        plt.xlabel("Classes", fontweight='bold')
        plt.ylabel("Accuracy", fontweight='bold')
        plt.title(f"Classwise Accuracies - Overall Accuracy: {test_accuracy:.2f}", fontweight='bold')
        plt.xticks(rotation=45)

        #Classification Report Plot
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).T
        report_df.drop(['support'], axis=1, inplace=True)
        
        plt.figure(figsize=(10, 8))
        y_tick_labels = self.config['CLASSES'] + ['accuracy', 'macro avg', 'weighted avg']
        x_tick_labels = ['Precision', 'Recall', 'F1-Score']
        
        sns.heatmap(report_df, annot=True, cmap="Blues", fmt=".2f",yticklabels=y_tick_labels,xticklabels=x_tick_labels)
        plt.xlabel("Classification Metrics", fontweight='bold')
        plt.ylabel("Classes", fontweight='bold')
        plt.title("Classification Report", fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks()
        plt.show()
        
    def evaluate_saved_model(self,saved_model_path):
        self.get_model()
        self.convert_data2tf()
        self.model.load_weights(saved_model_path)        
        test_loss, test_accuracy = self.model.evaluate(self.ds_test, steps=self.test_steps)
        print('Test Accuracy: ',test_accuracy)
        X, y = tuple(zip(*self.ds_test))
        y_pred = np.argmax(tf.nn.softmax(self.model.predict(tf.concat(X, axis=0)), axis=-1),axis=1)
        # Obtain true labels
        y_true = np.argmax(np.vstack(y), axis=1)
        self.plot_confusion_matrix(y_true,y_pred,test_accuracy)

    def hyperparameters_tuning_plots(self,encoder_layers_list,encoder_layer_val_acc_list,embed_dims_list,embed_dims_val_acc_list,dropout_percent_list,dropout_val_acc_list,mlp_dims_list,mlp_dim_val_acc_list,
    optimizers_list,lr_list,wd_list,adamw_val_acc_list,lamb_val_acc_list,lazyadam_val_acc_list,radam_val_acc_list,sgdw_val_acc_list,optimizers_best_val_acc_list,optimizers_best_lr_list,optimizers_best_wd_list,
        activation_functions_list,activation_function_val_acc_list,batch_size_list,batch_size_val_acc_list):
        
        # Plot for Encoder Layers
        plt.figure(figsize=(10, 8))
        plt.plot(encoder_layers_list, encoder_layer_val_acc_list, color='blue', label='Trendline')
        plt.scatter(encoder_layers_list, encoder_layer_val_acc_list, color='red', label='Data Points')
        for i, (x, y) in enumerate(zip(encoder_layers_list, encoder_layer_val_acc_list)):
            plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontweight='bold', color='black')
        plt.title("Encoder Layers Vs Validation Accuracy", fontweight='bold')
        plt.xlabel("Encoder Layers", fontweight='bold')
        plt.ylabel("Validation Accuracy", fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.legend()

        # Plot for Embedding Dimensions
        plt.figure(figsize=(10, 8))
        plt.plot(embed_dims_list, embed_dims_val_acc_list, color='blue', label='Trendline')
        plt.scatter(embed_dims_list, embed_dims_val_acc_list, color='red', label='Data Points')
        for i, (x, y) in enumerate(zip(embed_dims_list, embed_dims_val_acc_list)):
            plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontweight='bold', color='black')
        plt.title("Embedding Dimensions Vs Validation Accuracy", fontweight='bold')
        plt.xlabel("Embedding Dimensions", fontweight='bold')
        plt.ylabel("Validation Accuracy", fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.legend()

        # Plot for Dropout Percentage
        plt.figure(figsize=(10, 8))
        plt.plot(dropout_percent_list, dropout_val_acc_list, color='blue', label='Trendline')
        plt.scatter(dropout_percent_list, dropout_val_acc_list, color='red', label='Data Points')
        for i, (x, y) in enumerate(zip(dropout_percent_list, dropout_val_acc_list)):
            plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontweight='bold', color='black')
        plt.title("Dropout Percentage Vs Validation Accuracy", fontweight='bold')
        plt.xlabel("Dropout Percentage", fontweight='bold')
        plt.ylabel("Validation Accuracy", fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.legend()

        # Plot for MLP Dimensions
        plt.figure(figsize=(10, 8))
        plt.plot(mlp_dims_list, mlp_dim_val_acc_list, color='blue', label='Trendline')
        plt.scatter(mlp_dims_list, mlp_dim_val_acc_list, color='red', label='Data Points')
        for i, (x, y) in enumerate(zip(mlp_dims_list, mlp_dim_val_acc_list)):
            plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontweight='bold', color='black')
        plt.title("MLP Dimensions Vs Validation Accuracy", fontweight='bold')
        plt.xlabel("MLP Dimensions", fontweight='bold')
        plt.ylabel("Validation Accuracy", fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.legend()

        #AdamW: Weight Decay Versus Validation at Different LR
        # Create subplots
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))

        # Flatten axes for easier iteration
        axes = axes.flatten()

        # Plot for each learning rate
        for i, lr in enumerate(lr_list):
            ax = axes[i]
            lr_val_acc = adamw_val_acc_list[i * len(wd_list): (i + 1) * len(wd_list)]
            ax.scatter(wd_list, lr_val_acc, color='red')
            ax.plot(wd_list, lr_val_acc, marker='o', linestyle='-', color='blue')
            ax.set_title(f'Learning Rate: {lr}', fontweight='bold')
            ax.set_xlabel('Weight Decay', fontweight='bold')
            ax.set_ylabel('Validation Accuracy', fontweight='bold')
            ax.set_xscale('log')
            plt.suptitle(r'AdamW Optimizer: Weight Decay Vs Validation Accuracy at Different Learning Rate', fontweight = 'bold')
        
        #SGDW: Weight Decay Versus Validation at Different LR
        # Create subplots
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))

        # Flatten axes for easier iteration
        axes = axes.flatten()

        # Plot for each learning rate
        for i, lr in enumerate(lr_list):
            ax = axes[i]
            lr_val_acc = sgdw_val_acc_list[i * len(wd_list): (i + 1) * len(wd_list)]
            ax.scatter(wd_list, lr_val_acc, color='red')
            ax.plot(wd_list, lr_val_acc, marker='o', linestyle='-', color='blue')
            ax.set_title(f'Learning Rate: {lr}', fontweight='bold')
            ax.set_xlabel('Weight Decay', fontweight='bold')
            ax.set_ylabel('Validation Accuracy', fontweight='bold')
            ax.set_xscale('log')

        # Add a common title to the entire figure
        plt.suptitle("SGDW Optimizer: Weight Decay Vs Validation Accuracy at Different Learning Rates", fontweight='bold')

        plt.tight_layout()


        #LR VERSUS VALIDATION
        # Data for AdamW optimizer
        adamw_lr_max_acc = [max(adamw_val_acc_list[i * len(wd_list): (i + 1) * len(wd_list)]) for i in range(len(lr_list))]
        adamw_max_wd = [wd_list[np.argmax(adamw_val_acc_list[i * len(wd_list): (i + 1) * len(wd_list)])] for i in range(len(lr_list))]

        # Data for SGDW optimizer
        sgdw_lr_max_acc = [max(sgdw_val_acc_list[i * len(wd_list): (i + 1) * len(wd_list)]) for i in range(len(lr_list))]
        sgdw_max_wd = [wd_list[np.argmax(sgdw_val_acc_list[i * len(wd_list): (i + 1) * len(wd_list)])] for i in range(len(lr_list))]

        # Learning rates
        lr_labels = ['1e-1', '1e-2', '1e-3', '1e-4', '1e-5', '1e-6']

        # Width of the bars
        bar_width = 0.75

        # Index for the x-axis
        ind = np.arange(len(lr_labels))

        # Plotting the bar plots for AdamW
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot AdamW bars
        adamw_bars = ax.bar(ind, adamw_lr_max_acc, bar_width, label='AdamW', color='b')
        ax.set_title('AdamW Optimizer: Learning Rate Vs Validation Accuracy', fontweight='bold')

        # Add validation accuracy and maximum weight decay value inside the AdamW bars
        for bar, wd, acc in zip(adamw_bars, adamw_max_wd, adamw_lr_max_acc):
            height = bar.get_height()
            ax.annotate(f'{acc:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
            ax.annotate(f'WD: {wd}', xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                        xytext=(0, 0), textcoords="offset points",
                        ha='center', va='center', fontweight='bold', color='red')

        # Add some text for labels, title and axes ticks
        ax.set_ylabel('Validation Accuracy', fontweight='bold')
        ax.set_xlabel('Learning Rate', fontweight='bold')
        ax.set_xticks(ind)
        ax.set_xticklabels(lr_labels)
        ax.legend()

        plt.tight_layout()
        plt.show()

        # Plotting the bar plots for SGD with Warm Restarts
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot SGDW bars
        sgdw_bars = ax.bar(ind, sgdw_lr_max_acc, bar_width, label='SGD with Warm Restarts', color='b')
        ax.set_title('SGDW Optimizer: Learning Rate Vs Validation Accuracy', fontweight='bold')

        # Add validation accuracy and maximum weight decay value inside the SGD with Warm Restarts bars
        for bar, wd, acc in zip(sgdw_bars, sgdw_max_wd, sgdw_lr_max_acc):
            height = bar.get_height()
            ax.annotate(f'{acc:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
            ax.annotate(f'WD: {wd}', xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                        xytext=(0, 0), textcoords="offset points",
                        ha='center', va='center', fontweight='bold', color='red')

        # Add some text for labels, title and axes ticks
        ax.set_ylabel('Validation Accuracy', fontweight='bold')
        ax.set_xlabel('Learning Rate', fontweight='bold')
        ax.set_xticks(ind)
        ax.set_xticklabels(lr_labels)
        ax.legend()

        plt.tight_layout()

        # Activation Functions Vs Validation Accuracy
        plt.figure(figsize=(10, 6))

        # Plotting the bar graph
        bars = plt.bar(activation_functions_list, activation_function_val_acc_list, color='skyblue')

        # Annotating the bars with learning rate and weight decay
        for bar, val in zip(bars, activation_function_val_acc_list):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{val:.4f}', 
                    ha='center', va='bottom', weight='bold')

        plt.title('Activation Functions Vs Validation Accuracy', fontweight='bold')
        plt.xlabel('Activation Functions', fontweight='bold')
        plt.ylabel('Validation Accuracy', fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.tight_layout()

        # Batch Size Vs Validation Accuracy
        plt.figure(figsize=(10, 6))

        # Plot the test accuracies with specific styling
        plt.plot(batch_size_list, batch_size_val_acc_list, color='blue', linestyle='-', label='Validation Accuracy Trendline')
        # Set marker points to red
        plt.scatter(batch_size_list, batch_size_val_acc_list, color='red',label='Validation Accuracy Datapoints')
        plt.title('Batch-Size Vs Validation Accuracy', fontweight = 'bold')
        plt.xlabel('Batch-Size', fontweight = 'bold')
        plt.ylabel('Validation Accuracy', fontweight = 'bold')
        offset = 0.008
        # Annotate the first three datapoints on the right side and the rest on the top
        for i, acc in enumerate(batch_size_val_acc_list):
            if i < 4:
                plt.annotate(f'{acc:.4f}', (batch_size_list[i]+2, acc), weight='bold', color='black', ha='left', va='center')
            else:
                plt.annotate(f'{acc:.4f}', (batch_size_list[i], acc+offset), weight='bold', color='black', ha='center', va='bottom')
        # Add legend
        plt.legend()
        plt.show()

    def hyperparameters_tuning(self):
        self.get_model()
        self.convert_data2tf()
        if self.config['VAL_FLAG']:           
            model_history = self.model.fit(self.ds_train,
            epochs=self.config['EPOCHS'], initial_epoch=0, verbose=self.config['TRAINING_VERBOSE'],
            steps_per_epoch= int(self.train_steps),
            validation_data =self.ds_val,
            validation_steps = self.val_steps,
            )
        
        _, val_accuracy = self.model.evaluate(self.ds_val, steps=self.val_steps)
        return val_accuracy
            
        

    def do_training(self):
        self.get_model()
        self.convert_data2tf()

        if self.config['VAL_FLAG']:           
            model_history = self.model.fit(self.ds_train,
            epochs=self.config['EPOCHS'], initial_epoch=0, verbose=self.config['TRAINING_VERBOSE'],
            steps_per_epoch= int(self.train_steps),
            validation_data =self.ds_val,
            validation_steps = self.val_steps,
            )
            current_datetime = datetime.datetime.now()
            current_datetime = current_datetime.strftime("%d-%m-%Y_%H-%M-%S")
            model_name = self.config['MODEL_SIZE']+'_'
            extension = '.h5'
            save_model_path = os.path.join(self.config['SAVED_MODEL_WEIGHTS_PATH'],model_name+current_datetime+extension)
            self.model.save_weights(save_model_path)
            test_loss, test_accuracy = self.model.evaluate(self.ds_test, steps=self.test_steps)
            print('Test Accuracy: ',test_accuracy)
        else:
            model_history = self.model.fit(self.ds_train,
            epochs=self.config['EPOCHS'], initial_epoch=0, verbose=self.config['TRAINING_VERBOSE'],
            steps_per_epoch= int(self.train_steps)
            )
            current_datetime = datetime.datetime.now()
            current_datetime = current_datetime.strftime("%d-%m-%Y_%H-%M-%S")
            model_name = self.config['MODEL_SIZE']
            extension = '.pt'
            save_model_path = os.path.join(self.config['SAVED_MODEL_WEIGHTS_PATH'],model_name,current_datetime,extension)
            self.model.save_weights(save_model_path)
            test_loss, test_accuracy = self.model.evaluate(self.ds_test, steps=self.test_steps)
            print('Test Accuracy: ',test_accuracy)