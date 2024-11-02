import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import argparse
os.environ['HOME'] = os.getcwd()
import numpy as np
import tensorflow as tf
from utils.tools import read_yaml
from utils.trainer import Trainer
from utils.data import handle_input_settings, save_preprocess_input_data_settings, preprocess_data
from sklearn.model_selection import train_test_split
    
os.system('cls' if os.name == 'nt' else 'clear')
#------------PRE-PROCESSING BLOCK----------------------
#------------------------------------------------------
preprocessing_flag = False #Preprocessing Flag - Sequences Formation & Xdata and Ydata Formation 
#Path To Extracted Pose Data
extracted_pose_directory = 'datasets'
#Path To Save Generated Sequential Data
save_sequential_data_path = extracted_pose_directory
if preprocessing_flag: 
    #User-prompt to set preprocessing settings
    input_flag,last_load_file,last_load_file_path,selected_data_folder, dup_n_seq_verbose_selection, fp_seq_threshold, fp_seq, dup_flag = handle_input_settings(extracted_pose_directory)
    
    #Save User specified settings
    save_preprocess_input_data_settings(last_load_file_path,selected_data_folder,dup_n_seq_verbose_selection, fp_seq_threshold, fp_seq, dup_flag)
    Xdata,Ydata = preprocess_data(extracted_pose_directory,selected_data_folder,dup_n_seq_verbose_selection,fp_seq,dup_flag)
    print('Xdata Shape: ',Xdata.shape)
    print('Ydata Shape: ',Ydata.shape)

    #Save Xdata & Ydata
    np.save(save_sequential_data_path+'\Xdata.npy',Xdata)
    np.save(save_sequential_data_path+'\Ydata.npy',Ydata)


#------------RESHAPING & TRAIN-TEST-VALIDATION SETS GENERATION BLOCK----------------------
#-----------------------------------------------------------------------------------------
#Reshaping and Train-Test_Validation Sets Formation
reshaping_and_train_test_validation_sets_formation_flag = False
#SET TRAIN, TEST AND VALIDATION PERCENTAGES
train_percentage = 0.80
testval_percentage = round(1-train_percentage,2)
validation_percentage = 0.5 #Validation Set Out of TestVal Set
#Train Test Validation Sets Path To Save
traintestval_path = extracted_pose_directory 
if preprocessing_flag and reshaping_and_train_test_validation_sets_formation_flag:
    #DATA RESHAPING
    num_seq, num_frames_per_seq,num_kps,num_coords = Xdata.shape
    reshaped_Xdata = Xdata.reshape(num_seq,num_frames_per_seq,num_kps*num_coords)
    Xdata = reshaped_Xdata
    if 0<=validation_percentage<1:
        if validation_percentage == 0:
            test_percentage = round(testval_percentage,2)
        else:
            test_percentage = round(testval_percentage*(1-validation_percentage),2)

    print('Training Percentage: ',train_percentage)
    print('Test Percentage: ',test_percentage)
    print('Validation Percentage: ',round((testval_percentage*validation_percentage),2))
    #Split Xdata and Ydata
    X_train,X_testval,Y_train,Y_testval = train_test_split(Xdata,Ydata,test_size = testval_percentage,stratify=Ydata)
    X_test,X_val,Y_test,Y_val = train_test_split(X_testval,Y_testval,test_size=validation_percentage,stratify=Y_testval)
    print('X_train Shape: ',X_train.shape)
    print('Y_train Shape: ',Y_train.shape)
    print('X_test Shape: ',X_test.shape)
    print('Y_test Shape: ',Y_test.shape)
    print('X_val Shape: ',X_val.shape)
    print('Y_val Shape: ',Y_val.shape)

    #Save Train Test Validation Sets
    X_train_path = traintestval_path + '\X_train.npy'
    Y_train_path = traintestval_path + '\Y_train.npy'
    X_test_path = traintestval_path + '\X_test.npy'
    Y_test_path = traintestval_path + '\Y_test.npy'
    X_val_path = traintestval_path + '\X_val.npy'
    Y_val_path = traintestval_path + '\Y_val.npy'
    
    np.save(X_train_path,X_train)
    np.save(Y_train_path,Y_train)
    np.save(X_test_path,X_test)
    np.save(Y_test_path,Y_test)
    np.save(X_val_path,X_val)
    np.save(Y_val_path,Y_val)
    print('Train Test Validation Data Generated Successfully!')
    

elif not preprocessing_flag and reshaping_and_train_test_validation_sets_formation_flag:
    Xdata_path = save_sequential_data_path+'\Xdata.npy'
    Ydata_path = save_sequential_data_path+'\Ydata.npy'
    if os.path.exists(Xdata_path) and os.path.exists(Ydata_path):
        Xdata = np.load(Xdata_path)
        Ydata = np.load(Ydata_path)
        #DATA RESHAPING
        num_seq, num_frames_per_seq,num_kps,num_coords = Xdata.shape
        reshaped_Xdata = Xdata.reshape(num_seq,num_frames_per_seq,num_kps*num_coords)
        Xdata = reshaped_Xdata    
        if 0<=validation_percentage<1:
            if validation_percentage == 0:
                test_percentage = round(testval_percentage,2)
            else:
                test_percentage = round(testval_percentage*(1-validation_percentage),2)

        print('Training Percentage: ',train_percentage)
        print('Test Percentage: ',test_percentage)
        print('Validation Percentage: ',round((testval_percentage*validation_percentage),2))
        #Split Xdata and Ydata
        X_train,X_testval,Y_train,Y_testval = train_test_split(Xdata,Ydata,test_size = testval_percentage,stratify=Ydata)
        X_test,X_val,Y_test,Y_val = train_test_split(X_testval,Y_testval,test_size=validation_percentage,stratify=Y_testval)
        print('X_train Shape: ',X_train.shape)
        print('Y_train Shape: ',Y_train.shape)
        print('X_test Shape: ',X_test.shape)
        print('Y_test Shape: ',Y_test.shape)
        print('X_val Shape: ',X_val.shape)
        print('Y_val Shape: ',Y_val.shape)

        #Save Train Test Validation Sets
        X_train_path = traintestval_path + '\X_train.npy'
        Y_train_path = traintestval_path + '\Y_train.npy'
        X_test_path = traintestval_path + '\X_test.npy'
        Y_test_path = traintestval_path + '\Y_test.npy'
        X_val_path = traintestval_path + '\X_val.npy'
        Y_val_path = traintestval_path + '\Y_val.npy'
        
        np.save(X_train_path,X_train)
        np.save(Y_train_path,Y_train)
        np.save(X_test_path,X_test)
        np.save(Y_test_path,Y_test)
        np.save(X_val_path,X_val)
        np.save(Y_val_path,Y_val)
        print('Train Test Validation Data Generated Successfully!')
        
    else:
        print('Files Doesnot Exists. Please Generate Data OR Specify Data Path!')

else:
    X_train_path = traintestval_path + '\X_train.npy'
    Y_train_path = traintestval_path + '\Y_train.npy'
    X_test_path = traintestval_path + '\X_test.npy'
    Y_test_path = traintestval_path + '\Y_test.npy'
    X_val_path = traintestval_path + '\X_val.npy'
    Y_val_path = traintestval_path + '\Y_val.npy'

    if os.path.exists(X_train_path) and os.path.exists(Y_train_path)  and os.path.exists(X_test_path) and os.path.exists(Y_test_path)  and os.path.exists(X_val_path) and os.path.exists(Y_val_path):
        X_train = np.load(X_train_path)
        Y_train = np.load(Y_train_path)
        X_test = np.load(X_test_path)
        Y_test = np.load(Y_test_path)
        X_val = np.load(X_val_path)
        Y_val = np.load(Y_val_path)
        print('X_train Shape: ',X_train.shape)
        print('Y_train Shape: ',Y_train.shape)
        print('X_test Shape: ',X_test.shape)
        print('Y_test Shape: ',Y_test.shape)
        print('X_val Shape: ',X_val.shape)
        print('Y_val Shape: ',Y_val.shape)
        print('Train Test Validation Data Loaded Successfully!')
    else:
        print('X Y - Train Test Validation Files Doesnot Exits. Please Generate or Specify Path!')


#LOAD MODEL CONFIGURATION FILE
parser = argparse.ArgumentParser(description='Process some input')
parser.add_argument('--config', default='utils/config.yaml', type=str, help='Config path', required=False)
args = parser.parse_args()
config = read_yaml(args.config)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if not os.path.exists(config['SAVED_MODEL_WEIGHTS_PATH']):
    os.makedirs(config['SAVED_MODEL_WEIGHTS_PATH'])


#------------TRAINING MODEL BLOCK----------------------
#------------------------------------------------------
#TRAINING THE MODEL - 1 (THIS TRAINING IS MEANT TO TRAIN THE MODELS AS MENTIONED IN CONFIG.YAML FILE)
#DEFAULT MODEL IS PROPOSED 'NANO'
training_fixed_model_flag = False
if training_fixed_model_flag:
    #TO CHANGE MODEL, TRAINING AND EVALUATION SETTINGS, MAKE CHANGES IN CONFIG.YAML FILE
    #best_activation = get_activation_function(config['ACTIVATION_FUNCTION'])
    trainer = Trainer(config,X_train,Y_train,X_test,Y_test,X_val,Y_val)
    trainer.do_training()

#TRAINING THE MODEL - 2 (THIS TRAINING IS MEANT TO TRAIN THE ALTERED MODEL) - RANDOM CHOICES OF ARCHITECTURAL & TRAINING SETTINGS
training_random_model_flag = False
if training_random_model_flag:
    #SET MODEL WHOSE ARCHITECTURAL SETTINGS ARE TO CHANGE
    model = 'micro'
    config['MODEL_SIZE'] = model 

    #SET RANDOM CHOICE OF MLP HEADS
    config[config['MODEL_SIZE']]['N_HEADS'] = 1 

    #SET RANDOM CHOICE OF ENCODER LAYERS
    config[config['MODEL_SIZE']]['N_LAYERS'] = 4

    #SET RANDOM CHOICE OF EMBEDDED DIMENSIONS
    config[config['MODEL_SIZE']]['EMBED_DIM'] = 64 

    #SET RANDOM CHOICE OF DROPOUT PERCENTAGE
    config[config['MODEL_SIZE']]['DROPOUT'] = 0.30 

    #SET RANDOM CHOICE OF MLP DIMENSIONS
    config[config['MODEL_SIZE']]['MLP'] = 256 
    
    #SET RANDOM TRAINING SETTINGS
    #SET RANDOM CHOICE OF ACTIVATION FUNCTION
    config['ACTIVATION_FUNCTION'] = 'gelu'
    #Set Batch Size
    config['BATCHSIZE'] = 32 

    #Set Optimizer
    config['OPTIMIZER'] = 'SGDW'
    #Set Learning Rate
    config['LR_RATE'] = 0.0001 

    #Set Weight Decay (Its affect ony when Weighted Optimizer is selected)  
    config['WD'] = 0.000001 

    #Set Number of Training Epochs
    config['EPOCHS'] = 5
    trainer = Trainer(config,X_train,Y_train,X_test,Y_test,X_val,Y_val)
    trainer.do_training()


#------------EVALUATION OF SAVED MODEL ON TEST SET BLOCK----------------------
#-----------------------------------------------------------------------------
#EVALUATE SAVED MODEL ON TEST SET (TEST ACCURACY, CONFUSION MATRIX, CLASSWISE ACCURACIES, CLASSIFICATION REPORT)
evaluation_flag = True
if evaluation_flag:
    #Specify Saved Model Weights Path
    saved_model_weights_path = r'saved_model_weights\proposed_model_weights.h5'
    #PROPOSED MODEL
    #Set it to the model and data which was used for training, evaluation and was saved. (Specify all architectural and training settings if changed)
    config['MODEL_SIZE'] = 'nano' 
    trainer = Trainer(config,X_train,Y_train,X_test,Y_test,X_val,Y_val)
    trainer.evaluate_saved_model(saved_model_weights_path)

#------------HYPERPARAMETERS TUNING BLOCK----------------------
#------------------------------------------------------
#HYPERPARAMETERS TUNING
hyperparameters_tuning_flag = False
if hyperparameters_tuning_flag:
    #Set Validation Flag True if False
    config['VAL_FLAG'] = True

    #ARCHITECTURAL PARAMETERS TUNING (SEQUENTIAL TUNING)
    #CONSTANT VARS FOR ARCHITECTURAL TUNING
    config['EPOCHS'] = 1
    config['BATCHSIZE'] = 32
    config['LR_RATE'] = 0.0001
    config['OPTIMIZER'] = 'SGDW'
    config['WD'] = 0.00001
    config['ACTIVATION_FUNCTION'] = 'gelu'
    
    #SET MODEL WHOSE ARCHITECTURAL PARAMETERS ARE TO BE TUNED
    config['MODEL_SIZE'] = 'micro' #Default
    #Adjust List Values as Required (We took sets of values less than default values of model params)
    mlp_heads_list = [1] 
    encoder_layers_list = [1,2,3,4]
    embed_dims_list = [1,2,4,8,16,32,64]
    dropout_percent_list = [0.05,0.1,0.15,0.20,0.25,0.30]
    mlp_dims_list = [1,2,4,8,16,32,64,128,256]

    #Add for loop for mlp_heads_list (As 'micro' has mlp_head = 1 so we are not tuning it as it is already tuned)
    #MLP HEAD TUNING GOES HERE (MODIFY THE BELOW CODE & self.hyperparameters_tuning_plots() function ACCORDINGLY )

    #ENCODER LAYER TUNING
    encoder_layer_val_acc_list = []
    for encoder_layer in encoder_layers_list:
        print('IN PROGRESS ENCODER LAYER: ',encoder_layer)
        val_acc = [] #Variable to Store Current Validation Accuracy 
        config[config['MODEL_SIZE']]['N_LAYERS'] = encoder_layer
        trainer = Trainer(config,X_train,Y_train,X_test,Y_test,X_val,Y_val)
        val_acc = trainer.hyperparameters_tuning()
        encoder_layer_val_acc_list.append(val_acc)
    encoder_layer_max_val_acc_index = encoder_layer_val_acc_list.index(max(encoder_layer_val_acc_list))
    best_encoder_layer = encoder_layers_list[encoder_layer_max_val_acc_index]
    
    #EMBEDDED DIMENSIONS TUNING
    config[config['MODEL_SIZE']]['N_LAYERS'] = best_encoder_layer
    embed_dims_val_acc_list = []
    for embed_dim in embed_dims_list:
        print('IN PROGRESS EMBEDDED DIMENSIONS: ',embed_dim)
        val_acc = [] #Variable to Store Current Validation Accuracy 
        config[config['MODEL_SIZE']]['EMBED_DIM'] = embed_dim
        trainer = Trainer(config,X_train,Y_train,X_test,Y_test,X_val,Y_val)
        val_acc = trainer.hyperparameters_tuning()
        embed_dims_val_acc_list.append(val_acc)
    embed_dim_max_val_acc_index = embed_dims_val_acc_list.index(max(embed_dims_val_acc_list))
    best_embed_dim = embed_dims_list[embed_dim_max_val_acc_index]
    
    #DROPOUT PERCENTAGE TUNING
    config[config['MODEL_SIZE']]['EMBED_DIM'] = best_embed_dim
    dropout_val_acc_list = []
    for dropout in dropout_percent_list:
        print('IN PROGRESS DROPOUT PERCENTAGE: ',dropout)
        val_acc = [] #Variable to Store Current Validation Accuracy 
        config[config['MODEL_SIZE']]['DROPOUT'] = dropout
        trainer = Trainer(config,X_train,Y_train,X_test,Y_test,X_val,Y_val)
        val_acc = trainer.hyperparameters_tuning()
        dropout_val_acc_list.append(val_acc)
    dropout_max_val_acc_index = dropout_val_acc_list.index(max(dropout_val_acc_list))
    best_dropout = dropout_percent_list[dropout_max_val_acc_index]
    
    #MLP DIMENSIONS TUNING
    config[config['MODEL_SIZE']]['DROPOUT'] = best_dropout
    mlp_dim_val_acc_list = []
    for mlp_dim in mlp_dims_list:
        print('IN PROGRESS MLP DIMENSIONS: ',mlp_dim)
        val_acc = [] #Variable to Store Current Validation Accuracy 
        config[config['MODEL_SIZE']]['MLP'] = mlp_dim
        trainer = Trainer(config,X_train,Y_train,X_test,Y_test,X_val,Y_val)
        val_acc = trainer.hyperparameters_tuning()
        mlp_dim_val_acc_list.append(val_acc)
    mlp_dim_max_val_acc_index = mlp_dim_val_acc_list.index(max(mlp_dim_val_acc_list))
    best_mlp_dim = mlp_dims_list[mlp_dim_max_val_acc_index]

    #TRAINING PARAMETERS TUNING (SEQUENTIAL TUNING)
    #SELECT THE BEST ARCHITECTURAL PARAMETERS
    config[config['MODEL_SIZE']]['MLP'] = best_mlp_dim

    #OPTIMIZER & LR_RATE & WEIGHT DECAY(INCASE OF WEIGHTED OPTIMIZER) TUNING
    #ADJUST EACH TRAINING PARAMETERS LISTS
    optimizers_list = ['AdamW','LAMB','LazyAdam','RAdam','SGDW']
    
    adamw_val_acc_list = []
    lamb_val_acc_list = []
    lazyadam_val_acc_list = []
    radam_val_acc_list = []
    sgdw_val_acc_list = []
    lr_list = [0.1,0.01,0.001,0.0001,0.00001,0.000001]
    wd_list = [0.1,0.01,0.001,0.0001,0.00001,0.000001]
    #EXTENDED LR & WD LIST FOR WEIGHTED OPTIMIZERS
    lr_extended_list = []
    wd_extended_list = []
    for lr_v in lr_list:
        for wd_v in wd_list:
            lr_extended_list.append(lr_v)
            wd_extended_list.append(wd_v)
    
    for optimizer in optimizers_list:
        if optimizer == 'AdamW':
            config['OPTIMIZER'] = optimizer
            for lr in lr_list:
                for wd in wd_list:
                    print('IN PROGRESS OPTIMIZER: ',optimizer)
                    print('IN PROGRESS LR_RATE: ',lr)
                    print('IN PROGRESS WEIGHT DECAY: ',wd)
                    val_acc = []
                    config['LR_RATE'] = lr
                    config['WD'] = wd
                    trainer = Trainer(config,X_train,Y_train,X_test,Y_test,X_val,Y_val)
                    val_acc = trainer.hyperparameters_tuning()
                    adamw_val_acc_list.append(val_acc)
            adamw_max_val_acc_index = adamw_val_acc_list.index(max(adamw_val_acc_list))
            adamw_max_acc = adamw_val_acc_list[adamw_max_val_acc_index]
            adamw_best_lr = lr_extended_list[adamw_max_val_acc_index]
            adamw_best_wd = wd_extended_list[adamw_max_val_acc_index]

        elif optimizer == 'SGDW':
            config['OPTIMIZER'] = optimizer
            for lr in lr_list:
                for wd in wd_list:
                    print('IN PROGRESS OPTIMIZER: ',optimizer)
                    print('IN PROGRESS LR_RATE: ',lr)
                    print('IN PROGRESS WEIGHT DECAY: ',wd)
                    val_acc = []
                    config['LR_RATE'] = lr
                    config['WD'] = wd
                    trainer = Trainer(config,X_train,Y_train,X_test,Y_test,X_val,Y_val)
                    val_acc = trainer.hyperparameters_tuning()
                    sgdw_val_acc_list.append(val_acc)
            sgdw_max_val_acc_index = sgdw_val_acc_list.index(max(sgdw_val_acc_list))
            sgdw_max_acc = sgdw_val_acc_list[sgdw_max_val_acc_index]
            sgdw_best_lr = lr_extended_list[sgdw_max_val_acc_index]
            sgdw_best_wd = wd_extended_list[sgdw_max_val_acc_index]
        
        elif optimizer == 'LAMB':
            config['OPTIMIZER'] = optimizer
            for lr in lr_list:
                print('IN PROGRESS OPTIMIZER: ',optimizer)
                print('IN PROGRESS LR_RATE: ',lr)
                val_acc = []
                config['LR_RATE'] = lr
                trainer = Trainer(config,X_train,Y_train,X_test,Y_test,X_val,Y_val)
                val_acc = trainer.hyperparameters_tuning()
                lamb_val_acc_list.append(val_acc)
            lamb_max_val_acc_index = lamb_val_acc_list.index(max(lamb_val_acc_list))
            lamb_max_acc = lamb_val_acc_list[lamb_max_val_acc_index]
            lamb_best_lr = lr_list[lamb_max_val_acc_index]
                
        elif optimizer == 'LazyAdam':
            config['OPTIMIZER'] = optimizer
            for lr in lr_list:
                print('IN PROGRESS OPTIMIZER: ',optimizer)
                print('IN PROGRESS LR_RATE: ',lr)
                val_acc = []
                config['LR_RATE'] = lr
                trainer = Trainer(config,X_train,Y_train,X_test,Y_test,X_val,Y_val)
                val_acc = trainer.hyperparameters_tuning()
                lazyadam_val_acc_list.append(val_acc)
            lazyadam_max_val_acc_index = lazyadam_val_acc_list.index(max(lazyadam_val_acc_list))
            lazyadam_max_acc = lazyadam_val_acc_list[lazyadam_max_val_acc_index]
            lazyadam_best_lr = lr_list[lazyadam_max_val_acc_index]

        elif optimizer == 'RAdam':
            config['OPTIMIZER'] = optimizer
            for lr in lr_list:
                print('IN PROGRESS OPTIMIZER: ',optimizer)
                print('IN PROGRESS LR_RATE: ',lr)
                val_acc = []
                config['LR_RATE'] = lr
                trainer = Trainer(config,X_train,Y_train,X_test,Y_test,X_val,Y_val)
                val_acc = trainer.hyperparameters_tuning()
                radam_val_acc_list.append(val_acc)
            radam_max_val_acc_index = radam_val_acc_list.index(max(radam_val_acc_list))
            radam_max_acc = radam_val_acc_list[radam_max_val_acc_index]
            radam_best_lr = lr_list[radam_max_val_acc_index]
    
    optimizers_best_val_acc_list = [adamw_max_acc,lamb_max_acc,lazyadam_max_acc,radam_max_acc,sgdw_max_acc]
    optimizers_best_lr_list = [adamw_best_lr,lamb_best_lr,lazyadam_best_lr,radam_best_lr,sgdw_best_lr]
    optimizers_best_wd_list = [adamw_best_wd,'None','None','None',sgdw_best_wd]
    optimizer_max_val_acc_index = optimizers_best_val_acc_list.index(max(optimizers_best_val_acc_list))
    best_optimizer = optimizers_list[optimizer_max_val_acc_index]
    best_lr = optimizers_best_lr_list[optimizer_max_val_acc_index]
    if best_optimizer == 'AdamW' or best_optimizer == 'SGDW':
        best_wd = optimizers_best_wd_list[optimizer_max_val_acc_index]
    else:
        best_wd = 0

    #ACTIVATION FUNCTION TUNING
    config['LR_RATE'] = best_lr
    config['WD'] = best_wd
    config['OPTIMIZER'] = best_optimizer    
    activation_functions_list = ['gelu','swish','mish']
    activation_function_val_acc_list = []
    for activation_function_name in activation_functions_list:
        config['ACTIVATION_FUNCTION'] = activation_function_name
        print('IN PROGRESS ACTIVATION FUNCTION: ',activation_function_name)
        val_acc = []
        trainer = Trainer(config,X_train,Y_train,X_test,Y_test,X_val,Y_val)
        val_acc = trainer.hyperparameters_tuning()
        activation_function_val_acc_list.append(val_acc)
    activation_function_max_val_acc_index = activation_function_val_acc_list.index(max(activation_function_val_acc_list))
    best_activation_function = activation_functions_list[activation_function_max_val_acc_index]
    
    #BATCH SIZE TUNING
    config['ACTIVATION_FUNCTION'] = best_activation_function
    batch_size_list = [1,2,4,8,16,32,64,128,256]
    batch_size_val_acc_list = []
    for batch_size in batch_size_list:
        print('IN PROGRESS BATCHSIZE: ',batch_size)
        val_acc = []
        trainer = Trainer(config,X_train,Y_train,X_test,Y_test,X_val,Y_val)
        val_acc = trainer.hyperparameters_tuning()
        batch_size_val_acc_list.append(val_acc)
    batch_size_max_val_acc_index = batch_size_val_acc_list.index(max(batch_size_val_acc_list))
    best_batchsize = activation_functions_list[batch_size_max_val_acc_index]

    print('BEST NUMBER OF ENCODER LAYERS: ',best_encoder_layer)
    print('BEST EMBEDDED DIMENSIONS: ',best_embed_dim)
    print('BEST DROPOUT PERCENTAGE: ',best_dropout)
    print('BEST MLP DIMENSIONS: ',best_mlp_dim)
    print('BEST OPTIMIZER: ',best_optimizer)
    print('BEST WEIGHT DECAY: ',best_wd)
    print('BEST LR_RATE: ',best_lr)
    print('BEST ACTIVATION FUNCTION: ',best_activation_function)
    print('BEST BATCH SIZE: ',best_batchsize)
    
    #PLOT HYPERPARAMETERS TUNING PLOTS
    trainer.hyperparameters_tuning_plots(encoder_layers_list,encoder_layer_val_acc_list,embed_dims_list,embed_dims_val_acc_list,dropout_percent_list,dropout_val_acc_list,mlp_dims_list,mlp_dim_val_acc_list,
    optimizers_list,lr_list,wd_list,adamw_val_acc_list,lamb_val_acc_list,lazyadam_val_acc_list,radam_val_acc_list,sgdw_val_acc_list,optimizers_best_val_acc_list,optimizers_best_lr_list,optimizers_best_wd_list,
        activation_functions_list,activation_function_val_acc_list,batch_size_list,batch_size_val_acc_list)