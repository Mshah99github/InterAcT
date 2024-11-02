import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
import time
import datetime
from ultralytics import YOLO
import numpy as np
from sklearn.model_selection import train_test_split
import absl.logging
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from scipy import stats
import sklearn

# CUSTOM LIBRARIES
from utils.tools import read_yaml
from utils.trainer import Trainer
from utils.transformer import TransformerEncoder, PatchClassEmbedding
from utils.data import one_hot

#HIDE WARNINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['HOME'] = os.getcwd()

class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, progress_label, canvas, ax, epochs,stop_event, quit_event):
        super().__init__()
        self.progress_bar = progress_bar
        self.progress_label = progress_label
        self.canvas = canvas
        self.ax = ax
        self.epochs = epochs
        self.epoch = 0
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_losses = []
        self.val_losses = []
        self.stop_event = stop_event
        self.quit_event = quit_event

    def on_epoch_end(self, epoch, logs=None):
        # Ensure the logs are not None
        if logs is not None:
            self.train_accuracies.append(logs.get('accuracy', 0))
            self.val_accuracies.append(logs.get('val_accuracy', 0))
            self.train_losses.append(logs.get('loss', 0))
            self.val_losses.append(logs.get('val_loss', 0))
        self.progress_bar['value'] = (epoch + 1)/self.epochs * 100
        self.progress_label.config(text=f"Epoch {epoch + 1}/{self.epochs}")

        self.ax.clear()

        # Plot accuracy
        self.ax.plot(range(1, len(self.train_accuracies) + 1), self.train_accuracies, label='Training Accuracy', color='blue')
        self.ax.plot(range(1, len(self.val_accuracies) + 1), self.val_accuracies, label='Validation Accuracy', color='green')

        # Plot loss
        self.ax.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss', color='red')
        self.ax.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Loss', color='orange')

        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Value')
        self.ax.set_xlim(0, self.epochs)
        self.ax.set_ylim(0, 2.75)
        self.canvas.draw()
        self.canvas.flush_events()

        # Check if the stop event has been set
        if self.stop_event.is_set():
            self.model.stop_training = True
            self.stop_event.clear()  # Reset the event for future use

        # Check if the quit event has been set
        if self.quit_event.is_set():
            self.model.stop_training = True
            self.quit_event.clear()  # Reset the event for future use

class VideoPlayer:
    def __init__(self, master):
        self.master = master
        self.master.title("InterAcT")
        self.master.geometry("800x600")  # Set window size
        
        #MODEL ARCHITECTURE
        self.n_heads = 0
        self.n_layers = 0
        self.embed_dim = 0
        self.dropout = 0
        self.mlp_head_size = 0
        self.d_model = (self.embed_dim * self.n_heads)
        self.d_ff = self.d_model
        self.pos_emb = None
        self.batch_size = 0
        self.num_classes = 18
        self.activation_function = ''
        self.Xdata = []
    

        #PRE-PROCESSING TAB VARS
        self.pose_model_path = ""
        self.pose_model = None
        self.dataset_dir = ""
        self.output_preprocess_dir = ""
        self.in_progress_num = 0
        self.total_videos = 0
        self.video_capture = None
        self.confidence_score = []
        self.frame_size = 320
        self.thread = None
        self.pose_extraction_complete_flag = False
        self.action_type = []
        self.normalized_keypoints_check = tk.BooleanVar(value=True)  # Set default value to True
        self.global_keypoints_check = tk.BooleanVar(value=False)
        self.overwrite_keypoints_check = tk.BooleanVar(value=False)
        self.current_tab = "Inference"#"Preprocessing - Pose Extraction"  # Default tab
        self.event = threading.Event()
        self.stop_event = threading.Event()
        
        #TRANSFORMATION TAB VARS
        self.npy_file_data = []
        self.video_count = 0
        self.sequences = []
        #NORMALIZED
        self.handshaking_sequences_n = []
        self.handshaking_labels_n = []
        self.hugging_sequences_n = []
        self.hugging_labels_n = []
        self.kicking_sequences_n = []
        self.kicking_labels_n = []
        self.punching_sequences_n = []
        self.punching_labels_n = []
        self.pushing_sequences_n = []
        self.pushing_labels_n = []
        self.clapping_solo_sequences_n = []
        self.clapping_solo_labels_n = []
        self.hitting_bottle_solo_sequences_n = []
        self.hitting_bottle_solo_labels_n = []
        self.hitting_stick_solo_sequences_n = []
        self.hitting_stick_solo_labels_n = []
        self.jogging_f_b_solo_sequences_n = []
        self.jogging_f_b_solo_labels_n = []
        self.jogging_side_solo_sequences_n = []
        self.jogging_side_solo_labels_n = []
        self.kicking_solo_sequences_n = []
        self.kicking_solo_labels_n = []
        self.punching_solo_sequences_n = []
        self.punching_solo_labels_n = []
        self.running_f_b_solo_sequences_n = []
        self.running_f_b_solo_labels_n = []
        self.running_side_solo_sequences_n = []
        self.running_side_solo_labels_n = []
        self.stabbing_solo_sequences_n = []
        self.stabbing_solo_labels_n = []
        self.walking_f_b_solo_sequences_n = []
        self.walking_f_b_solo_labels_n = []
        self.walking_side_solo_sequences_n = []
        self.walking_side_solo_labels_n = []
        self.waving_hands_solo_sequences_n = []
        self.waving_hands_solo_labels_n = []

        #GLOBAL
        self.handshaking_sequences_g = []
        self.handshaking_labels_g = []
        self.hugging_sequences_g = []
        self.hugging_labels_g = []
        self.kicking_sequences_g = []
        self.kicking_labels_g = []
        self.punching_sequences_g = []
        self.punching_labels_g = []
        self.pushing_sequences_g = []
        self.pushing_labels_g = []
        self.clapping_solo_sequences_g = []
        self.clapping_solo_labels_g = []
        self.hitting_bottle_solo_sequences_g = []
        self.hitting_bottle_solo_labels_g = []
        self.hitting_stick_solo_sequences_g = []
        self.hitting_stick_solo_labels_g = []
        self.jogging_f_b_solo_sequences_g = []
        self.jogging_f_b_solo_labels_g = []
        self.jogging_side_solo_sequences_g = []
        self.jogging_side_solo_labels_g = []
        self.kicking_solo_sequences_g = []
        self.kicking_solo_labels_g = []
        self.punching_solo_sequences_g = []
        self.punching_solo_labels_g = []
        self.running_f_b_solo_sequences_g = []
        self.running_f_b_solo_labels_g = []
        self.running_side_solo_sequences_g = []
        self.running_side_solo_labels_g = []
        self.stabbing_solo_sequences_g = []
        self.stabbing_solo_labels_g = []
        self.walking_f_b_solo_sequences_g = []
        self.walking_f_b_solo_labels_g = []
        self.walking_side_solo_sequences_g = []
        self.walking_side_solo_labels_g = []
        self.waving_hands_solo_sequences_g = []
        self.waving_hands_solo_labels_g = []

        self.fixed_seq_check = tk.BooleanVar(value=False)
        self.window_seq_check = tk.BooleanVar(value=False)
        
        self.testset_percent = []
        self.trainset_percent = []
        self.validationset_percent = []
        self.duplicate_frames_threshold = []
        self.sequence_length_value = []
        self.total_npy_files = 0
        self.in_progress_npy_num = 0
        self.transformation_complete_flag = False
        self.pose_dir = ""
        self.output_transformation_dir = ""
        self.kps_checkbox_1_var = tk.BooleanVar(value=True)
        self.kps_checkbox_2_var = tk.BooleanVar(value=True)
        self.kps_checkbox_3_var = tk.BooleanVar(value=True)
        self.kps_checkbox_4_var = tk.BooleanVar(value=True)
        self.kps_checkbox_5_var = tk.BooleanVar(value=True)
        self.kps_checkbox_6_var = tk.BooleanVar(value=True)
        self.kps_checkbox_7_var = tk.BooleanVar(value=True)
        self.kps_checkbox_8_var = tk.BooleanVar(value=True)
        self.kps_checkbox_9_var = tk.BooleanVar(value=True)
        self.kps_checkbox_10_var = tk.BooleanVar(value=True)
        self.kps_checkbox_11_var = tk.BooleanVar(value=True)
        self.kps_checkbox_12_var = tk.BooleanVar(value=True)
        self.kps_checkbox_13_var = tk.BooleanVar(value=True)
        self.kps_checkbox_14_var = tk.BooleanVar(value=True)
        self.kps_checkbox_15_var = tk.BooleanVar(value=True)
        self.kps_checkbox_16_var = tk.BooleanVar(value=True)
        self.kps_checkbox_17_var = tk.BooleanVar(value=True)
        self.duplicate_frames_check = tk.BooleanVar(value=False)
        self.validation_set_flag = tk.BooleanVar(value=True)

        self.kps_indices_to_drop = []
        self.kps_type = []

        #TRAINING BLOCK VARS
        self.in_progress_epoch = 0
        self.epochs = 0
        self.optimizer_val = ''
        self.lr = 0
        self.wd = 0
        self.training_browse_transformed_output_directory_path = ''
        self.training_browse_model_output_directory_path = ''
        self.training_frame_size_height = 320
        self.training_frame_size_width = 450
        self.training_normalized_keypoints_check = tk.BooleanVar(value=False)
        self.training_global_keypoints_check = tk.BooleanVar(value=False)
        self.training_fixed_seq_check = tk.BooleanVar(value=False)
        self.training_window_seq_check = tk.BooleanVar(value=False)
        self.training_bal_seq_check = tk.BooleanVar(value=False)
        self.training_imbal_seq_check = tk.BooleanVar(value=False)
        self.training_validation_set_flag = tk.BooleanVar(value=False)
        self.training_complete_flag = False
        #INFERENCE BLOCK VARS
        self.config = []
        self.inference_complete_flag = False
        self.video_path = ''
        self.count = 0
        self.video_keypoints = []
        self.normalized_keypoints_check_inf = tk.BooleanVar(value=False)  # Set default value to True
        self.global_keypoints_check_inf = tk.BooleanVar(value=False)
        self.model_weights_path = ''
        self.predicted_label = None
        self.predicted_label_txt = 'None'


        # Create notebook (tab manager)
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.tabs = {
            "Preprocessing - Pose Extraction": ttk.Frame(self.notebook),
            "Data Transformation": ttk.Frame(self.notebook),
            "Model Training": ttk.Frame(self.notebook),
            "Inference": ttk.Frame(self.notebook)
        }

        # Add tabs to notebook
        for tab_name, tab_frame in self.tabs.items():
            self.notebook.add(tab_frame, text=tab_name)

        # Show default tab
        self.show_tab(self.current_tab)
        # Bind the tab change event to a function
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)
    
    def on_tab_change(self,event):
        # Get the current selected tab
        tab_index = self.notebook.index(self.notebook.select())
        tab_name = self.notebook.tab(tab_index, "text")
        # Show the selected tab and its widgets
        self.show_tab(tab_name)
        
    def show_tab(self, tab_name):
        self.current_tab = tab_name
        self.notebook.select(self.tabs[tab_name])
        
        if tab_name == "Inference":
            self.setup_inference_tab()
        elif tab_name == "Preprocessing - Pose Extraction":
            self.setup_preprocessing_tab()
        elif tab_name == "Data Transformation":
            self.setup_data_transformation_tab()
        elif tab_name == "Model Training":
            self.setup_model_training_tab()


    #--------------------------MODEL TRAINING BLOCK-------------------------------------------
    #----------------------------------------------------------------------------------
    def setup_model_training_tab(self):
        model_training_tab = self.tabs["Model Training"]
            # Style for buttons
        style = ttk.Style()
        style.configure("Red.TButton", foreground="black", background="red")
        style.configure("Blue.TButton", foreground="black", background="blue")
        style.configure("Green.TButton", foreground="black", background="green")
        style.configure("Orange.TButton", foreground="black", background="orange")
        style.configure("Yellow.TButton", foreground="black", background="yellow")

        # Frames for ACCURACY LOSS CURVE
        original_frame_label = tk.Label(model_training_tab, text="Accuracy Loss Curve", font=("Arial", 10, "bold"))
        original_frame_label.place(x=340, y=50)

        self.training_original_frame_container = tk.Frame(model_training_tab, bd=2, relief=tk.GROOVE)
        self.training_original_frame_container.place(x=175, y=80)
        self.training_original_frame_canvas = tk.Canvas(self.training_original_frame_container, width=self.training_frame_size_width, height=self.training_frame_size_height)
        self.training_original_frame_canvas.pack(fill=tk.BOTH, expand=True)

        ############
        # Setup Matplotlib Figure and Axis
        fig, self.ax = plt.subplots(figsize=(4, 3))
        self.canvas = FigureCanvasTkAgg(fig, master=self.training_original_frame_canvas)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)        
        ##############

        ngo_checkboxes_y_offset = 10
        ys_off = 5
        xoff = 55
        yoffs = 20
        yoff = 20
        y_offset_bottom_widgets = 30
        yy = 30
        self.attention_layers_label = ttk.Label(model_training_tab, text="N-SAL: ")
        self.attention_layers_label.place(x=30,y=130+y_offset_bottom_widgets)
        self.attention_layers_entry_frame = ttk.Frame(model_training_tab, borderwidth=2, relief="solid")
        self.attention_layers_entry_frame.place(x=70, y=128+y_offset_bottom_widgets)
        self.attention_layers_entry = ttk.Entry(self.attention_layers_entry_frame, width=5)
        self.attention_layers_entry.pack()        
        self.attention_layers_entry.bind("<KeyRelease>", self.get_attention_layers)

        self.enc_layers_label = ttk.Label(model_training_tab, text="N-ENC: ")
        self.enc_layers_label.place(x=27,y=yy+130+y_offset_bottom_widgets)
        self.enc_layers_entry_frame = ttk.Frame(model_training_tab, borderwidth=2, relief="solid")
        self.enc_layers_entry_frame.place(x=70, y=yy+128+y_offset_bottom_widgets)
        self.enc_layers_entry = ttk.Entry(self.enc_layers_entry_frame, width=5)
        self.enc_layers_entry.pack()        
        self.enc_layers_entry.bind("<KeyRelease>", self.get_enc_layers)

        self.embed_label = ttk.Label(model_training_tab, text="EMB-DIM: ")
        self.embed_label.place(x=12,y=2*yy+130+y_offset_bottom_widgets)
        self.embed_entry_frame = ttk.Frame(model_training_tab, borderwidth=2, relief="solid")
        self.embed_entry_frame.place(x=70, y=2*yy+128+y_offset_bottom_widgets)
        self.embed_entry = ttk.Entry(self.embed_entry_frame, width=5)
        self.embed_entry.pack()        
        self.embed_entry.bind("<KeyRelease>", self.get_emb_dims)

        self.drop_label = ttk.Label(model_training_tab, text="DROPOUT: ")
        self.drop_label.place(x=10,y=3*yy+130+y_offset_bottom_widgets)
        self.drop_entry_frame = ttk.Frame(model_training_tab, borderwidth=2, relief="solid")
        self.drop_entry_frame.place(x=70, y=3*yy+128+y_offset_bottom_widgets)
        self.drop_entry = ttk.Entry(self.drop_entry_frame, width=5)
        self.drop_entry.pack()        
        self.drop_entry.bind("<KeyRelease>", self.get_dropout)

        self.mlp_label = ttk.Label(model_training_tab, text="MLP-DIM: ")
        self.mlp_label.place(x=12,y=4*yy+130+y_offset_bottom_widgets)
        self.mlp_entry_frame = ttk.Frame(model_training_tab, borderwidth=2, relief="solid")
        self.mlp_entry_frame.place(x=70, y=4*yy+128+y_offset_bottom_widgets)
        self.mlp_entry = ttk.Entry(self.mlp_entry_frame, width=5)
        self.mlp_entry.pack()        
        self.mlp_entry.bind("<KeyRelease>", self.get_mlp_dims)

        self.batch_label = ttk.Label(model_training_tab, text="BATCH-S: ")
        self.batch_label.place(x=12,y=5*yy+130+y_offset_bottom_widgets)
        self.batch_entry_frame = ttk.Frame(model_training_tab, borderwidth=2, relief="solid")
        self.batch_entry_frame.place(x=70, y=5*yy+128+y_offset_bottom_widgets)
        self.batch_entry = ttk.Entry(self.batch_entry_frame, width=5)
        self.batch_entry.pack()        
        self.batch_entry.bind("<KeyRelease>", self.get_batchsize)


        self.training_normalized_checkbox = tk.Checkbutton(model_training_tab, text="Normalized Keypoints",  variable=self.training_normalized_keypoints_check, command=self.training_toggle_normalized)
        self.training_normalized_checkbox.place(x=50, y=400+ngo_checkboxes_y_offset)
        self.training_global_checkbox = tk.Checkbutton(model_training_tab, text="Global Keypoints", variable=self.training_global_keypoints_check, command=self.training_toggle_global)
        self.training_global_checkbox.place(x=200, y=400+ngo_checkboxes_y_offset)
        
        self.seq_type_label = ttk.Label(model_training_tab, text="Sequence Type:")
        self.seq_type_label.place(x=50,y=438-ys_off)
        self.training_fixed_seq_checkbox = tk.Checkbutton(model_training_tab, text="Fixed", variable=self.training_fixed_seq_check, command=self.training_toggle_fixed_seq)
        self.training_fixed_seq_checkbox.place(x=135, y=436-ys_off)
        self.training_window_seq_checkbox = tk.Checkbutton(model_training_tab, text="Sliding Window", variable=self.training_window_seq_check, command=self.training_toggle_window_seq)
        self.training_window_seq_checkbox.place(x=200, y=436-ys_off)
        

        self.class_seq_type_label = ttk.Label(model_training_tab, text="Sequences per Class Type:")
        self.class_seq_type_label.place(x=50,y=438-ys_off+yoffs)
        self.training_bal_seq_checkbox = tk.Checkbutton(model_training_tab, text="Balanced", variable=self.training_bal_seq_check, command=self.training_toggle_bal_seq)
        self.training_bal_seq_checkbox.place(x=135+xoff, y=436-ys_off+yoffs)
        self.training_imbal_seq_checkbox = tk.Checkbutton(model_training_tab, text="Imbalanced", variable=self.training_imbal_seq_check, command=self.training_toggle_imbal_seq)
        self.training_imbal_seq_checkbox.place(x=207+xoff, y=436-ys_off+yoffs)

        self.training_validationset_checkbox = tk.Checkbutton(model_training_tab, text="Validation Set",  variable=self.training_validation_set_flag, command=self.training_toggle_validation_set_flag)
        self.training_validationset_checkbox.place(x=50, y=442+ngo_checkboxes_y_offset+yoff)

        self.epoch_label = ttk.Label(model_training_tab, text="Epochs: ")
        self.epoch_label.place(x=160,y=445+ngo_checkboxes_y_offset+yoff)
        self.epoch_entry_frame = ttk.Frame(model_training_tab, borderwidth=2, relief="solid")
        self.epoch_entry_frame.place(x=205, y=442+ngo_checkboxes_y_offset+yoff)
        self.epoch_entry = ttk.Entry(self.epoch_entry_frame, width=5)
        self.epoch_entry.pack()        
        self.epoch_entry.bind("<KeyRelease>", self.get_epoch)

        # Optimizer combobox
        self.optimizer_label = ttk.Label(model_training_tab, text="Optimizer: ")
        self.optimizer_label.place(x=250, y=445 + ngo_checkboxes_y_offset + yoff)

        self.optimizer_combobox = ttk.Combobox(model_training_tab, values=["AdamW", "LAMB", "LazyAdam", "RAdam", "SGDW"],width=8)
        self.optimizer_combobox.place(x=315, y=445 + ngo_checkboxes_y_offset + yoff)
        self.optimizer_combobox.current(0)  # Set default selection
        self.optimizer_combobox.bind("<<ComboboxSelected>>", self.get_optimizer)

        self.lr_label = ttk.Label(model_training_tab, text="LR (0-1): ")
        self.lr_label.place(x=390,y=445+ngo_checkboxes_y_offset+yoff)
        self.lr_entry_frame = ttk.Frame(model_training_tab, borderwidth=2, relief="solid")
        self.lr_entry_frame.place(x=440, y=442+ngo_checkboxes_y_offset+yoff)
        self.lr_entry = ttk.Entry(self.lr_entry_frame, width=5)
        self.lr_entry.pack()        
        self.lr_entry.bind("<KeyRelease>", self.get_learning_rate)
        
        self.wd_label = ttk.Label(model_training_tab, text="WD (0-1): ")
        self.wd_label.place(x=485,y=445+ngo_checkboxes_y_offset+yoff)
        self.wd_entry_frame = ttk.Frame(model_training_tab, borderwidth=2, relief="solid")
        self.wd_entry_frame.place(x=540, y=442+ngo_checkboxes_y_offset+yoff)
        self.wd_entry = ttk.Entry(self.wd_entry_frame, width=5, state='disabled')
        self.wd_entry.pack()        
        self.wd_entry.bind("<KeyRelease>", self.get_weight_decay)
        #Activation ComboBox
        self.activation_label = ttk.Label(model_training_tab, text="Activation Function: ")
        self.activation_label.place(x=585, y=443 + ngo_checkboxes_y_offset + yoff)
        self.activation_combobox = ttk.Combobox(model_training_tab, values=["gelu", "leaky-relu", "selu", "sigmoid", "silu", "swish", "elu", "relu", "mish", "softmax", "softplus"],width=8)
        self.activation_combobox.place(x=700, y=443 + ngo_checkboxes_y_offset + yoff)
        self.activation_combobox.current(0)  # Set default selection
        self.activation_combobox.bind("<<ComboboxSelected>>", self.get_activ_function)


        self.training_browse_transformed_dir_button = ttk.Button(model_training_tab, text="Browse Transformed Data Directory", command=self.browse_transformed_data_directory, style="Red.TButton")
        self.training_browse_transformed_dir_button.place(x=190, y=10)
        
        self.training_browse_model_output_dir_button = ttk.Button(model_training_tab, text="Browse Model Output Directory", command=self.browse_model_output_directory, style="Red.TButton")
        self.training_browse_model_output_dir_button.place(x=410, y=10)
        

        
        progress_text = str(self.in_progress_epoch)+ '/' + str(self.epochs) + ': '
        self.progress_bar_label = tk.Label(model_training_tab, text=progress_text, font=("Arial", 10, "bold"))
        self.progress_bar_label.place(x=40, y=470+y_offset_bottom_widgets+yoff-7)
        
        self.progress_bar = ttk.Progressbar(model_training_tab, length=600, mode="determinate")
        self.progress_bar.place(x=100, y=470+y_offset_bottom_widgets+yoff-8)
        
        self.show_confusion_matrix_button = ttk.Button(model_training_tab, text="Confusion Matrix", command=self.show_confusion_matrix, width=17, style="Red.TButton",state='disabled')
        self.show_confusion_matrix_button.place(x=660, y=180+y_offset_bottom_widgets)
        
        self.show_classification_report_button = ttk.Button(model_training_tab, text="Classification Report", command=self.show_classification_report, width=20, style="Red.TButton",state='disabled')
        self.show_classification_report_button.place(x=650, y=220+y_offset_bottom_widgets)
        
        # Buttons for running, stopping, and quitting
        self.run_training_button = ttk.Button(model_training_tab, text="Run", command=self.run_training, width=10, style="Red.TButton")
        self.run_training_button.place(x=100, y=510+y_offset_bottom_widgets)
        
        self.stop_button = ttk.Button(model_training_tab, text="Stop", command=self.stop_preprocessing, width=10, style="Red.TButton")
        self.stop_button.place(x=200, y=510+y_offset_bottom_widgets)
        self.stop_button.config(state="disabled")

        self.quit_button = ttk.Button(model_training_tab, text="Quit", command=self.quit_gui, width=10, style="Red.TButton")
        self.quit_button.place(x=680, y=510+y_offset_bottom_widgets)

        self.get_optimizer()
        self.get_activ_function()
        # Check optimizer state initially
        self.update_wd_entry_state()

    def get_attention_layers(self,event):
        self.n_heads = int(self.attention_layers_entry.get())
        self.d_model = (self.embed_dim * self.n_heads)
        self.d_ff = self.d_model
        self.update_button_styles_training_tab()

    def get_enc_layers(self,event):
        self.n_layers = int(self.enc_layers_entry.get())
        self.update_button_styles_training_tab()
        
    def get_emb_dims(self,event):
        self.embed_dim = int(self.embed_entry.get())
        self.d_model = (self.embed_dim * self.n_heads)
        self.d_ff = self.d_model
        self.update_button_styles_training_tab()

    def get_dropout(self,event):
        self.dropout = float(self.drop_entry.get())
        self.update_button_styles_training_tab()

    def get_mlp_dims(self,event):
        self.mlp_head_size = int(self.mlp_entry.get())
        self.update_button_styles_training_tab()

    def get_batchsize(self,event):
        self.batch_size = int(self.batch_entry.get())
        self.update_button_styles_training_tab()

    def get_activ_function(self, event=None):
        self.activation_function = self.activation_combobox.get()
        self.update_button_styles_training_tab()

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

        train_steps = np.ceil(float(train_len)/self.batch_size)
        test_steps = np.ceil(float(test_len)/self.batch_size)
        val_steps = np.ceil(float(val_len)/self.batch_size)
        
        print("Value of Train_steps: ",train_steps)
        print("Value of Test_steps: ",test_steps)
        print("Value of Val_steps: ",val_steps)

        ds_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        ds_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        ds_val = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
                    
        ds_train = ds_train.map(lambda x,y : one_hot(x,y,self.num_classes), 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(X_train.shape[0])
        
        ds_test = ds_test.map(lambda x,y : one_hot(x,y,self.num_classes), 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.cache()

        ds_val = ds_val.map(lambda x,y : one_hot(x,y,self.num_classes), 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.cache()

        ds_train = ds_train.batch(self.batch_size)
        ds_test = ds_test.batch(self.batch_size)
        ds_val = ds_val.batch(self.batch_size)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE) 
        return ds_train,ds_test,ds_val,train_steps,test_steps,val_steps   

    def build_act(self, transformer):
        inputs = tf.keras.layers.Input(shape=(self.Xdata.shape[1], self.Xdata.shape[2]))
        x = tf.keras.layers.Dense(self.d_model)(inputs)
        x = PatchClassEmbedding(self.d_model, self.Xdata.shape[1], 
                                pos_emb=None)(x)
        x = transformer(x)
        x = tf.keras.layers.Lambda(lambda x: x[:,0,:])(x)
        x = tf.keras.layers.Dense(self.mlp_head_size)(x)
        outputs = tf.keras.layers.Dense(self.num_classes)(x)
        return tf.keras.models.Model(inputs, outputs)

    def get_model(self):
        transformer = TransformerEncoder(self.d_model, self.n_heads, self.d_ff, self.dropout, self.get_activation_function(self.activation_function), self.n_layers)
        self.model = self.build_act(transformer)
        
        if self.optimizer_val == 'AdamW':
            print('Optimizer in Get Model(): ','AdamW')
            optimizer = tfa.optimizers.AdamW(learning_rate=self.lr,weight_decay=self.wd)
        elif self.optimizer_val == 'LAMB':
            print('Optimizer in Get Model(): ','LAMB')
            optimizer = tfa.optimizers.LAMB(learning_rate=self.lr)
        elif self.optimizer_val == 'LazyAdam':
            print('Optimizer in Get Model(): ','LazyAdam')
            optimizer = tfa.optimizers.LazyAdam(learning_rate=self.lr)
        elif self.optimizer_val == 'RAdam':
            print('Optimizer in Get Model(): ','RAdam')
            optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.lr)
        elif self.optimizer_val == 'SGDW':
            print('Optimizer in Get Model(): ','SGDW')
            optimizer = tfa.optimizers.SGDW(learning_rate=self.lr, weight_decay=self.wd, momentum=0.9)
        else:
            print(f'Optimizer {self.optimizer_val} not found!')
            print('Using Default AdamW Optimizer')
            optimizer = tfa.optimizers.AdamW(learning_rate=self.lr,weight_decay=self.wd)

        self.model.compile(optimizer=optimizer,
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
                        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])
        return self.model
        
    def train_proposed_model(self):
        os.system('cls')
        self.in_progress_epoch = 0
        
        if self.training_normalized_keypoints_check.get():
            if self.training_fixed_seq_check.get():
                if self.training_bal_seq_check.get():
                    self.Xdata = np.load(self.training_browse_transformed_output_directory_path + r'\Normalized\Fixed_Window\Balanced_Sequences\X_data.npy')
                    X_train = np.load(self.training_browse_transformed_output_directory_path + r'\Normalized\Fixed_Window\Balanced_Sequences\X_train.npy')
                    Y_train = np.load(self.training_browse_transformed_output_directory_path + r'\Normalized\Fixed_Window\Balanced_Sequences\Y_train.npy')
                    X_test = np.load(self.training_browse_transformed_output_directory_path + r'\Normalized\Fixed_Window\Balanced_Sequences\X_test.npy')
                    Y_test = np.load(self.training_browse_transformed_output_directory_path + r'\Normalized\Fixed_Window\Balanced_Sequences\Y_test.npy')
                    X_val = np.load(self.training_browse_transformed_output_directory_path + r'\Normalized\Fixed_Window\Balanced_Sequences\X_val.npy')
                    Y_val = np.load(self.training_browse_transformed_output_directory_path + r'\Normalized\Fixed_Window\Balanced_Sequences\Y_val.npy')
                    ds_train,ds_test,ds_val,train_steps,test_steps,val_steps = self.convert_data2tf(X_train, Y_train, X_test, Y_test, X_val, Y_val)
                    self.get_model()
                    if self.training_validation_set_flag.get():
                        model_history = self.model.fit(ds_train,
                        epochs=self.epochs, initial_epoch=0, verbose=True,
                        steps_per_epoch= int(train_steps),
                        validation_data =ds_val,
                        validation_steps = val_steps,
                        callbacks=[TrainingCallback(self.progress_bar, self.progress_bar_label, self.canvas, self.ax, self.epochs,self.stop_event,
                    self.event)])
                        
                    else:
                        model_history = self.model.fit(ds_train,
                        epochs=self.epochs, initial_epoch=0, verbose=True,
                        steps_per_epoch= int(train_steps),
                        callbacks=[TrainingCallback(self.progress_bar, self.progress_bar_label, self.canvas, self.ax, self.epochs,self.stop_event,
                    self.event)])
                    
                elif self.training_imbal_seq_check.get():
                    self.Xdata = np.load(self.training_browse_transformed_output_directory_path + r'\Normalized\Fixed_Window\Imbalanced_Sequences\X_data.npy')
                    X_train = np.load(self.training_browse_transformed_output_directory_path + r'\Normalized\Fixed_Window\Imbalanced_Sequences\X_train.npy')
                    Y_train = np.load(self.training_browse_transformed_output_directory_path + r'\Normalized\Fixed_Window\Imbalanced_Sequences\Y_train.npy')
                    X_test = np.load(self.training_browse_transformed_output_directory_path + r'\Normalized\Fixed_Window\Imbalanced_Sequences\X_test.npy')
                    Y_test = np.load(self.training_browse_transformed_output_directory_path + r'\Normalized\Fixed_Window\Imbalanced_Sequences\Y_test.npy')
                    X_val = np.load(self.training_browse_transformed_output_directory_path + r'\Normalized\Fixed_Window\Imbalanced_Sequences\X_val.npy')
                    Y_val = np.load(self.training_browse_transformed_output_directory_path + r'\Normalized\Fixed_Window\Imbalanced_Sequences\Y_val.npy')
                    ds_train,ds_test,ds_val,train_steps,test_steps,val_steps = self.convert_data2tf(X_train, Y_train, X_test, Y_test, X_val, Y_val)
                    self.get_model()
                    if self.training_validation_set_flag.get():
                        model_history = self.model.fit(ds_train,
                        epochs=self.epochs, initial_epoch=0, verbose=True,
                        steps_per_epoch= int(train_steps),
                        validation_data =ds_val,
                        validation_steps = val_steps,
                        callbacks=[TrainingCallback(self.progress_bar, self.progress_bar_label, self.canvas, self.ax, self.epochs,self.stop_event,
                    self.event)])
                    else:
                        model_history = self.model.fit(ds_train,
                        epochs=self.epochs, initial_epoch=0, verbose=True,
                        steps_per_epoch= int(train_steps),
                        callbacks=[TrainingCallback(self.progress_bar, self.progress_bar_label, self.canvas, self.ax, self.epochs,self.stop_event,
                    self.event)])
                    


    def run_training(self):
        self.training_complete_flag = False
        self.stop_event.clear()
        self.training_widgets_state_change()
        self.train_proposed_model()
        self.training_complete_flag = True
        self.training_widgets_state_change()
            
    def training_widgets_state_change(self):
        
        if not self.training_complete_flag:
            total_tabs = self.notebook.index('end')
            for tab_no in range(total_tabs):
                if tab_no!=self.notebook.index(self.notebook.select()):
                    self.notebook.tab(tab_no,state='disabled') 
            self.in_progress_epoch = 0
            self.progress_bar['value'] = self.in_progress_epoch
            self.training_browse_transformed_dir_button.config(state="disabled")
            self.training_browse_model_output_dir_button.config(state="disabled")
            self.show_confusion_matrix_button.config(state="disabled")
            self.show_classification_report_button.config(state="disabled")
            self.training_normalized_checkbox.config(state="disabled")
            self.training_global_checkbox.config(state="disabled")
            self.training_fixed_seq_checkbox.config(state="disabled")
            self.training_window_seq_checkbox.config(state="disabled")
            self.training_bal_seq_checkbox.config(state="disabled")
            self.training_imbal_seq_checkbox.config(state="disabled")
            self.training_validationset_checkbox.config(state="disabled")
            self.epoch_entry.config(state="disabled")
            self.optimizer_combobox.config(state="disabled")
            self.lr_entry.config(state="disabled")
            self.wd_entry.config(state="disabled")
            self.stop_button.config(state="normal")
        else:
            total_tabs = self.notebook.index('end')
            for tab_no in range(total_tabs):
                if tab_no!=self.notebook.index(self.notebook.select()):
                    self.notebook.tab(tab_no,state='normal') 
            self.training_browse_transformed_dir_button.config(state="normal")
            self.training_browse_model_output_dir_button.config(state="normal")
            self.show_confusion_matrix_button.config(state="normal")
            self.show_classification_report_button.config(state="normal")
            self.training_normalized_checkbox.config(state="normal")
            self.training_global_checkbox.config(state="normal")
            self.training_fixed_seq_checkbox.config(state="normal")
            self.training_window_seq_checkbox.config(state="normal")
            self.training_bal_seq_checkbox.config(state="normal")
            self.training_imbal_seq_checkbox.config(state="normal")
            self.training_validationset_checkbox.config(state="normal")
            self.epoch_entry.config(state="normal")
            self.optimizer_combobox.config(state="normal")
            self.lr_entry.config(state="normal")
            self.wd_entry.config(state="normal")
            self.stop_button.config(state="normal")
            self.update_button_styles_training_tab()



        

    def get_learning_rate(self,event):
        self.lr = float(self.lr_entry.get())
        self.update_button_styles_training_tab()

    def get_weight_decay(self,event):
        self.wd = float(self.wd_entry.get())
        self.update_button_styles_training_tab()

    def get_optimizer(self, event=None):
        self.optimizer_val = self.optimizer_combobox.get()
        self.update_wd_entry_state()

    def update_wd_entry_state(self):
        if hasattr(self, 'optimizer_val') and self.optimizer_val[-1] == 'W':
            self.wd_entry.config(state='normal')
        else:
            self.wd_entry.config(state='disabled')
        self.update_button_styles_training_tab()

    def get_epoch(self,event):
        self.epochs = int(self.epoch_entry.get())
        self.training_update_progress_bar()

    def show_confusion_matrix(self):
        pass
    
    def show_classification_report(self):
        pass

    def training_toggle_validation_set_flag(self):
        self.training_validation_set_flag.set(self.training_validation_set_flag.get())

    def training_toggle_bal_seq(self):
        self.training_bal_seq_check.set(self.training_bal_seq_check.get())
        if not self.training_bal_seq_check.get():
            self.training_imbal_seq_checkbox.config(state="normal")
        else:
            self.training_imbal_seq_checkbox.config(state="disabled")
        self.update_button_styles_training_tab()

    def training_toggle_imbal_seq(self):
        self.training_imbal_seq_check.set(self.training_imbal_seq_check.get())
        if not self.training_imbal_seq_check.get():
            self.training_bal_seq_checkbox.config(state="normal")
        else:
            self.training_bal_seq_checkbox.config(state="disabled")
        self.update_button_styles_training_tab()

    def training_toggle_fixed_seq(self):
        self.training_fixed_seq_check.set(self.training_fixed_seq_check.get())
        if not self.training_fixed_seq_check.get():
            self.training_window_seq_checkbox.config(state="normal")
        else:
            self.training_window_seq_checkbox.config(state="disabled")
        self.update_button_styles_training_tab()

    def training_toggle_window_seq(self):
        self.training_window_seq_check.set(self.training_window_seq_check.get())
        if not self.training_window_seq_check.get():
            self.training_fixed_seq_checkbox.config(state="normal")
        else:
            self.training_fixed_seq_checkbox.config(state="disabled")
        self.update_button_styles_training_tab()
    
    def training_toggle_normalized(self):
        self.training_normalized_keypoints_check.set(self.training_normalized_keypoints_check.get())
        if not self.training_normalized_keypoints_check.get():
            self.training_global_checkbox.config(state="normal")
        else:
            self.training_global_checkbox.config(state="disabled")
        self.update_button_styles_training_tab()

    def training_toggle_global(self):
        self.training_global_keypoints_check.set(self.training_global_keypoints_check.get())
        if not self.training_global_keypoints_check.get():
            self.training_normalized_checkbox.config(state="normal")
        else:
            self.training_normalized_checkbox.config(state="disabled")
        self.update_button_styles_training_tab()

    def browse_transformed_data_directory(self):
        self.training_browse_transformed_output_directory_path = filedialog.askdirectory()
        self.update_button_styles_training_tab()

    def browse_model_output_directory(self):
        self.training_browse_model_output_directory_path = filedialog.askdirectory()
        self.update_button_styles_training_tab()
    
    def update_button_styles_training_tab(self):
        if self.training_browse_transformed_output_directory_path == "":
            self.training_browse_transformed_dir_button.configure(style="Red.TButton")
        else:
            self.training_browse_transformed_dir_button.configure(style="Blue.TButton")
        
        if self.training_browse_model_output_directory_path == "":
            self.training_browse_model_output_dir_button.configure(style="Red.TButton")
        else:
            self.training_browse_model_output_dir_button.configure(style="Blue.TButton")

        if self.training_browse_transformed_output_directory_path != "" and self.training_browse_model_output_directory_path != "" and (self.training_global_keypoints_check.get() or self.training_normalized_keypoints_check.get() ) and (self.training_fixed_seq_check.get() or self.training_window_seq_check.get()) and (self.training_bal_seq_check.get() or self.training_imbal_seq_check.get()) and (self.epochs>0) and (0<self.lr<=1) and (0<=self.wd<=1) and (self.n_heads>0) and (self.n_layers>0) and (self.embed_dim>0) and (0<=self.dropout<=1) and (self.mlp_head_size>0) and (self.batch_size>0) and (self.activation_function!=''):
            self.run_training_button.configure(style="Green.TButton")
        else:
            self.run_training_button.configure(style="Red.TButton")

    #--------------------------INFERENCE BLOCK-------------------------------------------
    #----------------------------------------------------------------------------------
    def setup_inference_tab(self):
        inference_tab = self.tabs["Inference"]
        
        # Style for buttons
        style = ttk.Style()
        style.configure("Red.TButton", foreground="black", background="red")
        style.configure("Blue.TButton", foreground="black", background="blue")
        style.configure("Green.TButton", foreground="black", background="green")
        style.configure("Orange.TButton", foreground="black", background="orange")
        style.configure("Yellow.TButton", foreground="black", background="yellow")

        top_buttons_x_offset = 40
        self.inference_browse_video_button = ttk.Button(inference_tab, text="Browse Video", command=self.browse_video, style="Red.TButton")
        self.inference_browse_video_button.place(x=100+top_buttons_x_offset, y=10)

        self.inference_browse_pose_model_button = ttk.Button(inference_tab, text="Browse Pose Model", command=self.browse_pose_model_inference, style="Red.TButton")
        self.inference_browse_pose_model_button.place(x=285+top_buttons_x_offset, y=10)

        self.inference_browse_model_weights_button = ttk.Button(inference_tab, text="Browse Model Weights", command=self.browse_model_weights, style="Red.TButton")
        self.inference_browse_model_weights_button.place(x=500+top_buttons_x_offset, y=10)
        # Frames for original video and keypoints video
        original_frame_label = tk.Label(inference_tab, text="Original Video", font=("Arial", 10, "bold"))
        original_frame_label.place(x=150, y=50)

        self.inference_original_frame_container = tk.Frame(inference_tab, bd=2, relief=tk.GROOVE)
        self.inference_original_frame_container.place(x=40, y=80)
        self.inference_original_frame_canvas = tk.Canvas(self.inference_original_frame_container, width=self.frame_size, height=self.frame_size)
        self.inference_original_frame_canvas.pack(fill=tk.BOTH, expand=True)

        keypoints_frame_label = tk.Label(inference_tab, text="Keypoints Video", font=("Arial", 10, "bold"))
        keypoints_frame_label.place(x=550, y=50)

        self.inference_keypoints_frame_container = tk.Frame(inference_tab, bd=2, relief=tk.GROOVE)
        self.inference_keypoints_frame_container.place(x=430, y=80)
        self.inference_keypoints_frame_canvas = tk.Canvas(self.inference_keypoints_frame_container, width=self.frame_size, height=self.frame_size)
        self.inference_keypoints_frame_canvas.pack(fill=tk.BOTH, expand=True)

        self.confidence_label = ttk.Label(inference_tab, text="Confidence Score (0-1):")
        self.confidence_label.place(x=335+130,y=432)
        # Create a frame for the entry widget with a border
        self.confidence_entry_frame = ttk.Frame(inference_tab, borderwidth=2, relief="solid")
        self.confidence_entry_frame.place(x=465+130, y=428)
        # Create the entry widget inside the frame
        self.confidence_entry = ttk.Entry(self.confidence_entry_frame, width=5)
        self.confidence_entry.pack()        
        self.confidence_entry.bind("<KeyRelease>", self.conf_score)
        self.confidence_entry.config(state="disabled")
         # Add checkboxes
        self.inference_normalized_checkbox = tk.Checkbutton(inference_tab, text="Normalized Keypoints",  variable=self.normalized_keypoints_check_inf, command=self.inference_toggle_normalized)
        self.inference_normalized_checkbox.place(x=50, y=430)
        self.inference_normalized_checkbox.config(state="disabled")
        self.inference_global_checkbox = tk.Checkbutton(inference_tab, text="Global Keypoints", variable=self.global_keypoints_check_inf, command=self.inference_toggle_global)
        self.inference_global_checkbox.place(x=200, y=430)
        self.inference_global_checkbox.config(state="disabled")

        self.predicted_output_label = ttk.Label(inference_tab, text="Predicted Output:", font=("Arial", 10, "bold"))
        self.predicted_output_label.place(x=260, y=468)

        self.predicted_output_entry_frame = ttk.Frame(inference_tab, borderwidth=2, relief="solid")
        self.predicted_output_entry_frame.place(x=375, y=466)
        self.predicted_output_entry = tk.Text(self.predicted_output_entry_frame, width=20, height=1.3, bg="black", fg="white", state="disabled")
        self.predicted_output_entry.pack()
        self.insert_centered_text("None")
        # Buttons for running, stopping, and quitting
        self.run_button = ttk.Button(inference_tab, text="Run", command=self.run_inference, width=10, style="Red.TButton")
        self.run_button.place(x=100, y=510)
        
        self.stop_button = ttk.Button(inference_tab, text="Stop", command=self.stop_preprocessing, width=10, style="Red.TButton")
        self.stop_button.place(x=200, y=510)
        self.stop_button.config(state="disabled")

        self.quit_button = ttk.Button(inference_tab, text="Quit", command=self.quit_gui, width=10, style="Red.TButton")
        self.quit_button.place(x=680, y=510)



    def browse_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        self.count = 0
        self.video_keypoints = []
        self.update_button_styles_inference_tab()

    def get_model_architecture(self):        
        parser = argparse.ArgumentParser(description='Process some input')
        parser.add_argument('--config', default='utils/config.yaml', type=str, help='Config path', required=False)
        args = parser.parse_args()
        self.config = read_yaml(args.config)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        trainer = Trainer(self.config)
        self.model = trainer.get_model()
        #LOAD SAVED WEIGHTS INTO PROPOSED ARCHITECTURE
        self.model.load_weights(self.model_weights_path)

    def browse_model_weights(self):
        self.model_weights_path = filedialog.askopenfilename(filetypes=[("Model files", "*.h5")])
        path_parts = self.model_weights_path.split('/')
        kp_type = path_parts[-1].split('_')[0]
        self.confidence_score = path_parts[-1].split('_')[3]
        self.sequence_length_value = path_parts[-1].split('_')[5]
        self.confidence_entry.config(state="normal")
        self.confidence_entry.insert(0, self.confidence_score)
        self.confidence_entry.config(state="disabled")
        if kp_type == 'global':
            self.inference_toggle_global()
        elif kp_type == 'normalized':
            self.inference_toggle_normalized()
        self.get_model_architecture()
        self.update_button_styles_inference_tab()
        

    def load_pose_model_inference(self):
        self.pose_model = YOLO(self.pose_model_path)   
        
    def browse_pose_model_inference(self):
        self.pose_model_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pt")])
        if self.pose_model_path!="":
            self.load_pose_model_inference()
        self.update_button_styles_inference_tab()
 
    
    def inference_toggle_global(self):
        self.inference_normalized_checkbox.config(state="normal")
        self.inference_global_checkbox.config(state="normal")
        self.global_keypoints_check_inf.set(True)
        if not self.global_keypoints_check_inf.get():
            self.inference_normalized_checkbox.config(state="normal")
        else:
            self.inference_normalized_checkbox.config(state="disabled")
        self.inference_global_checkbox.config(state="disabled")
        self.update_button_styles_inference_tab()
    
    def inference_toggle_normalized(self):
        self.inference_normalized_checkbox.config(state="normal")
        self.inference_global_checkbox.config(state="normal")
        self.normalized_keypoints_check_inf.set(True)
        if not self.normalized_keypoints_check_inf.get():
            self.inference_global_checkbox.config(state="normal")
        else:
            self.inference_global_checkbox.config(state="disabled")
        self.inference_normalized_checkbox.config(state="disabled")
        self.update_button_styles_inference_tab()

    def insert_centered_text(self, text):        
        self.predicted_output_entry.config(state='normal')
        # Calculate the number of spaces needed to center the text
        num_spaces = (self.predicted_output_entry["width"] - len(text)) // 2
        centered_text = " " * num_spaces + text + "\n"
        self.predicted_output_entry.insert("1.0", centered_text)
        self.predicted_output_entry.config(state='disabled')

    def update_button_styles_inference_tab(self):
        if self.video_path == "":
            self.inference_browse_video_button.configure(style="Red.TButton")
        else:
            self.inference_browse_video_button.configure(style="Blue.TButton")
        
        if self.pose_model_path == "":
            self.inference_browse_pose_model_button.configure(style="Red.TButton")
        else:
            self.inference_browse_pose_model_button.configure(style="Blue.TButton")
        
        if self.model_weights_path == '':
            self.inference_browse_model_weights_button.configure(style="Red.TButton")
        else:
            self.inference_browse_model_weights_button.configure(style="Blue.TButton")
        
        if self.video_path != "" and self.model_weights_path != '' and (self.confidence_score != [] and self.confidence_score != '' and 0<float(self.confidence_score)<=1) and (self.normalized_keypoints_check_inf.get() == True  or self.global_keypoints_check_inf.get() == True):
            self.run_button.configure(style="Green.TButton")
        else:
            self.run_button.configure(style="Red.TButton")

    def widgets_state_change_inference(self):
        if not self.inference_complete_flag:
            total_tabs = self.notebook.index('end')
            for tab_no in range(total_tabs):
                if tab_no!=self.notebook.index(self.notebook.select()):
                    self.notebook.tab(tab_no,state='disabled') 
            self.inference_browse_video_button.config(state="disabled")
            self.inference_browse_pose_model_button.config(state="disabled")
            self.inference_browse_model_weights_button.config(state="disabled")
            self.inference_normalized_checkbox.config(state="disabled")
            self.inference_global_checkbox.config(state="disabled")
            self.confidence_entry.config(state="disabled")
            self.stop_button.config(state="normal")
            self.run_button.config(state="disabled")

        else:
            total_tabs = self.notebook.index('end')
            for tab_no in range(total_tabs):
                if tab_no!=self.notebook.index(self.notebook.select()):
                    self.notebook.tab(tab_no,state='normal') 
            self.inference_browse_video_button.config(state="normal")
            self.inference_browse_pose_model_button.config(state="normal")
            self.inference_browse_model_weights_button.config(state="normal")
            self.inference_normalized_checkbox.config(state="disabled")
            self.inference_global_checkbox.config(state="disabled")
            self.confidence_entry.config(state="disabled")
            self.stop_button.config(state="normal")
            self.run_button.config(state="normal")
            self.update_button_styles_inference_tab()

    def solo_actions_inference(self):
        self.video_capture = cv2.VideoCapture(self.video_path)
        video_normalized_kps = []
        video_global_kps = []
        frame = []
        results = []
        results_gray_image_frame_resized = []
        global_kps = []
        global_kps_frame_resized = []
    
        while(not self.stop_event.is_set()):
            ret, frame = self.video_capture.read()
            if self.event.is_set() or self.stop_event.is_set():
                break

            if ret:
                original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #MONO CHANNEL
                gray_image = cv2.merge([gray_image, gray_image, gray_image]) #THREE CHANNELS GRAYSCALE for YOLO
                original_image_frame_resized = cv2.resize(original_image, (self.frame_size, self.frame_size))
                gray_image_frame_resized = cv2.resize(gray_image, (self.frame_size, self.frame_size))
                
                results = self.pose_model(gray_image, conf = float(self.confidence_score),save=False)
                results_gray_image_frame_resized = self.pose_model(gray_image_frame_resized,conf = float(self.confidence_score),save=False)
                
                detected_boxes = results[0].boxes
                num_persons_detected = len(detected_boxes)
                normalized_kps_tensor = results[0].keypoints.xyn.squeeze() 
                global_kps_tensor = results[0].keypoints.xy.squeeze()
                global_kps_tensor_frame_resized = results_gray_image_frame_resized[0].keypoints.xy.squeeze()
                normalized_kps = normalized_kps_tensor.numpy()
                global_kps = global_kps_tensor.numpy()
                global_kps_frame_resized = global_kps_tensor_frame_resized.numpy()
                if num_persons_detected == 1 and normalized_kps.ndim == 2 and global_kps.ndim==2 and global_kps_frame_resized.ndim==2:
                    # print('Normalized KPS: ',normalized_kps)
                    # print('Global KPS: ',global_kps)
                    # print('Global KPS Resized: ',global_kps_frame_resized)
                    gray_image_frame_resized = np.array(gray_image_frame_resized) #To write keypoints on it                  
                    # Draw keypoints on the frame
                    colors = self.pose_kps_colors()
                    for i, (x, y) in enumerate(global_kps_frame_resized):
                        cv2.circle(gray_image_frame_resized, (int(x), int(y)), 4, self.hex_to_rgb(colors[i]), -1)  # Draw a green circle around each keypoint        

                    gray_image_frame_resized = Image.fromarray(gray_image_frame_resized) #Back converted to PIL Image
                    gray_image_frame_resized = ImageTk.PhotoImage(gray_image_frame_resized)
                    original_image_frame_resized = Image.fromarray(original_image_frame_resized)
                    original_image_frame_resized = ImageTk.PhotoImage(original_image_frame_resized)

                    self.inference_original_frame_canvas.create_image(0, 0, anchor=tk.NW, image=original_image_frame_resized)
                    self.inference_keypoints_frame_canvas.create_image(0, 0, anchor=tk.NW, image=gray_image_frame_resized)
                    self.inference_original_frame_canvas.image = original_image_frame_resized
                    self.inference_keypoints_frame_canvas.image = gray_image_frame_resized

                    if self.normalized_keypoints_check_inf.get():
                        print('Normalized KPS')
                        video_normalized_kps.append(normalized_kps)
                        if len(video_normalized_kps)>=int(self.sequence_length_value):
                            #PICK LAST (SEQUENCE LENGTH) FRAMES KEYPOINTS
                            video_sequence = np.array(video_normalized_kps[-30:])
                            
                            #RESHAPING TO DESIRED INPUT SHAPE TO MAKE PREDICTION
                            num_seq,num_persons,num_kps,num_coords = video_sequence.shape
                            reshaped_sequence = video_sequence.reshape(num_seq,num_persons*num_kps*num_coords)
                            reshaped_sequence = np.expand_dims(reshaped_sequence,axis=0)
                            print('Reshaped Normalized KPS Seq SHAPE: ',reshaped_sequence.shape)    
                            #PREDICT SEQUENCE OUTPUT LABEL
                            predictions = self.model.predict(reshaped_sequence)
                            self.predicted_label = self.config['CLASSES'][np.argmax(predictions)]
                            video_sequence = []
                        
                    if self.global_keypoints_check_inf.get():
                        print('Global KPS')
                        video_global_kps.append(global_kps)
                        if len(video_global_kps)>=int(self.sequence_length_value):
                            #PICK LAST (SEQUENCE LENGTH) FRAMES KEYPOINTS
                            video_sequence = np.array(video_global_kps[-30:])
                            zeros_array = np.zeros_like(video_sequence)
                            stacked_sequence = np.stack((video_sequence, zeros_array), axis=1)
                            transformed_sequence = stacked_sequence.transpose(0, 1, 2, 3)
                            video_sequence = transformed_sequence
                            num_seq,num_persons,num_kps,num_coords = video_sequence.shape
                            reshaped_sequence = video_sequence.reshape(num_seq,num_persons*num_kps*num_coords)
                            reshaped_sequence = np.expand_dims(reshaped_sequence,axis=0)
                            print('Reshaped Global KPS Seq SHAPE: ',reshaped_sequence.shape)    
                            #PREDICT SEQUENCE OUTPUT LABEL
                            predictions = self.model.predict(reshaped_sequence)
                            self.predicted_label = self.config['CLASSES'][np.argmax(predictions)]
                            video_sequence = []
                        
                    if not self.predicted_label:        
                        self.predicted_label_txt = 'None'
                    else:
                        self.predicted_label_txt = str(self.predicted_label)
                    
                    self.insert_centered_text(self.predicted_label_txt)
                    #self.master.update_idletasks()  # Update the GUI
                    self.master.update()

                else: #Skip Frame with No Two Persons
                        continue
            else:#Video Ends
                break 
        self.video_capture.release()
        video_normalized_kps = []
        video_global_kps = []

    def interaction_inference(self):
        
        self.video_capture = cv2.VideoCapture(self.video_path)
        video_normalized_kps = []
        video_global_kps = []
        frame = []
        results = []
        results_gray_image_frame_resized = []
        global_kps = []
        global_kps_frame_resized = []

        while(not self.stop_event.is_set()):
            ret, frame = self.video_capture.read()
            if self.event.is_set() or self.stop_event.is_set():
                break

            if ret:
                original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #MONO CHANNEL
                gray_image = cv2.merge([gray_image, gray_image, gray_image]) #THREE CHANNELS GRAYSCALE for YOLO
                original_image_frame_resized = cv2.resize(original_image, (self.frame_size, self.frame_size))
                gray_image_frame_resized = cv2.resize(gray_image, (self.frame_size, self.frame_size))
                
                results = self.pose_model(gray_image, conf = float(self.confidence_score),save=False)
                results_gray_image_frame_resized = self.pose_model(gray_image_frame_resized,conf = float(self.confidence_score),save=False)
                
                detected_boxes = results[0].boxes
                num_persons_detected = len(detected_boxes)
                normalized_kps_tensor = results[0].keypoints.xyn.squeeze() 
                global_kps_tensor = results[0].keypoints.xy.squeeze()
                global_kps_tensor_frame_resized = results_gray_image_frame_resized[0].keypoints.xy.squeeze()
                normalized_kps = normalized_kps_tensor.numpy()
                global_kps = global_kps_tensor.numpy()
                global_kps_frame_resized = global_kps_tensor_frame_resized.numpy()
                if len(global_kps) == 0 or len(global_kps_frame_resized)==0:
                    continue   
                
                elif num_persons_detected == 2 and global_kps[0].shape[0]==17 and global_kps[1].shape[0]==17 and global_kps_frame_resized[0].shape[0]==17 and global_kps_frame_resized[1].shape[0]==17:
                    gray_image_frame_resized = np.array(gray_image_frame_resized) #To write keypoints on it
                    colors = self.pose_kps_colors()
                    # Draw keypoints on the frame
                    for n_person_kps in global_kps_frame_resized:
                        for i, (x, y) in enumerate(n_person_kps):
                            cv2.circle(gray_image_frame_resized, (int(x), int(y)), 4, self.hex_to_rgb(colors[i]), -1)          
                    
                    gray_image_frame_resized = Image.fromarray(gray_image_frame_resized) #Back converted to PIL Image
                    gray_image_frame_resized = ImageTk.PhotoImage(gray_image_frame_resized)
                    
                    original_image_frame_resized = Image.fromarray(original_image_frame_resized)
                    original_image_frame_resized = ImageTk.PhotoImage(original_image_frame_resized)
                    self.inference_original_frame_canvas.create_image(0, 0, anchor=tk.NW, image=original_image_frame_resized)
                    self.inference_keypoints_frame_canvas.create_image(0, 0, anchor=tk.NW, image=gray_image_frame_resized)
                    self.inference_original_frame_canvas.image = original_image_frame_resized
                    self.inference_keypoints_frame_canvas.image = gray_image_frame_resized
                    if self.normalized_keypoints_check_inf.get():
                        print('Normalized KPS')
                        video_normalized_kps.append(normalized_kps)
                        if len(video_normalized_kps)>=int(self.sequence_length_value):
                            #PICK LAST (SEQUENCE LENGTH) FRAMES KEYPOINTS
                            video_sequence = np.array(video_normalized_kps[-30:])
                            
                            #RESHAPING TO DESIRED INPUT SHAPE TO MAKE PREDICTION
                            num_seq,num_persons,num_kps,num_coords = video_sequence.shape
                            reshaped_sequence = video_sequence.reshape(num_seq,num_persons*num_kps*num_coords)
                            reshaped_sequence = np.expand_dims(reshaped_sequence,axis=0)
                            print('Reshaped Normalized KPS Seq SHAPE: ',reshaped_sequence.shape)    
                            #PREDICT SEQUENCE OUTPUT LABEL
                            predictions = self.model.predict(reshaped_sequence)
                            self.predicted_label = self.config['CLASSES'][np.argmax(predictions)]
                            video_sequence = []
                        
                    if self.global_keypoints_check_inf.get():
                        print('Global KPS')
                        video_global_kps.append(global_kps)
                        if len(video_global_kps)>=int(self.sequence_length_value):
                            #PICK LAST (SEQUENCE LENGTH) FRAMES KEYPOINTS
                            video_sequence = np.array(video_global_kps[-30:])
                            
                            #RESHAPING TO DESIRED INPUT SHAPE TO MAKE PREDICTION
                            num_seq,num_persons,num_kps,num_coords = video_sequence.shape
                            reshaped_sequence = video_sequence.reshape(num_seq,num_persons*num_kps*num_coords)
                            reshaped_sequence = np.expand_dims(reshaped_sequence,axis=0)
                            print('Reshaped Global KPS Seq SHAPE: ',reshaped_sequence.shape)    
                            #PREDICT SEQUENCE OUTPUT LABEL
                            predictions = self.model.predict(reshaped_sequence)
                            self.predicted_label = self.config['CLASSES'][np.argmax(predictions)]
                            video_sequence = []
                        
                    if not self.predicted_label:        
                        self.predicted_label_txt = 'None'
                    else:
                        self.predicted_label_txt = str(self.predicted_label)
                    
                    self.insert_centered_text(self.predicted_label_txt)
                    #self.master.update_idletasks()  # Update the GUI
                    self.master.update()

                else: #Skip Frame with No Two Persons
                        continue
            else:#Video Ends
                break 
        self.video_capture.release()
        video_normalized_kps = []
        video_global_kps = []

    def _inference(self):
        video_path_parts = self.video_path.split('/')
        self.action_type = video_path_parts[-1].split('_')[0]
        self.predicted_label = 'None'
        if self.action_type == 'solo':
            self.solo_actions_inference()
        elif self.action_type == 'interaction':
            self.interaction_inference()
        else:
            print(f'Action Type : ({self.action_type}) Not Found!')


    def run_inference(self):
        self.inference_complete_flag = False
        self.stop_event.clear()
        if self.pose_model_path!="" and self.video_path!="" and self.model_weights_path!="" and (self.confidence_score != [] and self.confidence_score != '' and 0<float(self.confidence_score)<=1) and (self.normalized_keypoints_check_inf.get()==True or self.global_keypoints_check_inf.get()==True):
            self.widgets_state_change_inference()
            self.thread = threading.Thread(target=self._inference(),args=(self.event,))
            self.thread.start()
            self.inference_complete_flag = True
            self.widgets_state_change_inference()
        


    #--------------------------TRANSFORMATION BLOCK-------------------------------------------
    #----------------------------------------------------------------------------------
    def setup_data_transformation_tab(self):
        data_transformation_tab = self.tabs["Data Transformation"]
        
        # Style for buttons
        style = ttk.Style()
        style.configure("Red.TButton", foreground="black", background="red")
        style.configure("Blue.TButton", foreground="black", background="blue")
        style.configure("Green.TButton", foreground="black", background="green")
        style.configure("Orange.TButton", foreground="black", background="orange")
        style.configure("Yellow.TButton", foreground="black", background="yellow")

        #colors for kps
        colors = self.pose_kps_colors()
        #pose_kps_img = cv2.imread('utils\gui_images\pose_kps.jpg')

        
        #KPS Check Boxes
        kps_x_offset = 120
        kps_y_offset = 5
        
        self.pose_frame_container = tk.Frame(data_transformation_tab, bd=0, relief=tk.GROOVE)
        self.pose_frame_container.place(x=400, y=27+kps_y_offset)
        self.pose_frame_canvas = tk.Canvas(self.pose_frame_container, width=227, height=400)
        self.pose_frame_canvas.pack(fill=tk.BOTH, expand=True)
        self.person_kps_show()        

        self.kps_checkbox_1 = tk.Checkbutton(data_transformation_tab, text="Nose",  variable=self.kps_checkbox_1_var, command=self.toggle_kps_check_vars,fg=colors[0])
        self.kps_checkbox_1.place(x=50+kps_x_offset, y=50+kps_y_offset)

        self.kps_checkbox_2 = tk.Checkbutton(data_transformation_tab, text="Right Eye",  variable=self.kps_checkbox_2_var, command=self.toggle_kps_check_vars,fg=colors[1])
        self.kps_checkbox_2.place(x=50+kps_x_offset, y=70+kps_y_offset)

        self.kps_checkbox_3 = tk.Checkbutton(data_transformation_tab, text="Left Eye",  variable=self.kps_checkbox_3_var, command=self.toggle_kps_check_vars,fg=colors[2])
        self.kps_checkbox_3.place(x=50+kps_x_offset, y=90+kps_y_offset)

        self.kps_checkbox_4 = tk.Checkbutton(data_transformation_tab, text="Right Ear",  variable=self.kps_checkbox_4_var, command=self.toggle_kps_check_vars,fg=colors[3])
        self.kps_checkbox_4.place(x=50+kps_x_offset, y=110+kps_y_offset)

        self.kps_checkbox_5 = tk.Checkbutton(data_transformation_tab, text="Left Ear",  variable=self.kps_checkbox_5_var, command=self.toggle_kps_check_vars,fg=colors[4])
        self.kps_checkbox_5.place(x=50+kps_x_offset, y=130+kps_y_offset)

        self.kps_checkbox_6 = tk.Checkbutton(data_transformation_tab, text="Right Shoulder",  variable=self.kps_checkbox_6_var, command=self.toggle_kps_check_vars,fg=colors[5])
        self.kps_checkbox_6.place(x=50+kps_x_offset, y=150+kps_y_offset)

        self.kps_checkbox_7 = tk.Checkbutton(data_transformation_tab, text="Left Shoulder",  variable=self.kps_checkbox_7_var, command=self.toggle_kps_check_vars,fg=colors[6])
        self.kps_checkbox_7.place(x=50+kps_x_offset, y=170+kps_y_offset)

        self.kps_checkbox_8 = tk.Checkbutton(data_transformation_tab, text="Right Elbow",  variable=self.kps_checkbox_8_var, command=self.toggle_kps_check_vars,fg=colors[7])
        self.kps_checkbox_8.place(x=50+kps_x_offset, y=190+kps_y_offset)

        self.kps_checkbox_9 = tk.Checkbutton(data_transformation_tab, text="Left Elbow",  variable=self.kps_checkbox_9_var, command=self.toggle_kps_check_vars,fg=colors[8])
        self.kps_checkbox_9.place(x=50+kps_x_offset, y=210+kps_y_offset)

        self.kps_checkbox_10 = tk.Checkbutton(data_transformation_tab, text="Right Wrist",  variable=self.kps_checkbox_10_var, command=self.toggle_kps_check_vars,fg=colors[9])
        self.kps_checkbox_10.place(x=50+kps_x_offset, y=230+kps_y_offset)

        self.kps_checkbox_11 = tk.Checkbutton(data_transformation_tab, text="Left Wrist",  variable=self.kps_checkbox_11_var, command=self.toggle_kps_check_vars,fg=colors[10])
        self.kps_checkbox_11.place(x=50+kps_x_offset, y=250+kps_y_offset)

        self.kps_checkbox_12 = tk.Checkbutton(data_transformation_tab, text="Right Hip",  variable=self.kps_checkbox_12_var, command=self.toggle_kps_check_vars,fg=colors[11])
        self.kps_checkbox_12.place(x=50+kps_x_offset, y=270+kps_y_offset)

        self.kps_checkbox_13 = tk.Checkbutton(data_transformation_tab, text="Left Hip",  variable=self.kps_checkbox_13_var, command=self.toggle_kps_check_vars,fg=colors[12])
        self.kps_checkbox_13.place(x=50+kps_x_offset, y=290+kps_y_offset)

        self.kps_checkbox_14 = tk.Checkbutton(data_transformation_tab, text="Right Knee",  variable=self.kps_checkbox_14_var, command=self.toggle_kps_check_vars,fg=colors[13])
        self.kps_checkbox_14.place(x=50+kps_x_offset, y=310+kps_y_offset)

        self.kps_checkbox_15 = tk.Checkbutton(data_transformation_tab, text="Left Knee",  variable=self.kps_checkbox_15_var, command=self.toggle_kps_check_vars,fg=colors[14])
        self.kps_checkbox_15.place(x=50+kps_x_offset, y=330+kps_y_offset)

        self.kps_checkbox_16 = tk.Checkbutton(data_transformation_tab, text="Right Ankle",  variable=self.kps_checkbox_16_var, command=self.toggle_kps_check_vars,fg=colors[15])
        self.kps_checkbox_16.place(x=50+kps_x_offset, y=350+kps_y_offset)

        self.kps_checkbox_17 = tk.Checkbutton(data_transformation_tab, text="Left Ankle",  variable=self.kps_checkbox_17_var, command=self.toggle_kps_check_vars,fg=colors[16])
        self.kps_checkbox_17.place(x=50+kps_x_offset, y=370+kps_y_offset)

        ngo_checkboxes_y_offset = 10
        self.transformation_normalized_checkbox = tk.Checkbutton(data_transformation_tab, text="Normalized Keypoints",  variable=self.normalized_keypoints_check, command=self.transformation_toggle_normalized)
        self.transformation_normalized_checkbox.place(x=50, y=400+ngo_checkboxes_y_offset)
        self.transformation_global_checkbox = tk.Checkbutton(data_transformation_tab, text="Global Keypoints", variable=self.global_keypoints_check, command=self.transformation_toggle_global)
        self.transformation_global_checkbox.place(x=200, y=400+ngo_checkboxes_y_offset)
        # self.transformation_overwrite_keypoints_checkbox = tk.Checkbutton(data_transformation_tab, text="Overwrite Keypoints", variable=self.overwrite_keypoints_check, command=self.transformation_overwrite_keypoints)
        # self.transformation_overwrite_keypoints_checkbox.place(x=320, y=400+ngo_checkboxes_y_offset)

        #SEQ TYPE:
        ys_off = 5
        self.seq_type_label = ttk.Label(data_transformation_tab, text="Sequence Type:")
        self.seq_type_label.place(x=50,y=438-ys_off)
        self.transformation_fixed_seq_checkbox = tk.Checkbutton(data_transformation_tab, text="Fixed", variable=self.fixed_seq_check, command=self.transformation_toggle_fixed_seq)
        self.transformation_fixed_seq_checkbox.place(x=135, y=436-ys_off)
        self.transformation_window_seq_checkbox = tk.Checkbutton(data_transformation_tab, text="Sliding Window", variable=self.window_seq_check, command=self.transformation_toggle_window_seq)
        self.transformation_window_seq_checkbox.place(x=200, y=436-ys_off)

        self.sequence_length_label = ttk.Label(data_transformation_tab, text="Sequence Length (10-60):")
        self.sequence_length_label.place(x=320,y=438-ys_off)
        self.sequence_length_entry_frame = ttk.Frame(data_transformation_tab, borderwidth=2, relief="solid")
        self.sequence_length_entry_frame.place(x=457, y=436-ys_off)
        self.sequence_length_entry = ttk.Entry(self.sequence_length_entry_frame, width=5)
        self.sequence_length_entry.pack()        
        self.sequence_length_entry.bind("<KeyRelease>", self.sequence_length)


        yoff = 20
        self.duplicate_threshold_label = ttk.Label(data_transformation_tab, text="Duplication Threshold (0-1):")
        self.duplicate_threshold_label.place(x=205,y=438+yoff)
        self.duplicate_threshold_entry_frame = ttk.Frame(data_transformation_tab, borderwidth=2, relief="solid")
        self.duplicate_threshold_entry_frame.place(x=355, y=435+yoff)
        self.duplicate_threshold_entry = ttk.Entry(self.duplicate_threshold_entry_frame, width=5,state='disabled')
        self.duplicate_threshold_entry.pack()        
        self.duplicate_threshold_entry.bind("<KeyRelease>", self.duplication_threshold)
        
        
        self.transformation_duplicate_checkbox = tk.Checkbutton(data_transformation_tab, text="Duplicate Last Frames",  variable=self.duplicate_frames_check, command=self.toggle_duplicate_frames_flag,state='disabled')
        self.transformation_duplicate_checkbox.place(x=50, y=425+ngo_checkboxes_y_offset+yoff)
        
        self.trainset_label = ttk.Label(data_transformation_tab, text="Training Set (0-1):")
        self.trainset_label.place(x=160,y=453+ngo_checkboxes_y_offset+yoff)
        self.trainset_entry_frame = ttk.Frame(data_transformation_tab, borderwidth=2, relief="solid")
        self.trainset_entry_frame.place(x=256, y=451+ngo_checkboxes_y_offset+yoff)
        self.trainset_entry = ttk.Entry(self.trainset_entry_frame, width=5)
        self.trainset_entry.pack()        
        self.trainset_entry.bind("<KeyRelease>", self.trainset_percentage)

        self.testset_label = ttk.Label(data_transformation_tab, text="Test Set (0-1):")
        self.testset_label.place(x=310, y=453+ngo_checkboxes_y_offset+yoff)

        self.testset_entry_frame = ttk.Frame(data_transformation_tab, borderwidth=2, relief="solid")
        self.testset_entry_frame.place(x=385, y=451+ngo_checkboxes_y_offset+yoff)
        self.testset_entry = tk.Text(self.testset_entry_frame, width=4,height=1.3)
        self.testset_entry.pack()

        validation_x_offset = 275
        self.validationset_label = ttk.Label(data_transformation_tab, text="Validation % (0-1):")
        self.validationset_label.place(x=160+validation_x_offset,y=453+ngo_checkboxes_y_offset+yoff)
        self.validationset_entry_frame = ttk.Frame(data_transformation_tab, borderwidth=2, relief="solid")
        self.validationset_entry_frame.place(x=260+validation_x_offset, y=451+ngo_checkboxes_y_offset+yoff)
        self.validationset_entry = ttk.Entry(self.validationset_entry_frame, width=5)
        self.validationset_entry.pack()        
        self.validationset_entry.bind("<KeyRelease>", self.validationset_percentage)

        self.transformation_validationset_checkbox = tk.Checkbutton(data_transformation_tab, text="Validation Set",  variable=self.validation_set_flag, command=self.toggle_validation_set_flag)
        self.transformation_validationset_checkbox.place(x=50, y=450+ngo_checkboxes_y_offset+yoff)

        self.browse_pose_dir_button = ttk.Button(data_transformation_tab, text="Browse Extracted Pose Directory", command=self.browse_pose_directory, style="Red.TButton")
        self.browse_pose_dir_button.place(x=190, y=10)
        
        self.browse_transformation_output_dir_button = ttk.Button(data_transformation_tab, text="Browse Transformation Output Directory", command=self.browse_transformation_output_directory, style="Red.TButton")
        self.browse_transformation_output_dir_button.place(x=410, y=10)

        y_offset_bottom_widgets = 30
        progress_text = str(self.in_progress_npy_num)+ '/' + str(self.total_npy_files) + ': '
        self.progress_bar_label = tk.Label(data_transformation_tab, text=progress_text, font=("Arial", 10, "bold"))
        self.progress_bar_label.place(x=40, y=470+y_offset_bottom_widgets+yoff-7)
        
        self.progress_bar = ttk.Progressbar(data_transformation_tab, length=600, mode="determinate")
        self.progress_bar.place(x=100, y=470+y_offset_bottom_widgets+yoff-8)
        
        # Buttons for running, stopping, and quitting
        self.generate_button = ttk.Button(data_transformation_tab, text="Generate", command=self.run_transformation, width=10, style="Red.TButton")
        self.generate_button.place(x=100, y=510+y_offset_bottom_widgets)
        
        self.stop_button = ttk.Button(data_transformation_tab, text="Stop", command=self.stop_preprocessing, width=10, style="Red.TButton")
        self.stop_button.place(x=200, y=510+y_offset_bottom_widgets)
        self.stop_button.config(state="disabled")

        self.quit_button = ttk.Button(data_transformation_tab, text="Quit", command=self.quit_gui, width=10, style="Red.TButton")
        self.quit_button.place(x=680, y=510+y_offset_bottom_widgets)

    def transformation_toggle_fixed_seq(self):
        self.fixed_seq_check.set(self.fixed_seq_check.get())
        if not self.fixed_seq_check.get():
            self.duplicate_frames_check.set(False)
            self.window_seq_check.set(False)
            self.transformation_window_seq_checkbox.config(state="normal")
            self.transformation_duplicate_checkbox.config(state="disabled")
            self.duplicate_threshold_entry.config(state="disabled")

        else:
            self.duplicate_frames_check.set(True)
            self.transformation_window_seq_checkbox.config(state="disabled")
            self.transformation_duplicate_checkbox.config(state="normal")
            self.duplicate_threshold_entry.config(state="normal")
        self.update_button_styles_transformation_tab()



    def transformation_toggle_window_seq(self):
        self.window_seq_check.set(self.window_seq_check.get())
        if not self.window_seq_check.get():
            self.transformation_fixed_seq_checkbox.config(state="normal")
        else:
            self.transformation_fixed_seq_checkbox.config(state="disabled")
        self.update_button_styles_transformation_tab()

    def transformation_toggle_normalized(self):
        self.normalized_keypoints_check.set(self.normalized_keypoints_check.get())
        self.update_button_styles_transformation_tab()

    def transformation_toggle_global(self):
        self.global_keypoints_check.set(self.global_keypoints_check.get())
        self.update_button_styles_transformation_tab()

    def transformation_overwrite_keypoints(self):
        self.overwrite_keypoints_check.set(self.overwrite_keypoints_check.get())
        self.update_button_styles_transformation_tab()

    def pose_kps_colors(self):
        colors =  [
                            '#00FFFF', '#00FF00', '#0000FF', '#FF0000', '#FF00FF',
                            '#64C8FF', '#800000', '#008000', '#000080', '#808000',
                            '#800080', '#008080', '#FF8000', '#FF0080', '#80FF00',
                            '#00FF80', '#0080FF'
                        ]
        return colors

    def hex_to_rgb(self,hex_color):
        hex_color = hex_color.lstrip('#')  # Remove '#' if present
        # Convert hex to RGB
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return rgb

    def person_kps_show(self):
        #pose_image = cv2.imread('utils\gui_images\pose_kps.jpg')
        colors = self.pose_kps_colors()
        
        pose_image = cv2.imread('utils\gui_images\human_kps_img.jpg')
        height, width = pose_image.shape[:2]
        aspect_ratio = width / height
        new_height = 400
        new_width = int(new_height * aspect_ratio)
        pose_image = cv2.resize(pose_image, (new_width, new_height))
        pose_image = np.array(pose_image)
        pose_kps = np.load(r'utils\gui_images\gui_person_kps.npy')
        for i, (x, y) in enumerate(pose_kps):
            if i not in self.kps_indices_to_drop:
                cv2.circle(pose_image, (int(x), int(y)), 4, self.hex_to_rgb(colors[i]), -1) 

        pose_image = Image.fromarray(pose_image) 
        pose_image = ImageTk.PhotoImage(pose_image)
        self.pose_frame_canvas.create_image(0, 0, anchor=tk.NW, image=pose_image)
        self.pose_frame_canvas.image = pose_image
        self.master.update()


    def trainset_percentage(self,event):
        self.trainset_percent = self.trainset_entry.get()
        remaining_percent = None
        if self.trainset_percent!='' and self.trainset_percent!=[]:
            remaining_percent = round(1-float(self.trainset_percent),2)
        if self.trainset_percent!='' and self.trainset_percent!=[] and self.validation_set_flag.get() and self.validationset_percent!='' and self.validationset_percent!=[]:
            if self.validationset_percent!='' and self.validationset_percent!=[]:
                self.testset_percent = round(remaining_percent*(1-float(self.validationset_percent)),2)
                self.testset_entry.delete("1.0", tk.END)
                self.testset_entry.insert("1.0",str(self.testset_percent))
        else:
            self.testset_percent = remaining_percent
            self.testset_entry.delete("1.0", tk.END)
            self.testset_entry.insert("1.0",str(self.testset_percent))
        self.update_button_styles_transformation_tab()

    def validationset_percentage(self,event):
        self.validationset_percent = self.validationset_entry.get()
        if self.validationset_percent!='' and self.validationset_percent!=[]:
            self.trainset_percentage(None)
        self.update_button_styles_transformation_tab()

    def toggle_validation_set_flag(self):
        self.validation_set_flag.set(self.validation_set_flag.get())
        if self.validation_set_flag.get():
            self.validationset_entry.config(state="normal")
        else:
            self.validationset_entry.config(state="disabled")
        self.validationset_percentage(None)


    def duplication_threshold(self,event):
        self.duplicate_frames_threshold = self.duplicate_threshold_entry.get()
        self.update_button_styles_transformation_tab()

    def sequence_length(self,event):
        self.sequence_length_value = self.sequence_length_entry.get()
        self.update_button_styles_transformation_tab()

    def toggle_duplicate_frames_flag(self):
        self.duplicate_frames_check.set(self.duplicate_frames_check.get())
        if self.duplicate_frames_check.get():
            self.duplicate_threshold_entry.config(state="normal")
        else:
            self.duplicate_threshold_entry.config(state="disabled")
        self.update_button_styles_transformation_tab()

    def toggle_kps_check_vars(self):
        checkbox_vars = [
                            self.kps_checkbox_1_var, self.kps_checkbox_2_var, self.kps_checkbox_3_var,
                            self.kps_checkbox_4_var, self.kps_checkbox_5_var, self.kps_checkbox_6_var,
                            self.kps_checkbox_7_var, self.kps_checkbox_8_var, self.kps_checkbox_9_var,
                            self.kps_checkbox_10_var, self.kps_checkbox_11_var, self.kps_checkbox_12_var,
                            self.kps_checkbox_13_var, self.kps_checkbox_14_var, self.kps_checkbox_15_var,
                            self.kps_checkbox_16_var, self.kps_checkbox_17_var
                        ]
        self.get_kps_indices_to_drop(checkbox_vars)

    def browse_pose_directory(self):
        self.pose_dir = filedialog.askdirectory()
        self.update_button_styles_transformation_tab()

    def browse_transformation_output_directory(self):
        self.output_transformation_dir = filedialog.askdirectory()
        self.update_button_styles_transformation_tab()

    def get_kps_indices_to_drop(self,kps_checkbox_status):
        # Remove indices of checked kps checkboxes
        for i in self.kps_indices_to_drop[:]:
            if kps_checkbox_status[i].get():
                self.kps_indices_to_drop.remove(i)

        # Add indices of unchecked kps checkboxes
        for i, checkbox_var in enumerate(kps_checkbox_status):
            if not checkbox_var.get() and i not in self.kps_indices_to_drop:
                self.kps_indices_to_drop.append(i)
        self.person_kps_show()
        
    def transformation_widgets_state_change(self):
        
        if not self.transformation_complete_flag:
            total_tabs = self.notebook.index('end')
            for tab_no in range(total_tabs):
                if tab_no!=self.notebook.index(self.notebook.select()):
                    self.notebook.tab(tab_no,state='disabled') 
            self.in_progress_num = 0
            self.progress_bar['value'] = self.in_progress_num
            self.browse_pose_dir_button.config(state="disabled")
            self.browse_transformation_output_dir_button.config(state="disabled")
            self.kps_checkbox_1.config(state="disabled")
            self.kps_checkbox_2.config(state="disabled")
            self.kps_checkbox_3.config(state="disabled")
            self.kps_checkbox_4.config(state="disabled")
            self.kps_checkbox_5.config(state="disabled")
            self.kps_checkbox_6.config(state="disabled")
            self.kps_checkbox_7.config(state="disabled")
            self.kps_checkbox_8.config(state="disabled")
            self.kps_checkbox_9.config(state="disabled")
            self.kps_checkbox_10.config(state="disabled")
            self.kps_checkbox_11.config(state="disabled")
            self.kps_checkbox_12.config(state="disabled")
            self.kps_checkbox_13.config(state="disabled")
            self.kps_checkbox_14.config(state="disabled")
            self.kps_checkbox_15.config(state="disabled")
            self.kps_checkbox_16.config(state="disabled")
            self.kps_checkbox_17.config(state="disabled")
            self.transformation_normalized_checkbox.config(state="disabled")
            self.transformation_global_checkbox.config(state="disabled")
            self.transformation_window_seq_checkbox.config(state="disabled")
            self.transformation_fixed_seq_checkbox.config(state="disabled")
#            self.transformation_overwrite_keypoints_checkbox.config(state="disabled")
            self.duplicate_threshold_entry.config(state="disabled")
            self.sequence_length_entry.config(state="disabled")
            self.transformation_duplicate_checkbox.config(state="disabled")
            self.transformation_validationset_checkbox.config(state="disabled")
            self.validationset_entry.config(state="disabled")
            self.testset_entry.config(state="disabled")
            self.trainset_entry.config(state="disabled")
            self.generate_button.config(state="disabled")    
            self.stop_button.config(state="normal")
        else:
            total_tabs = self.notebook.index('end')
            for tab_no in range(total_tabs):
                if tab_no!=self.notebook.index(self.notebook.select()):
                    self.notebook.tab(tab_no,state='normal') 
            self.browse_pose_dir_button.config(state="normal")
            self.browse_transformation_output_dir_button.config(state="normal")
            self.kps_checkbox_1.config(state="normal")
            self.kps_checkbox_2.config(state="normal")
            self.kps_checkbox_3.config(state="normal")
            self.kps_checkbox_4.config(state="normal")
            self.kps_checkbox_5.config(state="normal")
            self.kps_checkbox_6.config(state="normal")
            self.kps_checkbox_7.config(state="normal")
            self.kps_checkbox_8.config(state="normal")
            self.kps_checkbox_9.config(state="normal")
            self.kps_checkbox_10.config(state="normal")
            self.kps_checkbox_11.config(state="normal")
            self.kps_checkbox_12.config(state="normal")
            self.kps_checkbox_13.config(state="normal")
            self.kps_checkbox_14.config(state="normal")
            self.kps_checkbox_15.config(state="normal")
            self.kps_checkbox_16.config(state="normal")
            self.kps_checkbox_17.config(state="normal")
            self.transformation_normalized_checkbox.config(state="normal")
            self.transformation_global_checkbox.config(state="normal")
            self.transformation_window_seq_checkbox.config(state="normal")
            self.transformation_fixed_seq_checkbox.config(state="normal")
 #           self.transformation_overwrite_keypoints_checkbox.config(state="normal") 
            self.transformation_duplicate_checkbox.config(state="normal")
            self.sequence_length_entry.config(state="normal")
            self.toggle_duplicate_frames_flag()   
            self.transformation_validationset_checkbox.config(state="normal")
            self.toggle_validation_set_flag()
            self.testset_entry.config(state="normal")
            self.trainset_entry.config(state="normal")
            self.generate_button.config(state="normal")    
            self.stop_button.config(state="normal")
            self.update_button_styles_transformation_tab()



    def dataset_count_npy_files(self,folder_path):
        total_npy_files = 0
        stack = [folder_path]

        while stack:
            current_dir = stack.pop()
            try:
                subfolders = os.listdir(current_dir)
            except PermissionError:
                continue

            npy_count = 0

            for item in subfolders:
                item_path = os.path.join(current_dir, item)
                if os.path.isdir(item_path):
                    stack.append(item_path)
                elif item.endswith('.npy'):
                    npy_count += 1
            
            total_npy_files += npy_count

        return total_npy_files

    def count_total_npy_files(self):
        datasets_folders = os.listdir(self.pose_dir)
        normalized_npy_files = 0
        global_npy_files = 0
        total_npy_files = 0
        for datasets_folder in datasets_folders:
            if self.event.is_set() or self.stop_event.is_set():
                        break            
            datasets_folder_path = os.path.join(self.pose_dir,datasets_folder)
            kps_type_folders = os.listdir(datasets_folder_path)
            for kps_type_folder in kps_type_folders:
                if self.event.is_set() or self.stop_event.is_set():
                            break                    
                kps_type = kps_type_folder.split('_')[0]
                if kps_type == 'normalized':
                    kps_type_folder_path = os.path.join(datasets_folder_path,kps_type_folder)
                    normalized_npy_files+= self.dataset_count_npy_files(kps_type_folder_path)
                if kps_type == 'global':
                    kps_type_folder_path = os.path.join(datasets_folder_path,kps_type_folder)
                    global_npy_files+= self.dataset_count_npy_files(kps_type_folder_path)
        
        # print('Normalized: ',normalized_npy_files)
        # print('Global: ',global_npy_files)
        # print('Total')
        if self.normalized_keypoints_check.get() and self.global_keypoints_check.get():
            total_npy_files = normalized_npy_files+global_npy_files
        
        if self.normalized_keypoints_check.get() and not self.global_keypoints_check.get():
            total_npy_files = normalized_npy_files
        
        if not self.normalized_keypoints_check.get() and self.global_keypoints_check.get():
            total_npy_files = global_npy_files
        
        return total_npy_files
#FIXED SEQUENCES FORMATION BEGIN
    def fixed_seq_formation(self,class_npy_file_path):
        self.npy_file_data = []
        if self.action_type == 'interactions':
            self.npy_file_data = np.load(class_npy_file_path)
            print('\nNEW FILE LOAD')
            print('NPY FILE DATA SHAPE: ',self.npy_file_data.shape)
            if self.npy_file_data.shape[0] >= int(self.sequence_length_value):
                #DROPPIND INDEXES
                for indx in self.kps_indices_to_drop:
                    self.npy_file_data = np.delete(self.npy_file_data,indx,axis=2)
                
                remaining_frames = []
                remaining_frames_percentage = []                
                num_frames_to_duplicate = []
                num_sequences = []
                self.sequences = []                
                start_index = []
                end_index = []
                duplicated_sequences = []
                remaining_frames = self.npy_file_data.shape[0]%int(self.sequence_length_value)
                remaining_frames_percentage = remaining_frames/int(self.sequence_length_value)
                if remaining_frames_percentage == 0: #COMPLETE SEQUENCE
                    print('COMPLETE SEQUENCE BLOCK')
                    #SEQUENCES FORMATION
                    #COMPLETE SEQUENCES
                    num_sequences = self.npy_file_data.shape[0]//int(self.sequence_length_value)
                    # Initialize an empty array to store the sequences
                    
                    self.sequences = np.zeros((num_sequences, int(self.sequence_length_value), self.npy_file_data.shape[1], self.npy_file_data.shape[2], self.npy_file_data.shape[3]))
                    #Create fixed complete sequences
                    for i in range(num_sequences):
                        start_index = i*int(self.sequence_length_value)
                        end_index = start_index + int(self.sequence_length_value)
                        self.sequences[i] = self.npy_file_data[start_index:end_index]
                    self.sequences = self.sequences
                    print('Interactions Seq Shape: ',self.sequences.shape)

                elif remaining_frames_percentage > 0: #INCOMPLETE SEQUENCES 
                    print('INCOMPLETE SEQUENCES BLOCK')       
                    #DUPLICATE (IF CHECKED)
                    if self.duplicate_frames_check.get():
                        print('DUPLICATION BLOCK')
                        if remaining_frames_percentage>= float(self.duplicate_frames_threshold):
                            print('PERCENTAGE IS HIGH, DUPLICATING BLOCK')
                            num_frames_to_duplicate = int(self.sequence_length_value) - remaining_frames
                            print('Frames to Duplicate: ', num_frames_to_duplicate)
                            
                            duplicated_sequences = self.npy_file_data[-num_frames_to_duplicate:]
                            merged_npy_file_data = np.concatenate((self.npy_file_data, duplicated_sequences), axis=0)
                            print('Merged NPY File Shape: ',merged_npy_file_data.shape)
                            num_sequences = (merged_npy_file_data.shape[0]//int(self.sequence_length_value))
                            # Initialize an empty array to store the sequences
                            
                            self.sequences = np.zeros((num_sequences, int(self.sequence_length_value), self.npy_file_data.shape[1], self.npy_file_data.shape[2], self.npy_file_data.shape[3]))
                            print('Sequences Shape: ', self.sequences.shape)
                            #Create fixed duplicated sequences
                            for i in range(num_sequences):
                                start_index = i*int(self.sequence_length_value)
                                end_index = start_index + int(self.sequence_length_value)
                                print('Start Index: ',start_index)
                                print('End Index: ',end_index)
                                self.sequences[i] = merged_npy_file_data[start_index:end_index]
                            self.sequences = self.sequences
                            print('Interactions Seq Shape: ',self.sequences.shape)

                        else:
                            print('PERCENTAGE IS LOW, GETTING COMPLETE SEQUENCES ONLY BLOCK')
                            
                            #GET COMPLETE SEQUENCES ONLY
                            num_sequences = self.npy_file_data.shape[0]//int(self.sequence_length_value)
                            # Initialize an empty array to store the sequences
                            self.sequences = np.zeros((num_sequences, int(self.sequence_length_value), self.npy_file_data.shape[1], self.npy_file_data.shape[2], self.npy_file_data.shape[3]))
                            #Create fixed complete sequences
                            for i in range(num_sequences):
                                start_index = i*int(self.sequence_length_value)
                                end_index = start_index + int(self.sequence_length_value)
                                self.sequences[i] = self.npy_file_data[start_index:end_index]
                            self.sequences = self.sequences
                            print('Interactions Seq Shape: ',self.sequences.shape)
        
                    else: #NO DUPLICATION
                        print('DUPLICATION FLAG OFF BLOCK')
                        #SEQUENCES FORMATION
                        #GET COMPLETE SEQUENCES ONLY
                        num_sequences = self.npy_file_data.shape[0]//int(self.sequence_length_value)
                        # Initialize an empty array to store the sequences
                        self.sequences = np.zeros((num_sequences, int(self.sequence_length_value), self.npy_file_data.shape[1], self.npy_file_data.shape[2], self.npy_file_data.shape[3]))
                        #Create fixed complete sequences
                        for i in range(num_sequences):
                            start_index = i*int(self.sequence_length_value)
                            end_index = start_index + int(self.sequence_length_value)
                            self.sequences[i] = self.npy_file_data[start_index:end_index]
                        self.sequences = self.sequences
                        print('Interactions Seq Shape: ',self.sequences.shape)    
                
        if self.action_type == 'solo':
            self.npy_file_data = np.load(class_npy_file_path)
            print('NPY FILE DATA SHAPE: ',self.npy_file_data.shape)
            if self.npy_file_data.shape[0] >= int(self.sequence_length_value):
                #RESHAPING
                self.npy_file_data = self.npy_file_data.reshape(self.npy_file_data.shape[0], 1, self.npy_file_data.shape[1], self.npy_file_data.shape[2])   
                zeros_array = []
                zeros_array = np.zeros_like(self.npy_file_data)
                # Concatenate along the second axis
                self.npy_file_data = np.concatenate((self.npy_file_data, zeros_array), axis=1)
                print('RESHAPED NPY FILE SHAPE: ',self.npy_file_data.shape)
                #print('ORG NPY DATA: ',self.npy_file_data[0][:][:][:])
                #DROPPIND INDEXES
                for indx in self.kps_indices_to_drop:
                    self.npy_file_data = np.delete(self.npy_file_data,indx,axis=2)
                
                remaining_frames = []
                remaining_frames_percentage = []                
                num_frames_to_duplicate = []
                num_sequences = []
                self.sequences = []                
                start_index = []
                end_index = []
                duplicated_sequences = []
                
                remaining_frames = self.npy_file_data.shape[0]%int(self.sequence_length_value)
                remaining_frames_percentage = remaining_frames/int(self.sequence_length_value)
            
                if remaining_frames_percentage == 0: #COMPLETE SEQUENCE
                    #SEQUENCES FORMATION
                    #COMPLETE SEQUENCES
                    num_sequences = self.npy_file_data.shape[0]//int(self.sequence_length_value)
                    # Initialize an empty array to store the sequences
                    self.sequences = np.zeros((num_sequences, int(self.sequence_length_value), self.npy_file_data.shape[1], self.npy_file_data.shape[2], self.npy_file_data.shape[3]))
                    #Create fixed complete sequences
                    for i in range(num_sequences):
                        start_index = i*int(self.sequence_length_value)
                        end_index = start_index + int(self.sequence_length_value)
                        self.sequences[i] = self.npy_file_data[start_index:end_index]
                    self.sequences = self.sequences
                    print('Solo Seq Shape: ',self.sequences.shape)

                elif remaining_frames_percentage > 0: #INCOMPLETE SEQUENCES        
                    #DUPLICATE (IF CHECKED)
                    if self.duplicate_frames_check.get():
                        if remaining_frames_percentage>= float(self.duplicate_frames_threshold):
                            num_frames_to_duplicate = int(self.sequence_length_value) - remaining_frames
                            duplicated_sequences = self.npy_file_data[-num_frames_to_duplicate:]
                            merged_npy_file_data = np.concatenate((self.npy_file_data, duplicated_sequences), axis=0)
                            num_sequences = merged_npy_file_data.shape[0]//int(self.sequence_length_value)
                            # Initialize an empty array to store the sequences
                            
                            self.sequences = np.zeros((num_sequences, int(self.sequence_length_value), self.npy_file_data.shape[1], self.npy_file_data.shape[2], self.npy_file_data.shape[3]))
                            #Create fixed duplicated sequences
                            for i in range(num_sequences):
                                start_index = i*int(self.sequence_length_value)
                                end_index = start_index + int(self.sequence_length_value)
                                self.sequences[i] = merged_npy_file_data[start_index:end_index]
                            self.sequences = self.sequences
                            print('Solo Seq Shape: ',self.sequences.shape)

                        else:
                            #GET COMPLETE SEQUENCES ONLY
                            num_sequences = self.npy_file_data.shape[0]//int(self.sequence_length_value)
                            # Initialize an empty array to store the sequences
                            self.sequences = np.zeros((num_sequences, int(self.sequence_length_value), self.npy_file_data.shape[1], self.npy_file_data.shape[2], self.npy_file_data.shape[3]))
                            #Create fixed complete sequences
                            for i in range(num_sequences):
                                start_index = i*int(self.sequence_length_value)
                                end_index = start_index + int(self.sequence_length_value)
                                self.sequences[i] = self.npy_file_data[start_index:end_index]
                            self.sequences = self.sequences
                            print('Solo Seq Shape: ',self.sequences.shape)
        
                    else: #NO DUPLICATION
                        #SEQUENCES FORMATION
                        #GET COMPLETE SEQUENCES ONLY
                        num_sequences = self.npy_file_data.shape[0]//int(self.sequence_length_value)
                        # Initialize an empty array to store the sequences
                        self.sequences = np.zeros((num_sequences, int(self.sequence_length_value), self.npy_file_data.shape[1], self.npy_file_data.shape[2], self.npy_file_data.shape[3]))
                        #Create fixed complete sequences
                        for i in range(num_sequences):
                            start_index = i*int(self.sequence_length_value)
                            end_index = start_index + int(self.sequence_length_value)
                            self.sequences[i] = self.npy_file_data[start_index:end_index]
                        self.sequences = self.sequences
                        print('Solo Seq Shape: ',self.sequences.shape)
#FIXED SEQUENCES FORMATION END

    def window_seq_formation(self,class_npy_file_path):
        self.npy_file_data = []
        if self.action_type == 'interactions':
            self.npy_file_data = np.load(class_npy_file_path)
            print('NPY FILE DATA SHAPE: ',self.npy_file_data.shape)
            if self.npy_file_data.shape[0] >= int(self.sequence_length_value):
                #DROPPIND INDEXES
                for indx in self.kps_indices_to_drop:
                    self.npy_file_data = np.delete(self.npy_file_data,indx,axis=2)
                
                #SEQUENCES FORMATION
                num_sequences = self.npy_file_data.shape[0] - int(self.sequence_length_value) + 1
                # Initialize an empty array to store the sequences
                self.sequences = []
                self.sequences = np.zeros((num_sequences, int(self.sequence_length_value), self.npy_file_data.shape[1], self.npy_file_data.shape[2], self.npy_file_data.shape[3]))

                # Create consecutive sequences
                for i in range(num_sequences):
                    self.sequences[i] = self.npy_file_data[i:i+int(self.sequence_length_value)]
                self.sequences = self.sequences
                print('Interactions Seq Shape: ',self.sequences.shape)
                
                
        if self.action_type == 'solo':
            self.npy_file_data = np.load(class_npy_file_path)
            print('NPY FILE DATA SHAPE: ',self.npy_file_data.shape)
            if self.npy_file_data.shape[0] >= int(self.sequence_length_value):
                #RESHAPING
                self.npy_file_data = self.npy_file_data.reshape(self.npy_file_data.shape[0], 1, self.npy_file_data.shape[1], self.npy_file_data.shape[2])   
                zeros_array = []
                zeros_array = np.zeros_like(self.npy_file_data)
                # Concatenate along the second axis
                self.npy_file_data = np.concatenate((self.npy_file_data, zeros_array), axis=1)
                print('RESHAPED NPY FILE SHAPE: ',self.npy_file_data.shape)
                #print('ORG NPY DATA: ',self.npy_file_data[0][:][:][:])
                #DROPPIND INDEXES
                for indx in self.kps_indices_to_drop:
                    self.npy_file_data = np.delete(self.npy_file_data,indx,axis=2)
                # print('NPY FILE SHAPE AFTER INDICES DROP: ',self.npy_file_data.shape)
                # print('NPY DATA AFTER INDICES DROP: ',self.npy_file_data[0][:][:][:])
                #SEQUENCES FORMATION
                num_sequences = self.npy_file_data.shape[0] - int(self.sequence_length_value) + 1
                # Initialize an empty array to store the sequences
                self.sequences = []
                self.sequences = np.zeros((num_sequences, int(self.sequence_length_value), self.npy_file_data.shape[1], self.npy_file_data.shape[2], self.npy_file_data.shape[3]))

                # Create consecutive sequences
                for i in range(num_sequences):
                    self.sequences[i] = self.npy_file_data[i:i+int(self.sequence_length_value)]
                self.sequences = self.sequences
                print('Solo Seq Shape: ',self.sequences.shape)
                #self.sequences = np.concatenate(self.sequences,axis=0)

    def sequential_data_and_labels_formation(self):
        if self.window_seq_check.get():
            #NORMALIZED
            if self.normalized_keypoints_check.get():
            #if self.kps_type == 'normalized' and self.normalized_keypoints_check.get():
                #INTERACTIONS
                self.handshaking_sequences_n = np.concatenate(self.handshaking_sequences_n,axis=0)
                self.handshaking_labels_n = np.ones(self.handshaking_sequences_n.shape[0],dtype=int)*0

                self.hugging_sequences_n = np.concatenate(self.hugging_sequences_n,axis=0)
                self.hugging_labels_n = np.ones(self.hugging_sequences_n.shape[0],dtype=int)*1
                
                self.kicking_sequences_n = np.concatenate(self.kicking_sequences_n,axis=0)
                self.kicking_labels_n = np.ones(self.kicking_sequences_n.shape[0],dtype=int)*2    
                
                self.punching_sequences_n = np.concatenate(self.punching_sequences_n,axis=0)
                self.punching_labels_n = np.ones(self.punching_sequences_n.shape[0],dtype=int)*3    
                
                self.pushing_sequences_n = np.concatenate(self.pushing_sequences_n,axis=0)
                self.pushing_labels_n = np.ones(self.pushing_sequences_n.shape[0],dtype=int)*4    
                
                #SOLO ACTIONS
                self.clapping_solo_sequences_n = np.concatenate(self.clapping_solo_sequences_n,axis=0)
                self.clapping_solo_labels_n = np.ones(self.clapping_solo_sequences_n.shape[0],dtype=int)*5
                
                self.hitting_bottle_solo_sequences_n = np.concatenate(self.hitting_bottle_solo_sequences_n,axis=0)
                self.hitting_bottle_solo_labels_n = np.ones(self.hitting_bottle_solo_sequences_n.shape[0],dtype=int)*6

                self.hitting_stick_solo_sequences_n = np.concatenate(self.hitting_stick_solo_sequences_n,axis=0)
                self.hitting_stick_solo_labels_n = np.ones(self.hitting_stick_solo_sequences_n.shape[0],dtype=int)*7

                self.jogging_f_b_solo_sequences_n = np.concatenate(self.jogging_f_b_solo_sequences_n,axis=0)
                self.jogging_f_b_solo_labels_n = np.ones(self.jogging_f_b_solo_sequences_n.shape[0],dtype=int)*8

                self.jogging_side_solo_sequences_n = np.concatenate(self.jogging_side_solo_sequences_n,axis=0)
                self.jogging_side_solo_labels_n = np.ones(self.jogging_side_solo_sequences_n.shape[0],dtype=int)*9

                self.kicking_solo_sequences_n = np.concatenate(self.kicking_solo_sequences_n,axis=0)
                self.kicking_solo_labels_n = np.ones(self.kicking_solo_sequences_n.shape[0],dtype=int)*10

                self.punching_solo_sequences_n = np.concatenate(self.punching_solo_sequences_n,axis=0)
                self.punching_solo_labels_n = np.ones(self.punching_solo_sequences_n.shape[0],dtype=int)*11

                self.running_f_b_solo_sequences_n = np.concatenate(self.running_f_b_solo_sequences_n,axis=0)
                self.running_f_b_solo_labels_n = np.ones(self.running_f_b_solo_sequences_n.shape[0],dtype=int)*12

                self.running_side_solo_sequences_n = np.concatenate(self.running_side_solo_sequences_n,axis=0)
                self.running_side_solo_labels_n = np.ones(self.running_side_solo_sequences_n.shape[0],dtype=int)*13

                self.stabbing_solo_sequences_n = np.concatenate(self.stabbing_solo_sequences_n,axis=0)
                self.stabbing_solo_labels_n = np.ones(self.stabbing_solo_sequences_n.shape[0],dtype=int)*14

                self.walking_f_b_solo_sequences_n = np.concatenate(self.walking_f_b_solo_sequences_n,axis=0)
                self.walking_f_b_solo_labels_n = np.ones(self.walking_f_b_solo_sequences_n.shape[0],dtype=int)*15

                self.walking_side_solo_sequences_n = np.concatenate(self.walking_side_solo_sequences_n,axis=0)
                self.walking_side_solo_labels_n = np.ones(self.walking_side_solo_sequences_n.shape[0],dtype=int)*16

                self.waving_hands_solo_sequences_n = np.concatenate(self.waving_hands_solo_sequences_n,axis=0)
                self.waving_hands_solo_labels_n = np.ones(self.waving_hands_solo_sequences_n.shape[0],dtype=int)*17

                #PRINTING
                print('NORMALIZED SEQUENTIAL DATA: ')
                #INTERACTIONS
                print('Handshaking Sequntial Data Shape: ',self.handshaking_sequences_n.shape)
                print('Handshaking Labels Shape: ', len(self.handshaking_labels_n))

                print('Hugging Sequntial Data Shape: ',self.hugging_sequences_n.shape)
                print('Hugging Labels Shape: ', len(self.hugging_labels_n))

                print('Kicking Sequntial Data Shape: ',self.kicking_sequences_n.shape)
                print('Kicking Labels Shape: ', len(self.kicking_labels_n))

                print('Punching Sequntial Data Shape: ',self.punching_sequences_n.shape)
                print('Punching Labels Shape: ', len(self.punching_labels_n))

                print('Pushing Sequntial Data Shape: ',self.pushing_sequences_n.shape)
                print('Pushing Labels Shape: ', len(self.pushing_labels_n))

                #SOLO ACTIONS
                print('Clapping Solo Sequntial Data Shape: ',self.clapping_solo_sequences_n.shape)
                print('Clapping Solo Labels Shape: ', len(self.clapping_solo_labels_n))

                print('Hitting Bottle Solo Sequntial Data Shape: ',self.hitting_bottle_solo_sequences_n.shape)
                print('Hitting Bottle Solo Labels Shape: ', len(self.hitting_bottle_solo_labels_n))

                print('Hitting Stick Solo Sequntial Data Shape: ',self.hitting_stick_solo_sequences_n.shape)
                print('Hitting Stick Solo Labels Shape: ', len(self.hitting_stick_solo_labels_n))

                print('Jogging FB Solo Sequntial Data Shape: ',self.jogging_f_b_solo_sequences_n.shape)
                print('Jogging FB Solo Labels Shape: ', len(self.jogging_f_b_solo_labels_n))

                print('Jogging Side Solo Sequntial Data Shape: ',self.jogging_side_solo_sequences_n.shape)
                print('Jogging Side Solo Labels Shape: ', len(self.jogging_side_solo_labels_n))

                print('Kicking Solo Sequntial Data Shape: ',self.kicking_solo_sequences_n.shape)
                print('Kicking Solo Labels Shape: ', len(self.kicking_solo_labels_n))

                print('Punching Solo Sequntial Data Shape: ',self.punching_solo_sequences_n.shape)
                print('Punching Solo Labels Shape: ', len(self.punching_solo_labels_n))

                print('Running FB Solo Sequntial Data Shape: ',self.running_f_b_solo_sequences_n.shape)
                print('Running FB Solo Labels Shape: ', len(self.running_f_b_solo_labels_n))

                print('Running Side Solo Sequntial Data Shape: ',self.running_side_solo_sequences_n.shape)
                print('Running Side Solo Labels Shape: ', len(self.running_side_solo_labels_n))

                print('Stabbing Solo Sequntial Data Shape: ',self.stabbing_solo_sequences_n.shape)
                print('Stabbing Solo Labels Shape: ', len(self.stabbing_solo_labels_n))

                print('Walking FB Solo Sequntial Data Shape: ',self.walking_f_b_solo_sequences_n.shape)
                print('Walking FB Solo Labels Shape: ', len(self.walking_f_b_solo_labels_n))

                print('Walking Side Solo Sequntial Data Shape: ',self.walking_side_solo_sequences_n.shape)
                print('Walking Side Solo Labels Shape: ', len(self.walking_side_solo_labels_n))

                print('Waving Hands Solo Sequntial Data Shape: ',self.waving_hands_solo_sequences_n.shape)
                print('Waving Hands Solo Labels Shape: ', len(self.waving_hands_solo_labels_n))

                classses_seq_lists_n = [
                self.handshaking_sequences_n.shape[0],
                self.hugging_sequences_n.shape[0],
                self.kicking_sequences_n.shape[0],
                self.punching_sequences_n.shape[0],
                self.pushing_sequences_n.shape[0],
                self.clapping_solo_sequences_n.shape[0],
                self.hitting_bottle_solo_sequences_n.shape[0],
                self.hitting_stick_solo_sequences_n.shape[0],
                self.jogging_f_b_solo_sequences_n.shape[0],
                self.jogging_side_solo_sequences_n.shape[0],
                self.kicking_solo_sequences_n.shape[0],
                self.punching_solo_sequences_n.shape[0],
                self.running_f_b_solo_sequences_n.shape[0],
                self.running_side_solo_sequences_n.shape[0],
                self.stabbing_solo_sequences_n.shape[0],
                self.walking_f_b_solo_sequences_n.shape[0],
                self.walking_side_solo_sequences_n.shape[0],
                self.waving_hands_solo_sequences_n.shape[0]]
                balanced_seq_num_n = min(classses_seq_lists_n)
                #IMBALANCED DATA
                imb_Xdata_n = np.concatenate((self.handshaking_sequences_n,self.hugging_sequences_n, 
                                            self.kicking_sequences_n,self.punching_sequences_n, self.pushing_sequences_n, 
                                            self.clapping_solo_sequences_n,self.hitting_bottle_solo_sequences_n,
                                            self.hitting_stick_solo_sequences_n,self.jogging_f_b_solo_sequences_n,
                                            self.jogging_side_solo_sequences_n,self.kicking_solo_sequences_n,
                                            self.punching_solo_sequences_n,self.running_f_b_solo_sequences_n,
                                            self.running_side_solo_sequences_n,self.stabbing_solo_sequences_n,
                                            self.walking_f_b_solo_sequences_n,self.walking_side_solo_sequences_n,
                                            self.waving_hands_solo_sequences_n),axis=0)
                #RESHAPING TO DESIRED SHAPE
                num_sequences, seq_length, num_persons, num_kps, num_coords = imb_Xdata_n.shape
                imb_Xdata_n = imb_Xdata_n.reshape(num_sequences,seq_length, num_persons*num_kps*num_coords)
                imb_Ydata_n = np.concatenate((self.handshaking_labels_n,self.hugging_labels_n,
                                            self.kicking_labels_n, self.punching_labels_n,self.pushing_labels_n, 
                                            self.clapping_solo_labels_n, self.hitting_bottle_solo_labels_n,
                                            self.hitting_stick_solo_labels_n,self.jogging_f_b_solo_labels_n,
                                            self.jogging_side_solo_labels_n,self.kicking_solo_labels_n,
                                            self.punching_solo_labels_n,self.running_f_b_solo_labels_n,
                                            self.running_side_solo_labels_n,self.stabbing_solo_labels_n,
                                            self.walking_f_b_solo_labels_n,self.walking_side_solo_labels_n,
                                            self.waving_hands_solo_labels_n))
                #BALANCING SEQUENCES ACROSS ALL CLASSES
                self.handshaking_sequences_n = self.handshaking_sequences_n[:balanced_seq_num_n]
                self.hugging_sequences_n = self.hugging_sequences_n[:balanced_seq_num_n]
                self.kicking_sequences_n = self.kicking_sequences_n[:balanced_seq_num_n]
                self.punching_sequences_n = self.punching_sequences_n[:balanced_seq_num_n]
                self.pushing_sequences_n = self.pushing_sequences_n[:balanced_seq_num_n]
                self.clapping_solo_sequences_n = self.clapping_solo_sequences_n[:balanced_seq_num_n]
                self.hitting_bottle_solo_sequences_n = self.hitting_bottle_solo_sequences_n[:balanced_seq_num_n]
                self.hitting_stick_solo_sequences_n = self.hitting_stick_solo_sequences_n[:balanced_seq_num_n]
                self.jogging_f_b_solo_sequences_n = self.jogging_f_b_solo_sequences_n[:balanced_seq_num_n]
                self.jogging_side_solo_sequences_n = self.jogging_side_solo_sequences_n[:balanced_seq_num_n]
                self.kicking_solo_sequences_n = self.kicking_solo_sequences_n[:balanced_seq_num_n]
                self.punching_solo_sequences_n = self.punching_solo_sequences_n[:balanced_seq_num_n]
                self.running_f_b_solo_sequences_n = self.running_f_b_solo_sequences_n[:balanced_seq_num_n]
                self.running_side_solo_sequences_n = self.running_side_solo_sequences_n[:balanced_seq_num_n]
                self.stabbing_solo_sequences_n = self.stabbing_solo_sequences_n[:balanced_seq_num_n]
                self.walking_f_b_solo_sequences_n = self.walking_f_b_solo_sequences_n[:balanced_seq_num_n]
                self.walking_side_solo_sequences_n = self.walking_side_solo_sequences_n[:balanced_seq_num_n]
                self.waving_hands_solo_sequences_n = self.waving_hands_solo_sequences_n[:balanced_seq_num_n]
                #BALANCING LABELS ACROSS ALL CLASSES
                self.handshaking_labels_n = self.handshaking_labels_n[:balanced_seq_num_n]
                self.hugging_labels_n = self.hugging_labels_n[:balanced_seq_num_n]
                self.kicking_labels_n = self.kicking_labels_n[:balanced_seq_num_n]
                self.punching_labels_n = self.punching_labels_n[:balanced_seq_num_n]
                self.pushing_labels_n = self.pushing_labels_n[:balanced_seq_num_n]
                self.clapping_solo_labels_n = self.clapping_solo_labels_n[:balanced_seq_num_n]
                self.hitting_bottle_solo_labels_n = self.hitting_bottle_solo_labels_n[:balanced_seq_num_n]
                self.hitting_stick_solo_labels_n = self.hitting_stick_solo_labels_n[:balanced_seq_num_n]
                self.jogging_f_b_solo_labels_n = self.jogging_f_b_solo_labels_n[:balanced_seq_num_n]
                self.jogging_side_solo_labels_n = self.jogging_side_solo_labels_n[:balanced_seq_num_n]
                self.kicking_solo_labels_n = self.kicking_solo_labels_n[:balanced_seq_num_n]
                self.punching_solo_labels_n = self.punching_solo_labels_n[:balanced_seq_num_n]
                self.running_f_b_solo_labels_n = self.running_f_b_solo_labels_n[:balanced_seq_num_n]
                self.running_side_solo_labels_n = self.running_side_solo_labels_n[:balanced_seq_num_n]
                self.stabbing_solo_labels_n = self.stabbing_solo_labels_n[:balanced_seq_num_n]
                self.walking_f_b_solo_labels_n = self.walking_f_b_solo_labels_n[:balanced_seq_num_n]
                self.walking_side_solo_labels_n = self.walking_side_solo_labels_n[:balanced_seq_num_n]
                self.waving_hands_solo_labels_n = self.waving_hands_solo_labels_n[:balanced_seq_num_n]
                bal_Xdata_n = np.concatenate((self.handshaking_sequences_n,self.hugging_sequences_n, 
                                            self.kicking_sequences_n,self.punching_sequences_n, self.pushing_sequences_n, 
                                            self.clapping_solo_sequences_n,self.hitting_bottle_solo_sequences_n,
                                            self.hitting_stick_solo_sequences_n,self.jogging_f_b_solo_sequences_n,
                                            self.jogging_side_solo_sequences_n,self.kicking_solo_sequences_n,
                                            self.punching_solo_sequences_n,self.running_f_b_solo_sequences_n,
                                            self.running_side_solo_sequences_n,self.stabbing_solo_sequences_n,
                                            self.walking_f_b_solo_sequences_n,self.walking_side_solo_sequences_n,
                                            self.waving_hands_solo_sequences_n),axis=0)
                
                #RESHAPING TO DESIRED SHAPE
                num_sequences, seq_length, num_persons, num_kps, num_coords = bal_Xdata_n.shape
                bal_Xdata_n = bal_Xdata_n.reshape(num_sequences,seq_length, num_persons*num_kps*num_coords)
                bal_Ydata_n = np.concatenate((self.handshaking_labels_n,self.hugging_labels_n,
                                            self.kicking_labels_n, self.punching_labels_n,self.pushing_labels_n, 
                                            self.clapping_solo_labels_n,self.hitting_bottle_solo_labels_n,
                                            self.hitting_stick_solo_labels_n,self.jogging_f_b_solo_labels_n,
                                            self.jogging_side_solo_labels_n,self.kicking_solo_labels_n,
                                            self.punching_solo_labels_n,self.running_f_b_solo_labels_n,
                                            self.running_side_solo_labels_n,self.stabbing_solo_labels_n,
                                            self.walking_f_b_solo_labels_n,self.walking_side_solo_labels_n,
                                            self.waving_hands_solo_labels_n))
                #Xtrain, Xtest and Xval
                # Split the data into training and testing sets, ensuring balanced samples
                imb_X_train_n, imb_X_testval_n, imb_y_train_n, imb_y_testval_n = train_test_split(imb_Xdata_n, imb_Ydata_n, test_size=1-float(self.trainset_percent), stratify=imb_Ydata_n)           
                imb_X_test_n, imb_X_val_n, imb_y_test_n, imb_y_val_n = train_test_split(imb_X_testval_n, imb_y_testval_n, test_size=float(self.validationset_percent), stratify=imb_y_testval_n)
                bal_X_train_n, bal_X_testval_n, bal_y_train_n, bal_y_testval_n = train_test_split(bal_Xdata_n, bal_Ydata_n, test_size=1-float(self.trainset_percent), stratify=bal_Ydata_n)           
                bal_X_test_n, bal_X_val_n, bal_y_test_n, bal_y_val_n = train_test_split(bal_X_testval_n, bal_y_testval_n, test_size=float(self.validationset_percent), stratify=bal_y_testval_n)
                print('Imbalance Xdata Shape: ',imb_Xdata_n.shape)
                print('Imbalance Ydata Shape: ',imb_Ydata_n.shape)
                print('Imbalance Xtrain Shape: ',imb_X_train_n.shape)
                print('Imbalance Ytrain Shape: ',imb_y_train_n.shape)
                print('Imbalance Xtest Shape: ',imb_X_test_n.shape)
                print('Imbalance Ytest Shape: ',imb_y_test_n.shape)
                print('Imbalance Xval Shape: ',imb_X_val_n.shape)
                print('Imbalance Yval Shape: ',imb_y_val_n.shape)
                
                print('Balanced Sequence Num (Samples per class): ',balanced_seq_num_n)
                print('Balanced Xdata Shape: ',bal_Xdata_n.shape)
                print('Balanced Ydata Shape: ',bal_Ydata_n.shape)
                print('Balanced Xtrain Shape: ',bal_X_train_n.shape)
                print('Balanced Ytrain Shape: ',bal_y_train_n.shape)
                print('Balanced Xtest Shape: ',bal_X_test_n.shape)
                print('Balanced Ytest Shape: ',bal_y_test_n.shape)
                print('Balanced Xval Shape: ',bal_X_val_n.shape)
                print('Balanced Yval Shape: ',bal_y_val_n.shape)
                #Saving Xdata Ydata
                window_imb_folder_n = 'Normalized\Sliding_Window\Imbalanced_Sequences'
                window_bal_folder_n = 'Normalized\Sliding_Window\Balanced_Sequences'
                window_imb_path_n = os.path.join(self.output_transformation_dir,window_imb_folder_n)
                window_bal_path_n = os.path.join(self.output_transformation_dir,window_bal_folder_n)
                # Check if the directory exists
                if not os.path.exists(window_imb_path_n) and not os.path.exists(window_bal_path_n):
                    # If not, create it
                    os.makedirs(window_imb_path_n)
                    os.makedirs(window_bal_path_n)
                np.save(os.path.join(window_imb_path_n, "X_data.npy"), imb_Xdata_n)
                np.save(os.path.join(window_imb_path_n, "Y_data.npy"), imb_Ydata_n)
                np.save(os.path.join(window_imb_path_n, "X_train.npy"), imb_X_train_n)
                np.save(os.path.join(window_imb_path_n, "Y_train.npy"), imb_y_train_n)
                np.save(os.path.join(window_imb_path_n, "X_test.npy"), imb_X_test_n)
                np.save(os.path.join(window_imb_path_n, "Y_test.npy"), imb_y_test_n)
                np.save(os.path.join(window_imb_path_n, "X_val.npy"), imb_X_val_n)
                np.save(os.path.join(window_imb_path_n, "Y_val.npy"), imb_y_val_n)
                
                np.save(os.path.join(window_bal_path_n, "X_data.npy"), bal_Xdata_n)
                np.save(os.path.join(window_bal_path_n, "Y_data.npy"), bal_Ydata_n)
                np.save(os.path.join(window_bal_path_n, "X_train.npy"), bal_X_train_n)
                np.save(os.path.join(window_bal_path_n, "Y_train.npy"), bal_y_train_n)
                np.save(os.path.join(window_bal_path_n, "X_test.npy"), bal_X_test_n)
                np.save(os.path.join(window_bal_path_n, "Y_test.npy"), bal_y_test_n)
                np.save(os.path.join(window_bal_path_n, "X_val.npy"), bal_X_val_n)
                np.save(os.path.join(window_bal_path_n, "Y_val.npy"), bal_y_val_n)

            #GLOBAL
            if self.global_keypoints_check.get():
            #if self.kps_type == 'global' and self.global_keypoints_check.get():
                #INTERACTIONS
                self.handshaking_sequences_g = np.concatenate(self.handshaking_sequences_g,axis=0)
                self.handshaking_labels_g = np.ones(self.handshaking_sequences_g.shape[0],dtype=int)*0

                self.hugging_sequences_g = np.concatenate(self.hugging_sequences_g,axis=0)
                self.hugging_labels_g = np.ones(self.hugging_sequences_g.shape[0],dtype=int)*1
                
                self.kicking_sequences_g = np.concatenate(self.kicking_sequences_g,axis=0)
                self.kicking_labels_g = np.ones(self.kicking_sequences_g.shape[0],dtype=int)*2    
                
                self.punching_sequences_g = np.concatenate(self.punching_sequences_g,axis=0)
                self.punching_labels_g = np.ones(self.punching_sequences_g.shape[0],dtype=int)*3    
                
                self.pushing_sequences_g = np.concatenate(self.pushing_sequences_g,axis=0)
                self.pushing_labels_g = np.ones(self.pushing_sequences_g.shape[0],dtype=int)*4    
                
                #SOLO ACTIONS
                self.clapping_solo_sequences_g = np.concatenate(self.clapping_solo_sequences_g,axis=0)
                self.clapping_solo_labels_g = np.ones(self.clapping_solo_sequences_g.shape[0],dtype=int)*5
                
                self.hitting_bottle_solo_sequences_g = np.concatenate(self.hitting_bottle_solo_sequences_g,axis=0)
                self.hitting_bottle_solo_labels_g = np.ones(self.hitting_bottle_solo_sequences_g.shape[0],dtype=int)*6

                self.hitting_stick_solo_sequences_g = np.concatenate(self.hitting_stick_solo_sequences_g,axis=0)
                self.hitting_stick_solo_labels_g = np.ones(self.hitting_stick_solo_sequences_g.shape[0],dtype=int)*7

                self.jogging_f_b_solo_sequences_g = np.concatenate(self.jogging_f_b_solo_sequences_g,axis=0)
                self.jogging_f_b_solo_labels_g = np.ones(self.jogging_f_b_solo_sequences_g.shape[0],dtype=int)*8

                self.jogging_side_solo_sequences_g = np.concatenate(self.jogging_side_solo_sequences_g,axis=0)
                self.jogging_side_solo_labels_g = np.ones(self.jogging_side_solo_sequences_g.shape[0],dtype=int)*9

                self.kicking_solo_sequences_g = np.concatenate(self.kicking_solo_sequences_g,axis=0)
                self.kicking_solo_labels_g = np.ones(self.kicking_solo_sequences_g.shape[0],dtype=int)*10

                self.punching_solo_sequences_g = np.concatenate(self.punching_solo_sequences_g,axis=0)
                self.punching_solo_labels_g = np.ones(self.punching_solo_sequences_g.shape[0],dtype=int)*11

                self.running_f_b_solo_sequences_g = np.concatenate(self.running_f_b_solo_sequences_g,axis=0)
                self.running_f_b_solo_labels_g = np.ones(self.running_f_b_solo_sequences_g.shape[0],dtype=int)*12

                self.running_side_solo_sequences_g = np.concatenate(self.running_side_solo_sequences_g,axis=0)
                self.running_side_solo_labels_g = np.ones(self.running_side_solo_sequences_g.shape[0],dtype=int)*13

                self.stabbing_solo_sequences_g = np.concatenate(self.stabbing_solo_sequences_g,axis=0)
                self.stabbing_solo_labels_g = np.ones(self.stabbing_solo_sequences_g.shape[0],dtype=int)*14

                self.walking_f_b_solo_sequences_g = np.concatenate(self.walking_f_b_solo_sequences_g,axis=0)
                self.walking_f_b_solo_labels_g = np.ones(self.walking_f_b_solo_sequences_g.shape[0],dtype=int)*15

                self.walking_side_solo_sequences_g = np.concatenate(self.walking_side_solo_sequences_g,axis=0)
                self.walking_side_solo_labels_g = np.ones(self.walking_side_solo_sequences_g.shape[0],dtype=int)*16

                self.waving_hands_solo_sequences_g = np.concatenate(self.waving_hands_solo_sequences_g,axis=0)
                self.waving_hands_solo_labels_g = np.ones(self.waving_hands_solo_sequences_g.shape[0],dtype=int)*17

                #PRINTING
                print('GLOBAL SEQUENTIAL DATA: ')
                #INTERACTIONS
                print('Handshaking Sequntial Data Shape: ',self.handshaking_sequences_g.shape)
                print('Handshaking Labels Shape: ', len(self.handshaking_labels_g))

                print('Hugging Sequntial Data Shape: ',self.hugging_sequences_g.shape)
                print('Hugging Labels Shape: ', len(self.hugging_labels_g))

                print('Kicking Sequntial Data Shape: ',self.kicking_sequences_g.shape)
                print('Kicking Labels Shape: ', len(self.kicking_labels_g))

                print('Punching Sequntial Data Shape: ',self.punching_sequences_g.shape)
                print('Punching Labels Shape: ', len(self.punching_labels_g))

                print('Pushing Sequntial Data Shape: ',self.pushing_sequences_g.shape)
                print('Pushing Labels Shape: ', len(self.pushing_labels_g))

                #SOLO ACTIONS
                print('Clapping Solo Sequntial Data Shape: ',self.clapping_solo_sequences_g.shape)
                print('Clapping Solo Labels Shape: ', len(self.clapping_solo_labels_g))

                print('Hitting Bottle Solo Sequntial Data Shape: ',self.hitting_bottle_solo_sequences_g.shape)
                print('Hitting Bottle Solo Labels Shape: ', len(self.hitting_bottle_solo_labels_g))

                print('Hitting Stick Solo Sequntial Data Shape: ',self.hitting_stick_solo_sequences_g.shape)
                print('Hitting Stick Solo Labels Shape: ', len(self.hitting_stick_solo_labels_g))

                print('Jogging FB Solo Sequntial Data Shape: ',self.jogging_f_b_solo_sequences_g.shape)
                print('Jogging FB Solo Labels Shape: ', len(self.jogging_f_b_solo_labels_g))

                print('Jogging Side Solo Sequntial Data Shape: ',self.jogging_side_solo_sequences_g.shape)
                print('Jogging Side Solo Labels Shape: ', len(self.jogging_side_solo_labels_g))

                print('Kicking Solo Sequntial Data Shape: ',self.kicking_solo_sequences_g.shape)
                print('Kicking Solo Labels Shape: ', len(self.kicking_solo_labels_g))

                print('Punching Solo Sequntial Data Shape: ',self.punching_solo_sequences_g.shape)
                print('Punching Solo Labels Shape: ', len(self.punching_solo_labels_g))

                print('Running FB Solo Sequntial Data Shape: ',self.running_f_b_solo_sequences_g.shape)
                print('Running FB Solo Labels Shape: ', len(self.running_f_b_solo_labels_g))

                print('Running Side Solo Sequntial Data Shape: ',self.running_side_solo_sequences_g.shape)
                print('Running Side Solo Labels Shape: ', len(self.running_side_solo_labels_g))

                print('Stabbing Solo Sequntial Data Shape: ',self.stabbing_solo_sequences_g.shape)
                print('Stabbing Solo Labels Shape: ', len(self.stabbing_solo_labels_g))

                print('Walking FB Solo Sequntial Data Shape: ',self.walking_f_b_solo_sequences_g.shape)
                print('Walking FB Solo Labels Shape: ', len(self.walking_f_b_solo_labels_g))

                print('Walking Side Solo Sequntial Data Shape: ',self.walking_side_solo_sequences_g.shape)
                print('Walking Side Solo Labels Shape: ', len(self.walking_side_solo_labels_g))

                print('Waving Hands Solo Sequntial Data Shape: ',self.waving_hands_solo_sequences_g.shape)
                print('Waving Hands Solo Labels Shape: ', len(self.waving_hands_solo_labels_g))

                classses_seq_lists_g = [
                self.handshaking_sequences_g.shape[0],
                self.hugging_sequences_g.shape[0],
                self.kicking_sequences_g.shape[0],
                self.punching_sequences_g.shape[0],
                self.pushing_sequences_g.shape[0],
                self.clapping_solo_sequences_g.shape[0],
                self.hitting_bottle_solo_sequences_g.shape[0],
                self.hitting_stick_solo_sequences_g.shape[0],
                self.jogging_f_b_solo_sequences_g.shape[0],
                self.jogging_side_solo_sequences_g.shape[0],
                self.kicking_solo_sequences_g.shape[0],
                self.punching_solo_sequences_g.shape[0],
                self.running_f_b_solo_sequences_g.shape[0],
                self.running_side_solo_sequences_g.shape[0],
                self.stabbing_solo_sequences_g.shape[0],
                self.walking_f_b_solo_sequences_g.shape[0],
                self.walking_side_solo_sequences_g.shape[0],
                self.waving_hands_solo_sequences_g.shape[0]]
                balanced_seq_num_g = min(classses_seq_lists_g)
                #IMBALANCED DATA
                imb_Xdata_g = np.concatenate((self.handshaking_sequences_g,self.hugging_sequences_g, 
                                            self.kicking_sequences_g,self.punching_sequences_g, self.pushing_sequences_g, 
                                            self.clapping_solo_sequences_g,self.hitting_bottle_solo_sequences_g,
                                            self.hitting_stick_solo_sequences_g,self.jogging_f_b_solo_sequences_g,
                                            self.jogging_side_solo_sequences_g,self.kicking_solo_sequences_g,
                                            self.punching_solo_sequences_g,self.running_f_b_solo_sequences_g,
                                            self.running_side_solo_sequences_g,self.stabbing_solo_sequences_g,
                                            self.walking_f_b_solo_sequences_g,self.walking_side_solo_sequences_g,
                                            self.waving_hands_solo_sequences_g),axis=0)
                #RESHAPING TO DESIRED SHAPE
                num_sequences, seq_length, num_persons, num_kps, num_coords = imb_Xdata_g.shape
                imb_Xdata_g = imb_Xdata_g.reshape(num_sequences,seq_length, num_persons*num_kps*num_coords)
                imb_Ydata_g = np.concatenate((self.handshaking_labels_g,self.hugging_labels_g,
                                            self.kicking_labels_g, self.punching_labels_g,self.pushing_labels_g, 
                                            self.clapping_solo_labels_g, self.hitting_bottle_solo_labels_g,
                                            self.hitting_stick_solo_labels_g,self.jogging_f_b_solo_labels_g,
                                            self.jogging_side_solo_labels_g,self.kicking_solo_labels_g,
                                            self.punching_solo_labels_g,self.running_f_b_solo_labels_g,
                                            self.running_side_solo_labels_g,self.stabbing_solo_labels_g,
                                            self.walking_f_b_solo_labels_g,self.walking_side_solo_labels_g,
                                            self.waving_hands_solo_labels_g))
                #BALANCING SEQUENCES ACROSS ALL CLASSES
                self.handshaking_sequences_g = self.handshaking_sequences_g[:balanced_seq_num_g]
                self.hugging_sequences_g = self.hugging_sequences_g[:balanced_seq_num_g]
                self.kicking_sequences_g = self.kicking_sequences_g[:balanced_seq_num_g]
                self.punching_sequences_g = self.punching_sequences_g[:balanced_seq_num_g]
                self.pushing_sequences_g = self.pushing_sequences_g[:balanced_seq_num_g]
                self.clapping_solo_sequences_g = self.clapping_solo_sequences_g[:balanced_seq_num_g]
                self.hitting_bottle_solo_sequences_g = self.hitting_bottle_solo_sequences_g[:balanced_seq_num_g]
                self.hitting_stick_solo_sequences_g = self.hitting_stick_solo_sequences_g[:balanced_seq_num_g]
                self.jogging_f_b_solo_sequences_g = self.jogging_f_b_solo_sequences_g[:balanced_seq_num_g]
                self.jogging_side_solo_sequences_g = self.jogging_side_solo_sequences_g[:balanced_seq_num_g]
                self.kicking_solo_sequences_g = self.kicking_solo_sequences_g[:balanced_seq_num_g]
                self.punching_solo_sequences_g = self.punching_solo_sequences_g[:balanced_seq_num_g]
                self.running_f_b_solo_sequences_g = self.running_f_b_solo_sequences_g[:balanced_seq_num_g]
                self.running_side_solo_sequences_g = self.running_side_solo_sequences_g[:balanced_seq_num_g]
                self.stabbing_solo_sequences_g = self.stabbing_solo_sequences_g[:balanced_seq_num_g]
                self.walking_f_b_solo_sequences_g = self.walking_f_b_solo_sequences_g[:balanced_seq_num_g]
                self.walking_side_solo_sequences_g = self.walking_side_solo_sequences_g[:balanced_seq_num_g]
                self.waving_hands_solo_sequences_g = self.waving_hands_solo_sequences_g[:balanced_seq_num_g]
                #BALANCING LABELS ACROSS ALL CLASSES
                self.handshaking_labels_g = self.handshaking_labels_g[:balanced_seq_num_g]
                self.hugging_labels_g = self.hugging_labels_g[:balanced_seq_num_g]
                self.kicking_labels_g = self.kicking_labels_g[:balanced_seq_num_g]
                self.punching_labels_g = self.punching_labels_g[:balanced_seq_num_g]
                self.pushing_labels_g = self.pushing_labels_g[:balanced_seq_num_g]
                self.clapping_solo_labels_g = self.clapping_solo_labels_g[:balanced_seq_num_g]
                self.hitting_bottle_solo_labels_g = self.hitting_bottle_solo_labels_g[:balanced_seq_num_g]
                self.hitting_stick_solo_labels_g = self.hitting_stick_solo_labels_g[:balanced_seq_num_g]
                self.jogging_f_b_solo_labels_g = self.jogging_f_b_solo_labels_g[:balanced_seq_num_g]
                self.jogging_side_solo_labels_g = self.jogging_side_solo_labels_g[:balanced_seq_num_g]
                self.kicking_solo_labels_g = self.kicking_solo_labels_g[:balanced_seq_num_g]
                self.punching_solo_labels_g = self.punching_solo_labels_g[:balanced_seq_num_g]
                self.running_f_b_solo_labels_g = self.running_f_b_solo_labels_g[:balanced_seq_num_g]
                self.running_side_solo_labels_g = self.running_side_solo_labels_g[:balanced_seq_num_g]
                self.stabbing_solo_labels_g = self.stabbing_solo_labels_g[:balanced_seq_num_g]
                self.walking_f_b_solo_labels_g = self.walking_f_b_solo_labels_g[:balanced_seq_num_g]
                self.walking_side_solo_labels_g = self.walking_side_solo_labels_g[:balanced_seq_num_g]
                self.waving_hands_solo_labels_g = self.waving_hands_solo_labels_g[:balanced_seq_num_g]
                bal_Xdata_g = np.concatenate((self.handshaking_sequences_g,self.hugging_sequences_g, 
                                            self.kicking_sequences_g,self.punching_sequences_g, self.pushing_sequences_g, 
                                            self.clapping_solo_sequences_g,self.hitting_bottle_solo_sequences_g,
                                            self.hitting_stick_solo_sequences_g,self.jogging_f_b_solo_sequences_g,
                                            self.jogging_side_solo_sequences_g,self.kicking_solo_sequences_g,
                                            self.punching_solo_sequences_g,self.running_f_b_solo_sequences_g,
                                            self.running_side_solo_sequences_g,self.stabbing_solo_sequences_g,
                                            self.walking_f_b_solo_sequences_g,self.walking_side_solo_sequences_g,
                                            self.waving_hands_solo_sequences_g),axis=0)
                #RESHAPING TO DESIRED SHAPE
                num_sequences, seq_length, num_persons, num_kps, num_coords = bal_Xdata_g.shape
                bal_Xdata_g = bal_Xdata_g.reshape(num_sequences,seq_length, num_persons*num_kps*num_coords)
                bal_Ydata_g = np.concatenate((self.handshaking_labels_g,self.hugging_labels_g,
                                            self.kicking_labels_g, self.punching_labels_g,self.pushing_labels_g, 
                                            self.clapping_solo_labels_g,self.hitting_bottle_solo_labels_g,
                                            self.hitting_stick_solo_labels_g,self.jogging_f_b_solo_labels_g,
                                            self.jogging_side_solo_labels_g,self.kicking_solo_labels_g,
                                            self.punching_solo_labels_g,self.running_f_b_solo_labels_g,
                                            self.running_side_solo_labels_g,self.stabbing_solo_labels_g,
                                            self.walking_f_b_solo_labels_g,self.walking_side_solo_labels_g,
                                            self.waving_hands_solo_labels_g))
                #Xtrain, Xtest and Xval
                # Split the data into training and testing sets, ensuring balanced samples
                imb_X_train_g, imb_X_testval_g, imb_y_train_g, imb_y_testval_g = train_test_split(imb_Xdata_g, imb_Ydata_g, test_size=1-float(self.trainset_percent), stratify=imb_Ydata_g)           
                imb_X_test_g, imb_X_val_g, imb_y_test_g, imb_y_val_g = train_test_split(imb_X_testval_g, imb_y_testval_g, test_size=float(self.validationset_percent), stratify=imb_y_testval_g)
                bal_X_train_g, bal_X_testval_g, bal_y_train_g, bal_y_testval_g = train_test_split(bal_Xdata_g, bal_Ydata_g, test_size=1-float(self.trainset_percent), stratify=bal_Ydata_g)           
                bal_X_test_g, bal_X_val_g, bal_y_test_g, bal_y_val_g = train_test_split(bal_X_testval_g, bal_y_testval_g, test_size=float(self.validationset_percent), stratify=bal_y_testval_g)
                print('Imbalance Xdata Shape: ',imb_Xdata_g.shape)
                print('Imbalance Ydata Shape: ',imb_Ydata_g.shape)
                print('Imbalance Xtrain Shape: ',imb_X_train_g.shape)
                print('Imbalance Ytrain Shape: ',imb_y_train_g.shape)
                print('Imbalance Xtest Shape: ',imb_X_test_g.shape)
                print('Imbalance Ytest Shape: ',imb_y_test_g.shape)
                print('Imbalance Xval Shape: ',imb_X_val_g.shape)
                print('Imbalance Yval Shape: ',imb_y_val_g.shape)
                
                print('Balanced Sequence Num (Samples per class): ',balanced_seq_num_g)
                print('Balanced Xdata Shape: ',bal_Xdata_g.shape)
                print('Balanced Ydata Shape: ',bal_Ydata_g.shape)
                print('Balanced Xtrain Shape: ',bal_X_train_g.shape)
                print('Balanced Ytrain Shape: ',bal_y_train_g.shape)
                print('Balanced Xtest Shape: ',bal_X_test_g.shape)
                print('Balanced Ytest Shape: ',bal_y_test_g.shape)
                print('Balanced Xval Shape: ',bal_X_val_g.shape)
                print('Balanced Yval Shape: ',bal_y_val_g.shape)
                #Saving Xdata Ydata
                window_imb_folder_g = 'Global\Sliding_Window\Imbalanced_Sequences'
                window_bal_folder_g = 'Global\Sliding_Window\Balanced_Sequences'
                window_imb_path_g = os.path.join(self.output_transformation_dir,window_imb_folder_g)
                window_bal_path_g = os.path.join(self.output_transformation_dir,window_bal_folder_g)
                # Check if the directory exists
                if not os.path.exists(window_imb_path_g) and not os.path.exists(window_bal_path_g):
                    # If not, create it
                    os.makedirs(window_imb_path_g)
                    os.makedirs(window_bal_path_g)
                np.save(os.path.join(window_imb_path_g, "X_data.npy"), imb_Xdata_g)
                np.save(os.path.join(window_imb_path_g, "Y_data.npy"), imb_Ydata_g)
                np.save(os.path.join(window_imb_path_g, "X_train.npy"), imb_X_train_g)
                np.save(os.path.join(window_imb_path_g, "Y_train.npy"), imb_y_train_g)
                np.save(os.path.join(window_imb_path_g, "X_test.npy"), imb_X_test_g)
                np.save(os.path.join(window_imb_path_g, "Y_test.npy"), imb_y_test_g)
                np.save(os.path.join(window_imb_path_g, "X_val.npy"), imb_X_val_g)
                np.save(os.path.join(window_imb_path_g, "Y_val.npy"), imb_y_val_g)
                
                np.save(os.path.join(window_bal_path_g, "X_data.npy"), bal_Xdata_g)
                np.save(os.path.join(window_bal_path_g, "Y_data.npy"), bal_Ydata_g)
                np.save(os.path.join(window_bal_path_g, "X_train.npy"), bal_X_train_g)
                np.save(os.path.join(window_bal_path_g, "Y_train.npy"), bal_y_train_g)
                np.save(os.path.join(window_bal_path_g, "X_test.npy"), bal_X_test_g)
                np.save(os.path.join(window_bal_path_g, "Y_test.npy"), bal_y_test_g)
                np.save(os.path.join(window_bal_path_g, "X_val.npy"), bal_X_val_g)
                np.save(os.path.join(window_bal_path_g, "Y_val.npy"), bal_y_val_g)
        
        if self.fixed_seq_check.get():
            #NORMALIZED
            if self.normalized_keypoints_check.get():
            #if self.kps_type == 'normalized' and self.normalized_keypoints_check.get():
                #INTERACTIONS
                self.handshaking_sequences_n = np.concatenate(self.handshaking_sequences_n,axis=0)
                self.handshaking_labels_n = np.ones(self.handshaking_sequences_n.shape[0],dtype=int)*0

                self.hugging_sequences_n = np.concatenate(self.hugging_sequences_n,axis=0)
                self.hugging_labels_n = np.ones(self.hugging_sequences_n.shape[0],dtype=int)*1
                
                self.kicking_sequences_n = np.concatenate(self.kicking_sequences_n,axis=0)
                self.kicking_labels_n = np.ones(self.kicking_sequences_n.shape[0],dtype=int)*2    
                
                self.punching_sequences_n = np.concatenate(self.punching_sequences_n,axis=0)
                self.punching_labels_n = np.ones(self.punching_sequences_n.shape[0],dtype=int)*3    
                
                self.pushing_sequences_n = np.concatenate(self.pushing_sequences_n,axis=0)
                self.pushing_labels_n = np.ones(self.pushing_sequences_n.shape[0],dtype=int)*4    
                
                #SOLO ACTIONS
                self.clapping_solo_sequences_n = np.concatenate(self.clapping_solo_sequences_n,axis=0)
                self.clapping_solo_labels_n = np.ones(self.clapping_solo_sequences_n.shape[0],dtype=int)*5
                
                self.hitting_bottle_solo_sequences_n = np.concatenate(self.hitting_bottle_solo_sequences_n,axis=0)
                self.hitting_bottle_solo_labels_n = np.ones(self.hitting_bottle_solo_sequences_n.shape[0],dtype=int)*6

                self.hitting_stick_solo_sequences_n = np.concatenate(self.hitting_stick_solo_sequences_n,axis=0)
                self.hitting_stick_solo_labels_n = np.ones(self.hitting_stick_solo_sequences_n.shape[0],dtype=int)*7

                self.jogging_f_b_solo_sequences_n = np.concatenate(self.jogging_f_b_solo_sequences_n,axis=0)
                self.jogging_f_b_solo_labels_n = np.ones(self.jogging_f_b_solo_sequences_n.shape[0],dtype=int)*8

                self.jogging_side_solo_sequences_n = np.concatenate(self.jogging_side_solo_sequences_n,axis=0)
                self.jogging_side_solo_labels_n = np.ones(self.jogging_side_solo_sequences_n.shape[0],dtype=int)*9

                self.kicking_solo_sequences_n = np.concatenate(self.kicking_solo_sequences_n,axis=0)
                self.kicking_solo_labels_n = np.ones(self.kicking_solo_sequences_n.shape[0],dtype=int)*10

                self.punching_solo_sequences_n = np.concatenate(self.punching_solo_sequences_n,axis=0)
                self.punching_solo_labels_n = np.ones(self.punching_solo_sequences_n.shape[0],dtype=int)*11

                self.running_f_b_solo_sequences_n = np.concatenate(self.running_f_b_solo_sequences_n,axis=0)
                self.running_f_b_solo_labels_n = np.ones(self.running_f_b_solo_sequences_n.shape[0],dtype=int)*12

                self.running_side_solo_sequences_n = np.concatenate(self.running_side_solo_sequences_n,axis=0)
                self.running_side_solo_labels_n = np.ones(self.running_side_solo_sequences_n.shape[0],dtype=int)*13

                self.stabbing_solo_sequences_n = np.concatenate(self.stabbing_solo_sequences_n,axis=0)
                self.stabbing_solo_labels_n = np.ones(self.stabbing_solo_sequences_n.shape[0],dtype=int)*14

                self.walking_f_b_solo_sequences_n = np.concatenate(self.walking_f_b_solo_sequences_n,axis=0)
                self.walking_f_b_solo_labels_n = np.ones(self.walking_f_b_solo_sequences_n.shape[0],dtype=int)*15

                self.walking_side_solo_sequences_n = np.concatenate(self.walking_side_solo_sequences_n,axis=0)
                self.walking_side_solo_labels_n = np.ones(self.walking_side_solo_sequences_n.shape[0],dtype=int)*16

                self.waving_hands_solo_sequences_n = np.concatenate(self.waving_hands_solo_sequences_n,axis=0)
                self.waving_hands_solo_labels_n = np.ones(self.waving_hands_solo_sequences_n.shape[0],dtype=int)*17

                #PRINTING
                print('NORMALIZED SEQUENTIAL DATA: ')
                #INTERACTIONS
                print('Handshaking Sequntial Data Shape: ',self.handshaking_sequences_n.shape)
                print('Handshaking Labels Shape: ', len(self.handshaking_labels_n))

                print('Hugging Sequntial Data Shape: ',self.hugging_sequences_n.shape)
                print('Hugging Labels Shape: ', len(self.hugging_labels_n))

                print('Kicking Sequntial Data Shape: ',self.kicking_sequences_n.shape)
                print('Kicking Labels Shape: ', len(self.kicking_labels_n))

                print('Punching Sequntial Data Shape: ',self.punching_sequences_n.shape)
                print('Punching Labels Shape: ', len(self.punching_labels_n))

                print('Pushing Sequntial Data Shape: ',self.pushing_sequences_n.shape)
                print('Pushing Labels Shape: ', len(self.pushing_labels_n))

                #SOLO ACTIONS
                print('Clapping Solo Sequntial Data Shape: ',self.clapping_solo_sequences_n.shape)
                print('Clapping Solo Labels Shape: ', len(self.clapping_solo_labels_n))

                print('Hitting Bottle Solo Sequntial Data Shape: ',self.hitting_bottle_solo_sequences_n.shape)
                print('Hitting Bottle Solo Labels Shape: ', len(self.hitting_bottle_solo_labels_n))

                print('Hitting Stick Solo Sequntial Data Shape: ',self.hitting_stick_solo_sequences_n.shape)
                print('Hitting Stick Solo Labels Shape: ', len(self.hitting_stick_solo_labels_n))

                print('Jogging FB Solo Sequntial Data Shape: ',self.jogging_f_b_solo_sequences_n.shape)
                print('Jogging FB Solo Labels Shape: ', len(self.jogging_f_b_solo_labels_n))

                print('Jogging Side Solo Sequntial Data Shape: ',self.jogging_side_solo_sequences_n.shape)
                print('Jogging Side Solo Labels Shape: ', len(self.jogging_side_solo_labels_n))

                print('Kicking Solo Sequntial Data Shape: ',self.kicking_solo_sequences_n.shape)
                print('Kicking Solo Labels Shape: ', len(self.kicking_solo_labels_n))

                print('Punching Solo Sequntial Data Shape: ',self.punching_solo_sequences_n.shape)
                print('Punching Solo Labels Shape: ', len(self.punching_solo_labels_n))

                print('Running FB Solo Sequntial Data Shape: ',self.running_f_b_solo_sequences_n.shape)
                print('Running FB Solo Labels Shape: ', len(self.running_f_b_solo_labels_n))

                print('Running Side Solo Sequntial Data Shape: ',self.running_side_solo_sequences_n.shape)
                print('Running Side Solo Labels Shape: ', len(self.running_side_solo_labels_n))

                print('Stabbing Solo Sequntial Data Shape: ',self.stabbing_solo_sequences_n.shape)
                print('Stabbing Solo Labels Shape: ', len(self.stabbing_solo_labels_n))

                print('Walking FB Solo Sequntial Data Shape: ',self.walking_f_b_solo_sequences_n.shape)
                print('Walking FB Solo Labels Shape: ', len(self.walking_f_b_solo_labels_n))

                print('Walking Side Solo Sequntial Data Shape: ',self.walking_side_solo_sequences_n.shape)
                print('Walking Side Solo Labels Shape: ', len(self.walking_side_solo_labels_n))

                print('Waving Hands Solo Sequntial Data Shape: ',self.waving_hands_solo_sequences_n.shape)
                print('Waving Hands Solo Labels Shape: ', len(self.waving_hands_solo_labels_n))

                classses_seq_lists_n = [
                self.handshaking_sequences_n.shape[0],
                self.hugging_sequences_n.shape[0],
                self.kicking_sequences_n.shape[0],
                self.punching_sequences_n.shape[0],
                self.pushing_sequences_n.shape[0],
                self.clapping_solo_sequences_n.shape[0],
                self.hitting_bottle_solo_sequences_n.shape[0],
                self.hitting_stick_solo_sequences_n.shape[0],
                self.jogging_f_b_solo_sequences_n.shape[0],
                self.jogging_side_solo_sequences_n.shape[0],
                self.kicking_solo_sequences_n.shape[0],
                self.punching_solo_sequences_n.shape[0],
                self.running_f_b_solo_sequences_n.shape[0],
                self.running_side_solo_sequences_n.shape[0],
                self.stabbing_solo_sequences_n.shape[0],
                self.walking_f_b_solo_sequences_n.shape[0],
                self.walking_side_solo_sequences_n.shape[0],
                self.waving_hands_solo_sequences_n.shape[0]]
                balanced_seq_num_n = min(classses_seq_lists_n)
                #IMBALANCED DATA
                imb_Xdata_n = np.concatenate((self.handshaking_sequences_n,self.hugging_sequences_n, 
                                            self.kicking_sequences_n,self.punching_sequences_n, self.pushing_sequences_n, 
                                            self.clapping_solo_sequences_n,self.hitting_bottle_solo_sequences_n,
                                            self.hitting_stick_solo_sequences_n,self.jogging_f_b_solo_sequences_n,
                                            self.jogging_side_solo_sequences_n,self.kicking_solo_sequences_n,
                                            self.punching_solo_sequences_n,self.running_f_b_solo_sequences_n,
                                            self.running_side_solo_sequences_n,self.stabbing_solo_sequences_n,
                                            self.walking_f_b_solo_sequences_n,self.walking_side_solo_sequences_n,
                                            self.waving_hands_solo_sequences_n),axis=0)
                #RESHAPING TO DESIRED SHAPE
                num_sequences, seq_length, num_persons, num_kps, num_coords = imb_Xdata_n.shape
                imb_Xdata_n = imb_Xdata_n.reshape(num_sequences,seq_length, num_persons*num_kps*num_coords)
                imb_Ydata_n = np.concatenate((self.handshaking_labels_n,self.hugging_labels_n,
                                            self.kicking_labels_n, self.punching_labels_n,self.pushing_labels_n, 
                                            self.clapping_solo_labels_n, self.hitting_bottle_solo_labels_n,
                                            self.hitting_stick_solo_labels_n,self.jogging_f_b_solo_labels_n,
                                            self.jogging_side_solo_labels_n,self.kicking_solo_labels_n,
                                            self.punching_solo_labels_n,self.running_f_b_solo_labels_n,
                                            self.running_side_solo_labels_n,self.stabbing_solo_labels_n,
                                            self.walking_f_b_solo_labels_n,self.walking_side_solo_labels_n,
                                            self.waving_hands_solo_labels_n))
                #BALANCING SEQUENCES ACROSS ALL CLASSES
                self.handshaking_sequences_n = self.handshaking_sequences_n[:balanced_seq_num_n]
                self.hugging_sequences_n = self.hugging_sequences_n[:balanced_seq_num_n]
                self.kicking_sequences_n = self.kicking_sequences_n[:balanced_seq_num_n]
                self.punching_sequences_n = self.punching_sequences_n[:balanced_seq_num_n]
                self.pushing_sequences_n = self.pushing_sequences_n[:balanced_seq_num_n]
                self.clapping_solo_sequences_n = self.clapping_solo_sequences_n[:balanced_seq_num_n]
                self.hitting_bottle_solo_sequences_n = self.hitting_bottle_solo_sequences_n[:balanced_seq_num_n]
                self.hitting_stick_solo_sequences_n = self.hitting_stick_solo_sequences_n[:balanced_seq_num_n]
                self.jogging_f_b_solo_sequences_n = self.jogging_f_b_solo_sequences_n[:balanced_seq_num_n]
                self.jogging_side_solo_sequences_n = self.jogging_side_solo_sequences_n[:balanced_seq_num_n]
                self.kicking_solo_sequences_n = self.kicking_solo_sequences_n[:balanced_seq_num_n]
                self.punching_solo_sequences_n = self.punching_solo_sequences_n[:balanced_seq_num_n]
                self.running_f_b_solo_sequences_n = self.running_f_b_solo_sequences_n[:balanced_seq_num_n]
                self.running_side_solo_sequences_n = self.running_side_solo_sequences_n[:balanced_seq_num_n]
                self.stabbing_solo_sequences_n = self.stabbing_solo_sequences_n[:balanced_seq_num_n]
                self.walking_f_b_solo_sequences_n = self.walking_f_b_solo_sequences_n[:balanced_seq_num_n]
                self.walking_side_solo_sequences_n = self.walking_side_solo_sequences_n[:balanced_seq_num_n]
                self.waving_hands_solo_sequences_n = self.waving_hands_solo_sequences_n[:balanced_seq_num_n]
                #BALANCING LABELS ACROSS ALL CLASSES
                self.handshaking_labels_n = self.handshaking_labels_n[:balanced_seq_num_n]
                self.hugging_labels_n = self.hugging_labels_n[:balanced_seq_num_n]
                self.kicking_labels_n = self.kicking_labels_n[:balanced_seq_num_n]
                self.punching_labels_n = self.punching_labels_n[:balanced_seq_num_n]
                self.pushing_labels_n = self.pushing_labels_n[:balanced_seq_num_n]
                self.clapping_solo_labels_n = self.clapping_solo_labels_n[:balanced_seq_num_n]
                self.hitting_bottle_solo_labels_n = self.hitting_bottle_solo_labels_n[:balanced_seq_num_n]
                self.hitting_stick_solo_labels_n = self.hitting_stick_solo_labels_n[:balanced_seq_num_n]
                self.jogging_f_b_solo_labels_n = self.jogging_f_b_solo_labels_n[:balanced_seq_num_n]
                self.jogging_side_solo_labels_n = self.jogging_side_solo_labels_n[:balanced_seq_num_n]
                self.kicking_solo_labels_n = self.kicking_solo_labels_n[:balanced_seq_num_n]
                self.punching_solo_labels_n = self.punching_solo_labels_n[:balanced_seq_num_n]
                self.running_f_b_solo_labels_n = self.running_f_b_solo_labels_n[:balanced_seq_num_n]
                self.running_side_solo_labels_n = self.running_side_solo_labels_n[:balanced_seq_num_n]
                self.stabbing_solo_labels_n = self.stabbing_solo_labels_n[:balanced_seq_num_n]
                self.walking_f_b_solo_labels_n = self.walking_f_b_solo_labels_n[:balanced_seq_num_n]
                self.walking_side_solo_labels_n = self.walking_side_solo_labels_n[:balanced_seq_num_n]
                self.waving_hands_solo_labels_n = self.waving_hands_solo_labels_n[:balanced_seq_num_n]
                bal_Xdata_n = np.concatenate((self.handshaking_sequences_n,self.hugging_sequences_n, 
                                            self.kicking_sequences_n,self.punching_sequences_n, self.pushing_sequences_n, 
                                            self.clapping_solo_sequences_n,self.hitting_bottle_solo_sequences_n,
                                            self.hitting_stick_solo_sequences_n,self.jogging_f_b_solo_sequences_n,
                                            self.jogging_side_solo_sequences_n,self.kicking_solo_sequences_n,
                                            self.punching_solo_sequences_n,self.running_f_b_solo_sequences_n,
                                            self.running_side_solo_sequences_n,self.stabbing_solo_sequences_n,
                                            self.walking_f_b_solo_sequences_n,self.walking_side_solo_sequences_n,
                                            self.waving_hands_solo_sequences_n),axis=0)
                
                #RESHAPING TO DESIRED SHAPE
                num_sequences, seq_length, num_persons, num_kps, num_coords = bal_Xdata_n.shape
                bal_Xdata_n = bal_Xdata_n.reshape(num_sequences,seq_length, num_persons*num_kps*num_coords)
                bal_Ydata_n = np.concatenate((self.handshaking_labels_n,self.hugging_labels_n,
                                            self.kicking_labels_n, self.punching_labels_n,self.pushing_labels_n, 
                                            self.clapping_solo_labels_n,self.hitting_bottle_solo_labels_n,
                                            self.hitting_stick_solo_labels_n,self.jogging_f_b_solo_labels_n,
                                            self.jogging_side_solo_labels_n,self.kicking_solo_labels_n,
                                            self.punching_solo_labels_n,self.running_f_b_solo_labels_n,
                                            self.running_side_solo_labels_n,self.stabbing_solo_labels_n,
                                            self.walking_f_b_solo_labels_n,self.walking_side_solo_labels_n,
                                            self.waving_hands_solo_labels_n))
                #Xtrain, Xtest and Xval
                # Split the data into training and testing sets, ensuring balanced samples
                imb_X_train_n, imb_X_testval_n, imb_y_train_n, imb_y_testval_n = train_test_split(imb_Xdata_n, imb_Ydata_n, test_size=1-float(self.trainset_percent), stratify=imb_Ydata_n)           
                imb_X_test_n, imb_X_val_n, imb_y_test_n, imb_y_val_n = train_test_split(imb_X_testval_n, imb_y_testval_n, test_size=float(self.validationset_percent), stratify=imb_y_testval_n)
                bal_X_train_n, bal_X_testval_n, bal_y_train_n, bal_y_testval_n = train_test_split(bal_Xdata_n, bal_Ydata_n, test_size=1-float(self.trainset_percent), stratify=bal_Ydata_n)           
                bal_X_test_n, bal_X_val_n, bal_y_test_n, bal_y_val_n = train_test_split(bal_X_testval_n, bal_y_testval_n, test_size=float(self.validationset_percent), stratify=bal_y_testval_n)
                print('Imbalance Xdata Shape: ',imb_Xdata_n.shape)
                print('Imbalance Ydata Shape: ',imb_Ydata_n.shape)
                print('Imbalance Xtrain Shape: ',imb_X_train_n.shape)
                print('Imbalance Ytrain Shape: ',imb_y_train_n.shape)
                print('Imbalance Xtest Shape: ',imb_X_test_n.shape)
                print('Imbalance Ytest Shape: ',imb_y_test_n.shape)
                print('Imbalance Xval Shape: ',imb_X_val_n.shape)
                print('Imbalance Yval Shape: ',imb_y_val_n.shape)
                
                print('Balanced Sequence Num (Samples per class): ',balanced_seq_num_n)
                print('Balanced Xdata Shape: ',bal_Xdata_n.shape)
                print('Balanced Ydata Shape: ',bal_Ydata_n.shape)
                print('Balanced Xtrain Shape: ',bal_X_train_n.shape)
                print('Balanced Ytrain Shape: ',bal_y_train_n.shape)
                print('Balanced Xtest Shape: ',bal_X_test_n.shape)
                print('Balanced Ytest Shape: ',bal_y_test_n.shape)
                print('Balanced Xval Shape: ',bal_X_val_n.shape)
                print('Balanced Yval Shape: ',bal_y_val_n.shape)
                #Saving Xdata Ydata
                window_imb_folder_n = 'Normalized\Fixed_Window\Imbalanced_Sequences'
                window_bal_folder_n = 'Normalized\Fixed_Window\Balanced_Sequences'
                window_imb_path_n = os.path.join(self.output_transformation_dir,window_imb_folder_n)
                window_bal_path_n = os.path.join(self.output_transformation_dir,window_bal_folder_n)
                # Check if the directory exists
                if not os.path.exists(window_imb_path_n) and not os.path.exists(window_bal_path_n):
                    # If not, create it
                    os.makedirs(window_imb_path_n)
                    os.makedirs(window_bal_path_n)
                np.save(os.path.join(window_imb_path_n, "X_data.npy"), imb_Xdata_n)
                np.save(os.path.join(window_imb_path_n, "Y_data.npy"), imb_Ydata_n)
                np.save(os.path.join(window_imb_path_n, "X_train.npy"), imb_X_train_n)
                np.save(os.path.join(window_imb_path_n, "Y_train.npy"), imb_y_train_n)
                np.save(os.path.join(window_imb_path_n, "X_test.npy"), imb_X_test_n)
                np.save(os.path.join(window_imb_path_n, "Y_test.npy"), imb_y_test_n)
                np.save(os.path.join(window_imb_path_n, "X_val.npy"), imb_X_val_n)
                np.save(os.path.join(window_imb_path_n, "Y_val.npy"), imb_y_val_n)
                
                np.save(os.path.join(window_bal_path_n, "X_data.npy"), bal_Xdata_n)
                np.save(os.path.join(window_bal_path_n, "Y_data.npy"), bal_Ydata_n)
                np.save(os.path.join(window_bal_path_n, "X_train.npy"), bal_X_train_n)
                np.save(os.path.join(window_bal_path_n, "Y_train.npy"), bal_y_train_n)
                np.save(os.path.join(window_bal_path_n, "X_test.npy"), bal_X_test_n)
                np.save(os.path.join(window_bal_path_n, "Y_test.npy"), bal_y_test_n)
                np.save(os.path.join(window_bal_path_n, "X_val.npy"), bal_X_val_n)
                np.save(os.path.join(window_bal_path_n, "Y_val.npy"), bal_y_val_n)

            #GLOBAL
            if self.global_keypoints_check.get():
            #if self.kps_type == 'global' and self.global_keypoints_check.get():
                #INTERACTIONS
                self.handshaking_sequences_g = np.concatenate(self.handshaking_sequences_g,axis=0)
                self.handshaking_labels_g = np.ones(self.handshaking_sequences_g.shape[0],dtype=int)*0

                self.hugging_sequences_g = np.concatenate(self.hugging_sequences_g,axis=0)
                self.hugging_labels_g = np.ones(self.hugging_sequences_g.shape[0],dtype=int)*1
                
                self.kicking_sequences_g = np.concatenate(self.kicking_sequences_g,axis=0)
                self.kicking_labels_g = np.ones(self.kicking_sequences_g.shape[0],dtype=int)*2    
                
                self.punching_sequences_g = np.concatenate(self.punching_sequences_g,axis=0)
                self.punching_labels_g = np.ones(self.punching_sequences_g.shape[0],dtype=int)*3    
                
                self.pushing_sequences_g = np.concatenate(self.pushing_sequences_g,axis=0)
                self.pushing_labels_g = np.ones(self.pushing_sequences_g.shape[0],dtype=int)*4    
                
                #SOLO ACTIONS
                self.clapping_solo_sequences_g = np.concatenate(self.clapping_solo_sequences_g,axis=0)
                self.clapping_solo_labels_g = np.ones(self.clapping_solo_sequences_g.shape[0],dtype=int)*5
                
                self.hitting_bottle_solo_sequences_g = np.concatenate(self.hitting_bottle_solo_sequences_g,axis=0)
                self.hitting_bottle_solo_labels_g = np.ones(self.hitting_bottle_solo_sequences_g.shape[0],dtype=int)*6

                self.hitting_stick_solo_sequences_g = np.concatenate(self.hitting_stick_solo_sequences_g,axis=0)
                self.hitting_stick_solo_labels_g = np.ones(self.hitting_stick_solo_sequences_g.shape[0],dtype=int)*7

                self.jogging_f_b_solo_sequences_g = np.concatenate(self.jogging_f_b_solo_sequences_g,axis=0)
                self.jogging_f_b_solo_labels_g = np.ones(self.jogging_f_b_solo_sequences_g.shape[0],dtype=int)*8

                self.jogging_side_solo_sequences_g = np.concatenate(self.jogging_side_solo_sequences_g,axis=0)
                self.jogging_side_solo_labels_g = np.ones(self.jogging_side_solo_sequences_g.shape[0],dtype=int)*9

                self.kicking_solo_sequences_g = np.concatenate(self.kicking_solo_sequences_g,axis=0)
                self.kicking_solo_labels_g = np.ones(self.kicking_solo_sequences_g.shape[0],dtype=int)*10

                self.punching_solo_sequences_g = np.concatenate(self.punching_solo_sequences_g,axis=0)
                self.punching_solo_labels_g = np.ones(self.punching_solo_sequences_g.shape[0],dtype=int)*11

                self.running_f_b_solo_sequences_g = np.concatenate(self.running_f_b_solo_sequences_g,axis=0)
                self.running_f_b_solo_labels_g = np.ones(self.running_f_b_solo_sequences_g.shape[0],dtype=int)*12

                self.running_side_solo_sequences_g = np.concatenate(self.running_side_solo_sequences_g,axis=0)
                self.running_side_solo_labels_g = np.ones(self.running_side_solo_sequences_g.shape[0],dtype=int)*13

                self.stabbing_solo_sequences_g = np.concatenate(self.stabbing_solo_sequences_g,axis=0)
                self.stabbing_solo_labels_g = np.ones(self.stabbing_solo_sequences_g.shape[0],dtype=int)*14

                self.walking_f_b_solo_sequences_g = np.concatenate(self.walking_f_b_solo_sequences_g,axis=0)
                self.walking_f_b_solo_labels_g = np.ones(self.walking_f_b_solo_sequences_g.shape[0],dtype=int)*15

                self.walking_side_solo_sequences_g = np.concatenate(self.walking_side_solo_sequences_g,axis=0)
                self.walking_side_solo_labels_g = np.ones(self.walking_side_solo_sequences_g.shape[0],dtype=int)*16

                self.waving_hands_solo_sequences_g = np.concatenate(self.waving_hands_solo_sequences_g,axis=0)
                self.waving_hands_solo_labels_g = np.ones(self.waving_hands_solo_sequences_g.shape[0],dtype=int)*17

                #PRINTING
                print('GLOBAL SEQUENTIAL DATA: ')
                #INTERACTIONS
                print('Handshaking Sequntial Data Shape: ',self.handshaking_sequences_g.shape)
                print('Handshaking Labels Shape: ', len(self.handshaking_labels_g))

                print('Hugging Sequntial Data Shape: ',self.hugging_sequences_g.shape)
                print('Hugging Labels Shape: ', len(self.hugging_labels_g))

                print('Kicking Sequntial Data Shape: ',self.kicking_sequences_g.shape)
                print('Kicking Labels Shape: ', len(self.kicking_labels_g))

                print('Punching Sequntial Data Shape: ',self.punching_sequences_g.shape)
                print('Punching Labels Shape: ', len(self.punching_labels_g))

                print('Pushing Sequntial Data Shape: ',self.pushing_sequences_g.shape)
                print('Pushing Labels Shape: ', len(self.pushing_labels_g))

                #SOLO ACTIONS
                print('Clapping Solo Sequntial Data Shape: ',self.clapping_solo_sequences_g.shape)
                print('Clapping Solo Labels Shape: ', len(self.clapping_solo_labels_g))

                print('Hitting Bottle Solo Sequntial Data Shape: ',self.hitting_bottle_solo_sequences_g.shape)
                print('Hitting Bottle Solo Labels Shape: ', len(self.hitting_bottle_solo_labels_g))

                print('Hitting Stick Solo Sequntial Data Shape: ',self.hitting_stick_solo_sequences_g.shape)
                print('Hitting Stick Solo Labels Shape: ', len(self.hitting_stick_solo_labels_g))

                print('Jogging FB Solo Sequntial Data Shape: ',self.jogging_f_b_solo_sequences_g.shape)
                print('Jogging FB Solo Labels Shape: ', len(self.jogging_f_b_solo_labels_g))

                print('Jogging Side Solo Sequntial Data Shape: ',self.jogging_side_solo_sequences_g.shape)
                print('Jogging Side Solo Labels Shape: ', len(self.jogging_side_solo_labels_g))

                print('Kicking Solo Sequntial Data Shape: ',self.kicking_solo_sequences_g.shape)
                print('Kicking Solo Labels Shape: ', len(self.kicking_solo_labels_g))

                print('Punching Solo Sequntial Data Shape: ',self.punching_solo_sequences_g.shape)
                print('Punching Solo Labels Shape: ', len(self.punching_solo_labels_g))

                print('Running FB Solo Sequntial Data Shape: ',self.running_f_b_solo_sequences_g.shape)
                print('Running FB Solo Labels Shape: ', len(self.running_f_b_solo_labels_g))

                print('Running Side Solo Sequntial Data Shape: ',self.running_side_solo_sequences_g.shape)
                print('Running Side Solo Labels Shape: ', len(self.running_side_solo_labels_g))

                print('Stabbing Solo Sequntial Data Shape: ',self.stabbing_solo_sequences_g.shape)
                print('Stabbing Solo Labels Shape: ', len(self.stabbing_solo_labels_g))

                print('Walking FB Solo Sequntial Data Shape: ',self.walking_f_b_solo_sequences_g.shape)
                print('Walking FB Solo Labels Shape: ', len(self.walking_f_b_solo_labels_g))

                print('Walking Side Solo Sequntial Data Shape: ',self.walking_side_solo_sequences_g.shape)
                print('Walking Side Solo Labels Shape: ', len(self.walking_side_solo_labels_g))

                print('Waving Hands Solo Sequntial Data Shape: ',self.waving_hands_solo_sequences_g.shape)
                print('Waving Hands Solo Labels Shape: ', len(self.waving_hands_solo_labels_g))

                classses_seq_lists_g = [
                self.handshaking_sequences_g.shape[0],
                self.hugging_sequences_g.shape[0],
                self.kicking_sequences_g.shape[0],
                self.punching_sequences_g.shape[0],
                self.pushing_sequences_g.shape[0],
                self.clapping_solo_sequences_g.shape[0],
                self.hitting_bottle_solo_sequences_g.shape[0],
                self.hitting_stick_solo_sequences_g.shape[0],
                self.jogging_f_b_solo_sequences_g.shape[0],
                self.jogging_side_solo_sequences_g.shape[0],
                self.kicking_solo_sequences_g.shape[0],
                self.punching_solo_sequences_g.shape[0],
                self.running_f_b_solo_sequences_g.shape[0],
                self.running_side_solo_sequences_g.shape[0],
                self.stabbing_solo_sequences_g.shape[0],
                self.walking_f_b_solo_sequences_g.shape[0],
                self.walking_side_solo_sequences_g.shape[0],
                self.waving_hands_solo_sequences_g.shape[0]]
                balanced_seq_num_g = min(classses_seq_lists_g)
                #IMBALANCED DATA
                imb_Xdata_g = np.concatenate((self.handshaking_sequences_g,self.hugging_sequences_g, 
                                            self.kicking_sequences_g,self.punching_sequences_g, self.pushing_sequences_g, 
                                            self.clapping_solo_sequences_g,self.hitting_bottle_solo_sequences_g,
                                            self.hitting_stick_solo_sequences_g,self.jogging_f_b_solo_sequences_g,
                                            self.jogging_side_solo_sequences_g,self.kicking_solo_sequences_g,
                                            self.punching_solo_sequences_g,self.running_f_b_solo_sequences_g,
                                            self.running_side_solo_sequences_g,self.stabbing_solo_sequences_g,
                                            self.walking_f_b_solo_sequences_g,self.walking_side_solo_sequences_g,
                                            self.waving_hands_solo_sequences_g),axis=0)
                #RESHAPING TO DESIRED SHAPE
                num_sequences, seq_length, num_persons, num_kps, num_coords = imb_Xdata_g.shape
                imb_Xdata_g = imb_Xdata_g.reshape(num_sequences,seq_length, num_persons*num_kps*num_coords)
                imb_Ydata_g = np.concatenate((self.handshaking_labels_g,self.hugging_labels_g,
                                            self.kicking_labels_g, self.punching_labels_g,self.pushing_labels_g, 
                                            self.clapping_solo_labels_g, self.hitting_bottle_solo_labels_g,
                                            self.hitting_stick_solo_labels_g,self.jogging_f_b_solo_labels_g,
                                            self.jogging_side_solo_labels_g,self.kicking_solo_labels_g,
                                            self.punching_solo_labels_g,self.running_f_b_solo_labels_g,
                                            self.running_side_solo_labels_g,self.stabbing_solo_labels_g,
                                            self.walking_f_b_solo_labels_g,self.walking_side_solo_labels_g,
                                            self.waving_hands_solo_labels_g))
                #BALANCING SEQUENCES ACROSS ALL CLASSES
                self.handshaking_sequences_g = self.handshaking_sequences_g[:balanced_seq_num_g]
                self.hugging_sequences_g = self.hugging_sequences_g[:balanced_seq_num_g]
                self.kicking_sequences_g = self.kicking_sequences_g[:balanced_seq_num_g]
                self.punching_sequences_g = self.punching_sequences_g[:balanced_seq_num_g]
                self.pushing_sequences_g = self.pushing_sequences_g[:balanced_seq_num_g]
                self.clapping_solo_sequences_g = self.clapping_solo_sequences_g[:balanced_seq_num_g]
                self.hitting_bottle_solo_sequences_g = self.hitting_bottle_solo_sequences_g[:balanced_seq_num_g]
                self.hitting_stick_solo_sequences_g = self.hitting_stick_solo_sequences_g[:balanced_seq_num_g]
                self.jogging_f_b_solo_sequences_g = self.jogging_f_b_solo_sequences_g[:balanced_seq_num_g]
                self.jogging_side_solo_sequences_g = self.jogging_side_solo_sequences_g[:balanced_seq_num_g]
                self.kicking_solo_sequences_g = self.kicking_solo_sequences_g[:balanced_seq_num_g]
                self.punching_solo_sequences_g = self.punching_solo_sequences_g[:balanced_seq_num_g]
                self.running_f_b_solo_sequences_g = self.running_f_b_solo_sequences_g[:balanced_seq_num_g]
                self.running_side_solo_sequences_g = self.running_side_solo_sequences_g[:balanced_seq_num_g]
                self.stabbing_solo_sequences_g = self.stabbing_solo_sequences_g[:balanced_seq_num_g]
                self.walking_f_b_solo_sequences_g = self.walking_f_b_solo_sequences_g[:balanced_seq_num_g]
                self.walking_side_solo_sequences_g = self.walking_side_solo_sequences_g[:balanced_seq_num_g]
                self.waving_hands_solo_sequences_g = self.waving_hands_solo_sequences_g[:balanced_seq_num_g]
                #BALANCING LABELS ACROSS ALL CLASSES
                self.handshaking_labels_g = self.handshaking_labels_g[:balanced_seq_num_g]
                self.hugging_labels_g = self.hugging_labels_g[:balanced_seq_num_g]
                self.kicking_labels_g = self.kicking_labels_g[:balanced_seq_num_g]
                self.punching_labels_g = self.punching_labels_g[:balanced_seq_num_g]
                self.pushing_labels_g = self.pushing_labels_g[:balanced_seq_num_g]
                self.clapping_solo_labels_g = self.clapping_solo_labels_g[:balanced_seq_num_g]
                self.hitting_bottle_solo_labels_g = self.hitting_bottle_solo_labels_g[:balanced_seq_num_g]
                self.hitting_stick_solo_labels_g = self.hitting_stick_solo_labels_g[:balanced_seq_num_g]
                self.jogging_f_b_solo_labels_g = self.jogging_f_b_solo_labels_g[:balanced_seq_num_g]
                self.jogging_side_solo_labels_g = self.jogging_side_solo_labels_g[:balanced_seq_num_g]
                self.kicking_solo_labels_g = self.kicking_solo_labels_g[:balanced_seq_num_g]
                self.punching_solo_labels_g = self.punching_solo_labels_g[:balanced_seq_num_g]
                self.running_f_b_solo_labels_g = self.running_f_b_solo_labels_g[:balanced_seq_num_g]
                self.running_side_solo_labels_g = self.running_side_solo_labels_g[:balanced_seq_num_g]
                self.stabbing_solo_labels_g = self.stabbing_solo_labels_g[:balanced_seq_num_g]
                self.walking_f_b_solo_labels_g = self.walking_f_b_solo_labels_g[:balanced_seq_num_g]
                self.walking_side_solo_labels_g = self.walking_side_solo_labels_g[:balanced_seq_num_g]
                self.waving_hands_solo_labels_g = self.waving_hands_solo_labels_g[:balanced_seq_num_g]
                bal_Xdata_g = np.concatenate((self.handshaking_sequences_g,self.hugging_sequences_g, 
                                            self.kicking_sequences_g,self.punching_sequences_g, self.pushing_sequences_g, 
                                            self.clapping_solo_sequences_g,self.hitting_bottle_solo_sequences_g,
                                            self.hitting_stick_solo_sequences_g,self.jogging_f_b_solo_sequences_g,
                                            self.jogging_side_solo_sequences_g,self.kicking_solo_sequences_g,
                                            self.punching_solo_sequences_g,self.running_f_b_solo_sequences_g,
                                            self.running_side_solo_sequences_g,self.stabbing_solo_sequences_g,
                                            self.walking_f_b_solo_sequences_g,self.walking_side_solo_sequences_g,
                                            self.waving_hands_solo_sequences_g),axis=0)
                #RESHAPING TO DESIRED SHAPE
                num_sequences, seq_length, num_persons, num_kps, num_coords = bal_Xdata_g.shape
                bal_Xdata_g = bal_Xdata_g.reshape(num_sequences,seq_length, num_persons*num_kps*num_coords)
                bal_Ydata_g = np.concatenate((self.handshaking_labels_g,self.hugging_labels_g,
                                            self.kicking_labels_g, self.punching_labels_g,self.pushing_labels_g, 
                                            self.clapping_solo_labels_g,self.hitting_bottle_solo_labels_g,
                                            self.hitting_stick_solo_labels_g,self.jogging_f_b_solo_labels_g,
                                            self.jogging_side_solo_labels_g,self.kicking_solo_labels_g,
                                            self.punching_solo_labels_g,self.running_f_b_solo_labels_g,
                                            self.running_side_solo_labels_g,self.stabbing_solo_labels_g,
                                            self.walking_f_b_solo_labels_g,self.walking_side_solo_labels_g,
                                            self.waving_hands_solo_labels_g))
                #Xtrain, Xtest and Xval
                # Split the data into training and testing sets, ensuring balanced samples
                imb_X_train_g, imb_X_testval_g, imb_y_train_g, imb_y_testval_g = train_test_split(imb_Xdata_g, imb_Ydata_g, test_size=1-float(self.trainset_percent), stratify=imb_Ydata_g)           
                imb_X_test_g, imb_X_val_g, imb_y_test_g, imb_y_val_g = train_test_split(imb_X_testval_g, imb_y_testval_g, test_size=float(self.validationset_percent), stratify=imb_y_testval_g)
                bal_X_train_g, bal_X_testval_g, bal_y_train_g, bal_y_testval_g = train_test_split(bal_Xdata_g, bal_Ydata_g, test_size=1-float(self.trainset_percent), stratify=bal_Ydata_g)           
                bal_X_test_g, bal_X_val_g, bal_y_test_g, bal_y_val_g = train_test_split(bal_X_testval_g, bal_y_testval_g, test_size=float(self.validationset_percent), stratify=bal_y_testval_g)
                print('Imbalance Xdata Shape: ',imb_Xdata_g.shape)
                print('Imbalance Ydata Shape: ',imb_Ydata_g.shape)
                print('Imbalance Xtrain Shape: ',imb_X_train_g.shape)
                print('Imbalance Ytrain Shape: ',imb_y_train_g.shape)
                print('Imbalance Xtest Shape: ',imb_X_test_g.shape)
                print('Imbalance Ytest Shape: ',imb_y_test_g.shape)
                print('Imbalance Xval Shape: ',imb_X_val_g.shape)
                print('Imbalance Yval Shape: ',imb_y_val_g.shape)
                
                print('Balanced Sequence Num (Samples per class): ',balanced_seq_num_g)
                print('Balanced Xdata Shape: ',bal_Xdata_g.shape)
                print('Balanced Ydata Shape: ',bal_Ydata_g.shape)
                print('Balanced Xtrain Shape: ',bal_X_train_g.shape)
                print('Balanced Ytrain Shape: ',bal_y_train_g.shape)
                print('Balanced Xtest Shape: ',bal_X_test_g.shape)
                print('Balanced Ytest Shape: ',bal_y_test_g.shape)
                print('Balanced Xval Shape: ',bal_X_val_g.shape)
                print('Balanced Yval Shape: ',bal_y_val_g.shape)
                #Saving Xdata Ydata
                window_imb_folder_g = 'Global\Fixed_Window\Imbalanced_Sequences'
                window_bal_folder_g = 'Global\Fixed_Window\Balanced_Sequences'
                window_imb_path_g = os.path.join(self.output_transformation_dir,window_imb_folder_g)
                window_bal_path_g = os.path.join(self.output_transformation_dir,window_bal_folder_g)
                # Check if the directory exists
                if not os.path.exists(window_imb_path_g) and not os.path.exists(window_bal_path_g):
                    # If not, create it
                    os.makedirs(window_imb_path_g)
                    os.makedirs(window_bal_path_g)
                np.save(os.path.join(window_imb_path_g, "X_data.npy"), imb_Xdata_g)
                np.save(os.path.join(window_imb_path_g, "Y_data.npy"), imb_Ydata_g)
                np.save(os.path.join(window_imb_path_g, "X_train.npy"), imb_X_train_g)
                np.save(os.path.join(window_imb_path_g, "Y_train.npy"), imb_y_train_g)
                np.save(os.path.join(window_imb_path_g, "X_test.npy"), imb_X_test_g)
                np.save(os.path.join(window_imb_path_g, "Y_test.npy"), imb_y_test_g)
                np.save(os.path.join(window_imb_path_g, "X_val.npy"), imb_X_val_g)
                np.save(os.path.join(window_imb_path_g, "Y_val.npy"), imb_y_val_g)
                
                np.save(os.path.join(window_bal_path_g, "X_data.npy"), bal_Xdata_g)
                np.save(os.path.join(window_bal_path_g, "Y_data.npy"), bal_Ydata_g)
                np.save(os.path.join(window_bal_path_g, "X_train.npy"), bal_X_train_g)
                np.save(os.path.join(window_bal_path_g, "Y_train.npy"), bal_y_train_g)
                np.save(os.path.join(window_bal_path_g, "X_test.npy"), bal_X_test_g)
                np.save(os.path.join(window_bal_path_g, "Y_test.npy"), bal_y_test_g)
                np.save(os.path.join(window_bal_path_g, "X_val.npy"), bal_X_val_g)
                np.save(os.path.join(window_bal_path_g, "Y_val.npy"), bal_y_val_g)

    def reshaping_and_sequences_formation(self,dataset_name,class_name,class_npy_file_name,class_npy_file_path):
        if self.window_seq_check.get():
            #NORMALIZED
            #INTERACTIONS
            if self.action_type == 'interactions' and self.kps_type == 'normalized' and self.normalized_keypoints_check.get():
                if class_name == 'handshaking':
                    self.window_seq_formation(class_npy_file_path)
                    self.handshaking_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'hugging':
                    self.window_seq_formation(class_npy_file_path)
                    self.hugging_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'kicking':
                    self.window_seq_formation(class_npy_file_path)
                    self.kicking_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'punching':
                    self.window_seq_formation(class_npy_file_path)
                    self.punching_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'pushing':
                    self.window_seq_formation(class_npy_file_path)
                    self.pushing_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()
                    
            #SOLO ACTIONS
            if self.action_type == 'solo' and self.kps_type == 'normalized' and self.normalized_keypoints_check.get():
                if class_name == 'clapping':
                    self.window_seq_formation(class_npy_file_path)
                    self.clapping_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'hitting_bottle':
                    self.window_seq_formation(class_npy_file_path)
                    self.hitting_bottle_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'hitting_stick':
                    self.window_seq_formation(class_npy_file_path)
                    self.hitting_stick_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'jogging_f_b':
                    self.window_seq_formation(class_npy_file_path)
                    self.jogging_f_b_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'jogging_side':
                    self.window_seq_formation(class_npy_file_path)
                    self.jogging_side_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'kicking':
                    self.window_seq_formation(class_npy_file_path)
                    self.kicking_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'punching':
                    self.window_seq_formation(class_npy_file_path)
                    self.punching_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'running_f_b':
                    self.window_seq_formation(class_npy_file_path)
                    self.running_f_b_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'running_side':
                    self.window_seq_formation(class_npy_file_path)
                    self.running_side_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'stabbing':
                    self.window_seq_formation(class_npy_file_path)
                    self.stabbing_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'walking_f_b':
                    self.window_seq_formation(class_npy_file_path)
                    self.walking_f_b_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'walking_side':
                    self.window_seq_formation(class_npy_file_path)
                    self.walking_side_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'waving_hands':
                    self.window_seq_formation(class_npy_file_path)
                    self.waving_hands_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()
            
            #GLOBAL
            #INTERACTIONS
            if self.action_type == 'interactions' and self.kps_type == 'global' and self.global_keypoints_check.get():
                if class_name == 'handshaking':
                    self.window_seq_formation(class_npy_file_path)
                    self.handshaking_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'hugging':
                    self.window_seq_formation(class_npy_file_path)
                    self.hugging_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'kicking':
                    self.window_seq_formation(class_npy_file_path)
                    self.kicking_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'punching':
                    self.window_seq_formation(class_npy_file_path)
                    self.punching_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'pushing':
                    self.window_seq_formation(class_npy_file_path)
                    self.pushing_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()
                    
            #SOLO ACTIONS
            if self.action_type == 'solo' and self.kps_type == 'global' and self.global_keypoints_check.get():
                if class_name == 'clapping':
                    self.window_seq_formation(class_npy_file_path)
                    self.clapping_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'hitting_bottle':
                    self.window_seq_formation(class_npy_file_path)
                    self.hitting_bottle_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'hitting_stick':
                    self.window_seq_formation(class_npy_file_path)
                    self.hitting_stick_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'jogging_f_b':
                    self.window_seq_formation(class_npy_file_path)
                    self.jogging_f_b_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'jogging_side':
                    self.window_seq_formation(class_npy_file_path)
                    self.jogging_side_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'kicking':
                    self.window_seq_formation(class_npy_file_path)
                    self.kicking_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'punching':
                    self.window_seq_formation(class_npy_file_path)
                    self.punching_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'running_f_b':
                    self.window_seq_formation(class_npy_file_path)
                    self.running_f_b_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'running_side':
                    self.window_seq_formation(class_npy_file_path)
                    self.running_side_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'stabbing':
                    self.window_seq_formation(class_npy_file_path)
                    self.stabbing_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'walking_f_b':
                    self.window_seq_formation(class_npy_file_path)
                    self.walking_f_b_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'walking_side':
                    self.window_seq_formation(class_npy_file_path)
                    self.walking_side_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'waving_hands':
                    self.window_seq_formation(class_npy_file_path)
                    self.waving_hands_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()
            
            if self.in_progress_npy_num == self.total_npy_files:
                self.sequential_data_and_labels_formation()    
            
        
        if self.fixed_seq_check.get():
            #NORMALIZED
            #INTERACTIONS
            if self.action_type == 'interactions' and self.kps_type == 'normalized' and self.normalized_keypoints_check.get():
                if class_name == 'handshaking':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.handshaking_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'hugging':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.hugging_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'kicking':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.kicking_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'punching':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.punching_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'pushing':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.pushing_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()
                    
            #SOLO ACTIONS
            if self.action_type == 'solo' and self.kps_type == 'normalized' and self.normalized_keypoints_check.get():
                if class_name == 'clapping':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.clapping_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'hitting_bottle':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.hitting_bottle_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'hitting_stick':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.hitting_stick_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'jogging_f_b':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.jogging_f_b_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'jogging_side':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.jogging_side_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'kicking':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.kicking_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'punching':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.punching_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'running_f_b':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.running_f_b_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'running_side':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.running_side_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'stabbing':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.stabbing_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'walking_f_b':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.walking_f_b_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'walking_side':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.walking_side_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'waving_hands':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.waving_hands_solo_sequences_n.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()
            
            #GLOBAL
            #INTERACTIONS
            if self.action_type == 'interactions' and self.kps_type == 'global' and self.global_keypoints_check.get():
                if class_name == 'handshaking':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.handshaking_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'hugging':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.hugging_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'kicking':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.kicking_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'punching':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.punching_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'pushing':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.pushing_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()
                    
            #SOLO ACTIONS
            if self.action_type == 'solo' and self.kps_type == 'global' and self.global_keypoints_check.get():
                if class_name == 'clapping':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.clapping_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'hitting_bottle':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.hitting_bottle_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'hitting_stick':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.hitting_stick_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'jogging_f_b':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.jogging_f_b_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'jogging_side':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.jogging_side_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'kicking':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.kicking_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'punching':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.punching_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'running_f_b':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.running_f_b_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'running_side':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.running_side_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'stabbing':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.stabbing_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'walking_f_b':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.walking_f_b_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'walking_side':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.walking_side_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()

                if class_name == 'waving_hands':
                    self.fixed_seq_formation(class_npy_file_path)
                    self.waving_hands_solo_sequences_g.append(self.sequences)
                    self.in_progress_npy_num += 1
                    self.transformation_progress_bar_update()
            
            if self.in_progress_npy_num == self.total_npy_files:
                self.sequential_data_and_labels_formation()    
        


            

            

    def transformation_progress_bar_update(self):
        self.progress_bar['maximum'] = self.total_npy_files
        progress_text = f"{str(self.in_progress_npy_num)}/{str(self.total_npy_files)}: "
        self.progress_bar_label.config(text=progress_text)
        self.progress_bar['value'] = self.in_progress_npy_num
        self.master.update()

    def solo_transformation(self,datasets_folder_path):
        kps_type_folders = os.listdir(datasets_folder_path)
        for kps_type_folder in kps_type_folders:
            if self.event.is_set() or self.stop_event.is_set():
                break
            self.kps_type = kps_type_folder.split('_')[0]
            if self.kps_type == 'normalized' and self.normalized_keypoints_check.get():
                self.transformation_progress_bar_update()
                print('SOLO NORMALIZED ONLY')
                kps_type_folder_path = os.path.join(datasets_folder_path,kps_type_folder)
                solo_datasets = os.listdir(kps_type_folder_path)
                for solo_dataset in solo_datasets:
                    if self.event.is_set() or self.stop_event.is_set():
                        break
                    solo_dataset_path = os.path.join(kps_type_folder_path,solo_dataset)
                    dataset_classes_folders = os.listdir(solo_dataset_path)
                    for class_folder in dataset_classes_folders:
                        if self.event.is_set() or self.stop_event.is_set():
                            break
                        class_folder_path = os.path.join(solo_dataset_path,class_folder)
                        class_npy_files = os.listdir(class_folder_path)
                        for class_npy_file in class_npy_files:
                            if self.event.is_set() or self.stop_event.is_set():
                                break
                            class_npy_file_path = os.path.join(class_folder_path,class_npy_file)
                            self.reshaping_and_sequences_formation(solo_dataset,class_folder,class_npy_file,class_npy_file_path)

            #GLOBAL KPS
            if self.kps_type == 'global' and self.global_keypoints_check.get():
                print('SOLO-GLOBAL KPS ONLY')
                self.transformation_progress_bar_update()
                kps_type_folder_path = os.path.join(datasets_folder_path,kps_type_folder)
                solo_datasets = os.listdir(kps_type_folder_path)
                for solo_dataset in solo_datasets:
                    if self.event.is_set() or self.stop_event.is_set():
                        break
                    solo_dataset_path = os.path.join(kps_type_folder_path,solo_dataset)
                    dataset_classes_folders = os.listdir(solo_dataset_path)
                    for class_folder in dataset_classes_folders:
                        if self.event.is_set() or self.stop_event.is_set():
                            break
                        class_folder_path = os.path.join(solo_dataset_path,class_folder)
                        class_npy_files = os.listdir(class_folder_path)
                        for class_npy_file in class_npy_files:
                            if self.event.is_set() or self.stop_event.is_set():
                                break
                            class_npy_file_path = os.path.join(class_folder_path,class_npy_file)
                            self.reshaping_and_sequences_formation(solo_dataset,class_folder,class_npy_file,class_npy_file_path)


    def interactions_transformation(self,datasets_folder_path):
        kps_type_folders = os.listdir(datasets_folder_path)
        for kps_type_folder in kps_type_folders:
            if self.event.is_set() or self.stop_event.is_set():
                break
            self.kps_type = kps_type_folder.split('_')[0]
            if self.kps_type == 'normalized' and self.normalized_keypoints_check.get():
                self.transformation_progress_bar_update()
                print('INTERACTIONS NORMALIZED ONLY')
                kps_type_folder_path = os.path.join(datasets_folder_path,kps_type_folder)
                interactions_datasets = os.listdir(kps_type_folder_path)
                for interactions_dataset in interactions_datasets:
                    if self.event.is_set() or self.stop_event.is_set():
                        break
                    interactions_dataset_path = os.path.join(kps_type_folder_path,interactions_dataset)
                    dataset_classes_folders = os.listdir(interactions_dataset_path)
                    for class_folder in dataset_classes_folders:
                        if self.event.is_set() or self.stop_event.is_set():
                            break
                        class_folder_path = os.path.join(interactions_dataset_path,class_folder)
                        class_npy_files = os.listdir(class_folder_path)
                        for class_npy_file in class_npy_files:
                            if self.event.is_set() or self.stop_event.is_set():
                                break
                            class_npy_file_path = os.path.join(class_folder_path,class_npy_file)
                            self.reshaping_and_sequences_formation(interactions_dataset,class_folder,class_npy_file,class_npy_file_path)

            #GLOBAL KPS
            if self.kps_type == 'global' and self.global_keypoints_check.get():
                print('INTERACTIONS GLOBAL KPS ONLY')
                self.transformation_progress_bar_update()
                kps_type_folder_path = os.path.join(datasets_folder_path,kps_type_folder)
                interactions_datasets = os.listdir(kps_type_folder_path)
                for interactions_dataset in interactions_datasets:
                    if self.event.is_set() or self.stop_event.is_set():
                        break
                    interactions_dataset_path = os.path.join(kps_type_folder_path,interactions_dataset)
                    dataset_classes_folders = os.listdir(interactions_dataset_path)
                    for class_folder in dataset_classes_folders:
                        if self.event.is_set() or self.stop_event.is_set():
                            break
                        class_folder_path = os.path.join(interactions_dataset_path,class_folder)
                        class_npy_files = os.listdir(class_folder_path)
                        for class_npy_file in class_npy_files:
                            if self.event.is_set() or self.stop_event.is_set():
                                break
                            class_npy_file_path = os.path.join(class_folder_path,class_npy_file)
                            self.reshaping_and_sequences_formation(interactions_dataset,class_folder,class_npy_file,class_npy_file_path)

           
    def data_transformation(self):
        os.system('cls')
        print('DATA TRANSFORMATION')
        self.total_npy_files = self.count_total_npy_files()
        self.in_progress_npy_num = 0
        #self.transformation_progress_bar_update()
            
        datasets_folders = os.listdir(self.pose_dir)
        for datasets_folder in datasets_folders:
            if self.event.is_set() or self.stop_event.is_set():
                break
            self.action_type = datasets_folder.split('_')[0]
            datasets_folder_path = os.path.join(self.pose_dir,datasets_folder)
            if self.action_type == 'solo':
                self.solo_transformation(datasets_folder_path)
            elif self.action_type == 'interactions':
                self.interactions_transformation(datasets_folder_path)
            else:
                print(f'Action Type : ({self.action_type}) Not Found!')
        


    def run_transformation(self):
        self.transformation_complete_flag = False
        self.stop_event.clear()
        self.kps_indices_to_drop = []
        
        if (self.normalized_keypoints_check.get() or self.global_keypoints_check.get()) and self.pose_dir!="" and self.output_transformation_dir!="" and self.duplicate_frames_check.get()  and (self.duplicate_frames_threshold != [] and self.duplicate_frames_threshold != '' and 0<float(self.duplicate_frames_threshold)<=1)  and (self.sequence_length_value != [] and self.sequence_length_value != '' and 10<float(self.sequence_length_value)<=60) and self.validation_set_flag.get()  and (self.validationset_percent != [] and self.validationset_percent != '' and 0<float(self.validationset_percent)<=1) and (self.trainset_percent != [] and self.trainset_percent != '' and 0<float(self.trainset_percent)<=1):
            self.transformation_widgets_state_change()
            self.data_transformation()
            self.transformation_complete_flag = True
            self.transformation_widgets_state_change()
        
        elif (self.normalized_keypoints_check.get() or self.global_keypoints_check.get()) and self.pose_dir!="" and self.output_transformation_dir!="" and (not self.duplicate_frames_check.get())  and (self.sequence_length_value != [] and self.sequence_length_value != '' and 10<float(self.sequence_length_value)<=60) and self.validation_set_flag.get()  and (self.validationset_percent != [] and self.validationset_percent != '' and 0<float(self.validationset_percent)<=1) and (self.trainset_percent != [] and self.trainset_percent != '' and 0<float(self.trainset_percent)<=1):
            self.transformation_widgets_state_change()
            self.data_transformation()
            self.transformation_complete_flag = True
            self.transformation_widgets_state_change()
        
        elif (self.normalized_keypoints_check.get() or self.global_keypoints_check.get()) and self.pose_dir!="" and self.output_transformation_dir!="" and (self.duplicate_frames_check.get()) and (not self.validation_set_flag.get())  and (self.duplicate_frames_threshold != [] and self.duplicate_frames_threshold != '' and 0<float(self.duplicate_frames_threshold)<=1)  and (self.sequence_length_value != [] and self.sequence_length_value != '' and 10<float(self.sequence_length_value)<=60) and (self.trainset_percent != [] and self.trainset_percent != '' and 0<float(self.trainset_percent)<=1):
            self.transformation_widgets_state_change()
            self.data_transformation()
            self.transformation_complete_flag = True
            self.transformation_widgets_state_change()
        
        elif (self.normalized_keypoints_check.get() or self.global_keypoints_check.get()) and self.pose_dir!="" and self.output_transformation_dir!="" and (not self.duplicate_frames_check.get()) and (not self.validation_set_flag.get())  and (self.sequence_length_value != [] and self.sequence_length_value != '' and 10<float(self.sequence_length_value)<=60) and (self.trainset_percent != [] and self.trainset_percent != '' and 0<float(self.trainset_percent)<=1):
            self.transformation_widgets_state_change()
            self.data_transformation()
            self.transformation_complete_flag = True
            self.transformation_widgets_state_change()
        


    def update_button_styles_transformation_tab(self):
        if self.pose_dir == "":
            self.browse_pose_dir_button.configure(style="Red.TButton")
        else:
            self.browse_pose_dir_button.configure(style="Blue.TButton")
       
        if self.output_transformation_dir == "":
            self.browse_transformation_output_dir_button.configure(style="Red.TButton")
        else:
            self.browse_transformation_output_dir_button.configure(style="Blue.TButton")
        
        if (self.normalized_keypoints_check.get() or self.global_keypoints_check.get()) and (self.fixed_seq_check.get() or self.window_seq_check.get()) and self.pose_dir!="" and self.output_transformation_dir!="" and self.duplicate_frames_check.get()  and (self.duplicate_frames_threshold != [] and self.duplicate_frames_threshold != '' and 0<float(self.duplicate_frames_threshold)<=1) and (self.sequence_length_value != [] and self.sequence_length_value != '' and 10<float(self.sequence_length_value)<=60) and self.validation_set_flag.get()  and (self.validationset_percent != [] and self.validationset_percent != '' and 0<float(self.validationset_percent)<=1) and (self.trainset_percent != [] and self.trainset_percent != '' and 0<float(self.trainset_percent)<=1):
            self.generate_button.configure(style="Green.TButton")
        
        elif (self.normalized_keypoints_check.get() or self.global_keypoints_check.get()) and (self.fixed_seq_check.get() or self.window_seq_check.get()) and self.pose_dir!="" and self.output_transformation_dir!="" and (not self.duplicate_frames_check.get()) and self.validation_set_flag.get()  and (self.validationset_percent != [] and self.validationset_percent != '' and 0<float(self.validationset_percent)<=1) and (self.sequence_length_value != [] and self.sequence_length_value != '' and 10<float(self.sequence_length_value)<=60) and (self.trainset_percent != [] and self.trainset_percent != '' and 0<float(self.trainset_percent)<=1):
            self.generate_button.configure(style="Green.TButton")
        
        elif (self.normalized_keypoints_check.get() or self.global_keypoints_check.get()) and (self.fixed_seq_check.get() or self.window_seq_check.get()) and self.pose_dir!="" and self.output_transformation_dir!="" and (self.duplicate_frames_check.get()) and (not self.validation_set_flag.get())  and (self.duplicate_frames_threshold != [] and self.duplicate_frames_threshold != '' and 0<float(self.duplicate_frames_threshold)<=1) and (self.trainset_percent != [] and self.trainset_percent != '' and 0<float(self.trainset_percent)<=1):
            self.generate_button.configure(style="Green.TButton")

        elif (self.normalized_keypoints_check.get() or self.global_keypoints_check.get()) and (self.fixed_seq_check.get() or self.window_seq_check.get()) and self.pose_dir!="" and self.output_transformation_dir!="" and (not self.duplicate_frames_check.get()) and (not self.validation_set_flag.get()) and (self.trainset_percent != [] and self.trainset_percent != '' and 0<float(self.trainset_percent)<=1):
            self.generate_button.configure(style="Green.TButton")
            
        else:
            self.generate_button.configure(style="Red.TButton")
        

        # if self.output_preprocess_dir == "":
        #     self.browse_preprocess_output_dir_button.configure(style="Red.TButton")
        # else:
        #     self.browse_preprocess_output_dir_button.configure(style="Blue.TButton")
        
        # if self.pose_model_path!="" and self.dataset_dir!="" and self.output_preprocess_dir!="" and (self.confidence_score != [] and self.confidence_score != '' and 0<float(self.confidence_score)<=1) and (self.normalized_keypoints_check.get()==True or self.global_keypoints_check.get()==True):
        #     self.run_button.configure(style="Green.TButton")
        # else:
        #     self.run_button.configure(style="Red.TButton")
                


#--------------------------PREPROCESSING BLOCK-------------------------------------------
#----------------------------------------------------------------------------------    
    def setup_preprocessing_tab(self):
        
        preprocessing_tab = self.tabs["Preprocessing - Pose Extraction"]
        
        # Style for buttons
        style = ttk.Style()
        style.configure("Red.TButton", foreground="black", background="red")
        style.configure("Blue.TButton", foreground="black", background="blue")
        style.configure("Green.TButton", foreground="black", background="green")
        style.configure("Orange.TButton", foreground="black", background="orange")
        style.configure("Yellow.TButton", foreground="black", background="yellow")
        
        # Buttons for browsing pose model, video directory, and output directory
        self.browse_pose_model_button = ttk.Button(preprocessing_tab, text="Browse Pose Model", command=self.browse_pose_model, style="Red.TButton")
        self.browse_pose_model_button.place(x=100, y=10)
        
        self.browse_video_dir_button = ttk.Button(preprocessing_tab, text="Browse Datasets Directory", command=self.browse_video_directory, style="Red.TButton")
        self.browse_video_dir_button.place(x=280, y=10)
        
        self.browse_preprocess_output_dir_button = ttk.Button(preprocessing_tab, text="Browse Preprocessing Output Directory", command=self.browse_output_directory, style="Red.TButton")
        self.browse_preprocess_output_dir_button.place(x=500, y=10)
        
        # Frames for original video and keypoints video
        original_frame_label = tk.Label(preprocessing_tab, text="Original Video", font=("Arial", 10, "bold"))
        original_frame_label.place(x=150, y=50)
        
        self.original_frame_container = tk.Frame(preprocessing_tab, bd=2, relief=tk.GROOVE)
        self.original_frame_container.place(x=40, y=80)
        self.original_frame_canvas = tk.Canvas(self.original_frame_container, width=self.frame_size, height=self.frame_size)
        self.original_frame_canvas.pack(fill=tk.BOTH, expand=True)
        
        keypoints_frame_label = tk.Label(preprocessing_tab, text="Keypoints Video", font=("Arial", 10, "bold"))
        keypoints_frame_label.place(x=550, y=50)
        
        self.keypoints_frame_container = tk.Frame(preprocessing_tab, bd=2, relief=tk.GROOVE)
        self.keypoints_frame_container.place(x=430, y=80)
        self.keypoints_frame_canvas = tk.Canvas(self.keypoints_frame_container, width=self.frame_size, height=self.frame_size)
        self.keypoints_frame_canvas.pack(fill=tk.BOTH, expand=True)
          
        self.confidence_label = ttk.Label(preprocessing_tab, text="Confidence Score (0-1):")
        self.confidence_label.place(x=335+130,y=432)
        # Create a frame for the entry widget with a border
        self.confidence_entry_frame = ttk.Frame(preprocessing_tab, borderwidth=2, relief="solid")
        self.confidence_entry_frame.place(x=465+130, y=428)
        # Create the entry widget inside the frame
        self.confidence_entry = ttk.Entry(self.confidence_entry_frame, width=5)
        self.confidence_entry.pack()        
        self.confidence_entry.bind("<KeyRelease>", self.conf_score)
         # Add checkboxes
        self.normalized_checkbox = tk.Checkbutton(preprocessing_tab, text="Normalized Keypoints",  variable=self.normalized_keypoints_check, command=self.toggle_normalized)
        self.normalized_checkbox.place(x=50, y=430)
        self.global_checkbox = tk.Checkbutton(preprocessing_tab, text="Global Keypoints", variable=self.global_keypoints_check, command=self.toggle_global)
        self.global_checkbox.place(x=200, y=430)
        self.overwrite_keypoints_checkbox = tk.Checkbutton(preprocessing_tab, text="Overwrite Keypoints", variable=self.overwrite_keypoints_check, command=self.overwrite_keypoints)
        self.overwrite_keypoints_checkbox.place(x=320, y=430)
        

        progress_text = str(self.in_progress_num)+ '/' + str(self.total_videos) + ': '
        self.progress_bar_label = tk.Label(preprocessing_tab, text=progress_text, font=("Arial", 10, "bold"))
        self.progress_bar_label.place(x=50, y=470)
        
        self.progress_bar = ttk.Progressbar(preprocessing_tab, length=600, mode="determinate")
        self.progress_bar.place(x=100, y=470)
        
        # Buttons for running, stopping, and quitting
        self.run_button = ttk.Button(preprocessing_tab, text="Run", command=self.run_preprocessing, width=10, style="Red.TButton")
        self.run_button.place(x=100, y=510)
        
        self.stop_button = ttk.Button(preprocessing_tab, text="Stop", command=self.stop_preprocessing, width=10, style="Red.TButton")
        self.stop_button.place(x=200, y=510)
        self.stop_button.config(state="disabled")

        self.quit_button = ttk.Button(preprocessing_tab, text="Quit", command=self.quit_gui, width=10, style="Red.TButton")
        self.quit_button.place(x=680, y=510)
        
    # def setup_hyperparameters_tuning_tab(self):
    #     # Your code for setting up the Hyperparameters Tuning tab
    #     pass
        
    # def setup_model_training_tab(self):
    #     # Your code for setting up the Model Training tab
    #     pass


    def quit_gui(self):
        self.event.set()
        self.master.quit()

    def stop_preprocessing(self):
        self.stop_event.set()


    def conf_score(self,event):
        self.confidence_score = self.confidence_entry.get()
        if self.current_tab == 'Preprocessing - Pose Extraction':
            self.update_button_styles()
        elif self.current_tab == 'Inference':
            print('CF: ',self.confidence_score)
            self.update_button_styles_inference_tab()

    def toggle_normalized(self):
        self.normalized_keypoints_check.set(self.normalized_keypoints_check.get())
        self.update_button_styles()

    def toggle_global(self):
        self.global_keypoints_check.set(self.global_keypoints_check.get())
        self.update_button_styles()

    def overwrite_keypoints(self):
        self.overwrite_keypoints_check.set(self.overwrite_keypoints_check.get())
        self.update_button_styles()

    def load_pose_model(self):
        self.pose_model = YOLO(self.pose_model_path)   
        
    def browse_pose_model(self):
        self.pose_model_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pt")])
        if self.pose_model_path!="":
            self.load_pose_model()
        self.update_button_styles()
        
    def browse_video_directory(self):
        self.dataset_dir = filedialog.askdirectory()
        self.update_button_styles()
    
    def browse_output_directory(self):
        self.output_preprocess_dir = filedialog.askdirectory()
        self.update_button_styles()
    

    def count_video_files(self):
        total_video_files = 0
        stack = [self.dataset_dir]

        while stack:
            current_dir = stack.pop()
            try:
                subfolders = os.listdir(current_dir)
            except PermissionError:
                continue

            mp4_count = 0
            avi_count = 0

            for item in subfolders:
                item_path = os.path.join(current_dir, item)
                if os.path.isdir(item_path):
                    stack.append(item_path)
                elif item.endswith('.mp4'):
                    mp4_count += 1
                elif item.endswith('.avi'):
                    avi_count += 1
            
            if mp4_count > 0 or avi_count > 0:
                total_video_files += mp4_count + avi_count

        return total_video_files
    
    def training_update_progress_bar(self):
        if not self.stop_event.is_set():
            progress_text = f"{str(self.in_progress_epoch)}/{str(self.epochs)}: "
            self.progress_bar_label.config(text=progress_text)
            self.progress_bar['value'] = self.in_progress_num
    
    def update_progress_bar(self):
        if not self.stop_event.is_set():
            progress_text = f"{str(self.in_progress_num)}/{str(self.total_videos)}: "
            self.progress_bar_label.config(text=progress_text)
            self.progress_bar['value'] = self.in_progress_num
    
    def check_video_npy_file(self,dataset_name,class_folder_name,video_file_name):
        
        if self.action_type == 'solo':
            video_file_name_npy = video_file_name + '.npy'

            if self.normalized_keypoints_check.get() and self.global_keypoints_check.get():
                kps_type_n = 'normalized_kps'
                kps_type_g = 'global_kps'
                datasets_type = 'solo_actions_datasets'
                video_npy_path_n = os.path.join(self.output_preprocess_dir,datasets_type,kps_type_n,dataset_name,class_folder_name,video_file_name_npy)
                video_npy_path_g = os.path.join(self.output_preprocess_dir,datasets_type,kps_type_g,dataset_name,class_folder_name,video_file_name_npy)
                # Check if exists
                if not os.path.exists(video_npy_path_n):
                    self.solo_keypoints(dataset_name,class_folder_name,video_file_name)

                if not os.path.exists(video_npy_path_g):
                    self.solo_keypoints(dataset_name,class_folder_name,video_file_name)
                self.in_progress_num+=1


            if self.normalized_keypoints_check.get() and not self.global_keypoints_check.get():
                kps_type = 'normalized_kps'
                datasets_type = 'solo_actions_datasets'
                video_npy_path_n = os.path.join(self.output_preprocess_dir,datasets_type,kps_type,dataset_name,class_folder_name,video_file_name_npy)
                # Check if exists
                if not os.path.exists(video_npy_path_n):
                    self.solo_keypoints(dataset_name,class_folder_name,video_file_name)
                self.in_progress_num+=1
                    
            if self.global_keypoints_check.get() and not self.normalized_keypoints_check.get():
                kps_type = 'global_kps'
                datasets_type = 'solo_actions_datasets'
                video_npy_path_g = os.path.join(self.output_preprocess_dir,datasets_type,kps_type,dataset_name,class_folder_name,video_file_name_npy)
                # Check if exists
                if not os.path.exists(video_npy_path_g):
                    self.solo_keypoints(dataset_name,class_folder_name,video_file_name)
                self.in_progress_num+=1
                

        if self.action_type == 'interactions':
            video_file_name_npy = video_file_name + '.npy'

            if self.normalized_keypoints_check.get() and self.global_keypoints_check.get():
                kps_type_n = 'normalized_kps'
                kps_type_g = 'global_kps'
                datasets_type = 'interactions_datasets'
                video_npy_path_n = os.path.join(self.output_preprocess_dir,datasets_type,kps_type_n,dataset_name,class_folder_name,video_file_name_npy)
                video_npy_path_g = os.path.join(self.output_preprocess_dir,datasets_type,kps_type_g,dataset_name,class_folder_name,video_file_name_npy)
                # Check if exists
                if not os.path.exists(video_npy_path_n):
                    self.interactions_keypoints(dataset_name,class_folder_name,video_file_name)
                    
                if not os.path.exists(video_npy_path_g):
                    self.interactions_keypoints(dataset_name,class_folder_name,video_file_name)
                self.in_progress_num+=1
                    
            if self.normalized_keypoints_check.get() and not self.global_keypoints_check.get():
                kps_type = 'normalized_kps'
                datasets_type = 'interactions_datasets'
                video_npy_path_n = os.path.join(self.output_preprocess_dir,datasets_type,kps_type,dataset_name,class_folder_name,video_file_name_npy)
                # Check if exists
                if not os.path.exists(video_npy_path_n):
                    self.interactions_keypoints(dataset_name,class_folder_name,video_file_name)
                self.in_progress_num+=1
                    
            if self.global_keypoints_check.get() and not self.normalized_keypoints_check.get():
                kps_type = 'global_kps'
                datasets_type = 'interactions_datasets'
                video_npy_path_g = os.path.join(self.output_preprocess_dir,datasets_type,kps_type,dataset_name,class_folder_name,video_file_name_npy)
                # Check if exists
                if not os.path.exists(video_npy_path_g):
                    self.interactions_keypoints(dataset_name,class_folder_name,video_file_name)
                self.in_progress_num+=1
                    
    def interactions_keypoints(self,dataset_name,class_folder_name,video_file_name):
        
        if self.normalized_keypoints_check.get() and self.global_keypoints_check.get() and not self.event.is_set() and not self.stop_event.is_set():
            datasets_type = 'interactions_datasets'
            kps_type_n = 'normalized_kps'
            kps_type_g = 'global_kps'
            output_directory_n = os.path.join(self.output_preprocess_dir,datasets_type,kps_type_n,dataset_name,class_folder_name)
            output_directory_g = os.path.join(self.output_preprocess_dir,datasets_type,kps_type_g,dataset_name,class_folder_name)
            # Check if the directory exists
            if not os.path.exists(output_directory_n) and not os.path.exists(output_directory_g):
                # If the directory doesn't exist, create it along with any necessary parent directories
                os.makedirs(output_directory_n,exist_ok=True)
                os.makedirs(output_directory_g,exist_ok=True)


        if self.normalized_keypoints_check.get() and not self.global_keypoints_check.get() and not self.event.is_set() and not self.stop_event.is_set():
            datasets_type = 'interactions_datasets'
            kps_type = 'normalized_kps'
            output_directory_n = os.path.join(self.output_preprocess_dir,datasets_type,kps_type,dataset_name,class_folder_name)
            # Check if the directory exists
            if not os.path.exists(output_directory_n):
                # If the directory doesn't exist, create it along with any necessary parent directories
                os.makedirs(output_directory_n,exist_ok=True)
                

        if self.global_keypoints_check.get() and not self.normalized_keypoints_check.get() and not self.event.is_set()  and not self.stop_event.is_set():
            datasets_type = 'interactions_datasets'
            kps_type = 'global_kps'
            output_directory_g = os.path.join(self.output_preprocess_dir,datasets_type,kps_type,dataset_name,class_folder_name)
            # Check if the directory exists
            if not os.path.exists(output_directory_g):
                # If the directory doesn't exist, create it along with any necessary parent directories
                os.makedirs(output_directory_g,exist_ok=True)

        video_normalized_kps = []
        video_global_kps = []
        frame = []
        results = []
        results_gray_image_frame_resized = []
        global_kps = []
        global_kps_frame_resized = []

        while(not self.stop_event.is_set()):
            ret, frame = self.video_capture.read()
            if self.event.is_set() or self.stop_event.is_set():
                break

            if ret:
                original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #MONO CHANNEL
                gray_image = cv2.merge([gray_image, gray_image, gray_image]) #THREE CHANNELS GRAYSCALE for YOLO
                original_image_frame_resized = cv2.resize(original_image, (self.frame_size, self.frame_size))
                gray_image_frame_resized = cv2.resize(gray_image, (self.frame_size, self.frame_size))
                
                results = self.pose_model(gray_image, conf = float(self.confidence_score),save=False)
                results_gray_image_frame_resized = self.pose_model(gray_image_frame_resized,conf = float(self.confidence_score),save=False)
                
                detected_boxes = results[0].boxes
                num_persons_detected = len(detected_boxes)
                normalized_kps_tensor = results[0].keypoints.xyn.squeeze() 
                global_kps_tensor = results[0].keypoints.xy.squeeze()
                global_kps_tensor_frame_resized = results_gray_image_frame_resized[0].keypoints.xy.squeeze()
                normalized_kps = normalized_kps_tensor.numpy()
                global_kps = global_kps_tensor.numpy()
                global_kps_frame_resized = global_kps_tensor_frame_resized.numpy()
                if len(global_kps) == 0 or len(global_kps_frame_resized)==0:
                    continue   
                
                elif num_persons_detected == 2 and global_kps[0].shape[0]==17 and global_kps[1].shape[0]==17 and global_kps_frame_resized[0].shape[0]==17 and global_kps_frame_resized[1].shape[0]==17:
                    gray_image_frame_resized = np.array(gray_image_frame_resized) #To write keypoints on it
                    colors = self.pose_kps_colors()
                    # Draw keypoints on the frame
                    for n_person_kps in global_kps_frame_resized:
                        for i, (x, y) in enumerate(n_person_kps):
                            cv2.circle(gray_image_frame_resized, (int(x), int(y)), 4, self.hex_to_rgb(colors[i]), -1)          
                    
                    gray_image_frame_resized = Image.fromarray(gray_image_frame_resized) #Back converted to PIL Image
                    gray_image_frame_resized = ImageTk.PhotoImage(gray_image_frame_resized)
                    
                    original_image_frame_resized = Image.fromarray(original_image_frame_resized)
                    original_image_frame_resized = ImageTk.PhotoImage(original_image_frame_resized)
                    self.original_frame_canvas.create_image(0, 0, anchor=tk.NW, image=original_image_frame_resized)
                    self.keypoints_frame_canvas.create_image(0, 0, anchor=tk.NW, image=gray_image_frame_resized)
                    self.original_frame_canvas.image = original_image_frame_resized
                    self.keypoints_frame_canvas.image = gray_image_frame_resized
                    #self.master.update_idletasks()  # Update the GUI
                    self.master.update()
                    if self.normalized_keypoints_check.get():
                        video_normalized_kps.append(normalized_kps)
                    
                    if self.global_keypoints_check.get():
                        video_global_kps.append(global_kps)
                
                else: #Skip Frame with No Two Persons
                        continue
            else:#Video Ends
                break 
        self.video_capture.release()
        video_normalized_kps = np.array(video_normalized_kps)
        video_global_kps = np.array(video_global_kps)

        
        if self.normalized_keypoints_check.get() and self.global_keypoints_check.get() and not self.event.is_set() and not self.stop_event.is_set():
            video_file_name = video_file_name + '.npy'
            kps_path_n = os.path.join(output_directory_n,video_file_name)
            kps_path_g = os.path.join(output_directory_g,video_file_name)
            np.save(kps_path_n,video_normalized_kps)
            np.save(kps_path_g,video_global_kps)

        if self.normalized_keypoints_check.get() and not self.global_keypoints_check.get() and not self.event.is_set() and not self.stop_event.is_set():
            video_file_name = video_file_name + '.npy'
            kps_path = os.path.join(output_directory_n,video_file_name)
            np.save(kps_path,video_normalized_kps)

        if self.global_keypoints_check.get() and not self.normalized_keypoints_check.get() and not self.event.is_set()  and not self.stop_event.is_set():
            video_file_name = video_file_name + '.npy'
            kps_path = os.path.join(output_directory_g,video_file_name)
            np.save(kps_path,video_global_kps)

    def solo_keypoints(self,dataset_name,class_folder_name,video_file_name):
        
        if self.normalized_keypoints_check.get() and self.global_keypoints_check.get() and not self.event.is_set() and not self.stop_event.is_set():
            datasets_type = 'solo_actions_datasets'
            kps_type_n = 'normalized_kps'
            kps_type_g = 'global_kps'
            output_directory_n = os.path.join(self.output_preprocess_dir,datasets_type,kps_type_n,dataset_name,class_folder_name)
            output_directory_g = os.path.join(self.output_preprocess_dir,datasets_type,kps_type_g,dataset_name,class_folder_name)
            if not os.path.exists(output_directory_n) and not os.path.exists(output_directory_g):
                os.makedirs(output_directory_n,exist_ok=True)
                os.makedirs(output_directory_g,exist_ok=True)

            
        if self.normalized_keypoints_check.get() and not self.global_keypoints_check.get() and not self.event.is_set() and not self.stop_event.is_set():
            datasets_type = 'solo_actions_datasets'
            kps_type = 'normalized_kps'
            output_directory_n = os.path.join(self.output_preprocess_dir,datasets_type,kps_type,dataset_name,class_folder_name)
            # Check if the directory exists
            if not os.path.exists(output_directory_n):
                # If the directory doesn't exist, create it along with any necessary parent directories
                os.makedirs(output_directory_n,exist_ok=True)

        if self.global_keypoints_check.get() and not self.normalized_keypoints_check.get() and not self.event.is_set()  and not self.stop_event.is_set():
            datasets_type = 'solo_actions_datasets'
            kps_type = 'global_kps'
            output_directory_g = os.path.join(self.output_preprocess_dir,datasets_type,kps_type,dataset_name,class_folder_name)
            # Check if the directory exists
            if not os.path.exists(output_directory_g):
                # If the directory doesn't exist, create it along with any necessary parent directories
                os.makedirs(output_directory_g,exist_ok=True)

        video_normalized_kps = []
        video_global_kps = []
        frame = []
        results = []
        results_gray_image_frame_resized = []
        global_kps = []
        global_kps_frame_resized = []
    
        while(not self.stop_event.is_set()):
            ret, frame = self.video_capture.read()
            if self.event.is_set() or self.stop_event.is_set():
                break

            if ret:
                original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #MONO CHANNEL
                gray_image = cv2.merge([gray_image, gray_image, gray_image]) #THREE CHANNELS GRAYSCALE for YOLO
                original_image_frame_resized = cv2.resize(original_image, (self.frame_size, self.frame_size))
                gray_image_frame_resized = cv2.resize(gray_image, (self.frame_size, self.frame_size))
                
                results = self.pose_model(gray_image, conf = float(self.confidence_score),save=False)
                results_gray_image_frame_resized = self.pose_model(gray_image_frame_resized,conf = float(self.confidence_score),save=False)
                
                detected_boxes = results[0].boxes
                num_persons_detected = len(detected_boxes)
                normalized_kps_tensor = results[0].keypoints.xyn.squeeze() 
                global_kps_tensor = results[0].keypoints.xy.squeeze()
                global_kps_tensor_frame_resized = results_gray_image_frame_resized[0].keypoints.xy.squeeze()
                normalized_kps = normalized_kps_tensor.numpy()
                global_kps = global_kps_tensor.numpy()
                global_kps_frame_resized = global_kps_tensor_frame_resized.numpy()
                if num_persons_detected == 1 and normalized_kps.ndim == 2 and global_kps.ndim==2 and global_kps_frame_resized.ndim==2:
                    # print('Normalized KPS: ',normalized_kps)
                    # print('Global KPS: ',global_kps)
                    # print('Global KPS Resized: ',global_kps_frame_resized)
                    gray_image_frame_resized = np.array(gray_image_frame_resized) #To write keypoints on it                  
                    # Draw keypoints on the frame
                    colors = self.pose_kps_colors()
                    for i, (x, y) in enumerate(global_kps_frame_resized):
                        cv2.circle(gray_image_frame_resized, (int(x), int(y)), 4, self.hex_to_rgb(colors[i]), -1)  # Draw a green circle around each keypoint        

                    gray_image_frame_resized = Image.fromarray(gray_image_frame_resized) #Back converted to PIL Image
                    gray_image_frame_resized = ImageTk.PhotoImage(gray_image_frame_resized)
                    original_image_frame_resized = Image.fromarray(original_image_frame_resized)
                    original_image_frame_resized = ImageTk.PhotoImage(original_image_frame_resized)

                    self.original_frame_canvas.create_image(0, 0, anchor=tk.NW, image=original_image_frame_resized)
                    self.keypoints_frame_canvas.create_image(0, 0, anchor=tk.NW, image=gray_image_frame_resized)
                    self.original_frame_canvas.image = original_image_frame_resized
                    self.keypoints_frame_canvas.image = gray_image_frame_resized
                    self.master.update()

                    if self.normalized_keypoints_check.get():
                        video_normalized_kps.append(normalized_kps)
                    
                    if self.global_keypoints_check.get():
                        video_global_kps.append(global_kps)
                else: #Skip Frame with No Two Persons
                    continue
    
            else:#Video Ends
                break
        self.video_capture.release()
        video_normalized_kps = np.array(video_normalized_kps)
        video_global_kps = np.array(video_global_kps)

        
        if self.normalized_keypoints_check.get() and self.normalized_keypoints_check.get() and not self.event.is_set() and not self.stop_event.is_set():
            video_file_name = video_file_name + '.npy'
            kps_path_n = os.path.join(output_directory_n,video_file_name)
            kps_path_g = os.path.join(output_directory_g,video_file_name)
            np.save(kps_path_n,video_normalized_kps)
            np.save(kps_path_g,video_global_kps)

        if self.normalized_keypoints_check.get() and not self.global_keypoints_check and not self.event.is_set() and not self.stop_event.is_set():
            video_file_name = video_file_name + '.npy'
            kps_path = os.path.join(output_directory_n,video_file_name)
            np.save(kps_path,video_normalized_kps)

        if self.global_keypoints_check.get() and not self.normalized_keypoints_check.get() and not self.event.is_set()  and not self.stop_event.is_set():
            video_file_name = video_file_name + '.npy'
            kps_path = os.path.join(output_directory_g,video_file_name)
            np.save(kps_path,video_global_kps)

    def interactions_keypoints_extraction(self,folder):
        folder_path = os.path.join(self.dataset_dir,folder)
        interactions_datasets_name = os.listdir(folder_path)
        for dataset in interactions_datasets_name:
            if self.event.is_set() or self.stop_event.is_set():
                break
            dataset_path = os.path.join(folder_path,dataset)
            class_folders = os.listdir(dataset_path)
            for class_folder in class_folders:
                if self.event.is_set() or self.stop_event.is_set():
                    break
                class_folder_path = os.path.join(dataset_path,class_folder)
                videos_list = os.listdir(class_folder_path)
                for video in videos_list:
                    if self.event.is_set() or self.stop_event.is_set():
                        break
                    video_filename = os.path.splitext(video)[0]
                    video_path = os.path.join(class_folder_path,video)
                    self.video_capture = cv2.VideoCapture(video_path)
                    if self.overwrite_keypoints_check.get():
                        self.interactions_keypoints(dataset,class_folder,video_filename)
                    else:
                        self.check_video_npy_file(dataset,class_folder,video_filename)

                    self.update_progress_bar()


    def solo_actions_keypoints_extraction(self,folder):
        folder_path = os.path.join(self.dataset_dir,folder)
        solo_datasets_name = os.listdir(folder_path)
        for dataset in solo_datasets_name:
            if self.event.is_set() or self.stop_event.is_set():
                break
            dataset_path = os.path.join(folder_path,dataset)
            class_folders = os.listdir(dataset_path)
            for class_folder in class_folders:
                if self.event.is_set() or self.stop_event.is_set():
                    break
                class_folder_path = os.path.join(dataset_path,class_folder)
                videos_list = os.listdir(class_folder_path)
                for video in videos_list:
                    if self.event.is_set() or self.stop_event.is_set():
                        break
                    video_filename = os.path.splitext(video)[0]
                    video_path = os.path.join(class_folder_path,video)
                    self.video_capture = cv2.VideoCapture(video_path)
                    if self.overwrite_keypoints_check.get():
                        self.solo_keypoints(dataset,class_folder,video_filename)
                    else:
                        self.check_video_npy_file(dataset,class_folder,video_filename)

                    self.update_progress_bar()


    def _keypoints_extraction(self):
        os.system('cls')
        self.in_progress_num = 0
        self.total_videos = self.count_video_files()
        self.progress_bar['maximum'] = self.total_videos
        progress_text = f"{str(self.in_progress_num)}/{str(self.total_videos)}: "
        self.progress_bar_label.config(text=progress_text)
        self.dataset_folders = os.listdir(self.dataset_dir)

        for folder in self.dataset_folders:
            if self.event.is_set() or self.stop_event.is_set():
                break
            self.action_type = folder.split('_')[0]
            if self.action_type == 'solo':
                self.solo_actions_keypoints_extraction(folder)
            elif self.action_type == 'interactions':
                self.interactions_keypoints_extraction(folder)
            else:
                print(f'Action Type : ({self.action_type}) Not Found!')

    def widgets_state_change(self):

        if not self.pose_extraction_complete_flag:
            total_tabs = self.notebook.index('end')
            for tab_no in range(total_tabs):
                if tab_no!=self.notebook.index(self.notebook.select()):
                    self.notebook.tab(tab_no,state='disabled') 
            self.in_progress_num = 0
            self.progress_bar['value'] = self.in_progress_num
            self.browse_pose_model_button.config(state="disabled")
            self.browse_video_dir_button.config(state="disabled")
            self.browse_preprocess_output_dir_button.config(state="disabled")
            self.confidence_entry.config(state="disabled")
            self.normalized_checkbox.config(state="disabled")
            self.global_checkbox.config(state="disabled")
            self.overwrite_keypoints_checkbox.config(state="disabled")    
            self.run_button.config(state="disabled")    
            self.stop_button.config(state="normal")
        else:
            total_tabs = self.notebook.index('end')
            for tab_no in range(total_tabs):
                if tab_no!=self.notebook.index(self.notebook.select()):
                    self.notebook.tab(tab_no,state='normal') 
            self.browse_pose_model_button.config(state="normal")
            self.browse_video_dir_button.config(state="normal")
            self.browse_preprocess_output_dir_button.config(state="normal")
            self.confidence_entry.config(state="normal")
            self.normalized_checkbox.config(state="normal")
            self.global_checkbox.config(state="normal")
            self.overwrite_keypoints_checkbox.config(state="normal")    
            self.run_button.config(state="normal")    
            self.stop_button.config(state="disabled")
            self.update_button_styles()

    #PRE PROCESSING TAB MAIN FUNCTION   
    def run_preprocessing(self):
        self.pose_extraction_complete_flag = False
        self.stop_event.clear()
        if self.pose_model_path!="" and self.dataset_dir!="" and self.output_preprocess_dir!="" and (self.confidence_score != [] and self.confidence_score != '' and 0<float(self.confidence_score)<=1) and (self.normalized_keypoints_check.get()==True or self.global_keypoints_check.get()==True):
            self.widgets_state_change()
            self.thread = threading.Thread(target=self._keypoints_extraction(),args=(self.event,))
            self.thread.start()
            self.pose_extraction_complete_flag = True
            self.widgets_state_change()
            
            
    def update_button_styles(self):
        if self.pose_model_path == "":
            self.browse_pose_model_button.configure(style="Red.TButton")
        else:
            self.browse_pose_model_button.configure(style="Blue.TButton")
        
        if self.dataset_dir == "":
            self.browse_video_dir_button.configure(style="Red.TButton")
        else:
            self.browse_video_dir_button.configure(style="Blue.TButton")
       
        if self.output_preprocess_dir == "":
            self.browse_preprocess_output_dir_button.configure(style="Red.TButton")
        else:
            self.browse_preprocess_output_dir_button.configure(style="Blue.TButton")
        
        if self.pose_model_path!="" and self.dataset_dir!="" and self.output_preprocess_dir!="" and (self.confidence_score != [] and self.confidence_score != '' and 0<float(self.confidence_score)<=1) and (self.normalized_keypoints_check.get() or self.global_keypoints_check.get()):
            self.run_button.configure(style="Green.TButton")
        else:
            self.run_button.configure(style="Red.TButton")
            
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayer(root)
    root.mainloop()
