# IMPORT ALL LIBRARIES
import threading
import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import numpy as np
import os
import time
import absl.logging
import argparse
import numpy as np
import tensorflow as tf

# CUSTOM LIBRARIES
from utils.tools import read_yaml
from utils.trainer_inference import Trainer

#HIDE WARNINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
absl.logging.set_verbosity(absl.logging.ERROR)

os.environ['HOME'] = os.getcwd()

#GET ACTIVATION FUNCTION
def get_activation_function(name):
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

class VideoPlayer:
    def __init__(self, master):
        self.master = master
        self.master.title("InterAcT")
        
        self.stop = False
        #CLASS LABELS
        self.train_xdata = np.load(r'traintest_data\2P_AcT_train_test_val\trainset_data.npy')
        self.classes_names = ['handshaking','hugging','kicking','punching','pushing','clapping_solo','hitting_bottle_solo','hitting_stick_solo','jogging_f_b_solo','jogging_side_solo','kicking_solo','punching_solo','running_f_b_solo','running_side_solo','stabbing_solo','walking_f_b_solo','walking_side_solo','waving_hands_solo']
        self.num_classes = 18 #Number of Classes
    
        self.VAL_FLAG = 0
        #MODEL ARCHITECTURE SETTINGS 
        self.m = 'micro' #Model Architecture Name
        self.n_layers = 4
        self.emb_dim = 2
        self.dropout = 0.2
        self.mlp_dim = 128

        #TRAINING SETTINGS
        self.lr = 10**-2 #Learning Rate
        self.wd = 1e-06 #Weight Decay
        self.batch = 32 #Batch-size
        self.best_activation = get_activation_function('gelu') #Activation Function Name
        self.best_opt_name = 'SGDW' #Optimizer Name
        self.target_size = 320 #GUI Video Frames Size (i.e. self.target_size x self.target_size)
        self.model = None
        self.pose_model = None

        #PREDICTIONS AND GUI VARIABLES
        self.predicted_output_label = None
        self.video_keypoints = []
        self.predicted_label = None
        self.predicted_label_txt = 'None'
        self.video_path = ""
        self.pose_model_path = ""
        self.model_weights_path = ""
        self.video_capture = None
        self.thread = None
        self.playing = False
        self.video_complete = False
        
        #GUI VIDEO FRAMES, LABELS AND THEIR POSITIONS
        self.original_frame_container = tk.Frame(master, bd=2, relief=tk.GROOVE)  # Container for Original RGB Video frame
        self.gray_frame_container = tk.Frame(master, bd=2, relief=tk.GROOVE)      # Container for Grayscale Video frame
        
        self.original_frame_container.grid(row=3, column=0, padx=5, pady=5)
        self.gray_frame_container.grid(row=3, column=1, padx=5, pady=5)
        
        self.original_frame_canvas = tk.Canvas(self.original_frame_container, width=self.target_size, height=self.target_size)
        self.gray_frame_canvas = tk.Canvas(self.gray_frame_container, width=self.target_size, height=self.target_size)
        
        self.original_frame_canvas.pack(fill=tk.BOTH, expand=True)
        self.gray_frame_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.original_label = tk.Label(master, text="Original RGB Video", font=("Arial", 10, "bold"))
        self.gray_label = tk.Label(master, text="Grayscale Keypoints Video", font=("Arial", 10, "bold"))
        
        self.original_label.grid(row=2, column=0, padx=5, pady=5)
        self.gray_label.grid(row=2, column=1, padx=5, pady=5)
        
        self.predicted_output_label = tk.Label(master, text="Predicted Output:", font=("Arial", 10, "bold"))
        self.predicted_output_text = tk.Text(master, width=20, height=1, bg="black", fg="white")
        
        #GUI BUTTONS, LABELS AND THEIR POSITIONS
        button_width = 20 
        self.browse_video_button = ttk.Button(master, text="Browse Video", command=self.browse_video, width=button_width, style="RedBorder.TButton")
        self.browse_pose_model_button = ttk.Button(master, text="Browse Pose Model", command=self.browse_pose_model, width=button_width, style="RedBorder.TButton")
        self.browse_model_weights_button = ttk.Button(master, text="Browse Model Weights", command=self.browse_model_weights, width=button_width, style="RedBorder.TButton")
        self.run_button = ttk.Button(master, text="Run", command=self.play_video, width=10, style="RedBorder.TButton")
        self.stop_button = ttk.Button(master, text="Stop", command=self.stop_video, width=10, style="RedBorder.TButton")
        self.quit_button = ttk.Button(master, text="Quit", command=self.master.quit, width=10, style="RedBorder.TButton")
        self.browse_video_button.grid(row=1, column=0, padx=(5, 1), pady=5, sticky="ew")
        self.browse_pose_model_button.grid(row=1, column=1, padx=1, pady=5, sticky="ew")
        self.browse_model_weights_button.grid(row=1, column=2, padx=(1, 5), pady=5, sticky="ew")
        self.predicted_output_label.grid(row=4, column=0, padx=(5, 5), pady=(10, 5), sticky="e")
        self.predicted_output_text.grid(row=4, column=1, padx=(5, 5), pady=(10, 5), sticky="w")
        self.run_button.grid(row=5, column=0, padx=5, pady=5, sticky="e")
        self.stop_button.grid(row=5, column=1, padx=5, pady=5, sticky="w")
        self.quit_button.grid(row=5, column=2, padx=5, pady=5, sticky="w")
        
        self.predicted_output_text.insert(tk.END, self.predicted_label_txt)
        self.predicted_output_text.config(state=tk.DISABLED)
        
        self.style = ttk.Style()
        self.style.configure("RedBorder.TButton", borderwidth=5, relief="flat", foreground="black", background="red")
        self.style.map("RedBorder.TButton", background=[("active", "red")])
        self.style.configure("BlueBorder.TButton", borderwidth=5, relief="flat", foreground="black", background="blue")
        self.style.map("BlueBorder.TButton", background=[("active", "blue")])
        self.style.configure("GreenBorder.TButton", borderwidth=5, relief="flat", foreground="black", background="green")
        self.style.map("GreenBorder.TButton", background=[("active", "green")])
        self.style.configure("OrangeBorder.TButton", borderwidth=5, relief="flat", foreground="black", background="orange")
        self.style.map("OrangeBorder.TButton", background=[("active", "orange")])
        self.style.configure("YellowBorder.TButton", borderwidth=5, relief="flat", foreground="black", background="yellow")
        self.style.map("YellowBorder.TButton", background=[("active", "yellow")])
        self.style.configure("TButton", padding=6, font=("Arial", 10))
        self.style.configure("TLabel", font=("Arial", 10, "bold"))
    
    
    #BROWSE VIDEO BUTTON - FUNCTION
    def browse_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        self.count = 0
        self.video_keypoints = []
        self.update_button_styles()
    
    #BROWSE POSE MODEL BUTTON - FUNCTION
    def browse_pose_model(self):
        self.pose_model_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pt")])
        self.load_pose_model_and_trained_model()
        self.update_button_styles()
    
    #BROWSE SAVED MODEL WEIGHTS BUTTON- FUNCTION
    def browse_model_weights(self):
        self.model_weights_path = filedialog.askopenfilename(filetypes=[("Weight files", "*.h5;*.weights")])
        self.load_pose_model_and_trained_model()
        self.update_button_styles()
    
    #LOAD POSE MODEL FOR KEYPOINTS EXTRACTION
    def get_yolov8_pose_model(self):
        self.pose_model = YOLO(self.pose_model_path)

    #GET TRAINED MODEL ARCHITECTURE
    def get_model_architecture(self):
        
        parser = argparse.ArgumentParser(description='Process some input')
        parser.add_argument('--config', default='utils/config.yaml', type=str, help='Config path', required=False)
        args = parser.parse_args()
        config = read_yaml(args.config)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        #PROPOSED MODEL ARCHITECTURE SETTINGS   
        config['MODEL_SIZE'] = self.m
        config[self.m]['N_LAYERS'] = self.n_layers
        config[self.m]['EMBED_DIM'] = self.emb_dim
        config[self.m]['DROPOUT'] = self.dropout
        config[self.m]['MLP'] = self.mlp_dim

        trainer = Trainer(config,self.best_activation,self.best_opt_name,self.train_xdata,self.num_classes)
        self.model = trainer.get_model(self.lr,self.wd)
        
        #LOAD SAVED WEIGHTS INTO PROPOSED ARCHITECTURE
        self.model.load_weights(self.model_weights_path)

    #GET POSE MODEL AND TRAINED MODEL
    def load_pose_model_and_trained_model(self):
        if self.pose_model_path:
            self.get_yolov8_pose_model()
        
        if self.model_weights_path:
            self.get_model_architecture()
        os.system('cls')
    
    #RUN BUTTON - FUNCTION
    def play_video(self):
        if not self.playing:
            if self.video_complete:
                self.video_complete = False
                self.run_button.config(text="Run", style="RedBorder.TButton")
            self.playing = True
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._play_video)
                self.thread.start()
            self.run_button.config(text="Pause", style="OrangeBorder.TButton")
        else:
            self.playing = False
            self.run_button.config(text="Resume" if not self.video_complete else "Run", style="YellowBorder.TButton")
    
    #STOP BUTTON - FUNCTION
    def stop_video(self):
        self.stop = True

    #PREDICTED OUTPUT LABEL - UPDATE TEXTBOX FUNCTION
    def update_predicted_output_text_box(self,text):
        self.predicted_label_txt = text
        self.predicted_output_text.config(state=tk.NORMAL)
        self.predicted_output_text.delete(1.0, tk.END)
        self.predicted_output_text.insert(tk.END, text)
        self.predicted_output_text.config(state=tk.DISABLED)
        
    #RECOGNITION FUNCTION - INTERACTION DATASET VIDEOS
    def interaction_recognition(self):
        while(not self.stop):
            if self.playing:
                ret, frame = self.video_capture.read()
                if ret:
                    #CHANGE COLOR FROM BGR TO RGB 
                    original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    #CHANGE RGB COLOR TO MONO GRAYSCALE 
                    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #MONO-CHANNEL
                    
                    #THREE CHANNELS GRAYSCALE (REQUIRED FOR YOLO V8 POSE MODEL)
                    gray_image_pred = cv2.merge([gray_image, gray_image, gray_image]) 
                    gray_image = cv2.merge([gray_image, gray_image, gray_image]) 

                    #RESIZING IMAGES TO TARGET GUI FRAME SIZE
                    original_image = cv2.resize(original_image, (self.target_size, self.target_size))
                    gray_image = cv2.resize(gray_image, (self.target_size, self.target_size))
                    
                    #CONVERT ARRAYS TO PIL IMAGES
                    original_image = Image.fromarray(original_image)
                    gray_image = Image.fromarray(gray_image)
                    original_image = ImageTk.PhotoImage(original_image)

                    #APPLY POSE MODEL ON IMAGES
                    results = self.pose_model(gray_image, save=False)
                    results_gray_image_pred = self.pose_model(gray_image_pred,save=False)
                    detected_boxes = results_gray_image_pred[0].boxes
                    num_persons_detected = len(detected_boxes)
                    
                    #NORMALIZED AND GLOBAL KEYPOINTS FOR PREDICTION AND GUI DISPLAY RESPECTIVELY
                    normalized_kps_tensor = results_gray_image_pred[0].keypoints.xyn.squeeze() #Keypoints to Model For Predciting Output Label
                    global_kps_tensor = results[0].keypoints.xy.squeeze() #Keypoints for Visualization in GUI
                    normalized_kps = normalized_kps_tensor.numpy()
                    global_kps = global_kps_tensor.numpy()   

                    #CHECKING TWO PERSONS DETECTION AND 17 KEYPOINTS PER PERSON
                    if num_persons_detected == 2 and global_kps[0].shape[0] == 17 and global_kps[1].shape[0] == 17:
                        
                        gray_image = np.array(gray_image) #To write keypoints on it                     
                        #WRITE KEYPOINTS ON GRAY IMAGE
                        for n_person_kps in global_kps:
                            for x, y in n_person_kps:
                                cv2.circle(gray_image, (int(x), int(y)), 4, (0, 0, 255), -1) #DRAW KEYPOINTS      

                        #CONVERT KEYPOINTS WRITTEN GRAY IMAGE TO PIL IMAGE
                        gray_image = Image.fromarray(gray_image)
                        gray_image = ImageTk.PhotoImage(gray_image)

                        #APPEND FRAME KEYPOINTS
                        self.video_keypoints.append(normalized_kps)

                        #CHECK FRAMES KEYPOINTS REACH SEQUENCE LENGTH
                        if len(self.video_keypoints) >= 30:

                            #PICK LAST (SEQUENCE LENGTH) FRAMES KEYPOINTS
                            video_sequence = np.array(self.video_keypoints[-30:])
                            
                            #RESHAPING TO DESIRED INPUT SHAPE TO MAKE PREDICTION
                            num_seq,num_persons,num_kps,num_coords = video_sequence.shape
                            reshaped_sequence = video_sequence.reshape(num_seq,num_persons*num_kps*num_coords)
                            reshaped_sequence = np.expand_dims(reshaped_sequence,axis=0)
                            
                            #PREDICT SEQUENCE OUTPUT LABEL
                            predictions = self.model.predict(reshaped_sequence)
                            self.predicted_label = self.classes_names[np.argmax(predictions)]
                            video_sequence = []
                        
                        if not self.predicted_label:        
                            self.predicted_label_txt = 'None'
                        else:
                            self.predicted_label_txt = str(self.predicted_label)
                        
                        #DISPLAY RESULTS ON GUI
                        self.original_frame_canvas.create_image(0, 0, anchor=tk.NW, image=original_image)
                        self.gray_frame_canvas.create_image(0, 0, anchor=tk.NW, image=gray_image)
                        self.original_frame_canvas.image = original_image
                        self.gray_frame_canvas.image = gray_image
                        self.update_predicted_output_text_box(self.predicted_label_txt)
                        self.master.update()  #UPDATE GUI TO REFLECT CHANGES

                    else: #Skip Frame with No Two Persons
                        continue
        
                else:#Video Ends
                    break

            else:#Pause Video
                time.sleep(0.1)

        self.video_capture.release()
        #VIDEO ENDS HERE

        self.playing = False
        self.video_complete = True
        self.run_button.config(text="Run", style="GreenBorder.TButton")
        self.video_keypoints = []
        video_sequence = []
        frame = []
        self.stop = False

    #RECOGNITION FUNCTION - SOLO ACTION DATASET VIDEOS
    def solo_action_recognition(self):
        while(not self.stop):
            if self.playing:
                ret, frame = self.video_capture.read()
                if ret:
                    #CHANGE COLOR FROM BGR TO RGB
                    original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    #CHANGE RGB COLOR TO MONO GRAYSCALE
                    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    #THREE CHANNELS GRAYSCALE (REQUIRED FOR YOLO V8 POSE MODEL)
                    gray_image_pred = cv2.merge([gray_image, gray_image, gray_image])
                    gray_image = cv2.merge([gray_image, gray_image, gray_image])
                    
                    #RESIZING IMAGES TO TARGET GUI FRAME SIZE
                    original_image = cv2.resize(original_image, (self.target_size, self.target_size))
                    gray_image = cv2.resize(gray_image, (self.target_size, self.target_size))
                    
                    #CONVERT ARRAYS TO PIL IMAGES
                    original_image = Image.fromarray(original_image)
                    gray_image = Image.fromarray(gray_image)
                    original_image = ImageTk.PhotoImage(original_image)

                    #APPLY POSE MODEL ON IMAGES
                    results = self.pose_model(gray_image,save=False)
                    results_gray_image_pred = self.pose_model(gray_image_pred,save=False)
                    detected_boxes = results_gray_image_pred[0].boxes
                    num_persons_detected = len(detected_boxes)
                    
                    #NORMALIZED AND GLOBAL KEYPOINTS FOR PREDICTION AND GUI DISPLAY RESPECTIVELY
                    normalized_kps_tensor = results_gray_image_pred[0].keypoints.xyn.squeeze() #KPS to Model For Predciting Output Label
                    global_kps_tensor = results[0].keypoints.xy.squeeze() #KPS for Visualization in GUI
                    normalized_kps = normalized_kps_tensor.numpy()
                    global_kps = global_kps_tensor.numpy()   

                    #CHECKING SINGLE PERSON DETECTION
                    if num_persons_detected == 1 and global_kps.ndim == 2:
                        gray_image = np.array(gray_image) #To write keypoints on it                  
                        #WRITE KEYPOINTS ON GRAY IMAGE
                        for x, y in global_kps:
                           cv2.circle(gray_image, (int(x), int(y)), 4, (0, 0, 255), -1)  #DRAW KEYPOINTS   

                        #CONVERT KEYPOINTS WRITTEN GRAY IMAGE TO PIL IMAGE
                        gray_image = Image.fromarray(gray_image) #Back converted to PIL Image
                        gray_image = ImageTk.PhotoImage(gray_image)
                        
                        #APPEND FRAME KEYPOINTS
                        self.video_keypoints.append(normalized_kps)
                        
                        #CHECK FRAMES KEYPOINTS REACH SEQUENCE LENGTH
                        if len(self.video_keypoints)>=30:
                            #PICK LAST (SEQUENCE LENGTH) FRAMES KEYPOINTS
                            video_sequence = np.array(self.video_keypoints[-30:])
                            
                            #RESHAPING TO DESIRED INPUT SHAPE TO MAKE PREDICTION
                            zeros_array = np.zeros_like(video_sequence)
                            stacked_sequence = np.stack((video_sequence, zeros_array), axis=1)
                            transformed_sequence = stacked_sequence.transpose(0, 1, 2, 3)
                            video_sequence = transformed_sequence
                            num_seq,num_persons,num_kps,num_coords = video_sequence.shape
                            reshaped_sequence = video_sequence.reshape(num_seq,num_persons*num_kps*num_coords)
                            reshaped_sequence = np.expand_dims(reshaped_sequence,axis=0)
                            
                            #PREDICT SEQUENCE OUTPUT LABEL
                            predictions = self.model.predict(reshaped_sequence)
                            self.predicted_label = self.classes_names[np.argmax(predictions)]
                        if not self.predicted_label:        
                            self.predicted_label_txt = 'None'
                        else:
                            self.predicted_label_txt = str(self.predicted_label)

                        #DISPLAY RESULTS ON GUI
                        self.original_frame_canvas.create_image(0, 0, anchor=tk.NW, image=original_image)
                        self.gray_frame_canvas.create_image(0, 0, anchor=tk.NW, image=gray_image)
                        self.original_frame_canvas.image = original_image
                        self.gray_frame_canvas.image = gray_image
                        self.update_predicted_output_text_box(self.predicted_label_txt)
                        self.master.update()  #UPDATE GUI TO REFLECT CHANGES
                    else: #Skip Frame with No Single Person Detected
                        continue
        
                else:#Video Ends
                    break

            else:#Pause Video
                time.sleep(0.1)
        self.video_capture.release()
        #VIDEO ENDS HERE
        
        self.playing = False
        self.video_complete = True
        self.run_button.config(text="Run", style="GreenBorder.TButton")
        self.video_keypoints = []
        video_sequence = []
        frame = []
        self.stop = False

    #CHECK DATASET TYPE (SOLO/INTERACTION) FUNCTION CALLED IN RUN BUTTON-FUNCTION
    def _play_video(self):
        os.system('cls')
        #CAPTURE VIDEO
        self.video_capture = cv2.VideoCapture(self.video_path)
        self.predicted_label = 'None'

        #SPLIT DATASET TYPE (SOLO/INTERACTION) FROM VIDEO FILENAME
        _,videoname = os.path.split(self.video_path)
        action_type = videoname.split('_')[0]
        
        #VIDEO BELONGING TO INTERACTION DATASET
        if action_type == 'interaction':
            self.interaction_recognition()
        
        #VIDEO BELONGING TO SOLO ACTION DATASET
        elif action_type == 'solo':
            self.solo_action_recognition()

        #VIDEO FILENAME DOESNOT CONTAIN ACTION TYPE OR ACTION TYPE DOESNOT EXISTS
        else:
            print('Action Type Not Found or Doesnot Exists!')

    #UPDATE BUTTON COLORS - FUNCTION
    def update_button_styles(self):
        #CHECK VIDEO PATH
        if self.video_path == "":
            self.browse_video_button.configure(style="RedBorder.TButton")
        else:
            self.browse_video_button.configure(style="BlueBorder.TButton")
        
        #CHECK POSE MODEL PATH
        if self.pose_model_path == "":
            self.browse_pose_model_button.configure(style="RedBorder.TButton")
        else:
            self.browse_pose_model_button.configure(style="BlueBorder.TButton")
        
        #CHECK MODEL WEIGHTS PATH
        if self.model_weights_path == "":
            self.browse_model_weights_button.configure(style="RedBorder.TButton")
        else:
            self.browse_model_weights_button.configure(style="BlueBorder.TButton")

        #IF ALL PATH SELECTED, TURN RUN BUTTON GREEN INDICATING USER CAN PROCESS VIDEO
        if self.video_path != "" and self.pose_model_path != "" and self.model_weights_path != "":
            self.run_button.configure(style="GreenBorder.TButton")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayer(root)
    root.mainloop()
