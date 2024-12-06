## Acknowledgments

We would like to express our gratitude to the creators of the datasets, tools, and baseline models that were utilized in this project:

1. **UT-Interaction Dataset**
   | ![Fig 4b - Sample Images of UT Interaction dataset](https://github.com/user-attachments/assets/3647cbc8-5b5e-4bf2-935a-59cfb3986277)
   |:-------------------------------------------------------------------------------------------------------------------------------:|
   | *Figure 1. Sample image frames of human-human interaction classes from the UT Interaction dataset.* |

   - **Reference:** Ryoo MS, Aggarwal JK. Spatio-temporal relationship match: Video structure comparison for recognition of complex human activities. In: 2009 IEEE 12th International Conference on Computer Vision; 2009; Kyoto, Japan. IEEE; p. 1593-1600. Available from: [IEEE Xplore](https://ieeexplore.ieee.org/document/5459361)
   - **Download Link:** [UT-Interaction Dataset](https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html)  

3. **Drone-Action Dataset**
   | ![Fig 4a - Sample Images of Drone Action dataset-min](https://github.com/user-attachments/assets/29c77b1e-8961-4382-97c1-1df260946349) |
   |:-------------------------------------------------------------------------------------------------------------------------------:|
   | *Figure 2. Sample image frames of solo action classes from the Drone Action dataset.* |


   - **Reference:** Perera AG, Law YW, Chahl J. Drone-Action: An Outdoor Recorded Drone Video Dataset for Action Recognition. Drones [Internet]. 2019;3(4). Available from: [https://www.mdpi.com/2504-446X/3/4/82](https://www.mdpi.com/2504-446X/3/4/82)  
   - **Download Link:** [Drone-Action Dataset](https://asankagp.github.io/droneaction/)  

4. **YOLOv8 Pose Model**  
   - Developed by Ultralytics, providing state-of-the-art pose detection capabilities.  
   - **Link:** [YOLOv8 Pose Model](https://docs.ultralytics.com/tasks/pose/#models)  

5. **Baseline Model: Action Transformer (AcT)**  
   - **Reference:** Mazzia V, Angarano S, Salvetti F, Angelini F, Chiaberge M. Action Transformer: A self-attention model for short-time pose-based human action recognition. Pattern Recognit [Internet]. 2022;124:108487. Available from: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0031320321006634  
   - **Link:** [Action Transformer (AcT)](https://github.com/PIC4SeR/AcT)  

We deeply appreciate the efforts of these researchers and developers, whose work laid the foundation for this project.

---

**Project Title :**
InterAcT: A Generic Keypoints-Based Lightweight Transformer Model for Recognition of Human Solo Actions and Interactions in Aerial Videos


**Project Description:** 
It is an activity recognition project which utilizes keypoints data, extracted from RGB videos using YOLO v8 pose estimation model. This project introduce a generic transformer model known as InterAcT, capable of recognizing solo human actions as well as human-human interactions in aerial videos.

## InterAcT Code Guidelines

**Requirements and Dependencies:**
- Python Version: 3.9.9
- IDE: Visual Studio (VS)
- Operating System: Windows 

**Virtual Environment Creation and Installing Dependencies:**
- Create Virtual Environment using following command in VS Terminal:

`python -m venv virtual_environment_name`

- Activate the virtual environment using the below commmand in VS Terminal:

`virtual_environment_name\Scripts\activate`

- Install libraries and dependencies using following command in VS Terminal:

`pip install -r requirements.txt`
*Note: requirements.txt file can be found in "Code" directory* 

## Code: `main.py`

### 1. Preprocessing: Keypoints Sequences Formation
#### (A) Preprocessing Flag:
1. **Dataset Preparation:**
   - Ensure the `datasets\all_classes_original` folder contains the required solo or interaction datasets.
   - Each class folder should include extracted pose data in `.npy` format for each video.

2. **Sequence Generation:**
   - Modify the `duplicate_and_sequence_chunking` function in the `utils\data.py` file to include the required classes for your dataset.
   - Adjust the `num_sequences` value at line **644** to specify the number of sequences per class for balanced data generation.

3. **Save Directory:**
   - Define your directory path to save `Xdata` and `Ydata`.

4. **Preprocessing Flag Settings:**
   - Set the `preprocessing_flag` to `True`.
   - Select the **User-Defined Input Settings** option to configure sequential data formation.

5. **Output:**
   - After specifying the settings, sequential data will be generated and saved in the specified directory.

---

### 2. Script Blocks in `main.py`
The following blocks are included in the script, each controlled by its respective flag:

- **Preprocessing Block:** Generates sequential data.
- **Reshaping & Train-Test-Validation Sets Generation Block:** Reshapes data and creates splits for training, testing, and validation.
- **Training Model Block:** Trains the deep learning model.
- **Evaluation of Saved Model on Test Set Block:** Evaluates a saved model on the test set.
- **Hyperparameters Tuning Block:** Allows optimization of model hyperparameters.

#### Usage:
To use any of these blocks, set its flag to `True`. Set the flag to `False` if the block is not required.

---

### Notes
- Ensure the dataset and directories are properly configured before running the script.
- Modify code parameters as needed to align with the dataset structure and desired configurations.

### Steps for Running Inference:
1. **Start the Inference GUI:**
   - Run the `Inference_GUI.py` file to launch the inference interface.

2. **Load Saved Model Weights:**
   - Click the **"Browse Model Weights"** button and load the `proposed_model_weights.h5` file from the `inference` folder.
   - Ensure the button border turns **BLUE** after loading the weights successfully.

3. **Load Pose Extraction Model (YOLOv8):**
   - Click the **"Browse Pose Model"** button and load the `yolov8n-pose.pt` file from the `inference` folder.
   - Ensure the button border turns **BLUE** after loading the pose model successfully.

4. **Load a Video for Inference:**
   - Click the **"Browse Video"** button and select a video file from the `inference_sample_videos` folder within the `inference` directory. 
   - Alternatively, load any solo_action or interaction video. Ensure the video filename starts with either **"solo"** or **"interaction"**, otherwise, inference will not work.

5. **Note:**  
   - If using a different variant of the InterAcT model with modified architecture or training settings, ensure both settings are updated in the initialization section of `Inference_GUI.py`.

---

## Datasets Folder Overview

### Preprocessed Pose Data:
The `Datasets` folder contains the preprocessed pose data, including training, testing, and validation sets used in the training and evaluation of the proposed model.

#### 1. Preprocessed Data Splits:
- **Training Set Percentage:** 80%  
- **Test Set Percentage:** 10%  
- **Validation Set Percentage:** 10%

#### 2. Datasets:
##### **(A) UT-INTERACTION**
- **Action Category:** Interactions (Human-Human)  
- **Number of Classes:** 5  
- **Classes Names:**  
  - handshaking  
  - hugging  
  - kicking  
  - punching  
  - pushing  
- **Download Link:**  
  [UT-Interaction Dataset](https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html)

##### **(B) DRONE-ACTION**
- **Action Category:** Solo/Single Person Actions  
- **Number of Classes:** 13  
- **Classes Names:**  
  - clapping  
  - hitting_bottle  
  - hitting_stick  
  - jogging_front_back_view  
  - jogging_sideview  
  - kicking  
  - punching  
  - running_front_back_view  
  - running_sideview  
  - stabbing  
  - walking_front_back_view  
  - walking_sideview  
  - waving_hands  
- **Download Link:**  
  [Drone-Action Dataset](https://asankagp.github.io/droneaction/)
  *Note:* Drone Action dataset can be requested on the provided link.

### 3. Usage Instructions:
#### (A) Using Preprocessed Data in the Proposed Model:
- Copy the `.npy` files from the `train_test_validation_data` folder to the `datasets` folder (create if it does not exist).

#### (B) Using Preprocessed Data in Other Models:
- Copy the `.npy` files from the `train_test_validation_data` folder to the desired directory for use in your model.

---
---

## InterAcT GUI Version:
You can access the GUI version of **InterAcT** code through the following link:

[InterAcT GUI Version](https://drive.google.com/file/d/1fgkv9yZ5eK0lO8dSPH2X3uOSL52v2ESZ/view?usp=sharing)
