**Project Title :**
InterAcT: A Generic Keypoints-Based Lightweight Transformer Model for Recognition of Human Solo Actions and Interactions in Aerial Videos

**Project Description:** 
It is an activity recognition project which utilizes keypoints data, extracted from RGB videos using YOLO v8 pose estimation model. This project introduce a generic transformer model known as InterAcT, capable of recognizing solo human actions as well as human-human interactions in aerial videos.

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
