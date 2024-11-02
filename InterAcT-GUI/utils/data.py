import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys

def handle_input_settings(root_directory):
    last_load_file = "preprocess_last_input_data_settings.txt"
    last_load_file_path = os.path.join(root_directory,last_load_file)
    if os.path.isfile(last_load_file_path):
        input_flag = input("Do You Want to Use Default Input Settings (Y) OR Use User-Defined Input Settings (N)? OR Load Last Saved Input Settings (L)?: ")
        while len(input_flag)==0 or (input_flag!='Yes' and input_flag!='YES' and input_flag!='yes' and input_flag!='Y' and input_flag!='y' and input_flag!='No' and input_flag!='NO' and input_flag!='no' and input_flag!='N' and input_flag!='n' and input_flag!='Last' and input_flag!='LAST' and input_flag!='last' and input_flag!='L' and input_flag!='l'): 
            input_flag = input("Enter Valid Input Settings Flag (Y/N/L): ")

        if input_flag =='Yes' or input_flag =='YES' or input_flag =='yes' or input_flag =='Y' or input_flag =='y':
            selected_data_folder, dup_n_seq_verbose_selection, fp_seq_threshold, fp_seq, dup_flag = default_inputs()
        elif input_flag =='Last' or input_flag =='LAST' or input_flag =='last' or input_flag =='L' or input_flag =='l':
            selected_data_folder, dup_n_seq_verbose_selection, fp_seq_threshold, fp_seq, dup_flag = load_last_settings(last_load_file_path)

        else:
            selected_data_folder, dup_n_seq_verbose_selection, fp_seq_threshold, fp_seq, dup_flag = user_def_inputs(root_directory)

    else:
        input_flag = input("Do You Want to Use Default Input Settings (Y) OR Use User-Defined Input Settings (N)? : ")
        while len(input_flag)==0 or (input_flag!='Yes' and input_flag!='YES' and input_flag!='yes' and input_flag!='Y' and input_flag!='y' and input_flag!='No' and input_flag!='NO' and input_flag!='no' and input_flag!='N' and input_flag!='n'): 
            input_flag = input("Enter Valid Input Settings Flag (Y/N): ")

        if input_flag =='Yes' or input_flag =='YES' or input_flag =='yes' or input_flag =='Y' or input_flag =='y':
            selected_data_folder, dup_n_seq_verbose_selection, fp_seq_threshold, fp_seq, dup_flag = default_inputs()
        else:
            selected_data_folder, dup_n_seq_verbose_selection, fp_seq_threshold, fp_seq, dup_flag = user_def_inputs()

    return input_flag,last_load_file,last_load_file_path,selected_data_folder, dup_n_seq_verbose_selection, fp_seq_threshold, fp_seq, dup_flag


def save_preprocess_input_data_settings(path,selected_data_folder, dup_n_seq_verbose_selection, fp_seq_threshold, fp_seq, dup_flag):
    with open(path, "w") as file:
        file.write(f"selected_data_folder: {selected_data_folder}\n")
        file.write(f"dup_n_seq_verbose_selection: {dup_n_seq_verbose_selection}\n")
        file.write(f"fp_seq_threshold: {fp_seq_threshold}\n")
        file.write(f"fp_seq: {fp_seq}\n")
        file.write(f"dup_flag: {dup_flag}\n")
    if dup_n_seq_verbose_selection in ['True', 'TRUE', 'true', 'T', 't']:
        print("Saved Preprocess Input Data Settings Successfully \n")


def load_last_settings(path):
# Initialize variables
    selected_data_folder = []
    dup_n_seq_verbose_selection = []
    fp_seq_threshold = []
    fp_seq = []
    dup_flag = []

    try:
        with open(path, "r") as file:
            lines = file.readlines()
            for line in lines:
                # Split each line into key and value
                key, value = line.strip().split(": ")
                if key == "selected_data_folder":
                    selected_data_folder = value
                elif key == "dup_n_seq_verbose_selection":
                    dup_n_seq_verbose_selection = value
                elif key == "fp_seq_threshold":
                    fp_seq_threshold = int(value)
                elif key == "fp_seq":
                    fp_seq = int(value)
                elif key == "dup_flag":
                    dup_flag = str(value)       
    except FileNotFoundError:
        print(f"File not found: {path}")
    print("Data Loaded Successfully")
    return  selected_data_folder, dup_n_seq_verbose_selection, fp_seq_threshold, fp_seq, dup_flag
    

# Define the default inputs
def default_inputs():
    selected_data_folder = 'all_classes_original'  # Default value
    dup_n_seq_verbose_selection = "True"  # Default value
    fp_seq_threshold = 15  # Default value
    fp_seq = 30  # Default value
    dup_flag = True  # Default value
    return selected_data_folder, dup_n_seq_verbose_selection, fp_seq_threshold, fp_seq,dup_flag

def user_def_inputs(directory):
    items_in_directory = os.listdir(directory)
    folders = [item for item in items_in_directory if os.path.isdir(os.path.join(directory, item))]
    num_folders = len(folders)
    print("Folders Found in Root Directory: ",num_folders)
    print("List of Folders:")
    for index, folder_name in enumerate(folders, start=1):
        print(f"{index}- {folder_name}")
    print(f"{index+1}- {'Terminate Program'}")
    folders_numbers = list(range(1, num_folders + 1))
    first_iteration = True
    while True:
        try:
            if first_iteration:
                prompt = "\nEnter the Choice Name to Process: "
            else:
                prompt = "Enter Valid Choice Name to Process: "

            selected_choice = input(prompt)
            if selected_choice in ['Terminate', 'terminate']:
                    sys.exit()
            elif selected_choice == 'all_classes_original':
                    first_iteration = False
                    break
            elif (selected_choice == [] or selected_choice == '') and (selected_choice != 'all_classes_original'):
                    first_iteration = False
            else:
                break
        except ValueError:
            first_iteration = False
    
    #dup_n_seq_verbose_selection
    dup_n_seq_verbose_selection = input("Enter Data Preprocessing Verbose (T/F): ")
    while len(dup_n_seq_verbose_selection)==0 or (dup_n_seq_verbose_selection!='True' and dup_n_seq_verbose_selection!='TRUE' and dup_n_seq_verbose_selection!='true' and dup_n_seq_verbose_selection!='T' and dup_n_seq_verbose_selection!='t' and dup_n_seq_verbose_selection!='False' and dup_n_seq_verbose_selection!='FALSE' and dup_n_seq_verbose_selection!='false' and dup_n_seq_verbose_selection!='F' and dup_n_seq_verbose_selection!='f'): 
        dup_n_seq_verbose_selection = input("Enter Valid Data Preprocessing Verbose (T/F): ")

    #fp_seq_threshold
    #Conditions for fp_seq_threshold Variable
    first_iteration = True
    while True:
        try:
            if first_iteration:
                prompt = "Enter Threshold for Number of Frames per Sequence: "
            else:
                prompt = "Enter Valid Threshold for Number of Frames per Sequence: "

            fp_seq_threshold = int(input(prompt))
            
            if fp_seq_threshold <= 0:
                first_iteration = False
            else:
                break
        except ValueError:
            first_iteration = False
    
    #fp_seq
    #Conditions for fp_seq Variable
    first_iteration = True
    threshold_warning = False
    while True:
        try:
            if first_iteration:
                prompt = "Enter Required Number of Frames per Sequence: "
            elif threshold_warning:
                prompt = f"Frames per Sequence Threshold Warning! Enter Number of Frames per Sequence >= {fp_seq_threshold}: "
            else:
                prompt = "Enter Valid Required Number of Frames per Sequence: "

            fp_seq = int(input(prompt))
            
            if fp_seq < fp_seq_threshold:
                first_iteration = False
                threshold_warning = True
            else:
                break
        except ValueError:
            first_iteration = False
            threshold_warning = False
    
    #dup_flag
    #Conditions for dup_flag Variable
    first_iteration = True
    while True:
        try:
            if first_iteration:
                prompt = "Enter Duplication Flag (T/F)?: "
            else:
                prompt = "Enter Valid Duplication Flag (T/F): "

            dup_flag = input(prompt)
            
            if len(dup_flag)==0 or (dup_flag!='True' and dup_flag!='TRUE' and dup_flag!='true' and dup_flag!='T' and dup_flag!='t'and dup_flag!='False' and dup_flag!='FALSE' and dup_flag!='false' and dup_flag!='F'and dup_flag!='f'):
                first_iteration = False
            else:
                break
        except ValueError:
            first_iteration = False
            
    if dup_flag == 'True' or dup_flag == 'true' or dup_flag =='T' or dup_flag =='t'or dup_flag =='TRUE':
        dup_flag = True
    else:
        dup_flag = False
    
    return selected_choice, dup_n_seq_verbose_selection, fp_seq_threshold, fp_seq ,dup_flag


def duplicate_and_sequence_chunking(dir,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag):
    def duplicates_n_sequence_chunks(path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag):
        data = np.load(path)
        if dup_n_seq_verbose:
            print("\nFile Shape Before Duplication: ",data.shape)
        
        if fixed_seq_flag:
            total_frames,num_persons,num_keypoints,num_coordinates = data.shape
            total_frames_bf = data.shape[0]
            frames_per_seq = fp_seq
            remaining_frames = total_frames % frames_per_seq
            num_duplicates = [0]
            if dup_flag:
                while total_frames < frames_per_seq:
                    # Calculate the number of frames to duplicate
                    frames_to_duplicate = frames_per_seq - total_frames
                    # Duplicate the frames by repeating the last frame
                    last_frames = data[-frames_to_duplicate:]
                    num_duplicates.append(last_frames.shape[0])
                    # Append the duplicated frames to the data
                    data = np.concatenate((data, last_frames), axis=0)
                    if dup_n_seq_verbose:
                        print("Total Frames<Frames per Seq Block : Data Shape After Duplication: ",data.shape)
                    # Update the total number of frames
                    total_frames = data.shape[0]
                remaining_frames = total_frames % frames_per_seq

                # If there are remaining frames, duplicate the last frames to fill a sequence
                if remaining_frames > 0:
                    frames_to_duplicate = frames_per_seq - remaining_frames
                    last_frames = data[-frames_to_duplicate:]
                    num_duplicates.append(last_frames.shape[0])
                    data = np.concatenate((data,last_frames),axis=0)
                    if dup_n_seq_verbose:
                        print("Remaining Frames>0 Block : Data Shape After Duplication: ",data.shape)
                    total_frames = data.shape[0]
                if dup_n_seq_verbose:
                    print("Total Frames Duplicated: ",sum(num_duplicates))
                    print(f"Data Shape After Duplication Function with {dup_flag} Flag : ",data.shape)
            else:
                if total_frames < frames_per_seq:
                    return None, None, None, None, None, None
                else:
                    total_frames = data.shape[0]  
                    remaining_frames = total_frames % frames_per_seq
                    if remaining_frames>0:
                        total_frames -= remaining_frames
                        data = data[:total_frames]
                        total_frames = data.shape[0]
        
            num_sequences = total_frames // frames_per_seq
            #Reshaping
            if dup_n_seq_verbose:
                print(f"Data Shape After Duplication Function with {dup_flag} Flag : ",data.shape)
            seq_reshape = (num_sequences,frames_per_seq,num_persons*num_keypoints,num_coordinates)
            reshaped_data = data.reshape(seq_reshape)
            if dup_n_seq_verbose:
                print("Data After Reshaping: ",reshaped_data.shape)

        else:
            print('WINDOW SEQUENCING BLOCK')
            print('DATA SHAPE: ',data.shape)
            total_frames,num_persons,num_keypoints,num_coordinates = data.shape
            frames_per_sequence = 30  # Number of frames per sequence
            if total_frames>=frames_per_sequence:
                # Calculate the number of sequences
                num_sequences = total_frames - frames_per_sequence + 1

                # Reshape the file data to (num_sequences, frames_per_sequence, num_persons*num_keypoints_per_person*num_coordinates)
                resultant_data = np.zeros((num_sequences, frames_per_sequence, num_persons * num_keypoints * num_coordinates))

                for i in range(num_sequences):
                    sequence = data[i:i + frames_per_sequence]  # Extract sequence of frames
                    flattened_sequence = sequence.reshape(frames_per_sequence, -1)
                    resultant_data[i] = flattened_sequence

                num_sequences,frames_per_seq,kps = resultant_data.shape
                reshaped_data = resultant_data.reshape(num_sequences,frames_per_seq,34,2)
                total_frames_bf = total_frames
                num_duplicates = [0]
                # Print the shape of the resultant data
                print("Reshaped data shape:", reshaped_data.shape)
            else:
                return None, None, None, None, None, None
        
        return data, reshaped_data, total_frames_bf,total_frames,sum(num_duplicates), num_sequences #Duplicates Frames Data,Seq Data, Num Frames before duplication,Num frames after duplication,Num frames duplicated 
    

    #DUPLICATE AND SEQUENCE CHUNKING FUNCTION STARTS HERE
    statistics = []
    #Interaction Frames Vars
    all_videos_frames_handshaking = []
    all_videos_frames_hugging = []
    all_videos_frames_kicking = []
    all_videos_frames_punching = []
    all_videos_frames_pushing = []
    #Solo Actions Frames Vars
    all_videos_frames_clapping_solo = []
    all_videos_frames_hitting_bottle_solo = []
    all_videos_frames_hitting_stick_solo = []
    all_videos_frames_hitting_solo = []
    all_videos_frames_jogging_f_b_solo = []
    all_videos_frames_jogging_side_solo = []
    all_videos_frames_jogging_solo = []
    all_videos_frames_kicking_solo = []
    all_videos_frames_punching_solo = []
    all_videos_frames_running_f_b_solo = []
    all_videos_frames_running_side_solo = []
    all_videos_frames_running_solo = []
    all_videos_frames_stabbing_solo = []
    all_videos_frames_walking_f_b_solo = []
    all_videos_frames_walking_side_solo = []
    all_videos_frames_walking_solo = []
    all_videos_frames_waving_hands_solo = []
    

    # #Interactions Seq Vars
    all_videos_sequences_handshaking = []
    all_videos_sequences_hugging = []
    all_videos_sequences_kicking = []
    all_videos_sequences_punching = []
    all_videos_sequences_pushing = []

    #Solo Actions Seq Vars
    all_videos_sequences_clapping_solo = []
    all_videos_sequences_hitting_bottle_solo = []
    all_videos_sequences_hitting_stick_solo = []
    all_videos_sequences_hitting_solo = []
    all_videos_sequences_jogging_f_b_solo = []
    all_videos_sequences_jogging_side_solo = []
    all_videos_sequences_jogging_solo = []
    all_videos_sequences_kicking_solo = []
    all_videos_sequences_punching_solo = []
    all_videos_sequences_running_f_b_solo = []
    all_videos_sequences_running_side_solo = []
    all_videos_sequences_running_solo = []
    all_videos_sequences_stabbing_solo = []
    all_videos_sequences_walking_f_b_solo = []
    all_videos_sequences_walking_side_solo = []
    all_videos_sequences_walking_solo = []
    all_videos_sequences_waving_hands_solo = []
    
    Xdata = []
    Ydata =[]
    for set_folder in os.listdir(dir):
        set_path = os.path.join(dir, set_folder)
        if os.path.isdir(set_path):
            for action_folder in os.listdir(set_path):
                action_path = os.path.join(set_path, action_folder)
                if os.path.isdir(action_path):
                    for video_npy_file in os.listdir(action_path):
                        if video_npy_file.endswith('.npy'):
                            npy_file_path = os.path.join(action_path, video_npy_file)
                            if action_folder == 'handshaking':
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_handshaking.append(frames_data)
                                    all_videos_sequences_handshaking.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                            'Total Frames Before Duplication': total_frames_bf_dup,
                                                            'Total Frames After Duplication': total_frames_af_dup,
                                                            'Total Duplicates Generated': dup , 
                                                            'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")
                                        
                                   
                            elif action_folder == 'hugging':
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_hugging.append(frames_data)
                                    all_videos_sequences_hugging.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                        'Total Frames Before Duplication': total_frames_bf_dup,
                                                        'Total Frames After Duplication': total_frames_af_dup,
                                                        'Total Duplicates Generated': dup , 
                                                        'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:    
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")
                               
                            elif action_folder == 'kicking':
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_kicking.append(frames_data)
                                    all_videos_sequences_kicking.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                        'Total Frames Before Duplication': total_frames_bf_dup,
                                                        'Total Frames After Duplication': total_frames_af_dup,
                                                        'Total Duplicates Generated': dup , 
                                                        'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")
                               
                            elif action_folder == 'punching':
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_punching.append(frames_data)
                                    all_videos_sequences_punching.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                        'Total Frames Before Duplication': total_frames_bf_dup,
                                                        'Total Frames After Duplication': total_frames_af_dup,
                                                        'Total Duplicates Generated': dup , 
                                                        'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")
                               
                            elif action_folder == 'pushing':
                                #  print(npy_file_path)
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_pushing.append(frames_data)
                                    all_videos_sequences_pushing.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                        'Total Frames Before Duplication': total_frames_bf_dup,
                                                        'Total Frames After Duplication': total_frames_af_dup,
                                                        'Total Duplicates Generated': dup , 
                                                        'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")

                            #SOLO ACTIONS    
                            elif action_folder == 'clapping_solo':
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_clapping_solo.append(frames_data)
                                    all_videos_sequences_clapping_solo.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                        'Total Frames Before Duplication': total_frames_bf_dup,
                                                        'Total Frames After Duplication': total_frames_af_dup,
                                                        'Total Duplicates Generated': dup , 
                                                        'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")
                            
                            elif action_folder == 'hitting_bottle_solo':
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_hitting_bottle_solo.append(frames_data)
                                    all_videos_sequences_hitting_bottle_solo.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                        'Total Frames Before Duplication': total_frames_bf_dup,
                                                        'Total Frames After Duplication': total_frames_af_dup,
                                                        'Total Duplicates Generated': dup , 
                                                        'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")
                                
                            elif action_folder == 'hitting_stick_solo':
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_hitting_stick_solo.append(frames_data)
                                    all_videos_sequences_hitting_stick_solo.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                        'Total Frames Before Duplication': total_frames_bf_dup,
                                                        'Total Frames After Duplication': total_frames_af_dup,
                                                        'Total Duplicates Generated': dup , 
                                                        'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")
                                
                            elif action_folder == 'jogging_f_b_solo':
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_jogging_f_b_solo.append(frames_data)
                                    all_videos_sequences_jogging_f_b_solo.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                        'Total Frames Before Duplication': total_frames_bf_dup,
                                                        'Total Frames After Duplication': total_frames_af_dup,
                                                        'Total Duplicates Generated': dup , 
                                                        'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")
                                
                            elif action_folder == 'jogging_side_solo':
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_jogging_side_solo.append(frames_data)
                                    all_videos_sequences_jogging_side_solo.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                        'Total Frames Before Duplication': total_frames_bf_dup,
                                                        'Total Frames After Duplication': total_frames_af_dup,
                                                        'Total Duplicates Generated': dup , 
                                                        'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")
                                
                            elif action_folder == 'kicking_solo':
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_kicking_solo.append(frames_data)
                                    all_videos_sequences_kicking_solo.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                        'Total Frames Before Duplication': total_frames_bf_dup,
                                                        'Total Frames After Duplication': total_frames_af_dup,
                                                        'Total Duplicates Generated': dup , 
                                                        'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")
                                
                            elif action_folder == 'punching_solo':
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_punching_solo.append(frames_data)
                                    all_videos_sequences_punching_solo.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                        'Total Frames Before Duplication': total_frames_bf_dup,
                                                        'Total Frames After Duplication': total_frames_af_dup,
                                                        'Total Duplicates Generated': dup , 
                                                        'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")
                                
                            elif action_folder == 'running_f_b_solo':
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_running_f_b_solo.append(frames_data)
                                    all_videos_sequences_running_f_b_solo.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                        'Total Frames Before Duplication': total_frames_bf_dup,
                                                        'Total Frames After Duplication': total_frames_af_dup,
                                                        'Total Duplicates Generated': dup , 
                                                        'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")
                                
                            elif action_folder == 'running_side_solo':
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_running_side_solo.append(frames_data)
                                    all_videos_sequences_running_side_solo.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                        'Total Frames Before Duplication': total_frames_bf_dup,
                                                        'Total Frames After Duplication': total_frames_af_dup,
                                                        'Total Duplicates Generated': dup , 
                                                        'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")
                                
                            elif action_folder == 'stabbing_solo':
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_stabbing_solo.append(frames_data)
                                    all_videos_sequences_stabbing_solo.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                        'Total Frames Before Duplication': total_frames_bf_dup,
                                                        'Total Frames After Duplication': total_frames_af_dup,
                                                        'Total Duplicates Generated': dup , 
                                                        'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")
                                
                            elif action_folder == 'walking_f_b_solo':
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_walking_f_b_solo.append(frames_data)
                                    all_videos_sequences_walking_f_b_solo.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                        'Total Frames Before Duplication': total_frames_bf_dup,
                                                        'Total Frames After Duplication': total_frames_af_dup,
                                                        'Total Duplicates Generated': dup , 
                                                        'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")
                                
                            elif action_folder == 'walking_side_solo':
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_walking_side_solo.append(frames_data)
                                    all_videos_sequences_walking_side_solo.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                        'Total Frames Before Duplication': total_frames_bf_dup,
                                                        'Total Frames After Duplication': total_frames_af_dup,
                                                        'Total Duplicates Generated': dup , 
                                                        'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")
                                
                            elif action_folder == 'waving_hands_solo':
                                frames_data, seq_data, total_frames_bf_dup,total_frames_af_dup,dup,num_seq = duplicates_n_sequence_chunks(npy_file_path,dup_n_seq_verbose,fp_seq,dup_flag,fixed_seq_flag)
                                if frames_data is not None:
                                    all_videos_frames_waving_hands_solo.append(frames_data)
                                    all_videos_sequences_waving_hands_solo.append(seq_data)
                                    statistics.append({'Set':set_folder,'Class-Folder':action_folder,'Video-Filename': video_npy_file,
                                                        'Total Frames Before Duplication': total_frames_bf_dup,
                                                        'Total Frames After Duplication': total_frames_af_dup,
                                                        'Total Duplicates Generated': dup , 
                                                        'Total Sequences Made':num_seq})
                                else:
                                    # Handle the case when duplicates_n_sequence_chunks returns None
                                    if dup_n_seq_verbose:
                                        print(f"Warning: Skipping '{npy_file_path}' due to None return value.")

                                        
    if dup_n_seq_verbose:
        print("\n Concatenating All Appended Files \n")
    all_videos_frames_handshaking = np.concatenate(all_videos_frames_handshaking,axis=0)
    all_videos_frames_hugging = np.concatenate(all_videos_frames_hugging,axis=0)
    all_videos_frames_kicking = np.concatenate(all_videos_frames_kicking,axis=0)
    all_videos_frames_punching = np.concatenate(all_videos_frames_punching,axis=0)
    all_videos_frames_pushing = np.concatenate(all_videos_frames_pushing,axis=0)

    #SOLO ACTIONS FRAMES
    all_videos_frames_clapping_solo = np.concatenate(all_videos_frames_clapping_solo,axis=0)
    all_videos_frames_hitting_bottle_solo = np.concatenate(all_videos_frames_hitting_bottle_solo,axis=0)
    all_videos_frames_hitting_stick_solo = np.concatenate(all_videos_frames_hitting_stick_solo,axis=0)
    #all_videos_frames_hitting_solo = np.concatenate((all_videos_frames_hitting_bottle_solo,all_videos_frames_hitting_stick_solo),axis=0)
    all_videos_frames_jogging_f_b_solo = np.concatenate(all_videos_frames_jogging_f_b_solo,axis=0)
    all_videos_frames_jogging_side_solo = np.concatenate(all_videos_frames_jogging_side_solo,axis=0)
    #all_videos_frames_jogging_solo = np.concatenate((all_videos_frames_jogging_f_b_solo,all_videos_frames_jogging_side_solo),axis=0)
    all_videos_frames_kicking_solo = np.concatenate(all_videos_frames_kicking_solo,axis=0)
    all_videos_frames_punching_solo = np.concatenate(all_videos_frames_punching_solo,axis=0)
    all_videos_frames_running_f_b_solo = np.concatenate(all_videos_frames_running_f_b_solo,axis=0)
    all_videos_frames_running_side_solo = np.concatenate(all_videos_frames_running_side_solo,axis=0)
    #all_videos_frames_running_solo = np.concatenate((all_videos_frames_running_f_b_solo,all_videos_frames_running_side_solo),axis=0)
    all_videos_frames_stabbing_solo = np.concatenate(all_videos_frames_stabbing_solo,axis=0)
    all_videos_frames_walking_f_b_solo = np.concatenate(all_videos_frames_walking_f_b_solo,axis=0)
    all_videos_frames_walking_side_solo = np.concatenate(all_videos_frames_walking_side_solo,axis=0)
    #all_videos_frames_walking_solo = np.concatenate((all_videos_frames_walking_f_b_solo,all_videos_frames_walking_side_solo),axis=0)
    all_videos_frames_waving_hands_solo = np.concatenate(all_videos_frames_waving_hands_solo,axis=0)    

    
    all_videos_sequences_handshaking = np.concatenate(all_videos_sequences_handshaking,axis=0)
    all_videos_sequences_hugging = np.concatenate(all_videos_sequences_hugging,axis=0)
    all_videos_sequences_kicking = np.concatenate(all_videos_sequences_kicking,axis=0)
    all_videos_sequences_punching = np.concatenate(all_videos_sequences_punching,axis=0)
    all_videos_sequences_pushing = np.concatenate(all_videos_sequences_pushing,axis=0)

    #SOLO ACTIONS SEQUENCES
    all_videos_sequences_clapping_solo = np.concatenate(all_videos_sequences_clapping_solo,axis=0)
    all_videos_sequences_hitting_bottle_solo = np.concatenate(all_videos_sequences_hitting_bottle_solo,axis=0)
    all_videos_sequences_hitting_stick_solo = np.concatenate(all_videos_sequences_hitting_stick_solo,axis=0)
    #all_videos_sequences_hitting_solo = np.concatenate((all_videos_sequences_hitting_bottle_solo,all_videos_sequences_hitting_stick_solo),axis=0)
    all_videos_sequences_jogging_f_b_solo = np.concatenate(all_videos_sequences_jogging_f_b_solo,axis=0)
    all_videos_sequences_jogging_side_solo = np.concatenate(all_videos_sequences_jogging_side_solo,axis=0)
    #all_videos_sequences_jogging_solo = np.concatenate((all_videos_sequences_jogging_f_b_solo,all_videos_sequences_jogging_side_solo),axis=0)
    all_videos_sequences_kicking_solo = np.concatenate(all_videos_sequences_kicking_solo,axis=0)
    all_videos_sequences_punching_solo = np.concatenate(all_videos_sequences_punching_solo,axis=0)
    all_videos_sequences_running_f_b_solo = np.concatenate(all_videos_sequences_running_f_b_solo,axis=0)
    all_videos_sequences_running_side_solo = np.concatenate(all_videos_sequences_running_side_solo,axis=0)
    #all_videos_sequences_running_solo = np.concatenate((all_videos_sequences_running_f_b_solo,all_videos_sequences_running_side_solo),axis=0)
    all_videos_sequences_stabbing_solo = np.concatenate(all_videos_sequences_stabbing_solo,axis=0)
    all_videos_sequences_walking_f_b_solo = np.concatenate(all_videos_sequences_walking_f_b_solo,axis=0)
    all_videos_sequences_walking_side_solo = np.concatenate(all_videos_sequences_walking_side_solo,axis=0)
    #all_videos_sequences_walking_solo = np.concatenate((all_videos_sequences_walking_f_b_solo,all_videos_sequences_walking_side_solo),axis=0)
    all_videos_sequences_waving_hands_solo = np.concatenate(all_videos_sequences_waving_hands_solo,axis=0) 
        
    # Get the number of sequences from the user
    #num_sequences = int(input("Enter the number of sequences you want for each class: "))
    if fixed_seq_flag:
        num_seq_list = [
                            all_videos_sequences_handshaking.shape[0], 
                            all_videos_sequences_hugging.shape[0], 
                            all_videos_sequences_kicking.shape[0],
                            all_videos_sequences_punching.shape[0], 
                            all_videos_sequences_pushing.shape[0],
                            
                            all_videos_sequences_clapping_solo.shape[0],
                            all_videos_sequences_hitting_bottle_solo.shape[0],
                            all_videos_sequences_hitting_stick_solo.shape[0],
                            all_videos_sequences_jogging_f_b_solo.shape[0],
                            all_videos_sequences_jogging_side_solo.shape[0],
                            all_videos_sequences_kicking_solo.shape[0],
                            all_videos_sequences_punching_solo.shape[0],
                            all_videos_sequences_running_f_b_solo.shape[0],
                            all_videos_sequences_running_side_solo.shape[0],
                            all_videos_sequences_stabbing_solo.shape[0],
                            all_videos_sequences_walking_f_b_solo.shape[0],
                            all_videos_sequences_walking_side_solo.shape[0],
                            all_videos_sequences_waving_hands_solo.shape[0] ]
        num_sequences = min(num_seq_list)

    # Select specific sequences for each class
    all_videos_sequences_handshaking = all_videos_sequences_handshaking[:num_sequences]
    all_videos_sequences_hugging = all_videos_sequences_hugging[:num_sequences]
    all_videos_sequences_kicking = all_videos_sequences_kicking[:num_sequences]
    all_videos_sequences_punching = all_videos_sequences_punching[:num_sequences]
    all_videos_sequences_pushing = all_videos_sequences_pushing[:num_sequences]

    # SOLO ACTIONS SEQUENCES
    all_videos_sequences_clapping_solo = all_videos_sequences_clapping_solo[:num_sequences]
    all_videos_sequences_hitting_bottle_solo = all_videos_sequences_hitting_bottle_solo[:num_sequences]
    all_videos_sequences_hitting_stick_solo = all_videos_sequences_hitting_stick_solo[:num_sequences]
    all_videos_sequences_jogging_f_b_solo = all_videos_sequences_jogging_f_b_solo[:num_sequences]
    all_videos_sequences_jogging_side_solo = all_videos_sequences_jogging_side_solo[:num_sequences]
    all_videos_sequences_kicking_solo = all_videos_sequences_kicking_solo[:num_sequences]
    all_videos_sequences_punching_solo = all_videos_sequences_punching_solo[:num_sequences]
    all_videos_sequences_running_f_b_solo = all_videos_sequences_running_f_b_solo[:num_sequences]
    all_videos_sequences_running_side_solo = all_videos_sequences_running_side_solo[:num_sequences]
    all_videos_sequences_stabbing_solo = all_videos_sequences_stabbing_solo[:num_sequences]
    all_videos_sequences_walking_f_b_solo = all_videos_sequences_walking_f_b_solo[:num_sequences]
    all_videos_sequences_walking_side_solo = all_videos_sequences_walking_side_solo[:num_sequences]
    all_videos_sequences_waving_hands_solo = all_videos_sequences_waving_hands_solo[:num_sequences]


    y_handshaking = np.zeros(all_videos_sequences_handshaking.shape[0],dtype=int)
    y_hugging = np.ones(all_videos_sequences_hugging.shape[0],dtype=int)
    y_kicking = 2*np.ones(all_videos_sequences_kicking.shape[0],dtype=int)
    y_punching = 3*np.ones(all_videos_sequences_punching.shape[0],dtype=int)
    y_pushing = 4*np.ones(all_videos_sequences_pushing.shape[0],dtype=int)

    # #SOLO ACTIONS LABELS
    y_clapping_solo = 5*np.ones(all_videos_sequences_clapping_solo.shape[0],dtype=int)
    #y_hitting_solo = 6*np.ones(all_videos_sequences_hitting_solo.shape[0],dtype=int)
    y_hitting_bottle_solo = 6*np.ones(all_videos_sequences_hitting_bottle_solo.shape[0],dtype=int)
    y_hitting_stick_solo = 7*np.ones(all_videos_sequences_hitting_stick_solo.shape[0],dtype=int)
    #y_jogging_solo = 7*np.ones(all_videos_sequences_jogging_solo.shape[0],dtype=int)
    y_jogging_f_b_solo = 8*np.ones(all_videos_sequences_jogging_f_b_solo.shape[0],dtype=int)
    y_jogging_side_solo = 9*np.ones(all_videos_sequences_jogging_side_solo.shape[0],dtype=int)
    y_kicking_solo = 10*np.ones(all_videos_sequences_kicking_solo.shape[0],dtype=int)
    y_punching_solo = 11*np.ones(all_videos_sequences_punching_solo.shape[0],dtype=int)
    #y_running_solo = 10*np.ones(all_videos_sequences_running_solo.shape[0],dtype=int)
    y_running_f_b_solo = 12*np.ones(all_videos_sequences_running_f_b_solo.shape[0],dtype=int)
    y_running_side_solo = 13*np.ones(all_videos_sequences_running_side_solo.shape[0],dtype=int)
    y_stabbing_solo = 14*np.ones(all_videos_sequences_stabbing_solo.shape[0],dtype=int)
    #y_walking_solo = 12*np.ones(all_videos_sequences_walking_solo.shape[0],dtype=int)
    y_walking_f_b_solo = 15*np.ones(all_videos_sequences_walking_f_b_solo.shape[0],dtype=int)
    y_walking_side_solo = 16*np.ones(all_videos_sequences_walking_side_solo.shape[0],dtype=int)
    y_waving_hands_solo = 17*np.ones(all_videos_sequences_waving_hands_solo.shape[0],dtype=int)

    Xdata = np.concatenate((all_videos_sequences_handshaking,all_videos_sequences_hugging,all_videos_sequences_kicking,all_videos_sequences_punching,all_videos_sequences_pushing,
                            all_videos_sequences_clapping_solo,all_videos_sequences_hitting_bottle_solo,all_videos_sequences_hitting_stick_solo,all_videos_sequences_jogging_f_b_solo,all_videos_sequences_jogging_side_solo,all_videos_sequences_kicking_solo,all_videos_sequences_punching_solo,all_videos_sequences_running_f_b_solo,all_videos_sequences_running_side_solo,all_videos_sequences_stabbing_solo,all_videos_sequences_walking_f_b_solo,all_videos_sequences_walking_side_solo,all_videos_sequences_waving_hands_solo),axis=0)
    Ydata = np.concatenate((y_handshaking,y_hugging,y_kicking,y_punching,y_pushing,
                            y_clapping_solo,y_hitting_bottle_solo,y_hitting_stick_solo,y_jogging_f_b_solo,y_jogging_side_solo,y_kicking_solo,y_punching_solo,y_running_f_b_solo,y_running_side_solo,y_stabbing_solo,y_walking_f_b_solo,y_walking_side_solo,y_waving_hands_solo))        

    if dup_n_seq_verbose:
        print("Total Videos Frames Handshaking Shape: ",all_videos_frames_handshaking.shape)
        print("Total Videos Frames Hugging Shape: ",all_videos_frames_hugging.shape)
        print("Total Videos Frames Kicking Shape: ",all_videos_frames_kicking.shape)
        print("Total Videos Frames Punching Shape: ",all_videos_frames_punching.shape)
        print("Total Videos Frames Pushing Shape: ",all_videos_frames_pushing.shape)
        #SOLO ACTIONS FRAMES
        print("Total Videos Frames Solo Clapping Shape: ",all_videos_frames_clapping_solo.shape)
        print("Total Videos Frames Solo Hitting Bottle Shape: ",all_videos_frames_hitting_bottle_solo.shape)
        print("Total Videos Frames Solo Hitting Stick Shape: ",all_videos_frames_hitting_stick_solo.shape)
        #print("Total Videos Frames Solo Hitting Shape: ",all_videos_frames_hitting_solo.shape)
        print("Total Videos Frames Solo Jogging FB Shape: ",all_videos_frames_jogging_f_b_solo.shape)
        print("Total Videos Frames Solo Jogging SV Shape: ",all_videos_frames_jogging_side_solo.shape)
        #print("Total Videos Frames Solo Jogging Shape: ",all_videos_frames_jogging_solo.shape)
        print("Total Videos Frames Solo Kicking Shape: ",all_videos_frames_kicking_solo.shape)
        print("Total Videos Frames Solo Punching Shape: ",all_videos_frames_punching_solo.shape)
        print("Total Videos Frames Solo Running FB Shape: ",all_videos_frames_running_f_b_solo.shape)
        print("Total Videos Frames Solo Running SV Shape: ",all_videos_frames_running_side_solo.shape)
        #print("Total Videos Frames Solo Running Shape: ",all_videos_frames_running_solo.shape)
        print("Total Videos Frames Solo Stabbing Shape: ",all_videos_frames_stabbing_solo.shape)
        print("Total Videos Frames Solo Walking FB Shape: ",all_videos_frames_walking_f_b_solo.shape)
        print("Total Videos Frames Solo Walking SV Shape: ",all_videos_frames_walking_side_solo.shape)
        #print("Total Videos Frames Solo Walking Shape: ",all_videos_frames_walking_solo.shape)
        print("Total Videos Frames Solo Waving Shape: ",all_videos_frames_waving_hands_solo.shape)

        print("Total Videos Sequences Handshaking Shape: ",all_videos_sequences_handshaking.shape)
        print("Total Videos Sequences Hugging Shape: ",all_videos_sequences_hugging.shape)
        print("Total Videos Sequences Kicking Shape: ",all_videos_sequences_kicking.shape)
        print("Total Videos Sequences Punching Shape: ",all_videos_sequences_punching.shape)
        print("Total Videos Sequences Pushing Shape: ",all_videos_sequences_pushing.shape)
        # #SOLO ACTIONS SEQ
        print("Total Videos Sequences Solo Clapping Shape: ",all_videos_sequences_clapping_solo.shape)
        print("Total Videos Sequences Solo Hitting Bottle Shape: ",all_videos_sequences_hitting_bottle_solo.shape)
        print("Total Videos Sequences Solo Hitting Stick Shape: ",all_videos_sequences_hitting_stick_solo.shape)
        #print("Total Videos Sequences Solo Hitting Shape: ",all_videos_sequences_hitting_solo.shape)
        print("Total Videos Sequences Solo Jogging FB Shape: ",all_videos_sequences_jogging_f_b_solo.shape)
        print("Total Videos Sequences Solo Jogging SV Shape: ",all_videos_sequences_jogging_side_solo.shape)
        #print("Total Videos Sequences Solo Jogging Shape: ",all_videos_sequences_jogging_solo.shape)
        print("Total Videos Sequences Solo Kicking Shape: ",all_videos_sequences_kicking_solo.shape)
        print("Total Videos Sequences Solo Punching Shape: ",all_videos_sequences_punching_solo.shape)
        print("Total Videos Sequences Solo Running FB Shape: ",all_videos_sequences_running_f_b_solo.shape)
        print("Total Videos Sequences Solo Running SV Shape: ",all_videos_sequences_running_side_solo.shape)
        #print("Total Videos Sequences Solo Running Shape: ",all_videos_sequences_running_solo.shape)
        print("Total Videos Sequences Solo Stabbing Shape: ",all_videos_sequences_stabbing_solo.shape)
        print("Total Videos Sequences Solo Walking FB Shape: ",all_videos_sequences_walking_f_b_solo.shape)
        print("Total Videos Sequences Solo Walking SV Shape: ",all_videos_sequences_walking_side_solo.shape)
        #print("Total Videos Sequences Solo Walking Shape: ",all_videos_sequences_walking_solo.shape)
        print("Total Videos Sequences Solo Waving Shape: ",all_videos_sequences_waving_hands_solo.shape)

        print("Y-handshaking Shape: ",y_handshaking.shape)
        print("Y-hugging Shape: ",y_hugging.shape)
        print("Y-kicking Shape: ",y_kicking.shape)
        print("Y-punching Shape: ",y_punching.shape)
        print("Y-pushing Shape: ",y_pushing.shape)
        #SOLO ACTIONS LABELS
        print("Y-Solo Clapping Shape: ",y_clapping_solo.shape)
        print("Y-Solo Hitting Bottle Shape: ",y_hitting_bottle_solo.shape)
        print("Y-Solo Hitting Stick Shape: ",y_hitting_stick_solo.shape)
        print("Y-Solo Jogging FB Shape: ",y_jogging_f_b_solo.shape)
        print("Y-Solo Jogging SV Shape: ",y_jogging_side_solo.shape)
        print("Y-Solo Kicking Shape: ",y_kicking_solo.shape)
        print("Y-Solo Punching Shape: ",y_punching_solo.shape)
        print("Y-Solo Running FB Shape: ",y_running_f_b_solo.shape)
        print("Y-Solo Running SV Shape: ",y_running_side_solo.shape)
        print("Y-Solo Stabbing Shape: ",y_stabbing_solo.shape)
        print("Y-Solo Walking FB Shape: ",y_walking_f_b_solo.shape)
        print("Y-Solo Walking SV Shape: ",y_walking_side_solo.shape)
        print("Y-Solo Waving Shape: ",y_waving_hands_solo.shape)
        print('Before Data Split Shapes: \n')
        print("Xdata Shape: ",Xdata.shape)
        print("Ydata Shape: ",Ydata.shape)
        return Xdata,Ydata


def preprocess_data(directory, selected_data_folder, dup_n_seq_verbose_selection,fp_seq,dup_flag,fixed_seq_flag):    
    
    if isinstance(dup_flag,bool):
        dup_flag = str(dup_flag)
    
    if selected_data_folder == 'all_classes_original':
        folderpath = directory+r'\all_classes_original'
        if dup_n_seq_verbose_selection == 'True' or dup_n_seq_verbose_selection == 'true' or dup_n_seq_verbose_selection =='T' or dup_n_seq_verbose_selection =='t'or dup_n_seq_verbose_selection =='TRUE':
            if dup_flag == 'True' or dup_flag == 'true' or dup_flag =='T' or dup_flag =='t'or dup_flag =='TRUE':
               Xdata,Ydata = duplicate_and_sequence_chunking(folderpath, True, fp_seq,True,fixed_seq_flag)
            else:
                Xdata,Ydata = duplicate_and_sequence_chunking(folderpath, True, fp_seq,False,fixed_seq_flag)
        else:
            if dup_flag == 'True' or dup_flag == 'true' or dup_flag =='T' or dup_flag =='t'or dup_flag =='TRUE':
                Xdata,Ydata = duplicate_and_sequence_chunking(folderpath, False, fp_seq,True,fixed_seq_flag)
            else:
                Xdata,Ydata = duplicate_and_sequence_chunking(folderpath, False, fp_seq,False,fixed_seq_flag)
   
    else:
        print("No Folder Selected. Program Terminated!\n")
    
    return Xdata,Ydata

def one_hot(x, y, n_classes):
    print("ONE HOT FUNCTION RUNNING ")
    return x, tf.one_hot(y, n_classes)

