import pandas as pd
import scipy.io as sio
import numpy as np
import os

class DataLoader:
    def __init__(self, matlab_file,  pupil_csv=None, latent_states_file=None, server=None, animalID=None, date=None):
        self.pupil_csv = pupil_csv
        self.matlab_file = matlab_file
        self.latent_states_file = latent_states_file
        self.server = server
        self.animalID = animalID
        self.date = date

    def load_pupil_data(self):
        """Loads pupil data from a CSV file into a pandas DataFrame."""
        print(self.pupil_csv)
        pupil_df = pd.read_csv(self.pupil_csv)
        return pupil_df
    
    # Normalize trial_info and movement_in_imaging to remove unnecessary indexing
    def normalize_data(self, data):
        """Normalize nested structures to remove unnecessary indexing."""
        normalized_data = {}
        
        if isinstance(data, list):
            # If the data is a list, normalize each item in the list
            for index, item in enumerate(data):
                if isinstance(item, dict):  # Check if the item is a dictionary
                    # Normalize the dictionary item
                    normalized_data[index] = {k: v[0] if isinstance(v, (list, np.ndarray)) and len(v) > 0 else v for k, v in item.items()}
                else:
                    normalized_data[index] = item  # Directly append if not a dict
        elif isinstance(data, dict):  # If it's a dictionary
            for key, val in data.items():
                if isinstance(val, dict):  # Check if it's a dictionary
                    # Recursively apply normalization to nested dictionaries
                    normalized_data[key] = {k: v[0] if isinstance(v, (list, np.ndarray)) and len(v) > 0 else v for k, v in val.items()}
                else:
                    normalized_data[key] = val  # Assign directly

        return normalized_data
    
    def load_neural_data(self, neural_data_type='dff'):
        """Loads neural data from a MATLAB file."""
        data = sio.loadmat(self.matlab_file)
        imaging_data = data['imaging'][0]  # Assuming neural data is in 'imaging'
        neural_data = imaging_data[neural_data_type]  # Extract neural data from the structure

        # Initialize a dictionary to hold the dFF data for each trial
        dff_struct = {}
        trial_info = []
        movement_in_imaging = {}
        frame_id_events = {}
        file_num = {}
        velocity_struct = {}

        # Assuming imaging_data is a dictionary loaded previously
        num_trials = len(imaging_data)  # Get the number of trials from dff structure

        # Fields that require MATLAB-to-Python indexing adjustment
        adjust_fields = ['maze_frames', 'reward_frames', 'iti_frames','turn_frame','frame_indices']

        # Loop through each trial to populate the dictionary
        for trial in range(num_trials):

            # Initialize dictionaries for each trial to avoid KeyErrors
            movement_in_imaging[trial] = {}
            frame_id_events[trial] = {}
            
            # Get trial info and convert it to a DataFrame
            trial_info_raw = imaging_data['virmen_trial_info'][trial][0]
            
            # Convert the structured trial info into a dictionary for DataFrame creation
            trial_info_dict = {field: trial_info_raw[field][0][0] for field in trial_info_raw.dtype.names}
            trial_info.append(trial_info_dict)
            # Access the dff data safely

            # Check if dff data is present for the trial
            if imaging_data[neural_data_type][trial].size > 0:
                dff = imaging_data[neural_data_type][trial]  # Neurons x Frames
                dff_struct[trial] = dff  # Assign the dFF data to the trial key
                # Adjust for MATLAB indexing (subtract 1 where necessary)
                for field in imaging_data['movement_in_imaging_time'][trial][0].dtype.names:
                    
                    data = imaging_data['movement_in_imaging_time'][trial][0][field][0][0]
                    
                    # Apply MATLAB-to-Python indexing adjustment to specified fields only
                    if field in adjust_fields:
                        movement_in_imaging[trial][field] = data - 1
                    else:
                        movement_in_imaging[trial][field] = data

                # Calculate running velocity using x_velocity and y_velocity
                x_velocity = movement_in_imaging[trial].get('x_velocity', np.zeros_like(data))
                y_velocity = movement_in_imaging[trial].get('y_velocity', np.zeros_like(data))
                velocity = np.sqrt(x_velocity**2 + y_velocity**2)
                velocity_struct[trial] = velocity
                #frame_id_events[trial] = imaging_data['frame_id_events'][trial][0] 

                # Frame ID events (subtract 1 to adjust to Python's 0-based indexing)
                # Frame ID events: Extract start (first element) and end (last element) and adjust for 0-based indexing
                def get_start_end(array):
                    return array[0][0] - 1, array[-1][0] - 1 if len(array) > 0 else (None, None)
                
                frame_id_events[trial] = {
                    'maze': get_start_end(imaging_data['frame_id_events'][trial]['maze']),
                    'reward': get_start_end(imaging_data['frame_id_events'][trial]['reward']),
                    'iti': get_start_end(imaging_data['frame_id_events'][trial]['iti'])
                }
                file_num[trial] = imaging_data['file_num'][trial][0][0] -1
            else:
                
                # If dff is empty, initialize corresponding entries to None
                dff_struct[trial] = None
                movement_in_imaging[trial] = None
                frame_id_events[trial] = None
                file_num[trial] = None
                velocity_struct[trial] = None


        # Initialize a list to hold the indices of good trials
        good_trials = []

        # Loop through the dff_structure to find trials with valid data
        for trial, dff in dff_struct.items():
            if (dff is not None and 
                dff.size > 0 and 
                imaging_data[trial]['good_trial']):  # Check if both dFF data is valid and trial is marked as good
                good_trials.append(trial)  # Add the trial index to the list 
        print(good_trials)


        # Normalize the structures
        trial_info_normalized = self.normalize_data(trial_info)
        trial_info_df = pd.DataFrame(trial_info_normalized)
        frame_id_events_norm = self.normalize_data(frame_id_events)

        return dff_struct, good_trials, trial_info_df.T, movement_in_imaging, frame_id_events_norm, file_num, velocity_struct, imaging_data
    
    def load_alignment_data(self):
        """Loads frame alignment data"""
        base = f"{self.server}/Connie/ProcessedData/{self.animalID}/{self.date}/alignment_info.mat"
        try:
            data = sio.loadmat(base)
            frames_times = data['alignment_info']['frame_times'][0]
            imaging_length = []
            for acqusition in frames_times:
                imaging_length.append(len(acqusition[0]))
            return imaging_length
        except FileNotFoundError:
            print(f"Error: Alignment data file not found at {base}.")
            return []
        except Exception as e:
            print(f"Error loading alignment data: {e}")
            return []
    
    def align_frames_to_session(self, file_num, frame_id_events_norm, imaging_frames_per_file):
        """
        Align frame_id_events (maze, reward, iti) to the global session frame numbers.
        
        file_num: Dictionary of trial file numbers
        frame_id_events: Dictionary of local frame IDs (maze, reward, iti) for each trial
        imaging_frames_per_file: List of total number of frames for each file
        """
        # Compute the cumulative number of frames collected up to each file
        cumulative_frames = [0]  # Start with 0 frames before the first file
        for frames in imaging_frames_per_file:
            cumulative_frames.append(cumulative_frames[-1] + frames)

        # Align frames for each trial relative to the entire session
        global_frame_ids = {}
        for trial, local_frame_ids in frame_id_events_norm.items():
            file_idx = file_num[trial]

            if file_idx is None:
                continue
            cumulative_frames_before_file = cumulative_frames[file_idx]  # Cumulative frames before this file
            # Ensure local_frame_ids contains lists
            maze_frames = local_frame_ids['maze'].flatten() if hasattr(local_frame_ids['maze'], 'flatten') else np.array(local_frame_ids['maze']).flatten()
            reward_frames = local_frame_ids['reward'].flatten() if hasattr(local_frame_ids['reward'], 'flatten') else np.array(local_frame_ids['reward']).flatten()
            iti_frames = local_frame_ids['iti'].flatten() if hasattr(local_frame_ids['iti'], 'flatten') else np.array(local_frame_ids['iti']).flatten()


            # Convert local frame numbers to global session frame numbers and ensure uniqueness
            global_frame_ids[trial] = {
                'maze': np.unique(maze_frames + cumulative_frames_before_file),
                'reward': np.unique(reward_frames + cumulative_frames_before_file),
                'iti': np.unique(iti_frames + cumulative_frames_before_file)
            }
            # # Convert local frame numbers to global session frame numbers
            # global_frame_ids[trial] = {
            #     'maze': [frame + cumulative_frames_before_file for frame in local_frame_ids['maze']],
            #     'reward': [frame + cumulative_frames_before_file for frame in local_frame_ids['reward']],
            #     'iti': [frame + cumulative_frames_before_file for frame in local_frame_ids['iti']]
            # }

        return global_frame_ids
    
    def load_latent_states(self):
        """Loads latent behavioral states for each trial."""
        # Load the complete latent states data once
        latent_states_full = pd.read_csv(self.latent_states_file)  # Assuming CSV for simplicity
        
        # Convert date from YYYY-MM-DD to YYMMDD format
        formatted_date = pd.to_datetime(self.date).strftime('%y%m%d')

        latent_states_full['mouseID'] = latent_states_full['mouseID'].astype(str).str.strip()
        latent_states_full['Date'] = latent_states_full['Date'].astype(str).str.strip()
        
        # Filter the latent states based on animalID and formatted_date
        # if dataset is not found, return empty dataframe
        if not ((latent_states_full['mouseID'] == self.animalID) &
                (latent_states_full['Date'] == formatted_date)).any():
            print(f"Warning: No latent states found for animalID {self.animalID} on date {formatted_date}.")
            return pd.DataFrame()
        else:
            latent_states_dataset = latent_states_full[(latent_states_full['mouseID'] == self.animalID) & 
                                                   (latent_states_full['Date'] == formatted_date)]
        return latent_states_dataset

    
    #LOAD INFO STRUCTURE
    def load_info(self,directory):
        """
        Load mouse dates and associated keys from the specified directory.

        Parameters:
            directory (str): The directory containing the info.mat file.

        Returns:
            list: A list of tuples containing (animalID, date, server).
            list: A list of mouse date keys.
        """
        load_dir = directory #'V:/Connie/results/active/mod' #

        # Load the condition_array_trials structure
        mat_data = sio.loadmat(os.path.join(load_dir,'info.mat'))
        info = mat_data['info'][0][0]

        # Assuming your mouse_date structure is loaded as a numpy array
        mouse_dates_keys = [
            item[0].replace('\\', '_').replace('/', '_')  # Replace slashes with underscores for consistency
            for item in info['mouse_date'][0]
        ]

        mouse_dates = []
        for item,server in zip(info['mouse_date'][0],info['serverid'][0]):
            current_item = item[0].replace('\\', '_').replace('/', '_')  # Replace both slashes with underscores
            
            parts = current_item.split('_')  # Split the modified string by underscore
            
            # Assuming animalID is the first part and date is the last part
            animalID = parts[0]  # Assuming animal ID is the first part
            date = parts[-1]  # Assuming date is the last part

            # Append as a tuple in the format (animalID, date, server)
            mouse_dates.append((animalID, date, server[0]))

        return mouse_dates, mouse_dates_keys
    
    #LOAD INFO STRUCTURE
    def load_engagement_proj(self):
        """
        Load engagement project data, including projection structure, test trials, 
        and selected engagement frames from predefined files.

        Returns:
            engagement_proj (list): List or array of engagement projection data 
                loaded from 'engagement_proj_all.mat'.
            test_trials (list): List or array of test trial data 
                loaded from 'test_trials_all.mat' in MATLAB indexing.
            engagement_frames (list): List or array of selected frame indices 
                loaded from 'selected_frames.mat' in MATLAB indexing.
        """

        directory = 'W:/Connie/Analysis/engagement/'
        # Load the engagement_proj structure
        engagement_proj_data = sio.loadmat(os.path.join(directory,'engagement_proj_all.mat'))
        engagement_proj = engagement_proj_data['engagement_proj_all'][0] 

        test_trials_data = sio.loadmat(os.path.join(directory,'test_trials_all.mat'))
        test_trials = test_trials_data['test_trials_all'][0]

        engagement_frames_data = sio.loadmat(os.path.join(directory,'selected_frames.mat')) #shape is trials x frames
        engagement_frames = engagement_frames_data['selected_frames'][0]

        return engagement_proj, test_trials, engagement_frames
