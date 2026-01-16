import pandas as pd
import scipy.io as sio
import numpy as np
import os

class DataLoader:
    def __init__(self, matlab_file, server=None, animalID=None, date=None): #pupil_csv=None, latent_states_file=None,
        # self.pupil_csv = pupil_csv
        self.matlab_file = matlab_file
        # self.latent_states_file = latent_states_file
        self.server = server
        self.animalID = animalID
        self.date = date

    
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