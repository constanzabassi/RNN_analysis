import numpy as np
import pandas as pd
from scipy.signal import find_peaks

class DataAligner:
    def __init__(self, neural_data, movement_frames, velocity_data, global_frame_ids, good_trials, celltype_array=None):
        """
        Initializes the DataAligner with neural data, movement frame indices,
        global frame indices, good trials, and optional celltype array.

        Args:
            neural_data (np.ndarray): Neural activity data, organized by trials and frames.
            movement_frames (dict): A dictionary with keys 'maze_frames', 'reward_frames', 'iti_frames'.
            global_frame_ids (dict): Global frame indices for each trial.
            good_trials (list): List of trials considered good for analysis.
            celltype_array (np.ndarray, optional): Array representing neuron types (e.g., pyr, som, pv).
        """
        self.neural_data = neural_data
        self.movement_frames = movement_frames  # Expecting a dict with keys 'maze_frames', 'reward_frames', 'iti_frames'
        self.global_frame_ids = global_frame_ids  # Global frame indices for each trial
        self.good_trials = good_trials  # List of good trials
        self.celltype_array = celltype_array
        self.velocity_data = velocity_data

    def align_data(self, field='all', last_n_frames=None, velocity_threshold=None):
        """
        Align neural and velocity data based on movement frame indices.

        Args:
            field (str): The type of field to align to ('maze', 'reward', 'iti', 'all').
            last_n_frames (int, optional): Number of frames to use from the end of the trial.
            velocity_threshold (float, optional): Minimum average velocity to include the trial.

        Returns:
            pd.DataFrame: A DataFrame containing aligned neural and velocity data for each trial.
                        Columns are 'trial', 'neural_data', and 'velocity_data'.
        """
        aligned_data_list = []

        def _flatten_frames(frames):
            if frames is None:
                return np.array([], dtype=int)
            return np.asarray(frames, dtype=int).ravel()

        # Iterate only over good trials
        for trial in self.good_trials:
            frame_block = self.movement_frames.get(trial)
            if frame_block is None:
                continue

            if field == 'maze':
                frame_indices = _flatten_frames(frame_block.get('maze_frames'))
            elif field == 'reward':
                frame_indices = _flatten_frames(frame_block.get('reward_frames'))
            elif field == 'iti':
                frame_indices = _flatten_frames(frame_block.get('iti_frames'))
            else:  # 'all'
                frame_sets = [
                    _flatten_frames(frame_block.get('maze_frames')),
                    _flatten_frames(frame_block.get('reward_frames')),
                    _flatten_frames(frame_block.get('iti_frames'))
                ]
                frame_sets = [frames for frames in frame_sets if frames.size > 0]
                if not frame_sets:
                    continue
                frame_indices = np.concatenate(frame_sets)

            if frame_indices.size == 0:
                continue

            if last_n_frames is not None:
                frame_indices = frame_indices[-last_n_frames:]

            aligned_neural = self.neural_data[trial][:, frame_indices]
            aligned_velocity = self.velocity_data[trial][frame_indices]

            if velocity_threshold is not None:
                mean_velocity = np.nanmean(aligned_velocity)
                if mean_velocity < velocity_threshold:
                    print(f"Excluding trial {trial} due to mean velocity below threshold: {mean_velocity}")
                    continue

            aligned_data_list.append({
                'trial': trial,
                'neural_data': aligned_neural,
                'velocity_data': aligned_velocity,
                'trial_id': trial
            })

        return pd.DataFrame(aligned_data_list)
    
    def calculate_trial_means(self, aligned_data_df, trial_ids=None, trial_indices=None, usable_frames=None, neuron_type=None, neuron_group=None):
        """
        Calculate mean neural and velocity data for each trial.

        Returns:
            tuple: (trial_means_neural, trial_means_velocity, trial_means_neural_all)
        """
        trial_means_neural = {}
        trial_means_velocity = {}
        trial_means_neural_all = {}

        if trial_ids is None:
            trial_ids = aligned_data_df['trial'].unique()
            print(f'Trials used: {trial_ids}')

        selected_trials = aligned_data_df[aligned_data_df['trial'].isin(trial_ids)].copy()

        if trial_indices is None:
            selected_trials['trial_index'] = 0
        else:
            if isinstance(trial_indices, dict):
                trial_index_series = pd.Series(trial_indices)
                selected_trials['trial_index'] = selected_trials['trial'].map(trial_index_series)
            else:
                trial_indices_array = np.asarray(trial_indices)
                selected_trials['trial_index'] = trial_indices_array[selected_trials['trial'].to_numpy()]

        for trial_index in selected_trials['trial_index'].unique():
            index_trials = selected_trials[selected_trials['trial_index'] == trial_index]

            neural_means = []
            velocity_means = []
            neural_all_means = []

            for _, trial_data in index_trials.iterrows():
                trial_neural = trial_data['neural_data']
                trial_velocity = trial_data['velocity_data']

                if usable_frames is not None:
                    trial_neural = trial_neural[:, usable_frames]
                    trial_velocity = trial_velocity[usable_frames]

                if neuron_type is not None:
                    neuron_indices = np.asarray(neuron_group[neuron_type]).ravel()
                    trial_neural_mean = np.nanmean(trial_neural[neuron_indices, :], axis=(0, 1))
                    trial_neural_all_mean = np.nanmean(trial_neural[neuron_indices, :], axis=1)
                else:
                    trial_neural_mean = np.nanmean(trial_neural, axis=(0, 1))
                    trial_neural_all_mean = np.nanmean(trial_neural, axis=1)

                neural_means.append(trial_neural_mean)
                velocity_means.append(np.nanmean(trial_velocity))
                neural_all_means.append(trial_neural_all_mean)

            trial_means_neural[trial_index] = pd.Series(neural_means, index=index_trials['trial'])
            trial_means_velocity[trial_index] = pd.Series(velocity_means, index=index_trials['trial'])
            trial_means_neural_all[trial_index] = pd.DataFrame(neural_all_means, index=index_trials['trial'])

        return trial_means_neural, trial_means_velocity, trial_means_neural_all

    def align_behavior_data(self, imaging, align_info, alignment_frames, left_padding, right_padding, alignment, cell_ids=None):
        # Initial setup code same as before...
        
        if alignment['type'] == 'stimulus':
            frames = self.find_alignment_frames(alignment_frames, list(range(3)), 
                                        left_padding, right_padding)
        
        elif alignment['type'] == 'turn':
            frames = self.find_alignment_frames(alignment_frames, [3], 
                                        left_padding, right_padding)
        
        elif alignment['type'] == 'all':
            frames = self.find_alignment_frames(alignment_frames, list(range(6)), 
                                        left_padding, right_padding)
        
        elif alignment['type'] == 'pre':
            frames = self.find_alignment_frames(alignment_frames, list(range(5)), 
                                        left_padding, right_padding)
        
        # Initialize aligned imaging array
        empty_trials = [i for i in range(len(imaging)) if not imaging[i]['good_trial']]
        good_trials = [i for i in range(len(imaging)) if i not in empty_trials]
        n_trials = len(good_trials)
        if cell_ids is None:
            cell_ids = np.arange(imaging[good_trials[0]]['dff'].shape[0])
            n_cells = imaging[good_trials[0]]['dff'].shape[0] #len(cell_ids)
        else:
            n_cells = len(cell_ids)
        n_frames = frames.shape[1]
        aligned_imaging = np.zeros((n_trials, n_cells, n_frames))
        
        # Common alignment loop for all types
        for vr_trials in range(len(good_trials)):
            frames_to_include = frames[vr_trials]
            
            if alignment['data_type'] == 'dff':
                data = imaging[good_trials[vr_trials]]['dff']
            elif alignment['data_type'] == 'z_dff':
                data = imaging[good_trials[vr_trials]]['z_dff']
            else:  # deconv
                data = imaging[good_trials[vr_trials]]['deconv']
            print(data.shape)
                
            aligned_imaging[vr_trials] = data[cell_ids][:, frames_to_include]
       
        # Replace the simple list comprehension with a structured dictionary
        imaging_array = []
        for i in good_trials:
            trial_data = {}
            movement_data = imaging[i]['movement_in_imaging_time'][0][0]
            
            # Preserve all fields from the original structure
            for field_name in movement_data.dtype.names:
                trial_data[field_name] = movement_data[field_name]
            
            imaging_array.append(trial_data)

        return aligned_imaging, imaging_array, align_info, frames

    def find_alignment_frames(self, alignment_frames: np.ndarray, event_id: list, left_padding: np.ndarray, right_padding: np.ndarray):
        """
        Align frames based on events and padding.
        
        Args:
            alignment_frames: Array of frame indices for each event
            event_id: List of event indices to align
            left_padding: Padding before each event
            right_padding: Padding after each event
        
        Returns:
            frames: Array of aligned frame indices
        """
        # # Initialize frames array
        # n_trials = len(alignment_frames[1])
        # total_padding = sum(left_padding[event_id]) + sum(right_padding[event_id]) + len(event_id)
        # frames = np.zeros((n_trials, total_padding))
        
        # # Generate frame indices for each trial
        # for i in range(n_trials):
        #     temp_frames = []
        #     for event in event_id:
        #         # Create range of frames around event
        #         pad_range = np.arange(-left_padding[event], right_padding[event] + 1)
        #         temp_frames.extend(alignment_frames[event, i] + pad_range)
        #     frames[i, :] = temp_frames
        
        # # Remove zero frames (for passive condition)
        # zero_frames = np.where(frames[0, :] == 0)[0]
        # if len(zero_frames) > 1:
        #     frames = np.delete(frames, zero_frames, axis=1)
        
        # return frames
        # Calculate the size of the frames array
        # Calculate the size of the frames array
        num_trials = len(alignment_frames[0])
        print(num_trials)
        total_frame_length = (
            np.sum([left_padding[event] for event in event_id]) +
            np.sum([right_padding[event] for event in event_id]) +
            len(event_id)
        )

        frames = np.zeros((num_trials, total_frame_length), dtype=int)
        print(frames.shape) 

        for i in range(num_trials):
            temp_frames = []
            for event in event_id:
                left_pad = -left_padding[event]
                right_pad = right_padding[event]
                event_frames = alignment_frames[event, i] + np.arange(left_pad, right_pad + 1)
                temp_frames.extend(event_frames)
            frames[i, :] = temp_frames

        # Remove zero frames (for passive trials)
        if np.any(frames == 0):
            zero_frame_indices = np.where(frames[0, :] == 0)[0]
            frames = np.delete(frames, zero_frame_indices, axis=1)

        return frames



    def find_align_info(self,imaging, turn_frames, alternative_alignment=False):
        empty_trials = [i for i in range(len(imaging)) if not imaging[i]['good_trial']]
        good_trials = [i for i in range(len(imaging)) if i not in empty_trials]

        imaging_array = [imaging[i]['movement_in_imaging_time'][0][0] for i in good_trials]
        align_info = {}

        # Get maze length
        maze_length = [len(trial['maze_frames'][0]) for trial in imaging_array]

        # Get stimulus info
        stimulus_repeats_onsets = []
        for trial in imaging_array:
            stimulus = np.array(trial['stimulus'][0])
            diff_stimulus = np.diff(stimulus)
            peaks, _ = find_peaks(diff_stimulus)
            stimulus_repeats_onsets.append(peaks + 1)  # Adjust for diff offset

        total_stimulus_repeats = [len(onsets) + 1 for onsets in stimulus_repeats_onsets]
        stim_onset = [np.min(onsets) for onsets in stimulus_repeats_onsets]

        shortest_maze_length = min(
            maze_len - stim_on for maze_len, stim_on in zip(maze_length, stim_onset)
        )
        min_length_stim = min(stim_onset) - 1

        if min_length_stim == 0:
            min_length_stim = min(set(stim_onset) - {min(stim_onset)}) - 1

        # Get reward info
        max_length_reward = min(len(trial['reward_frames'][0]) for trial in imaging_array)
        reward_onset = [
            np.where(np.array(trial['is_reward'][0]) == 1)[0] for trial in imaging_array
        ]
        reward_trial = [i for i, onset in enumerate(reward_onset) if len(onset) > 0]
        min_length_reward = min(
            reward_onset[i][0] - np.min(imaging_array[i]['reward_frames'][0])
            for i in reward_trial
        )

        # Pure tones
        pure_onsets = [
            np.where(np.array(trial['pure_tones'][0]) == 1)[0] for trial in imaging_array
        ]
        pure_onsets = [np.min(onset) for onset in pure_onsets if len(onset) > 0]

        # Frame alignments
        frames_around = turn_frames
        align_info['turn_onset'] = frames_around + 1
        align_info['maze_length'] = maze_length
        align_info['min_length'] = shortest_maze_length
        align_info['stimulus_onset'] = min_length_stim + 1
        align_info['stimulus_repeats_onsets'] = stimulus_repeats_onsets
        align_info['total_stimulus_repeats'] = total_stimulus_repeats
        align_info['max_length_reward'] = max_length_reward
        align_info['reward_onset'] = min_length_reward + 1
        align_info['pure_onsets'] = pure_onsets
        align_info['good_trials'] = good_trials

        # Event alignment
        alignment_frames = np.zeros((6, len(good_trials)))  # 6 events, len(good_trials) trials
        left_padding = {}
        right_padding = {}

        for event in range(6):
            if event == 0:
                alignment_frames[event,:] = [onsets[0] for onsets in stimulus_repeats_onsets]
                left_padding[event] = 6
                right_padding[event] = 30

            elif event in [1,2]:
                alignment_frames[event,:] = [onsets[event] for onsets in stimulus_repeats_onsets]
                left_padding[event] = 1
                right_padding[event] = 30

            elif event == 3:
                alignment_frames[event,:] = [
                    trial['turn_frame'][0] - 1 if isinstance(trial['turn_frame'], list) else trial['turn_frame'] - 1
                    for trial in imaging_array
                ]
                left_padding[event] = 30 if not alternative_alignment else 90
                right_padding[event] = 12 if not alternative_alignment else 60


            elif event == 4:
                if all(len(onset) == 0 for onset in reward_onset):
                    alignment_frames[event,:] = pure_onsets
                    left_padding[event] = 1 if not alternative_alignment else 60
                    right_padding[event] = max_length_reward - 1
                else:
                    alignment_frames[event,:] = [None] * len(good_trials)
                    for i in reward_trial:
                        alignment_frames[event,i] = reward_onset[i][0]
                    incorrect_trials = set(range(len(good_trials))) - set(reward_trial)
                    for i in incorrect_trials:
                        alignment_frames[event,i] = pure_onsets[i]
                    left_padding[event] = 1
                    right_padding[event] = 23

            elif event == 5:
                alignment_frames[event,:] = [trial['iti_frames'][0][0] for trial in imaging_array]
                left_padding[event] = 1
                right_padding[event] = 80

        align_info['alignment_frames'] = alignment_frames
        align_info['left_padding'] = left_padding
        align_info['right_padding'] = right_padding

        return align_info, alignment_frames, left_padding, right_padding

