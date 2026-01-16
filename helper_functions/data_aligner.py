import numpy as np
import pandas as pd
from scipy.signal import find_peaks

class DataAligner:
    def __init__(self, neural_data, movement_frames, velocity_data, global_frame_ids, good_trials, pupil_data =None, celltype_array=None, engagement_proj = None, engagement_proj_trials = None, engagement_frames = None):
        """
        Initializes the DataAligner with neural data, movement frame indices, pupil data, global frame indices,
        good trials, and optional celltype array.
        
        Args:
            neural_data (np.ndarray): Neural activity data, organized by trials and frames.
            movement_frames (dict): A dictionary with keys 'maze_frames', 'reward_frames', 'iti_frames'.
            pupil_data (pd.DataFrame): Pupil data corresponding to each frame.
            global_frame_ids (dict): Global frame indices for each trial.
            good_trials (list): List of trials considered good for analysis.
            celltype_array (np.ndarray, optional): Array representing neuron types (e.g., pyr, som, pv).
        """
        self.neural_data = neural_data
        self.movement_frames = movement_frames  # Expecting a dict with keys 'maze_frames', 'reward_frames', 'iti_frames'
        self.pupil_data = pupil_data
        self.global_frame_ids = global_frame_ids  # Global frame indices for pupil data
        self.good_trials = good_trials  # List of good trials
        self.celltype_array = celltype_array
        self.velocity_data = velocity_data
        self.engagement_proj = engagement_proj
        self.engagement_proj_trials = engagement_proj_trials
        self.engagement_frames = engagement_frames

    def align_data(self, field='all', last_n_frames=None,max_interp_percentage=0.50, velocity_threshold=None, trial_engagement_add = 0):
        """
        Aligns neural and pupil data based on specified movement frame indices and global frame IDs.

        Args:
            field (str): The type of field to align to ('maze', 'reward', 'iti', 'all').
            last_n_frames (int, optional): Number of frames to use from the end of the trial.
            velocity_threshold (float, optional): Minimum average velocity to include the trial. Trials with mean velocity below this threshold will be excluded.
            normalize_mode (str, optional): Method to normalize pupil area ('max' for max-normalization, 'zscore' for z-score normalization).
            trial_engagement_add (float, optional): whether to add a constant to the trial ids for engagement projection alignment.

        Returns:
            pd.DataFrame: A DataFrame containing aligned neural and pupil data for each trial.
                        Columns are 'trial', 'neural_data', and 'pupil_data'.
        """
        aligned_data_list = []  # Initialize a list to collect data for each trial


        #frame within training session is this
        if field == 'all':
            current_frames = [
                frame for trial in self.good_trials for field in ['maze', 'reward', 'iti']
                for frame in self.global_frame_ids[trial][field]
            ]
        else:
            current_frames = [
                frame for trial in self.good_trials
                for frame in self.global_frame_ids[trial][field]
            ]
        
        # if normalize_mode == "max":
        # ✅ Max-normalization
        max_area = self.pupil_data['area'].iloc[current_frames].max()
        if max_area != 0:
            self.pupil_data['Normalized_Area'] = self.pupil_data['area'] / max_area
        else:
            self.pupil_data['Normalized_Area'] = self.pupil_data['area']  # handle all-zero case

        # ✅ Z-score normalization
        mean_area = self.pupil_data['area'].iloc[current_frames].mean()
        std_area = self.pupil_data['area'].iloc[current_frames].std()
        if std_area != 0:
            self.pupil_data['Normalized_Area_Z'] = (self.pupil_data['area'] - mean_area) / std_area
        else:
            self.pupil_data['Normalized_Area_Z'] = self.pupil_data['area'] - mean_area  # handle zero variance


        # Iterate only over good trials
        for trial in self.good_trials:
            # Get trial index based on good trials
            trial_id = trial

            # Get frame indices based on the specified field
            if field == 'maze':
                frame_indices = self.movement_frames[trial_id]['maze_frames']
            elif field == 'reward':
                frame_indices = self.movement_frames[trial_id]['reward_frames']
            elif field == 'iti':
                frame_indices = self.movement_frames[trial_id]['iti_frames']
            else:  # 'all'
                frame_indices = np.concatenate([
                    self.movement_frames[trial_id]['maze_frames'],
                    self.movement_frames[trial_id]['reward_frames'],
                    self.movement_frames[trial_id]['iti_frames']
                ])

            # If last_n_frames is specified, slice the frame indices accordingly
            if last_n_frames is not None:
                frame_indices = frame_indices[-last_n_frames:]

            # Align neural data for the selected frames
            aligned_neural = self.neural_data[trial][:, frame_indices]
            aligned_velocity = self.velocity_data[trial][frame_indices]

            # Check for velocity threshold if specified (using mean velocity)
            if velocity_threshold is not None:
                mean_velocity = np.mean(aligned_velocity)
                if mean_velocity < velocity_threshold:
                    print(f"Excluding trial {trial_id} due to mean velocity below threshold: {mean_velocity}")
                    continue  # Skip this trial if mean velocity is below the threshold

            # Align pupil data using global frame IDs for the trial
            pupil_frames = self.global_frame_ids[trial_id][field]
            if last_n_frames is not None:
                pupil_frames = pupil_frames[-last_n_frames:]
            aligned_pupil = self.pupil_data.iloc[pupil_frames]

            # Handle interpolation in pupil data
            interp_values = aligned_pupil['interp']  # Assuming 'interp' is the column for interpolated values

            # Calculate the total number of values and the number of interpolated values
            total_values = len(interp_values)
            num_interpolated = np.sum(interp_values)  # Assuming NaN indicates an interpolated value
            
            # Calculate the percentage of interpolated values
            interp_percentage = num_interpolated / total_values

            # Check if the percentage exceeds the threshold
            if interp_percentage > max_interp_percentage:
                # Option 1: Exclude this trial by continuing to the next iteration
                print(f"Excluding trial {trial_id} due to excessive interpolation: {interp_percentage:.2%} interpolated")
                continue  # Skip this trial

                # Option 2: Set interpolated values to NaN in aligned_pupil (if you don't want to exclude)
                # aligned_pupil.loc[np.isnan(interp_values), 'area'] = np.nan

            # Align engagement projection if available
            if self.engagement_proj is not None and self.engagement_proj_trials is not None:
                next_trial = trial + trial_engagement_add
                if next_trial in self.engagement_proj_trials:
                    idx = list(self.engagement_proj_trials).index(next_trial)
                    aligned_enagement_proj = self.engagement_proj[idx]
                else:
                    aligned_enagement_proj = None
            else:
                aligned_enagement_proj = None


            # Append to list as a dictionary for each trial
            aligned_data_list.append({
                'trial': trial,
                'neural_data': aligned_neural,
                'pupil_data': aligned_pupil,
                'velocity_data': aligned_velocity,
                'engagement_proj': aligned_enagement_proj,
                'trial_id': trial_id,
                'next_trial': next_trial
            })

        # Convert list of dictionaries to DataFrame
        aligned_data_df = pd.DataFrame(aligned_data_list)

        return aligned_data_df
    
    def align_data_by_engagement(
        self,
        max_interp_percentage=0.50,
        velocity_threshold=None,
        ):
        """
        Align neural, velocity, and pupil data using engagement_frames (trials x frames)
        that are aligned to sound onset (and can include pre-onset frames from the prior trial).
        REQUIRED class attributes
        -------------------------
        self.engagement_proj_trials : iterable of trial IDs (one per row in engagement_frames)
        self.engagement_proj        : iterable of mean projection values for each trial (same order)
        self.engagement_frames      : dict[int -> 1D array-like of GLOBAL frame IDs]
                                    OR 2D array-like (n_trials x n_frames) whose row order matches engagement_proj_trials
        self.global_frame_ids       : dict[trial_id -> {'maze': [...], 'reward': [...], 'iti': [...]}]  # GLOBAL frame IDs
        self.neural_data            : dict[trial_id -> np.ndarray]  # shape (n_neurons, n_local_frames)
        self.velocity_data          : dict[trial_id -> np.ndarray]  # shape (n_local_frames,)
        self.pupil_data             : pd.DataFrame indexed by GLOBAL frame ID, with columns:
                                    'area' (float), 'interp' (bool or 0/1)
        self.good_trials            : iterable of trial IDs to consider (subset/superset of engagement_proj_trials ok)
        Returns
        -------
        pd.DataFrame with columns:
            - trial (int)
            - neural_data (np.ndarray)         shape (n_neurons, T)
            - velocity_data (np.ndarray)       shape (T,)
            - pupil_data (pd.DataFrame)        index = global frame IDs (len T)
            - engagement_frames (np.ndarray)   the exact global frames used (len T)
            - engagement_proj (float)          mean projection value for that engagement trial
        """

        # -------------------------
        # Build global → (trial_id, local_idx) map
        # -------------------------
        global_to_local = {}
        for tr, epochs in self.global_frame_ids.items():
            per_trial_global = (
                list(epochs['maze']) + list(epochs['reward']) + list(epochs['iti'])
            )
            for local_idx, g in enumerate(per_trial_global):
                global_to_local[g] = (tr, local_idx)
        # -------------------------
        # Helper to get engagement frames for a trial
        # -------------------------
        frames_is_dict = isinstance(self.engagement_frames, dict)
        def get_engagement_frames_for_trial(trial_id, row_idx):
            if frames_is_dict:
                frames = self.engagement_frames[trial_id]
            else:
                frames = self.engagement_frames[row_idx]
            frames = np.asarray(frames)
            frames = frames[np.isfinite(frames) & (frames >= 0)].astype(int)
            frames = np.array([g for g in frames if g in global_to_local], dtype=int)
            return frames
        # -------------------------
        # Gather all frames for normalization scope
        # -------------------------
        all_used_globals = []
        per_trial_frames = {}
        trial_to_row = {tr: i for i, tr in enumerate(self.engagement_proj_trials)}
        for tr in self.engagement_proj_trials:
            if hasattr(self, 'good_trials') and (tr not in self.good_trials):
                continue
            row_idx = trial_to_row[tr]
            g_frames = get_engagement_frames_for_trial(tr, row_idx)
            if g_frames.size == 0:
                continue
            per_trial_frames[tr] = g_frames
            all_used_globals.extend(g_frames.tolist())
        if len(per_trial_frames) == 0:
            return pd.DataFrame(columns=[
                'trial', 'neural_data', 'velocity_data', 'pupil_data',
                'engagement_frames', 'engagement_proj'
            ])
        # -------------------------
        # Pupil normalization over the union of used frames
        # -------------------------
        union_idx = pd.Index(sorted(set(all_used_globals)))
        max_area = self.pupil_data['area'].reindex(union_idx).max()
        if np.isfinite(max_area) and (max_area != 0):
            self.pupil_data['Normalized_Area'] = self.pupil_data['area'] / max_area
        else:
            self.pupil_data['Normalized_Area'] = self.pupil_data['area']
        subset = self.pupil_data['area'].reindex(union_idx)
        mean_area = subset.mean()
        std_area = subset.std()
        if np.isfinite(std_area) and (std_area != 0):
            self.pupil_data['Normalized_Area_Z'] = (self.pupil_data['area'] - mean_area) / std_area
        else:
            self.pupil_data['Normalized_Area_Z'] = self.pupil_data['area'] - mean_area
        # -------------------------
        # Align data per engagement trial
        # -------------------------
        aligned_rows = []
        for tr, g_frames in per_trial_frames.items():
            n_neurons = self.neural_data[tr].shape[0] if tr in self.neural_data else \
                        next(iter(self.neural_data.values())).shape[0]
            T = len(g_frames)
            aligned_neural = np.full((n_neurons, T), np.nan, dtype=float)
            aligned_velocity = np.full((T,), np.nan, dtype=float)
            lookup = [global_to_local[g] for g in g_frames]
            by_src = {}
            for pos, (src_tr, loc_idx) in enumerate(lookup):
                by_src.setdefault(src_tr, []).append((pos, loc_idx))
            for src_tr, pos_loc in by_src.items():
                pos_loc.sort(key=lambda x: x[0])
                positions = [p for p, _ in pos_loc]
                local_idxs = [li for _, li in pos_loc]
                ndata = self.neural_data[src_tr][:, local_idxs]
                vdata = self.velocity_data[src_tr][local_idxs]
                aligned_neural[:, positions] = ndata
                aligned_velocity[positions] = vdata
            # Optional velocity filtering
            if velocity_threshold is not None:
                mean_v = np.nanmean(aligned_velocity)
                if np.isnan(mean_v) or (mean_v < velocity_threshold):
                    print(f"Excluding trial {tr} due to mean velocity below threshold: {mean_v}")
                    continue  # Skip this trial if mean velocity is below the threshold
            aligned_pupil = self.pupil_data.reindex(g_frames)
            interp_vals = np.asarray(aligned_pupil['interp'].fillna(0)).astype(float)
            interp_pct = float(np.nansum(interp_vals)) / max(1, interp_vals.size)
            if interp_pct > max_interp_percentage:
                print(f"Excluding trial {tr} due to excessive interpolation: {interp_pct:.2%} interpolated")
                continue
            # Grab engagement mean for this trial
            idx = list(self.engagement_proj_trials).index(tr)
            engagement_proj_val = self.engagement_proj[idx]
            aligned_rows.append({
                'trial': tr,
                'neural_data': aligned_neural,
                'velocity_data': aligned_velocity,
                'pupil_data': aligned_pupil,
                'engagement_frames': g_frames,
                'engagement_proj': engagement_proj_val,
            })
        return pd.DataFrame(aligned_rows)
    
    # def align_data_by_engagement(
    #     self,
    #     max_interp_percentage=0.50,
    #     velocity_threshold=None,
    #     trial_engagement_add=0,
    #     engagement_key="default",
    # ):
    #     """
    #     Align neural, velocity, and pupil data using engagement_frames (trials x frames)
    #     that were used to compute engagement projections aligned to sound onset
    #     (including pre-onset frames that may extend into the previous trial).
    #     Assumptions:
    #     - self.engagement_frames: dict OR array-like indexed by trial, each entry is a
    #         1D list/array of GLOBAL frame IDs for the engagement-aligned window for that trial.
    #         If it's a dict keyed by (engagement_key, trial), pass `engagement_key` or adapt below.
    #     - self.global_frame_ids[trial] has lists of GLOBAL frame IDs per epoch:
    #         {'maze': [...], 'reward': [...], 'iti': [...]}
    #     - self.neural_data[trial] is (n_neurons x n_local_frames) for that trial.
    #     - self.velocity_data[trial] is (n_local_frames,) for that trial.
    #     - self.pupil_data is a DataFrame indexed by GLOBAL frame number and includes:
    #         'area' (float) and 'interp' (bool or 0/1 indicating interpolated samples).
    #     - self.good_trials is an iterable of trial IDs you want to include.
    #     - Optional: self.engagement_proj, self.engagement_proj_trials (as in your existing function).
    #     Returns
    #     -------
    #     pd.DataFrame with columns:
    #         - trial (int): the base trial id (from self.good_trials)
    #         - neural_data (np.ndarray): shape (n_neurons, n_frames)
    #         - velocity_data (np.ndarray): shape (n_frames,)
    #         - pupil_data (pd.DataFrame): rows correspond to the same frames (index = global ids)
    #         - engagement_proj (np.ndarray or None)
    #         - engagement_frames (np.ndarray of global frame ids actually used)
    #         - next_trial (int): trial id + trial_engagement_add
    #     """
    #     # -------------------------
    #     # Helper: build a global→(trial, local_idx) map
    #     # -------------------------
    #     def _build_global_index():
    #         global_to_local = {}
    #         for tr in self.good_trials:
    #             # Concatenate per-trial global frames in the same order they appear locally
    #             per_trial_global = (
    #                 list(self.global_frame_ids[tr]['maze']) +
    #                 list(self.global_frame_ids[tr]['reward']) +
    #                 list(self.global_frame_ids[tr]['iti'])
    #             )
    #             for local_idx, g in enumerate(per_trial_global):
    #                 global_to_local[g] = (tr, local_idx)
    #         return global_to_local
    #     global_to_local = _build_global_index()
    #     # -------------------------
    #     # Collect the engagement frame sets we will actually use
    #     # -------------------------
    #     engagement_by_trial = {}
    #     all_used_global_frames = []
    #     for tr in self.good_trials:
    #         tr_key = tr + trial_engagement_add
    #         # Flexible access: handle dict keyed by trial or (engagement_key, trial),
    #         # or a 2D array-like indexed by trial.
    #         frames = None
    #         if isinstance(self.engagement_frames, dict):
    #             # try (engagement_key, trial) first, then trial
    #             if (engagement_key, tr_key) in self.engagement_frames:
    #                 frames = self.engagement_frames[(engagement_key, tr_key)]
    #             elif tr_key in self.engagement_frames:
    #                 frames = self.engagement_frames[tr_key]
    #         else:
    #             # assume array-like indexed by trial order matching self.good_trials ordering
    #             # map from absolute trial id to row index if needed
    #             try:
    #                 frames = self.engagement_frames[tr_key]
    #             except Exception:
    #                 # last resort: if engagement_frames has same length/order as good_trials
    #                 idx = list(self.good_trials).index(tr)
    #                 frames = self.engagement_frames[idx]
    #         if frames is None:
    #             # No engagement window for this trial; skip
    #             continue
    #         # Clean: remove NaNs/negatives if present
    #         frames = np.asarray(frames)
    #         valid_mask = np.isfinite(frames) & (frames >= 0)
    #         frames = frames[valid_mask].astype(int)
    #         # Keep only frames we can resolve into (trial, local_idx)
    #         resolvable = [g for g in frames if g in global_to_local]
    #         if len(resolvable) == 0:
    #             continue
    #         engagement_by_trial[tr] = np.array(resolvable, dtype=int)
    #         all_used_global_frames.extend(resolvable)
    #     if len(engagement_by_trial) == 0:
    #         # Nothing to align
    #         return pd.DataFrame(columns=[
    #             'trial', 'neural_data', 'velocity_data',
    #             'pupil_data', 'engagement_proj', 'engagement_frames', 'next_trial'
    #         ])
    #     # -------------------------
    #     # Pupil normalization over the union of frames actually used
    #     # -------------------------
    #     union_idx = pd.Index(sorted(set(all_used_global_frames)))
    #     # Max-normalization
    #     max_area = self.pupil_data['area'].reindex(union_idx).max()
    #     if max_area and np.isfinite(max_area) and max_area != 0:
    #         self.pupil_data['Normalized_Area'] = self.pupil_data['area'] / max_area
    #     else:
    #         self.pupil_data['Normalized_Area'] = self.pupil_data['area']
    #     # Z-score normalization
    #     subset = self.pupil_data['area'].reindex(union_idx)
    #     mean_area = subset.mean()
    #     std_area = subset.std()
    #     if std_area and np.isfinite(std_area) and std_area != 0:
    #         self.pupil_data['Normalized_Area_Z'] = (self.pupil_data['area'] - mean_area) / std_area
    #     else:
    #         self.pupil_data['Normalized_Area_Z'] = self.pupil_data['area'] - mean_area
    #     # -------------------------
    #     # Extract aligned data per trial window (handles cross-trial seamlessly)
    #     # -------------------------
    #     aligned_rows = []
    #     for tr in self.good_trials:
    #         if tr not in engagement_by_trial:
    #             continue
    #         g_frames = engagement_by_trial[tr]  # global frames for this trial’s engagement window
    #         # Map global frames to (trial_id, local_idx); group by trial for efficient slicing
    #         lookup = [global_to_local[g] for g in g_frames]
    #         # We must preserve the original order in g_frames
    #         segs = {}
    #         for pos, (t_id, l_idx) in enumerate(lookup):
    #             segs.setdefault(t_id, []).append((pos, l_idx))
    #         # Build neural and velocity arrays in order
    #         # We’ll assemble in chunks per contributing trial, then place into correct positions.
    #         # Determine shapes
    #         # Use any trial to get n_neurons
    #         some_trial = next(iter(self.neural_data))
    #         n_neurons = self.neural_data[some_trial].shape[0]
    #         T = len(g_frames)
    #         aligned_neural = np.empty((n_neurons, T), dtype=float)
    #         aligned_neural[:] = np.nan
    #         aligned_velocity = np.empty((T,), dtype=float)
    #         aligned_velocity[:] = np.nan
    #         for t_id, pos_lidx_list in segs.items():
    #             # sort by position to allow minimal slicing
    #             pos_lidx_list.sort(key=lambda x: x[0])
    #             positions = [p for p, _ in pos_lidx_list]
    #             local_idxs = [li for _, li in pos_lidx_list]
    #             # Pull from this trial’s local arrays
    #             # neural_data[t_id]: (neurons x local_frames)
    #             # velocity_data[t_id]: (local_frames,)
    #             ndata = self.neural_data[t_id][:, local_idxs]
    #             vdata = self.velocity_data[t_id][local_idxs]
    #             aligned_neural[:, positions] = ndata
    #             aligned_velocity[positions] = vdata
    #         # Velocity threshold (mean over the aligned window)
    #         if velocity_threshold is not None:
    #             mean_v = np.nanmean(aligned_velocity)
    #             if np.isnan(mean_v) or mean_v < velocity_threshold:
    #                 # Skip low-velocity windows
    #                 continue
    #         # Pupil segment (using global frames directly; index aligns to g_frames)
    #         aligned_pupil = self.pupil_data.reindex(g_frames)
    #         # Interp percentage check
    #         interp_col = aligned_pupil['interp']
    #         # Accept bool/0-1; treat True/1 as interpolated
    #         interp_vals = np.asarray(interp_col.fillna(0)).astype(float)
    #         total_vals = interp_vals.size if interp_vals.size else 1
    #         interp_pct = float(np.nansum(interp_vals)) / float(total_vals)
    #         if interp_pct > max_interp_percentage:
    #             # Exclude this trial/window
    #             continue
    #         # Engagement projection for this (possibly offset) trial if available
    #         if getattr(self, 'engagement_proj', None) is not None and getattr(self, 'engagement_proj_trials', None) is not None:
    #             next_trial = tr + trial_engagement_add
    #             if next_trial in self.engagement_proj_trials:
    #                 idx = list(self.engagement_proj_trials).index(next_trial)
    #                 aligned_engagement_proj = self.engagement_proj[idx]
    #             else:
    #                 aligned_engagement_proj = None
    #         else:
    #             next_trial = tr + trial_engagement_add
    #             aligned_engagement_proj = None
    #         aligned_rows.append({
    #             'trial': tr,
    #             'neural_data': aligned_neural,
    #             'velocity_data': aligned_velocity,
    #             'pupil_data': aligned_pupil,         # index = global frame ids
    #             'engagement_proj': aligned_engagement_proj,
    #             'engagement_frames': g_frames,       # the exact global frames used
    #             'next_trial': next_trial
    #         })
    #     return pd.DataFrame(aligned_rows)

    # def align_data(self, field='all', last_n_frames=None):
    #     """
    #     Aligns neural and pupil data based on specified movement frame indices and global frame IDs.

    #     Args:
    #         field (str): The type of field to align to ('maze', 'reward', 'iti', 'all').
    #         last_n_frames (int, optional): Number of frames to use from the end of the trial.

    #     Returns:
    #         dict: A dictionary containing aligned neural and pupil data for each trial.
    #     """
    #     # Initialize a dictionary to hold aligned data for each trial
    #     aligned_data = {}

    #     # Iterate only over good trials
    #     for trial in self.good_trials:  # Only process good trials
    #         aligned_data[trial] = {}  # Initialize trial entry

    #         # Get trial index based on good trials
    #         trial_id = trial

    #         if field == 'maze':
    #             frame_indices = self.movement_frames[trial_id]['maze_frames']  # Get maze frame indices
    #         elif field == 'reward':
    #             frame_indices = self.movement_frames[trial_id]['reward_frames']  # Get reward frame indices
    #         elif field == 'iti':
    #             frame_indices = self.movement_frames[trial_id]['iti_frames']  # Get ITI frame indices
    #         else:  # 'all'
    #             frame_indices = np.concatenate([
    #                 self.movement_frames[trial_id]['maze_frames'],
    #                 self.movement_frames[trial_id]['reward_frames'],
    #                 self.movement_frames[trial_id]['iti_frames']
    #             ])

    #         # If last_n_frames is specified, slice the frame indices accordingly
    #         if last_n_frames is not None:
    #             frame_indices = frame_indices[-last_n_frames:]

    #         # Align neural data
    #         aligned_neural = self.neural_data[trial]  # Get the neural data for the current trial
    #         aligned_neural = aligned_neural[:,frame_indices]  # Adjust for shape
    #         aligned_data[trial]['neural'] = aligned_neural  # Store neural data

    #         # Align pupil data using global frame IDs for the trial
    #         pupil_frames = self.global_frame_ids[trial_id][field]  # Get global frames for the field
    #         aligned_pupil = self.pupil_data.iloc[pupil_frames]  # Align pupil data
    #         aligned_data[trial]['pupil'] = aligned_pupil  # Store pupil data

    #     return aligned_data

    def calculate_trial_means(self, aligned_data_df, trial_ids=None, trial_indices=None, usable_frames=None, neuron_type=None, neuron_group = None):
        """
        Calculates the mean neural and pupil data for each trial based on provided trial ids,
        trial indices, and frames, optionally for a specific neuron type.
        """
        trial_means_neural = {}
        trial_means_pupil = {}
        trial_means_pupil_z = {} #uses zscored pupil data
        trial_means_velocity = {}
        trial_means_neural_all = {} #similar to means_neural but instead I have individual means for each cell
        trials_means_engagement = {}

        # If no trial_ids are provided, use all unique trials in aligned_data_df
        if trial_ids is None:
            trial_ids = aligned_data_df['trial'].unique()
            print(f'Trials used: {trial_ids}')
        
        # Filter trials in aligned_data_df based on trial_ids
        selected_trials = aligned_data_df[aligned_data_df['trial'].isin(trial_ids)]

        # If no trial_indices provided, assign a single group for all trials
        if trial_indices is None:
            selected_trials['trial_index'] = 0  # Assign a default index
        else:
            selected_trials['trial_index'] = trial_indices[trial_ids]

        # Iterate over each unique trial index
        for trial_index in selected_trials['trial_index'].unique():
            index_trials = selected_trials[selected_trials['trial_index'] == trial_index]
            
            neural_means = []
            pupil_means = []
            pupil_means_z = []
            velocity_means = []
            neural_all_means = [] 
            engagement_means = []
            
            # Iterate over each selected trial for this index
            for _, trial_data in index_trials.iterrows():
                trial_id = trial_data['trial']
                
                # Use provided usable_frames or default to all frames
                trial_neural = trial_data['neural_data']
                trial_pupil = trial_data['pupil_data']
                trial_velocity = trial_data['velocity_data']
                
                if usable_frames is not None:
                    trial_neural = trial_neural[:, usable_frames]
                    trial_pupil = trial_pupil.iloc[usable_frames]
                    trial_velocity = trial_velocity.iloc[usable_frames]
                
                # Subset neural data by neuron type if specified
                if neuron_type is not None:
                    neuron_indices = neuron_group[neuron_type][0]
                    trial_neural_mean = np.nanmean(trial_neural[neuron_indices, :],axis=(0, 1)) #will give one number per neuron
                    trial_neural_all_mean = np.nanmean(trial_neural[neuron_indices, :],axis=1) #gives mean value for each neuron
                else:
                    trial_neural_mean = np.nanmean(trial_neural,axis=(0, 1))  # axis = 1 gives mean value for each neuron
                    trial_neural_all_mean = np.nanmean(trial_neural,axis=1)
                
                # Calculate means for neural and pupil data
                neural_means.append(trial_neural_mean)
                pupil_means.append(np.nanmean(trial_pupil['Normalized_Area'])) #max normalized
                pupil_means_z.append(np.nanmean(trial_pupil['Normalized_Area_Z'])) #max normalized
                velocity_means.append(np.nanmean(trial_velocity)) #nan mean because could contain nans?
                neural_all_means.append(trial_neural_all_mean)

                # For engagement_proj, just append the value (already a mean)
                engagement_means.append(trial_data['engagement_proj'])
            
            # Store means for this trial index
            trial_means_neural[trial_index] = pd.Series(neural_means, index=index_trials['trial'])
            trial_means_pupil[trial_index] = pd.Series(pupil_means, index=index_trials['trial'])
            trial_means_pupil_z[trial_index] = pd.Series(pupil_means_z, index=index_trials['trial'])
            trial_means_velocity[trial_index] = pd.Series(velocity_means, index=index_trials['trial'])
            trial_means_neural_all[trial_index] = pd.DataFrame(neural_all_means, index=index_trials['trial']) #2 dimensional 
            trials_means_engagement[trial_index] = pd.Series(engagement_means, index=index_trials['trial']) #2 dimensional 
        
        return trial_means_neural, trial_means_pupil, trial_means_pupil_z, trial_means_velocity,trial_means_neural_all, trials_means_engagement

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


    # def calculate_trial_means(neural_data, pupil_data , trial_ids=None, trial_indices=None, usable_frames=None, neuron_type=None, celltype_array=None, good_trials=None):
    #     """
    #     Calculates the mean neural and pupil data for each trial based on provided trial ids,
    #     trial indices, and frames, optionally for a specific neuron type.
        
    #     Args:
    #         neural_data (dict): Dictionary where keys are trial IDs and values are neural data arrays for each trial.
    #         pupil_data (pd.DataFrame): DataFrame containing pupil data for all frames.
    #         trial_ids (list of int, optional): List of trial IDs to calculate means for.
    #                                         If None, uses all good trials.
    #         trial_indices (list of int, optional): List of trial indices (0 to 2) to filter trials by their index.
    #                                             If None, uses all trials determined from trial_ids.
    #         usable_frames (list of int, optional): List of frames to use for calculations.
    #                                             If None, uses all frames for each trial.
    #         neuron_type (int, optional): Specify a neuron type to subset (0 = pyr, 1 = som, 2 = pv). 
    #                                     Defaults to None for all neurons.
    #         celltype_array (np.ndarray, optional): Array specifying neuron types for each neuron.
    #                                             Required if neuron_type is specified.
    #         good_trials (list of int, optional): List of good trials to use if trial_ids is None.
        
    #     Returns:
    #         tuple: Trial means for neural data and pupil data.
    #     """
    #     trial_means_neural = []
    #     trial_means_pupil = []
        
    #     # If no trial_ids are provided, use all good trials
    #     if trial_ids is None:
    #         trial_ids = np.arange(len(neural_data)) # contains only the good_trials
        
    #     # If trial_indices are provided, filter the trial_ids accordingly
    #     if trial_indices is not None:
    #         trial_ids = [trial_id for trial_id in trial_ids if trial_id in good_trials]

    #     # Use the provided usable frames or default to all frames in the neural data for the first trial
    #     if usable_frames is None:
    #         trial_frames = np.arange(np.shape(neural_data[0])[1])  # Assuming frames are along axis 1
    #     else:
    #         trial_frames = usable_frames  # Use specified usable frames

    #     # Iterate over each trial ID
    #     for trial_id in trial_ids:
    #         # Calculate mean pupil data for this trial (ensure the frame count is aligned)
    #         trial_pupil = pupil_data.iloc[trial_frames].mean(axis=0)
    #         trial_means_pupil.append(trial_pupil)

    #         # Subset neural data by neuron type if specified
    #         if neuron_type is not None and celltype_array is not None:
    #             neuron_indices = np.where(celltype_array == neuron_type)[0]
    #             # Average across the selected neurons for the trial frames
    #             trial_neural = neural_data[trial_id][neuron_indices, :][:, trial_frames].mean(axis=1)
    #         else:
    #             # Use all neurons
    #             trial_neural = neural_data[trial_id][:, trial_frames].mean(axis=1)  # Average across neurons
            
    #         trial_means_neural.append(trial_neural)

    #     return np.array(trial_means_neural), pd.DataFrame(trial_means_pupil)










