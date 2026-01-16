import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import permutation_test, bootstrap
from scipy import stats
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Any, Optional
from helper_functions.data_aligner import DataAligner
from helper_functions.data_plotter import Plotter

class DataAnalyzer:
    def __init__(self, neural_means= None, pupil_means =None, latent_states = None):
        """
        Initializes the DataAnalyzer with trial-averaged neural data, pupil data, and latent states.
        
        Args:
            neural_means (np.ndarray): Neural data averaged across trials.
            pupil_means (pd.DataFrame): Pupil data averaged across trials.
            latent_states (pd.DataFrame): Latent behavioral states for each trial.
        """
        self.neural_means = neural_means
        self.pupil_means = pupil_means
        self.latent_states = latent_states
        # self.data_aligner = DataAligner()
        # self.plotter = Plotter()
    
    # def correlate_data(self):
    #     """
    #     Relates neural and pupil data to latent states using linear regression.
        
    #     Returns:
    #         tuple: Fitted model for neural data, fitted model for pupil data.
    #     """
    #     # Regress latent states onto neural data
    #     neural_model = LinearRegression().fit(self.latent_states, self.neural_means)
        
    #     # Regress latent states onto pupil data
    #     pupil_model = LinearRegression().fit(self.latent_states, self.pupil_means)
        
    #     return neural_model, pupil_model

    def correlate_data(self,trial_info_normalized):
        """
        Relates neural and pupil data to latent states using linear regression for each state.
        
        Returns:
            tuple: Dictionaries of fitted models for neural and pupil data for each state.
        """
        state_models = {}

        # Loop over each state
        for state in np.unique(trial_info_normalized['state']):
            # Filter the trials corresponding to the current state
            trials_in_state = trial_info_normalized[trial_info_normalized['state'] == state]

            # Ensure there are trials for this state
            if trials_in_state.empty:
                continue

            # Get neural and pupil data for the relevant trials
            neural_data =self.neural_means[state] #[trials_in_state.index]
            pupil_data = self.pupil_means[state] #[trials_in_state.index]

            # Ensure data is in the correct shape for linear regression
            neural_data_reshaped = neural_data.values.reshape(-1, 1)  # Reshape to 2D array for features
            pupil_data_reshaped = pupil_data.values.reshape(-1, 1)    # Reshape to 2D array for features


            # Fit linear regression models for each state
            state_model = LinearRegression().fit(pupil_data_reshaped, neural_data_reshaped)

            # Store the models for each state
            state_models[state] = state_model

        return state_models
    
    def plot_data(self):
        """
        Plots neural data, pupil data, and latent states to visualize relationships.
        """
        # Plot latent states vs neural data
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.neural_means, cmap='viridis', cbar=True)
        plt.title('Neural Data Across Latent States')
        plt.xlabel('Neurons')
        plt.ylabel('Trials')
        plt.show()
        
        # Plot latent states vs pupil data
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.pupil_means, cmap='coolwarm', cbar=True)
        plt.title('Pupil Data Across Latent States')
        plt.xlabel('Pupil Feature')
        plt.ylabel('Trials')
        plt.show()

    def aggregate_data_by_state(self,data_list, num_states, min_trials=0):
        """
        Aggregates data by state across multiple datasets.
        
        Args:
            data_list (list): List of dictionaries with data (e.g., neural, pupil, or velocity) by state.
            num_states (int): Number of unique states.
            min_trials (int, optional): Minimum number of trials needed for a state to be included in aggregation. Defaults to 0.
        
        Returns:
            aggregated_data (list): List of lists containing aggregated data by state.
            aggregated_indices (list): List of lists with dataset indices corresponding to each data point.
        """
        aggregated_data = [[] for _ in range(num_states)]
        aggregated_indices = [[] for _ in range(num_states)]
        
        for dataset in range(len(data_list)):
            for state in range(num_states):
                try:
                    # Retrieve data for the current state in the current dataset
                    state_data = data_list[dataset].get(state)
                    
                    # Check if the state data is valid and the number of trials meets the minimum requirement
                    if isinstance(state_data, pd.Series) and len(state_data) >= min_trials:
                        aggregated_data[state].extend(state_data.tolist())
                        aggregated_indices[state].extend([dataset] * len(state_data))
                    elif isinstance(state_data, pd.Series):
                        print(f"Dataset {dataset} state {state} has insufficient trials. Skipping state.")
                        
                except KeyError:
                    print(f"Dataset {dataset} does not have data for state {state}. Skipping.")
                except Exception as e:
                    print(f"An error occurred for dataset {dataset}, state {state}: {e}")
        
        return aggregated_data, aggregated_indices
    
    def analyze_correct_incorrect(self,
        data_loaders: List[Any],
        celltype_info: Dict[Tuple[str, str], Dict[str, Any]],
        engagement_proj: Dict[Tuple[str, str], Any],
        engagement_test_trials: Dict[Tuple[str, str], Any],
        *,
        # NEW: generalize which trial_info field is used to index/group trials
        trial_index_field: str = "correct",
        # NEW: allow custom colors/labels; defaults match your original
        state_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
        state_labels: Optional[Dict[int, str]] = None,
        # existing knobs
        align_field: str = "iti",
        last_n_frames: int = 10,
        velocity_threshold: float = 10,
        trial_engagement_add: float = 1,
        make_plots: bool = True,
        pupil_norm_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Aligns per-dataset signals, computes trial-wise means split by a chosen trial_info field,
        plots summaries (optional), and returns structured results.
        trial_index_field:
            Name of the column in trial_info to group by (e.g., 'correct', 'left_turn',
            'condition', 'is_stim_trial', 'state'). Values will be factorized to 0..K-1.
        state_colors / state_labels:
            Optional mappings keyed by state id (0..K-1). If omitted and the field is binary
            with values {0,1} or {False,True}, defaults to:
                labels: {0: 'Incorrect', 1: 'Correct'}
                colors: {1: (56/255,102/255,65/255), 0: (1,0,110/255)}
            Otherwise, simple auto labels are generated and colors must be provided if you
            want specific ones.
        """
        # ---- containers ----
        aligned_data: List[Any] = []
        good_trials_all: List[Any] = []
        trial_means_neural_list: List[Any] = []
        trial_means_neural_all_list: List[Any] = []
        trial_means_pupil_list: List[Any] = []
        trial_means_pupil_z_list: List[Any] = []
        trial_means_velocity_list: List[Any] = []
        trial_means_engagement_list: List[Any] = []
        all_state_percentages: Dict[str, Any] = {}
        state_data_all: Dict[str, Dict[str, Any]] = {}
        plotter_objects: List[Any] = []
        analyzer_objects: List[Any] = []
        # original latent-state fields to save if present
        dict_fields_state = ['Correctness','Correct', 'Incorrect']
        # default celltype palette (unchanged)
        celltype_palette = {
            'pyr': (0.37, 0.75, 0.49),
            'som': (0.17, 0.35, 0.8),
            'pv' : (0.82, 0.04, 0.04),
        }
        for (data_loader, (key, info)) in zip(data_loaders, celltype_info.items()):
            animalID, date = key
            print(f"Processing: {key}")
            # --- load data ---
            pupil_data = data_loader.load_pupil_data()
            neural_data, good_trials, trial_info, movement_in_imaging, frame_id_events, file_num, velocity_data, imaging_data = \
                data_loader.load_neural_data(neural_data_type='dff')
            good_trials_all.append(good_trials)
            latent_states = data_loader.load_latent_states()
            imaging_frame_lengths = data_loader.load_alignment_data()
            global_frame_ids = data_loader.align_frames_to_session(file_num, frame_id_events, imaging_frame_lengths)
            # movement frames
            movement_frames = {}
            for trial in range(len(neural_data)):
                if trial in good_trials:
                    movement_frames[trial] = {
                        'maze_frames': movement_in_imaging[trial].get('maze_frames'),
                        'reward_frames': movement_in_imaging[trial].get('reward_frames'),
                        'iti_frames': movement_in_imaging[trial].get('iti_frames')
                    }
                else:
                    movement_frames[trial] = None
            # --- align ---
            aligner = DataAligner(
                neural_data, movement_frames, velocity_data, global_frame_ids, good_trials,
                pupil_data, engagement_proj=engagement_proj[key],
                engagement_proj_trials=engagement_test_trials[key]
            )
            align_kwargs = dict(
                field=align_field,
                last_n_frames=last_n_frames,
                velocity_threshold=velocity_threshold,
                trial_engagement_add=trial_engagement_add,
            )
            if pupil_norm_mode is not None:
                align_kwargs["normalize_mode"] = pupil_norm_mode
            aligned_data_single_dataset = aligner.align_data(**align_kwargs)
            aligned_data.append(aligned_data_single_dataset)
            # --- build state indices from trial_info[trial_index_field] ---
            if trial_index_field not in trial_info:
                raise KeyError(f"`trial_info` is missing '{trial_index_field}' for dataset {animalID}, {date}")
            series = trial_info[trial_index_field]
     
            # factorize to contiguous 0..K-1 ids while preserving value order (sorted unique)
            codes, uniques = pd.factorize(series, sort=True)
            state_ids = codes.astype(int)
            n_states = int(len(uniques))
            # default labels/colors (match your original when it's binary correctness)
            if state_labels is None:
                # if binary and looks like correctness, use your labels
                if n_states == 2 and set(map(int, pd.Series(uniques).astype(int, errors='ignore'))) == {0,1} and trial_index_field == "correct":
                    my_state_labels = {0: "Incorrect", 1: "Correct"}
                else:
                    # generic: e.g., "condition=A", "condition=B"
                    my_state_labels = {i: f"{trial_index_field}={uniques[i]}" for i in range(n_states)}
            else:
                my_state_labels = state_labels
            if state_colors is None:
                # if binary correctness, use your exact original colors
                if n_states == 2 and set(map(int, pd.Series(uniques).astype(int, errors='ignore'))) == {0,1} and trial_index_field == "correct":
                    my_state_colors = {
                        1: (56/255, 102/255, 65/255),  # Correct
                        0: (255/255, 0/255, 110/255),  # Incorrect
                    }
                else:
                    # no automatic palette here to keep “specific colors” contract;
                    # require user to pass explicit colors when not the default two-state case
                    raise ValueError(
                        "Please provide `state_colors` for non-binary or non-`correct` groupings "
                        f"(found {n_states} states for '{trial_index_field}')."
                    )
            else:
                my_state_colors = state_colors
            # --- compute trial means grouped by state_ids ---
            (trial_means_neural,
            trial_means_pupil,
            trial_means_pupil_z,
            trial_means_velocity,
            trial_means_neural_all,
            trial_means_engagement) = aligner.calculate_trial_means(
                aligned_data_single_dataset,
                trial_indices=state_ids
            )
            trial_means_neural_list.append(trial_means_neural)
            trial_means_neural_all_list.append(trial_means_neural_all)
            trial_means_pupil_list.append(trial_means_pupil)
            trial_means_pupil_z_list.append(trial_means_pupil_z)
            trial_means_velocity_list.append(trial_means_velocity)
            trial_means_engagement_list.append(trial_means_engagement)
            # --- plotting ---
            the_plotter = None
            if make_plots:
                the_plotter = Plotter(
                    state_colors=my_state_colors,
                    state_labels=my_state_labels,
                    celltype_colors=celltype_palette
                )
                the_plotter.plot_box(
                    data=trial_means_neural, measure_label="Mean Activity",
                    y_lim=(0, 0.3), figsize=(3, 3)
                )
                the_plotter.plot_box(
                    data=trial_means_pupil, measure_label="Mean Pupil",
                    y_lim=(0, 1), figsize=(3, 3)
                )
                the_plotter.plot_box(
                    data=trial_means_velocity, measure_label="Mean Velocity",
                    y_lim=(0, 1), figsize=(3, 3)
                )
                # keep your original trial-stats plot when using 'correct'
                if trial_index_field == "correct":
                    _, _, state_percentages = the_plotter.plot_trial_statistics_by_state(
                        trial_info, trial_info['correct'], the_plotter, figsize=(9, 3)
                    )
                    all_state_percentages[f"{animalID}_{date}"] = state_percentages
            # --- stash latent state fields if present ---
            state_data_all[f"{animalID}_{date}"] = {}
            for field in dict_fields_state:
                state_data_all[f"{animalID}_{date}"][field] = (
                    latent_states[field].values if (latent_states is not None and field in latent_states) else None
                )
            # --- analysis & correlation plots ---
            analyzer = DataAnalyzer(trial_means_neural, trial_means_pupil, state_ids)
            state_models_fit = analyzer.correlate_data(trial_info)
            if make_plots and the_plotter is not None:
                the_plotter.plot_correlations(
                    trial_info, state_models_fit, trial_means_neural, trial_means_pupil, figsize=(6, 3)
                )
                the_plotter.plot_trial_averaged_data(
                    trial_info, trial_means_pupil, trial_means_neural,
                    title=f"Trail Data {animalID} {date}",
                    correct_trials=(trial_info["correct"] if "correct" in trial_info else None),
                    figsize=(6, 3),
                )
            if make_plots:
                plotter_objects.append(the_plotter)
            analyzer_objects.append(analyzer)
        return {
            "aligned_data": aligned_data,
            "good_trials_all": good_trials_all,
            "trial_means_neural_list": trial_means_neural_list,
            "trial_means_neural_all_list": trial_means_neural_all_list,
            "trial_means_pupil_list": trial_means_pupil_list,
            "trial_means_pupil_z_list": trial_means_pupil_z_list,
            "trial_means_velocity_list": trial_means_velocity_list,
            "trial_means_engagement_list": trial_means_engagement_list,
            "all_state_percentages": all_state_percentages,
            "state_data_all": state_data_all,
            "plotter_objects": plotter_objects if make_plots else None,
            "analyzer_objects": analyzer_objects,
        }

