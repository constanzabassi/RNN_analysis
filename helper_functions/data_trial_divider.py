import numpy as np
from typing import List, Tuple, Dict, Any
import pandas as pd

class TrialDivider:
    # def divide_trials_updated(self, imaging: Dict, fields_to_separate: List[str]) -> Tuple[List, np.ndarray]:
    #     """
    #     Divide trials based on specified fields and create condition arrays.
        
    #     Parameters:
    #     -----------
    #     imaging : dict
    #         Dictionary containing imaging data with trial information
    #     fields_to_separate : list
    #         List of field names to use for separation
            
    #     Returns:
    #     --------
    #     tuple
    #         (all_conditions, condition_array)
    #     """
    #     # Find empty trials
    #     empty_trials = [i for i, trial in enumerate(imaging) 
    #                    if not trial.get('good_trial')]
    #     good_trials = list(set(range(len(imaging))) - set(empty_trials))
        
    #     condition_array = []
    #     for t in good_trials:
    #         trial_conditions = [t]  # First column is good trial index
            
    #         for field in fields_to_separate:
    #             value = imaging[t]['virmen_trial_info'][field]
    #             # Binarize stimuli for condition field
    #             if field == 'condition':
    #                 value = value % 2
    #             trial_conditions.append(value)
                
    #         condition_array.append(trial_conditions)
        
    #     condition_array = np.array(condition_array)
        
    #     # Generate all possible combinations
    #     num_conditions = condition_array.shape[1] - 1
    #     all_combinations = np.array([list(map(int, format(i, f'0{num_conditions}b')))
    #                                for i in range(2**num_conditions)])
        
    #     # Initialize all_conditions list
    #     all_conditions = []
        
    #     # Find trials for each combination
    #     for comb in all_combinations:
    #         # Check which trials match all conditions
    #         matching = (condition_array[:, 1:] == comb).all(axis=1)
    #         matching_trials = np.where(matching)[0]
            
    #         # Get labels for current combination
    #         labels = self.get_condition_labels_updated(comb, fields_to_separate)
            
    #         all_conditions.append([matching_trials, [comb], labels])
            
    #     return all_conditions, condition_array

    # def get_condition_labels_updated(self, condition_values: np.ndarray, 
    #                                fields_to_separate: List[str]) -> str:
    #     """Generate labels for condition combinations."""
    #     labels = []
        
    #     for value, field in zip(condition_values, fields_to_separate):
    #         if field == 'correct':
    #             labels.append('Correct' if value == 1 else 'Incorrect')
    #         elif field == 'left_turn':
    #             labels.append('Left Turn' if value == 1 else 'Right Turn')
    #         elif field == 'condition':
    #             labels.append('Left' if value == 1 else 'Right')
    #         elif field == 'is_stim_trial':
    #             labels.append('Stim' if value == 1 else 'Control')
                
    #     return '/'.join(labels)
    

    # def divide_trials_from_df(self, trial_info_df: pd.DataFrame, 
    #                      good_trials: List[int],
    #                      fields_to_separate: List[str]) -> Tuple[List, np.ndarray]:
    #     """
    #     Divide trials based on specified fields using a pandas DataFrame, only for good trials.
        
    #     Parameters:
    #     -----------
    #     trial_info_df : pd.DataFrame
    #         DataFrame containing trial information
    #     good_trials : list
    #         List of good trial indices to include in the analysis
    #     fields_to_separate : list
    #         List of column names to use for separation
        
    #     Returns:
    #     --------
    #     tuple
    #         (all_conditions, condition_array)
    #     """
    #     # Filter DataFrame to only include good trials
    #     good_trials_df = trial_info_df.iloc[good_trials].copy()
        
    #     # Convert specified columns to binary values
    #     condition_df = good_trials_df[fields_to_separate].copy()
    #     if 'condition' in fields_to_separate:
    #         condition_df['condition'] = condition_df['condition'] % 2
            
    #     # Create condition array with trial indices
    #     condition_array = np.column_stack([
    #         good_trials_df.index.values,  # Original trial indices
    #         condition_df.values
    #     ])
        
    #     # Rest of the function remains the same...
    #     num_conditions = len(fields_to_separate)
    #     all_combinations = np.array([
    #         list(map(int, format(i, f'0{num_conditions}b')))
    #         for i in range(2**num_conditions)
    #     ])
        
    #     all_conditions = []
    #     for comb in all_combinations:
    #         # Check which trials match all conditions
    #         matching = (condition_df.values == comb).all(axis=1)
    #         matching_trials = good_trials_df.index.values[matching]
            
    #         labels = self.get_condition_labels_updated(comb, fields_to_separate)
    #         all_conditions.append([matching_trials, [comb], labels])
        
    #     return all_conditions, condition_array

    def divide_trials_from_df(self, trial_info_df: pd.DataFrame, 
                         good_trials: List[int],
                         fields_to_separate: List[str]) -> Tuple[List, np.ndarray]:
        """
        Divide trials based on specified fields, returning indices relative to good trials list.
        
        Parameters:
        -----------
        trial_info_df : pd.DataFrame
            DataFrame containing trial information
        good_trials : list
            List of good trial indices to include in the analysis
        fields_to_separate : list
            List of column names to use for separation
        
        Returns:
        --------
        tuple
            (all_conditions, condition_array) where indices are relative to good_trials
        """
        # Filter DataFrame to only include good trials
        good_trials_df = trial_info_df.iloc[good_trials].copy()
        
        # Convert specified columns to binary values
        condition_df = good_trials_df[fields_to_separate].copy()
        if 'condition' in fields_to_separate:
            condition_df['condition'] = condition_df['condition'] % 2
            
        # Create condition array with relative indices
        condition_array = np.column_stack([
            np.arange(len(good_trials)),  # Relative indices
            condition_df.values
        ])
        
        # Generate all possible combinations
        num_conditions = len(fields_to_separate)
        all_combinations = np.array([
            list(map(int, format(i, f'0{num_conditions}b')))
            for i in range(2**num_conditions)
        ])
        
        all_conditions = []
        for comb in all_combinations:
            # Check which trials match all conditions
            matching = (condition_df.values == comb).all(axis=1)
            matching_trials = np.where(matching)[0]  # Relative indices
            
            labels = self.get_condition_labels_updated(comb, fields_to_separate)
            all_conditions.append([matching_trials, [comb], labels])
        
        return all_conditions, condition_array

    def get_condition_labels_updated(self, condition_values: np.ndarray, 
                                   fields_to_separate: List[str]) -> str:
        """Generate labels for condition combinations."""
        label_map = {
            'correct': {1: 'Correct', 0: 'Incorrect'},
            'left_turn': {1: 'Left Turn', 0: 'Right Turn'},
            'condition': {1: 'Left', 0: 'Right'},
            'is_stim_trial': {1: 'Stim', 0: 'Control'}
        }
        
        labels = [
            label_map[field][value] 
            for field, value in zip(fields_to_separate, condition_values)
        ]
        
        return '/'.join(labels)