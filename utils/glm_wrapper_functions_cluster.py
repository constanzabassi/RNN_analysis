import numpy as np
import os
import scipy.io
from itertools import accumulate
from collections import Counter
from scipy import stats


class glm_wrapper_functions_cluster:

    def get_glm_predictors_data_from_each_CV_fold(
        self,
        train_directory,
        test_directory,
        fold,
        raw_deconvolved=0,
        ctx_value=1   # 1 = active, 0 = passive
    ):
        """
        Load GLM predictors and neural responses, then trialize for RNN use.
        Returns trial-resolved predictors and responses.
        """

        # =====================
        # LOAD TRAIN DATA
        # =====================
        directory_train = train_directory.format(fold)
        os.chdir(directory_train)

        # Behavioral predictors
        behav = scipy.io.loadmat('behav_big_matrix.mat')
        X_train = behav['behav_big_matrix']

        behav_ids = scipy.io.loadmat('behav_big_matrix_ids.mat')

        # Neural responses
        response = scipy.io.loadmat('combined_response.mat')
        Y_train = response['combined_response']
        if raw_deconvolved == 0:
            Y_train[Y_train > 0.05] = 1

        # Frame indices for trial segmentation
        frames = scipy.io.loadmat('combined_frames_included.mat')
        frames = frames['combined_frames_included'][0]

        trial_starts = self.get_trial_frames_from_combined_frames(frames)

        # Transpose to [time, features]
        X_train = X_train.T
        Y_train = Y_train.T

        # Normalize predictors (same as GLM)
        X_train = self.safe_zscore(X_train)

        # Split into variable-length trials
        X_train_trials, Y_train_trials = self.split_into_trials(
            X_train, Y_train, trial_starts
        )

        # Add context as last predictor
        X_train_trials = self.add_context(X_train_trials, ctx_value)

        # NEW: extract correct / incorrect labels
        if ctx_value == 1:
            # only for active trials
            correct_train = self.get_correct_from_behav(
                behav['behav_big_matrix'].T,   # IMPORTANT: same time axis as X
                trial_starts
            )
        else:
            correct_train = None
            print(correct_train)

        # =====================
        # BEHAVIOR IDS (unchanged)
        # =====================
        behav_IDS = []
        for trial in range(behav_ids['behav_big_matrix_ids'][0].shape[0]):
            behav_IDS.append(behav_ids['behav_big_matrix_ids'][0][trial][0])

        counter = Counter(behav_IDS)
        count_of_values = list(counter.values())

        IDS_index = np.array(list(accumulate(count_of_values))) - 1
        IDS_for_count_of_values = [
            behav_IDS[i] for i in IDS_index[:np.sum(count_of_values)]
        ]

        # =====================
        # LOAD TEST DATA
        # =====================
        directory_test = test_directory.format(fold)
        os.chdir(directory_test)

        behav = scipy.io.loadmat('behav_big_matrix.mat')
        X_test = behav['behav_big_matrix']

        response = scipy.io.loadmat('combined_response.mat')
        Y_test = response['combined_response']
        if raw_deconvolved == 0:
            Y_test[Y_test > 0.05] = 1

        frames = scipy.io.loadmat('combined_frames_included.mat')
        frames = frames['combined_frames_included'][0]

        trial_starts_test = self.get_trial_frames_from_combined_frames(frames)

        X_test = X_test.T
        Y_test = Y_test.T
        X_test = self.safe_zscore(X_test)

        X_test_trials, Y_test_trials = self.split_into_trials(
            X_test, Y_test, trial_starts_test
        )

        X_test_trials = self.add_context(X_test_trials, ctx_value)

        if ctx_value == 1:
            # only for active trials
            correct_test = self.get_correct_from_behav(
                behav['behav_big_matrix'].T,   # IMPORTANT: same time axis as X
                trial_starts_test
            )
        else:
            correct_test = None


        return (
            X_train_trials,
            Y_train_trials,
            X_test_trials,
            Y_test_trials,
            correct_train,
            correct_test,
            count_of_values,
            IDS_for_count_of_values
        )

    # =====================
    # HELPERS
    # =====================

    def safe_zscore(self, X):
        z_scored = np.zeros_like(X)
        stds = np.std(X, axis=0)
        non_zero = stds != 0
        z_scored[:, non_zero] = stats.zscore(X[:, non_zero], axis=0)
        return z_scored

    def get_trial_frames_from_combined_frames(self, combined_frames_included):
        """
        Identify trial start indices from discontinuities in frame indices.
        """
        diffs = np.diff(combined_frames_included)
        trial_starts = np.where(diffs > 1)[0] + 1
        return np.concatenate(([0], trial_starts))
    
    def get_correct_from_behav(self, behav_matrix, trial_starts):
        """
        Extract correct/incorrect trial information from behavior matrix.
        Assumes correct/incorrect info is in a specific column.
        """
        
        correct_col_index = 122 #first reward predictor
        correct_array = np.where(behav_matrix[:, correct_col_index] > 0) #look at reward predictor greater than 0
        correct_trials = []
        for i in range(len(trial_starts)):
            start = trial_starts[i]
            end = trial_starts[i + 1] if i < len(trial_starts) - 1 else behav_matrix.shape[0]
            trial_correct = np.any(np.isin(np.arange(start, end), correct_array))
            correct_trials.append(trial_correct)
        return correct_trials

    def split_into_trials(self, X, Y, trial_starts):
        X_trials, Y_trials = [], []
        for i in range(len(trial_starts)):
            start = trial_starts[i]
            end = trial_starts[i + 1] if i < len(trial_starts) - 1 else X.shape[0]
            X_trials.append(X[start:end])
            Y_trials.append(Y[start:end])
        return X_trials, Y_trials

    def add_context(self, X_trials, ctx_value):
        return [
            np.concatenate(
                [X, ctx_value * np.ones((X.shape[0], 1))],
                axis=1
            )
            for X in X_trials
        ]
    
    def load_celltypes(self,server,animalID,date):
        base_path = f"{server}/Connie/ProcessedData/{animalID}/{date}/"


        path = os.path.join(base_path, 'red_variables/')
        pyr_str = scipy.io.loadmat(path+'pyr_cells.mat')
        pyr = pyr_str['pyr_cells']-1 #convert to python indices
        pyr = np.transpose(pyr)

        som_str = scipy.io.loadmat(path+'mcherry_cells.mat')
        som = som_str['mcherry_cells']-1

        pv_str = scipy.io.loadmat(path+'tdtom_cells.mat')
        pv = pv_str['tdtom_cells']-1


        neuron_groups = {
            'pyr': pyr,
            'som': som,
            'pv': pv}

        # Define colors for each group
        colors = {'pyr': (0.37, 0.75, 0.49),   # pyr = 0
                'som': (0.17, 0.35, 0.8),    # som = 1
                'pv': (0.82, 0.04, 0.04)}    # pv = 2

        #combine celltypes into different indices
        celltype_array = np.zeros(np.shape(np.concatenate((pyr,som,pv)))[0])
        celltype_array[som] = 1
        celltype_array[pv] = 2

        return celltype_array, neuron_groups, colors

# import numpy as np
# import os 
# import tensorflow as tf
# import scipy.io
# import time
# from itertools import accumulate
# from collections import Counter
# from scipy import stats


# class glm_wrapper_functions_cluster:

#     def get_glm_predictors_data_from_each_CV_fold(self,train_directory, test_directory, fold, raw_deconvolved = 0, ctx = 0):
#         '''
#             function to locate and arrange the train and test datasets prior to training the GLM encoding model 
            
#             inputs:
#             train_directory: location of the training datasets 
#             test_directory: location of the testing datasets 
#             fold: cross validation fold for the specified dataset 
#             raw_deconvolved: whether to binarize the deconvolved neural data (0 = binarize, 1 = keep raw deconvolved values)
            
#             outputs:
#             X_train, Y_train: Variables + neural data for training set 
#             X_test, Y_test: Variables + neural data for testing set 
#             count_of_values: # and position of unique variables for fitting 
#             IDS_for_count_of_values: names of the unique variables for fitting - helps for B-weight analysis 
#         '''
#         directory_train = train_directory.format(fold)
#         os.chdir(directory_train)

#         #LOAD BEHAVIOR MATRIX
#         behav = scipy.io.loadmat('behav_big_matrix.mat')
#         behav_ids = scipy.io.loadmat('behav_big_matrix_ids.mat')

#         behav_matrix = behav['behav_big_matrix']
#         behav_ids_matrix = behav_ids['behav_big_matrix_ids'][0]

#         #LOAD NEURAL RESPONSES
#         response = scipy.io.loadmat('combined_response.mat')
#         response_matrix = response['combined_response']
#         if raw_deconvolved == 0:
#             response_matrix[response_matrix > 0.05] = 1

#         #get trial ends
#         combined_frames_included = scipy.io.loadmat('combined_frames_included.mat')
#         combined_frames_included = combined_frames_included['combined_frames_included'][0]
        
#         #trialize data
#         frame_relative_to_all = self.get_trial_frames(combined_frames_included)
#         T = frame_relative_to_all.shape[0]

#         X_train = behav_matrix  # +1 bc of MATLAB indices
#         ctx = np.ones((T, 1)) # active = 1# later: zeros for passive
#         X_trial = np.concatenate([X_trial, ctx], axis=1)

#         Y_train = response_matrix
        
#         behav_IDS = []
#         for trial in list(range(behav_ids['behav_big_matrix_ids'][0].shape[0])):
#             behav_IDS.append(behav_ids['behav_big_matrix_ids'][0][trial][0])
        

#         # Count the occurrences of each element in the list
#         counter = Counter(behav_IDS)

#         # Get the unique values
#         unique_values = list(counter.keys())

#         # Get the count of each unique value
#         count_of_values = list(counter.values())
#         #print(str(count_of_values) + '- count_of_values (counts each unique feature)')

#         #print(np.sum(count_of_values))

#         IDS_index = np.array(list(accumulate(count_of_values)))-1
#         IDS_for_count_of_values = []
#         for index in IDS_index[0:np.sum(count_of_values)]: #take this out to instead look at the size of the array
#             IDS_for_count_of_values.append(behav_IDS[index])

#         #print(IDS_for_count_of_values)
        
#         #LOAD TESTING DATA!
#         directory_test = test_directory.format(fold)
#         os.chdir(directory_test)

#         #LOAD BEHAVIOR MATRIX
#         behav = scipy.io.loadmat('behav_big_matrix.mat')
#         behav_ids = scipy.io.loadmat('behav_big_matrix_ids.mat')

#         behav_matrix = behav['behav_big_matrix']
#         behav_ids_matrix = behav_ids['behav_big_matrix_ids'][0]

#         response = scipy.io.loadmat('combined_response.mat')
#         response_matrix = response['combined_response']
#         if raw_deconvolved == 0:
#             response_matrix[response_matrix > 0.05] = 1

#         #get trial ends
#         combined_frames_included = scipy.io.loadmat('combined_frames_included.mat')
#         combined_frames_included = combined_frames_included['combined_frames_included'][0]

#         #INDEX INTO CURRENT NEURON!
#         X_test = behav_matrix  # +1 bc of MATLAB indices
#         Y_test = response_matrix
        
#         # Clean up design matrix and z-score along sample dimension
#         X_train = X_train.T
        
#         # Multiply deconvolved activity by 10 to mimic spike number
#         Y_train = 1 * Y_train.T


#         X_test = X_test.T
#         # Multiply deconvolved activity by 10 to mimic spike number
#         Y_test = 1 * Y_test.T

#         #Normalize predictors - zscore each predictor column
#         X_train = self.safe_zscore(X_train)
#         X_test = self.safe_zscore(X_test)
        
#         return X_train, Y_train, X_test, Y_test, count_of_values, IDS_for_count_of_values
    
#     # Replace zscore line with:
#     def safe_zscore(X):
#         """Z-score predictors while preserving zero columns."""
#         means = np.mean(X, axis=0)
#         stds = np.std(X, axis=0)
        
#         # Create output array
#         z_scored = np.zeros_like(X)
        
#         # Only z-score columns with non-zero std
#         non_zero_std = stds != 0
#         if np.any(non_zero_std):
#             z_scored[:, non_zero_std] = stats.zscore(X[:, non_zero_std], axis=0)
        
#         return z_scored
    
#     def get_trial_frames_from_combined_frames(self, combined_frames_included):
#         """
#         Assume combined_frames_included is a 1D array where values jump at trial boundaries
#         """
#         included = combined_frames_included
#         diffs = np.diff(included)

#         trial_starts = np.where(diffs > 1)[0] + 1
#         relative_trial_starts = np.concatenate(([0], trial_starts))

#         return relative_trial_starts



#         return frame_relative_to_all
#     def split_into_trials(self,X, Y, trial_starts):
#         X_trials, Y_trials = [], []
#         for i in range(len(trial_starts)):
#             start = trial_starts[i]
#             end = trial_starts[i+1] if i < len(trial_starts)-1 else X.shape[0]
#             X_trials.append(X[start:end])
#             Y_trials.append(Y[start:end])

#         return X_trials, Y_trials
    
#     def add_context(self,X_trials, ctx_value):
#         return [
#             np.concatenate([X, ctx_value * np.ones((X.shape[0], 1))], axis=1)
#             for X in X_trials
#             ]