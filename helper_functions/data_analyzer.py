import pandas as pd


class DataAnalyzer:
    def __init__(self, neural_means=None, velocity_means=None):
        """
        Initialize with trial-averaged neural and velocity data.
        """
        self.neural_means = neural_means
        self.velocity_means = velocity_means

    def aggregate_data_by_state(self, data_list, num_states, min_trials=0):
        """
        Aggregate per-state data across multiple datasets.
        """
        aggregated_data = [[] for _ in range(num_states)]
        aggregated_indices = [[] for _ in range(num_states)]

        for dataset in range(len(data_list)):
            for state in range(num_states):
                try:
                    state_data = data_list[dataset].get(state)
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

