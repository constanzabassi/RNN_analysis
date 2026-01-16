import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
from helper_functions.general_stats import GeneralStats
import os

class Plotter:
    def __init__(self,state_colors, state_labels, celltype_colors=None):
        self.celltype_colors = celltype_colors or {'all': 'gray'}
        self.state_colors = state_colors
        self.state_labels = state_labels

        self.gen_stats = GeneralStats()  # Instantiate GeneralStats
        
    def plot_bar(self, data, measure_label, cell_type_groups=None, bar_width=0.5, save_path=None, y_lim=(0, 0.5), figsize=(5,4), fontsize=7):
        """
        Creates bar plots with error bars for each state. Optionally separates by cell type if cell_type_groups is provided.
        
        Parameters:
        data: pd.DataFrame
            A DataFrame where rows correspond to trials and columns correspond to states.
        measure_label: str
            The label for the y-axis.
        cell_type_groups: dict, optional
            A dictionary where keys are cell types and values are indices of neurons in those groups.
        bar_width: float
            The width of the bars in the plot.
        save_path: str, optional
            The path to save the plot. If None, the plot is displayed instead of saved.
        y_lim: tuple
            The limits for the y-axis.
        """
        # Set colors if cell types are provided
        colors = self.celltype_colors if cell_type_groups else {'all': 'gray'}
        plt.rcParams.update({'font.size': fontsize, 'font.family': 'arial'})

        if cell_type_groups:
            # Create a figure with subplots for each cell type
            fig, axs = plt.subplots(1, len(cell_type_groups), figsize=(5 * len(cell_type_groups), 4), sharey=True)
        else:
            fig, axs = plt.subplots(1, 1, figsize=figsize)
            axs = [axs]  # Treat single plot as a list for consistent handling

        # Iterate over cell types or single plot case
        for i, (cell_type, indices) in enumerate(cell_type_groups.items() if cell_type_groups else [('all', slice(None))]):
            ax = axs[i]  # Select the subplot
            means = []
            errors = []

            # Calculate mean and error for each state
            for state in data:
                state_data = data[state].values[indices] if cell_type_groups else data[state].values
                means.append(np.nanmean(state_data))
                errors.append(np.nanstd(state_data) / np.sqrt(np.sum(~np.isnan(state_data))))  # Standard error ignoring NaNs

            # Create bar plot
            ax.bar(range(len(data)), means, yerr=errors, capsize=4, color=colors.get(cell_type, 'gray'),
                edgecolor='black', linewidth=1.5, width=bar_width, label=cell_type)

            # Set labels and titles
            ax.set_title(f'{cell_type}' if cell_type_groups else 'All Cells')
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data, rotation=45, ha='right')
            ax.set_ylabel(measure_label)
            ax.set_ylim(y_lim)

            # Clean up appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Add global title
        fig.suptitle(f'{measure_label} Across States' + ('' if cell_type_groups else ' (All Cells)'), fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save or display plot
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_box(self, data, measure_label, cell_type_groups=None, save_path=None, y_lim=None,figsize=(5,4), fontsize=7):
        """
        Creates boxplots for each state. Optionally separates by cell type if cell_type_groups is provided.

        Parameters:
        data: pd.DataFrame
            A DataFrame where rows correspond to trials and columns correspond to states.
        measure_label: str
            The label for the y-axis.
        cell_type_groups: dict, optional
            A dictionary where keys are cell types and values are indices of neurons in those groups.
        save_path: str, optional
            The path to save the plot. If None, the plot is displayed instead of saved.
        y_lim: tuple, optional
            The limits for the y-axis. If None, limits are determined automatically.
        """
        # Set colors if cell types are provided
        colors = self.celltype_colors if cell_type_groups else {'all': 'gray'}
        plt.rcParams.update({'font.size': fontsize, 'font.family': 'arial'})

        if cell_type_groups:
            # Create a figure with subplots for each cell type
            fig, axs = plt.subplots(1, len(cell_type_groups), figsize= (figsize[0] * len(cell_type_groups), figsize[1]), sharey=True)
        else:
            fig, axs = plt.subplots(1, 1, figsize=figsize)
            axs = [axs]  # Treat single plot as a list for consistent handling

        # Iterate over cell types or single plot case
        for i, (cell_type, indices) in enumerate(cell_type_groups.items() if cell_type_groups else [('all', slice(None))]):
            ax = axs[i]  # Select the subplot
            
            # Prepare data for boxplot
            boxplot_data = []
            for state, state_data in data.items():
                # Extract 'Area' data for the specified cell types
                if cell_type_groups:
                    state_data = state_data.values[indices]  # Select Area column and filter by indices
                else:
                    state_data = state_data.values  # Select Area column for all cells
                
                boxplot_data.append(state_data)

            # Create boxplot
            ax.boxplot(boxplot_data, labels=data.keys() if not cell_type_groups else [cell_type] * len(boxplot_data),
                    patch_artist=True, boxprops=dict(facecolor=colors.get(cell_type, 'gray')),
                    medianprops=dict(color='black'), flierprops=dict(marker='o', color='red', alpha=0.5))


            # Set labels and titles
            ax.set_title(f'{cell_type}' if cell_type_groups else 'All Cells')
            ax.set_ylabel(measure_label)
            
            # Set y-axis limits if provided
            if y_lim is not None:
                ax.set_ylim(y_lim)

            # Clean up appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Add global title
        fig.suptitle(f'{measure_label} Across States' + ('' if cell_type_groups else ' (All Cells)'), fontsize=8)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save or display plot
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_trial_statistics_by_state(self,trial_info_normalized, latent_states, plotter, figsize=(15,5), fontsize=7):
        """
        Creates bar plots for percentage of correct responses, left turns, and stimulus trials by state.

        Parameters:
        trial_info_normalized: pd.DataFrame
            DataFrame containing trial information with 'correct', 'left_turn', and 'is_stim_trial' columns.
        latent_states: pd.Series
            Series containing the state for each trial.
        plotter: Plotter
            An instance of the Plotter class to access state colors.

        Returns:
        tuple: (state_highest_left_turn, state_lowest_left_turn)
            State with the highest and lowest percentage of left turns.
        """
        plt.rcParams.update({'font.size': fontsize, 'font.family': 'arial'})
        # Align the latent states with trial_info_normalized
        trial_info_normalized['state'] = latent_states.iloc[trial_info_normalized.index].values

        overall_performance = np.nanmean(trial_info_normalized["correct"]) * 100
        print(f'Overall performance: {overall_performance:.4f}%')

        # Group by state and calculate percentages
        state_percentages = trial_info_normalized.groupby('state').agg(
            percent_correct=('correct', 'mean'),
            percent_left_turn=('left_turn', 'mean'),
            percent_is_stim=('is_stim_trial', 'mean')
        ) * 100  # Convert to percentage

        # Prepare for plotting
        states = state_percentages.index
        bar_width = 0.25  # Width of the bars
        x = np.arange(len(states))

        # Create figure and axes
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

        # Bar plots for each percentage using the defined state colors
        axes[0].bar(x, state_percentages['percent_correct'], width=bar_width, color=[plotter.state_colors[state] for state in states], label='% Correct')
        axes[0].set_title('% Correct')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(states)
        axes[0].set_ylim(0, 100)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        axes[1].bar(x, state_percentages['percent_left_turn'], width=bar_width, color=[plotter.state_colors[state] for state in states], label='% Left Turn')
        axes[1].set_title('% Left Turn')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(states)
        axes[1].set_ylim(0, 100)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        axes[2].bar(x, state_percentages['percent_is_stim'], width=bar_width, color=[plotter.state_colors[state] for state in states], label='% Stimulus Trial')
        axes[2].set_title('% Photostim Trial')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(states)
        axes[2].set_ylim(0, 100)
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)

        # Add labels and legends
        for ax in axes:
            ax.set_ylabel('Percentage (%)')

        plt.tight_layout()
        plt.show()

        # Determine states with highest and lowest percentage of left turns
        optimal_state = state_percentages['percent_correct'].idxmax()
        highest_left_turn_state = state_percentages['percent_left_turn'].idxmax()
        lowest_left_turn_state = state_percentages['percent_left_turn'].idxmin()
        #trial_info_normalized['state'] =

        return highest_left_turn_state, lowest_left_turn_state,state_percentages
    


    def plot_correlations(self, trial_info_normalized, state_models, neural_means, pupil_means, xlabel = 'Pupil Data', ylabel= 'Neural Data',figsize=(15,5), fontsize=7):
        """
        Plots the correlation between neural and pupil data for each state using the fitted models.

        Parameters:
        trial_info_normalized: pd.DataFrame
            DataFrame containing trial information with 'state' column.
        state_models: dict
            Dictionary of fitted LinearRegression models for each state.
        """
        plt.rcParams.update({'font.size': fontsize, 'font.family': 'arial'})
        unique_states = np.unique(trial_info_normalized['state'])
        
        # Create a plot for each state
        plt.figure(figsize=figsize)
        
        for i, state in enumerate(unique_states):
            # Filter the trials corresponding to the current state
            trials_in_state = trial_info_normalized[trial_info_normalized['state'] == state]

            # Get neural and pupil data for the relevant trials
            neural_data = neural_means[state]#[trials_in_state.index]
            pupil_data = pupil_means[state]#[trials_in_state.index]

            # Ensure data is in the correct shape for plotting
            neural_data_reshaped = neural_data.values
            pupil_data_reshaped = pupil_data.values

            # Create scatter plot
            plt.subplot(1, len(unique_states), i + 1)
            plt.scatter(pupil_data_reshaped, neural_data_reshaped, alpha=0.6, label='Data Points')

            # Plot the regression line
            if state in state_models:
                # Get the coefficients
                model = state_models[state]
                x_range = np.linspace(pupil_data_reshaped.min(), pupil_data_reshaped.max(), 100).reshape(-1, 1)
                y_pred = model.predict(x_range)

                plt.plot(x_range, y_pred, color='red', label='Regression Line')

                # Calculate R^2 and p-value
                slope, intercept, r_value, p_value, std_err = stats.linregress(pupil_data_reshaped.flatten(), neural_data_reshaped.flatten())
                r_squared = r_value**2

                # Display R^2 and p-value on the plot
                plt.text(0.05, 0.95, f'$R = {r_value:.2f}$\n$p = {p_value:.3f}$',  #f'$R^2 = {r_squared:.2f}
                        transform=plt.gca().transAxes, fontsize=fontsize-1, verticalalignment='top', 
                        bbox=dict(facecolor='white', alpha=0.5))

            plt.title(f'State {state}')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xlim(pupil_data_reshaped.min() - 0.1, pupil_data_reshaped.max() + 0.1)
            plt.ylim(neural_data_reshaped.min() - 0.1, neural_data_reshaped.max() + 0.1)
            plt.legend()

        plt.tight_layout()
        plt.show()


    def plot_trial_averaged_data(self,trial_info_normalized, pupil_data, neural_data,title = 'Trial Averaged Pupil Area and Neural Data by State',correct_trials = None, marker_y_pos = 0.5, figsize=(10,6), fontsize=7, marker_names = {'pupil': 'o', 'neural': 'x'},save_path=None):
        """
        Plot trial-averaged pupil area and neural data, colored by state.
        
        Parameters:
            trial_info_normalized (DataFrame): Data containing state information.
            pupil_data (Series): Pupil area data.
            neural_data (Series): Neural data.
        """
        plt.rcParams.update({'font.size': fontsize, 'font.family': 'arial'})
        # Create a figure and axis
        plt.figure(figsize=figsize)

        # Define markers and colors based on state
        markers = marker_names  # 'o' for pupil, 'x' for neural
        colors = self.state_colors  # Update with actual colors

        # Calculate trial-averaged pupil area and neural data
        trial_averaged_pupil = pupil_data
        trial_averaged_neural = neural_data

        # Loop through each unique state to plot data
        for state in trial_info_normalized['state'].unique():
            # Get indices for the current state
            state_trials = trial_averaged_pupil[state].index #trial_info_normalized[trial_info_normalized['state'] == state].index this gives all trials including interpolated ones!
            
            # Plot pupil data vs neural data for each trial in the current state
            plt.scatter(state_trials, 
                        trial_averaged_pupil[state], 
                        marker=markers['pupil'], 
                        color=colors[state], 
                        label=f'Pupil Area - {state}' if state == trial_info_normalized['state'].unique()[0] else "")
            
            plt.scatter(state_trials, 
                        trial_averaged_neural[state], 
                        marker=markers['neural'], 
                        color=colors[state], 
                        label=f'Neural Data - {state}' if state == trial_info_normalized['state'].unique()[0] else "")

        if correct_trials is not None:
            # Plot correctness markers
            # Plot markers for each trial
            for trial_idx in correct_trials.index:
                plt.scatter(trial_idx, 0.5, 
                            marker='*', 
                            color='green' if correct_trials[trial_idx] else 'red')
                            
        # Adding labels and title
        plt.xlabel('Trial')
        plt.ylabel('Average Values')
        plt.title(title)
        plt.legend()
        # # Clean up the appearance
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # plt.grid(True)

        # Show the plot
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()

    def plot_pupil_binned_fraction_correct(self,
        pupil_by_state,                   # list of arrays: [Correct, Incorrect, ...]
        dataset_idx_by_state=None,        # list of arrays same shapes as pupil_by_state entries; per-trial dataset ids
        correct_state_idx=0,              # which index is "Correct"
        incorrect_state_idx=1,            # which index is "Incorrect"
        *,
        zscore_x=True,
        n_bins=5,
        edge_input=None,                  # (xmin, xmax); if None use data min/max (post z-score if enabled)
        min_trials_per_bin=5,             # per-dataSET minimum to include that dataset's bin fraction
        regress_on="binpoints",           # 'binpoints' (default) or 'raw'
        scatter_alpha=0.10,
        binpoint_size=12,                 # small gray per-dataset points
        grandpoint_size=20,               # darker grand-mean per-bin point
        text_xy=(0.55, 0.45),
        ylim=(0.4, 1.0),
        figsize=(1.5, 1.5),
        save_path=None
    ):
        """
        Reproduces the MATLAB behavior:
        1) Bin by pupil (x),
        2) Compute fraction-correct *per dataset* in each bin (if >= min_trials_per_bin),
        3) Scatter those dataset-level bin fractions,
        4) Fit linear regression (default on bin points),
        5) Overlay grand mean per bin.
        Returns dict with regression + bin summaries.
        """
        # ---- 1) flatten trials & labels from state lists ----
        X_list, Y_list, DS_list = [], [], []
        for s, x_s in enumerate(pupil_by_state):
            x_s = np.asarray(x_s, float)
            if s == correct_state_idx:
                y_s = np.ones_like(x_s, float)
            elif s == incorrect_state_idx:
                y_s = np.zeros_like(x_s, float)
            else:
                # ignore other states for this plot
                continue
            if dataset_idx_by_state is not None and dataset_idx_by_state[s] is not None:
                ds_s = np.asarray(dataset_idx_by_state[s])
                assert len(ds_s) == len(x_s), "dataset_idx_by_state[state] must match pupil_by_state[state] length"
            else:
                # if no dataset ids provided, treat all trials as belonging to dataset 0
                ds_s = np.zeros_like(x_s, int)
            X_list.append(x_s); Y_list.append(y_s); DS_list.append(ds_s)
        if not X_list:
            raise ValueError("No trials found; check correct_state_idx/incorrect_state_idx and inputs.")
        x = np.concatenate(X_list)
        y = np.concatenate(Y_list)
        ds = np.concatenate(DS_list)
        # drop NaNs
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(ds)
        x, y, ds = x[m], y[m], ds[m].astype(int)
        # ---- 2) z-score x if requested ----
        if zscore_x:
            mu, sd = np.nanmean(x), np.nanstd(x, ddof=1)
            if np.isfinite(sd) and sd > 0:
                x = (x - mu) / sd
        # ---- 3) binning edges & centers ----
        xmin = np.nanmin(x) if edge_input is None else edge_input[0]
        xmax = np.nanmax(x) if edge_input is None else edge_input[1]
        edges = np.linspace(xmin, xmax, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        # ---- 4) compute per-dataset fraction correct per bin ----
        bin_ds_points_x = []  # x at bin center per dataset occurrence
        bin_ds_points_y = []  # fraction correct per dataset in that bin
        bin_counts_total = np.zeros(n_bins, int)
        bin_grand_means = np.full(n_bins, np.nan, float)
        for b in range(n_bins):
            in_bin = (x >= edges[b]) & (x < edges[b+1]) if b < n_bins-1 else (x >= edges[b]) & (x <= edges[b+1])
            bin_counts_total[b] = int(np.sum(in_bin))
            if not np.any(in_bin):
                continue
            # per-dataset fraction in this bin (only if dataset has enough trials in bin)
            ds_vals = []
            for u in np.unique(ds[in_bin]):
                mask_u = in_bin & (ds == u)
                if np.sum(mask_u) >= min_trials_per_bin:
                    frac_u = float(np.nanmean(y[mask_u]))
                    if np.isfinite(frac_u):
                        ds_vals.append(frac_u)
                        bin_ds_points_x.append(centers[b])
                        bin_ds_points_y.append(frac_u)
            # grand mean across datasets that cleared minimum
            if ds_vals:
                bin_grand_means[b] = float(np.nanmean(ds_vals))
        bin_ds_points_x = np.asarray(bin_ds_points_x)
        bin_ds_points_y = np.asarray(bin_ds_points_y)
        # ---- 5) choose what to regress on ----
        if regress_on == "binpoints":
            rx, ry = bin_ds_points_x, bin_ds_points_y
        elif regress_on == "raw":
            rx, ry = x, y
        else:
            raise ValueError("regress_on must be 'binpoints' or 'raw'.")
        # ---- 6) plot ----
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})
        fig, ax = plt.subplots(figsize=figsize)
        # light scatter of bin-level dataset points (what you expected to see)
        if rx.size and ry.size:
            ax.scatter(rx, ry, s=binpoint_size, facecolor=[0.6, 0.6, 0.6], edgecolor='none', alpha=scatter_alpha)
        # regression line
        slope = intercept = r = p = stderr = np.nan
        ok = np.isfinite(rx) & np.isfinite(ry)
        if np.count_nonzero(ok) >= 3:
            slope, intercept, r, p, stderr = stats.linregress(rx[ok], ry[ok])
            xfit = np.linspace(xmin, xmax, 200)
            yfit = slope * xfit + intercept
            ax.plot(xfit, yfit, 'k-', lw=1.5)
        # overlay grand means per bin (dark dots)
        gm_mask = np.isfinite(bin_grand_means)
        if np.any(gm_mask):
            ax.plot(centers[gm_mask], bin_grand_means[gm_mask],
                    'o', ms=grandpoint_size/6, markerfacecolor=[0.3, 0.3, 0.3], markeredgecolor='none')
        # annotate P, R
        ax.text(text_xy[0], text_xy[1],
                f'P = {p:.3g}\nR = {r:.2f}' if np.isfinite(p) else 'P = n/a\nR = n/a',
                transform=ax.transAxes, fontsize=6, va='top')
        ax.set_ylabel('Fraction Correct')
        ax.set_xlabel('Pupil (z-scored)' if zscore_x else 'Pupil')
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        #MATLAB-like aesthetics!!!
        self.matlabfy_plot_appearance(ax)
        #plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True) #TO SAVE LIKE MATLAB!

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
            # fig.savefig(save_path, bbox_inches='tight')
        plt.show()
        return dict(
            slope=slope, intercept=intercept, r=r, p=p, stderr=stderr,
            edges=edges, centers=centers,
            bin_counts_total=bin_counts_total,
            bin_dataset_points_x=bin_ds_points_x,  # the actual points used for scatter (per-dataset per-bin)
            bin_dataset_points_y=bin_ds_points_y,
            bin_grand_means=bin_grand_means
        )

    def plot_bar_scatter(
        self,
        data1,
        data2,
        scatter_data1,
        scatter_data2,
        state_labels,
        state_colors,
        xlabel='Pupil Data',
        ylabel='Neural Data',
        xaxis=None,
        yaxis=None,
        figsize1=(5, 3),
        figsize2=(8, 4),
        save_path=None,
        stats_test='permutation',
        sem_across='trials',                   # 'trials' (default) or 'datasets'
        aggregated_data_indices1=None,         # list per state, same length as data1[state]
        aggregated_data_indices2=None          # list per state, same length as data2[state]
    ):
        """
        Creates:
        - Bar plot for data1 (neural)
        - Bar plot for data2 (pupil)
        - Scatter with per-state regressions
        If sem_across='datasets' and aggregated_data_indices* are provided:
        - For each state, compute per-dataset means first,
            then plot bar = mean(per-ds means), yerr = SEM across dataset means,
            and run stats on those per-ds means.
        """

        def _per_dataset_means(values, ds_idx):
            """
            values: 1D array-like of trial values within a state
            ds_idx: 1D array-like of dataset IDs (same length as values)
            returns: np.array of per-dataset means (ignores NaNs)
            """
            values = np.asarray(values, dtype=float)
            ds_idx = np.asarray(ds_idx)
            # guard: keep only finite values
            mask = np.isfinite(values)
            values = values[mask]
            ds_idx = ds_idx[mask]
            if values.size == 0:
                return np.array([])
            uniq = np.unique(ds_idx)
            means = []
            for u in uniq:
                m = np.nanmean(values[ds_idx == u])
                if np.isfinite(m):
                    means.append(m)
            return np.asarray(means, dtype=float)
        def _bar_mean_sem(state_vals, state_ds_idx=None):
            """
            Returns (mean, sem, series_for_stats)
            - If sem_across=='datasets' and ds_idx provided: operate on per-dataset means
            - Else: operate directly across trials
            """
            if sem_across == 'datasets' and state_ds_idx is not None:
                per_ds = _per_dataset_means(state_vals, state_ds_idx)
                mu = np.nanmean(per_ds) if per_ds.size else np.nan
                se = stats.sem(per_ds, nan_policy='omit') if per_ds.size > 1 else np.nan
                series_for_stats = per_ds  # use dataset means for stats
                paired_logic = True  # when using per-dataset means, we can do paired tests
            else:
                vals = np.asarray(state_vals, dtype=float)
                vals = vals[np.isfinite(vals)]
                mu = np.nanmean(vals) if vals.size else np.nan
                se = stats.sem(vals, nan_policy='omit') if vals.size > 1 else np.nan
                series_for_stats = vals  # use trial values for stats
                paired_logic = False
            return mu, se, series_for_stats, paired_logic
        
        num_states = len(state_labels)
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})
        # ---------- BAR PLOT: data1 (neural) ----------
        # Prepare transformed data for stats (depending on sem_across)
        transformed1 = []
        fig1, ax1 = plt.subplots(figsize=figsize1)
        for s in range(num_states):
            ds_idx = None
            if sem_across == 'datasets' and aggregated_data_indices1 is not None:
                ds_idx = aggregated_data_indices1[s]
            mu, se, series_for_stats,paired_logic = _bar_mean_sem(data1[s], ds_idx)
            transformed1.append(series_for_stats)
            ax1.bar(s, mu, yerr=se, color=state_colors[s], capsize=3, alpha=1,error_kw=dict(lw=.7, capthick=.7))
        ax1.set_xticks(range(num_states))
        ax1.set_xticklabels(state_labels)
        ax1.set_ylabel(ylabel)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        self.matlabfy_plot_appearance(ax1)
        # Stats on transformed1 (list of arrays per state)
        results1, all_stats_dict1, comparisons1 = self.gen_stats.compare_states_aggregated_data(
            transformed1,
            state_labels=self.state_labels,
            paired=paired_logic,
            test=stats_test,
            n_permutations=10000
        )
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f'{save_path}/bar_plot_{ylabel}{sem_across}.pdf', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()
        # ---------- BAR PLOT: data2 (pupil) ----------
        transformed2 = []
        fig2, ax2 = plt.subplots(figsize=figsize1)
        for s in range(num_states):
            ds_idx = None
            if sem_across == 'datasets' and aggregated_data_indices2 is not None:
                ds_idx = aggregated_data_indices2[s]
            mu, se, series_for_stats,paired_logic = _bar_mean_sem(data2[s], ds_idx)
            transformed2.append(series_for_stats)
            ax2.bar(s, mu, yerr=se, color=state_colors[s], capsize=3, alpha=1,error_kw=dict(lw=.7, capthick=.7))
        ax2.set_xticks(range(num_states))
        ax2.set_xticklabels(state_labels)
        ax2.set_ylabel(xlabel)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        self.matlabfy_plot_appearance(ax2)
        results2, all_stats_dict2, comparisons2 = self.gen_stats.compare_states_aggregated_data(
            transformed2,
            state_labels=self.state_labels,
            paired=paired_logic,
            test=stats_test,
            n_permutations=10000
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}/bar_plot_{xlabel}{sem_across}.pdf', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()
        # ---------- SCATTER + REGRESSION ----------
        model_results = {}
        fig3, axs = plt.subplots(1, num_states, figsize=figsize2)
        # initialize axis trackers
        min_pupil, max_pupil = np.inf, -np.inf
        min_neural, max_neural = np.inf, -np.inf
        for s in range(num_states):
            neural_data = np.asarray(scatter_data1[s], dtype=float)
            pupil_data  = np.asarray(scatter_data2[s], dtype=float)
            if neural_data.size == 0 or pupil_data.size == 0:
                continue
            # update ranges if not fixed
            if xaxis is None and np.isfinite(pupil_data).any():
                min_pupil = min(min_pupil, np.nanmin(pupil_data))
                max_pupil = max(max_pupil, np.nanmax(pupil_data))
            if yaxis is None and np.isfinite(neural_data).any():
                min_neural = min(min_neural, np.nanmin(neural_data))
                max_neural = max(max_neural, np.nanmax(neural_data))
            axs[s].scatter(pupil_data, neural_data, edgecolors=state_colors[s],facecolors='none', alpha=0.6, s=5)
            # regression (drop NaNs)
            mask = np.isfinite(pupil_data) & np.isfinite(neural_data)
            n = np.count_nonzero(mask)
            if np.count_nonzero(mask) >= 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    pupil_data[mask], neural_data[mask]
                )
                x_fit = np.linspace(np.nanmin(pupil_data[mask]), np.nanmax(pupil_data[mask]), 100)
                y_fit = slope * x_fit + intercept
                axs[s].plot(x_fit, y_fit, color='black', linestyle='--')
            else:
                slope = intercept = r_value = p_value = std_err = np.nan
            axs[s].text(
                0.05, 0.95,
                f'R = {r_value:.2f}\np = {p_value:.2e}' if np.isfinite(r_value) else 'R = n/a\np = n/a',
                transform=axs[s].transAxes,
                fontsize=6,
                verticalalignment='top'
            )
            axs[s].set_xlabel(xlabel)
            axs[s].set_ylabel(ylabel)
            axs[s].spines['top'].set_visible(False)
            axs[s].spines['right'].set_visible(False)
            self.matlabfy_plot_appearance(axs[s])
            model_results[state_labels[s]] = dict(
                slope=slope, intercept=intercept, r_value=r_value, p_value=p_value, std_err=std_err, n=n
            )
        # consistent axes
        for ax in axs:
            if xaxis is None and np.isfinite(min_pupil) and np.isfinite(max_pupil):
                ax.set_xlim(min_pupil, max_pupil)
            else:
                ax.set_xlim(xaxis)
            if yaxis is None and np.isfinite(min_neural) and np.isfinite(max_neural):
                ax.set_ylim(min_neural, max_neural)
            else:
                ax.set_ylim(yaxis)
        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}/bar_plot_{xlabel}_vs_{ylabel}.pdf', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()
        # ---------- SAVE TABLES ----------
        if save_path:
            save_path_updated = save_path
            # data1
            all_p_values1 = [res['p'] for res in results1]
            test_stats1   = [res['stat'] for res in results1]
            self.gen_stats.to_table(comparisons1, test_stats1, all_p_values1,
                                    save_path=f'{save_path_updated}/stat_tests_bar_{ylabel}.csv',
                                    type=stats_test)
            self.gen_stats.basic_stats_to_table(all_stats_dict1,
                                    save_path=f'{save_path_updated}/basic_stats_{ylabel}.csv')
            # data2
            all_p_values2 = [res['p'] for res in results2]
            test_stats2   = [res['stat'] for res in results2]
            self.gen_stats.to_table(comparisons2, test_stats2, all_p_values2,
                                    save_path=f'{save_path_updated}/stat_tests_bar_{xlabel}.csv',
                                    type=stats_test)
            self.gen_stats.basic_stats_to_table(all_stats_dict2,
                                    save_path=f'{save_path_updated}/basic_stats_{xlabel}.csv')
            # regression
            pd.DataFrame.from_dict(model_results, orient='index') \
                .to_csv(f'{save_path_updated}/regression_model_results_{xlabel}_vs_{ylabel}.csv')
        return results1, results2, model_results
    

    def matlabfy_plot_appearance(self, ax):
        """Apply MATLAB-like aesthetics to a matplotlib axis."""
        ax.spines['bottom'].set_linewidth(.5)
        ax.spines['left'].set_linewidth(.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(width=.5, length=1)
        ax.tick_params(direction='in')

        #plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True) #TO SAVE LIKE MATLAB!
    # def plot_bar_scatter(self, data1, data2, scatter_data1, scatter_data2, state_labels, state_colors, xlabel='Pupil Data', ylabel='Neural Data', xaxis = None, yaxis = None, figsize1 = (5,3), figsize2 = (4*2,4), save_path=None, stats_test = 'permutation'):
    #     """
    #     Creates separate figures for bar plots of data1 and data2, and a third figure for scatter plots with regression lines.
        
    #     Args:
    #         data1 (list of lists): Aggregated data for neural activity by state.
    #         data2 (list of lists): Aggregated data for pupil size by state.
    #         scatter_data1 (list of lists): Aggregated data for neural scatter plots by state.
    #         scatter_data2 (list of lists): Aggregated data for pupil scatter plots by state.
    #         state_labels (list): List of labels for each state.
    #         state_colors (list): List of colors for each state.
    #     """
    #     num_states = len(state_labels)
    #     plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})

    #     # Figure 1: Bar plot for data1 (neural data)
    #     fig1, ax1 = plt.subplots(figsize=figsize1)
    #     for state in range(num_states):
    #         ax1.bar(state, np.nanmean(data1[state]), yerr=np.nanstd(data1[state]), color=state_colors[state],capsize=3, alpha=1)
    #     ax1.set_xticks(range(num_states))
    #     ax1.set_xticklabels(state_labels)
    #     # ax1.set_title(f"{ylabel} by State")
    #     ax1.set_ylabel(ylabel)
    #     ax1.spines['top'].set_visible(False)
    #     ax1.spines['right'].set_visible(False)

    #     #do stats?
    #     results1, all_stats_dict1, comparisons1 = self.gen_stats.compare_states_aggregated_data(data1, state_labels = self.state_labels, paired=False, test=stats_test, n_permutations=10000)

    #     # Show first bar plot
    #     plt.tight_layout()
    #     if save_path:
    #         os.makedirs(save_path, exist_ok=True)
    #         plt.savefig(f'{save_path}/bar_plot_{ylabel}.pdf', bbox_inches='tight')
    #     plt.show()

    #     # Figure 2: Bar plot for data2 (pupil data)
    #     fig2, ax2 = plt.subplots(figsize=figsize1)
    #     for state in range(num_states):
    #         ax2.bar(state, np.nanmean(data2[state]), yerr=np.nanstd(data2[state]), color=state_colors[state],capsize=3, alpha=1)
    #     ax2.set_xticks(range(num_states))
    #     ax2.set_xticklabels(state_labels)
    #     # ax2.set_title(f"{xlabel} by State")
    #     ax2.set_ylabel(xlabel)
    #     ax2.spines['top'].set_visible(False)
    #     ax2.spines['right'].set_visible(False)

    #     results2, all_stats_dict2, comparisons2 =self.gen_stats.compare_states_aggregated_data(data2,  state_labels = self.state_labels, paired=False, test=stats_test, n_permutations=10000)

    #     # Show second bar plot
    #     plt.tight_layout()
    #     if save_path:
    #         plt.savefig(f'{save_path}/bar_plot_{xlabel}.pdf', bbox_inches='tight')
    #     plt.show()

    #     # Figure 3: Scatter plots with regression for each state
    #     #aggregate the model results
    #     model_results = {}
    #     fig3, axs = plt.subplots(1, num_states, figsize=figsize2)
    #     for state in range(num_states):
    #         neural_data = np.array(scatter_data1[state])
    #         pupil_data = np.array(scatter_data2[state])

    #         if len(neural_data) == 0 or len(pupil_data) == 0:
    #             continue
            
    #         if xaxis is None:
    #             # Update min/max values for axis scaling
    #             min_pupil = min(min_pupil, min(pupil_data))
    #             max_pupil = max(max_pupil, max(pupil_data))

    #         if yaxis is None:
    #             min_neural = min(min_neural, min(neural_data))
    #             max_neural = max(max_neural, max(neural_data))
            
    #         axs[state].scatter(pupil_data, neural_data, color=state_colors[state], alpha=0.6,s=5)

    #         # Remove NaNs from both arrays before regression
    #         mask = ~np.isnan(pupil_data) & ~np.isnan(neural_data)
    #         slope, intercept, r_value, p_value, std_err = stats.linregress(pupil_data[mask], neural_data[mask])
    #         x_fit = np.linspace(min(pupil_data[mask]), max(pupil_data[mask]), 100)
    #         y_fit = slope * x_fit + intercept
    #         axs[state].plot(x_fit, y_fit, color='black', linestyle='--')
            
    #         axs[state].text(0.05, 0.95, f'R = {r_value:.2f}\np = {p_value:.2e}', #f'R² = {r_value**2:.2f}
    #                         transform=axs[state].transAxes, fontsize=6, verticalalignment='top') #, ,
    #                         #bbox=dict(facecolor='white', alpha=0.7)edgecolor='black'

    #         axs[state].set_xlabel(xlabel)
    #         axs[state].set_ylabel(ylabel)
    #         # axs[state].set_title(f"{state_labels[state]}")
    #         axs[state].spines['top'].set_visible(False)
    #         axs[state].spines['right'].set_visible(False)

    #         # Store model results
    #         model_results[state_labels[state]] = {
    #             'slope': slope,
    #             'intercept': intercept,
    #             'r_value': r_value,
    #             'p_value': p_value,
    #             'std_err': std_err
    #         }

    #     # Set consistent axis limits across all subplots
    #     for ax in axs:
    #         if xaxis is None:
    #             ax.set_xlim(min_pupil, max_pupil)
    #         else:
    #             ax.set_xlim(xaxis)
            
    #         if yaxis is None:
    #             ax.set_ylim(min_neural, max_neural)
    #         else:
    #             ax.set_ylim(yaxis)

    #     plt.tight_layout()
    #     if save_path:
    #         plt.savefig(f'{save_path}/bar_plot_{xlabel}_vs_{ylabel}.pdf', bbox_inches='tight')
    #     plt.show()

    #     # get actual save_path by getting the string in front of the last /
    #     if save_path:
    #         # Ensure save_path uses proper path separators

    #         save_path_updated = save_path #save_path[:save_path.rfind('/')]
    #         all_p_values = [res['p'] for res in results1]
    #         test_stats = [res['stat'] for res in results1]
    #         df_tests = self.gen_stats.to_table(comparisons1 , test_stats, all_p_values, save_path=f'{save_path_updated}/stat_tests_bar_{ylabel}.csv',type=stats_test)
    #         df_stats = self.gen_stats.basic_stats_to_table(all_stats_dict1, save_path=f'{save_path_updated}/basic_stats_{ylabel}.csv')
    #         # save the results2 as well
    #         all_p_values2 = [res['p'] for res in results2]
    #         test_stats2 = [res['stat'] for res in results2]
    #         df_tests2 = self.gen_stats.to_table(comparisons2 , test_stats2, all_p_values2, save_path=f'{save_path_updated}/stat_tests_bar_{xlabel}.csv',type=stats_test)
    #         df_stats2 = self.gen_stats.basic_stats_to_table(all_stats_dict2, save_path=f'{save_path_updated}/basic_stats_{xlabel}.csv')

    #         # Save model results to a CSV file
    #         model_results_df = pd.DataFrame.from_dict(model_results, orient='index')
    #         model_results_df.to_csv(f'{save_path_updated}/regression_model_results_{xlabel}_vs_{ylabel}.csv')

    #     return results1, results2, model_results



    def plot_bar_with_error_bars(self,data, metric_names, state_colors, title, y_label, state_labels, bar_width = 0.5, figsize=(15,5), save_path=None):
        """
        Creates bar plots with error bars for specified metrics.

        Parameters:
        data : pd.DataFrame
            DataFrame containing metrics for each state.
        metric_names : list
            List of metric names to plot.
        state_colors : dict
            Dictionary mapping state labels to their colors.
        title : str
            Title for the plot.
        y_label : str
            Label for the y-axis.
        """
        num_metrics = len(metric_names)
        x = np.arange(len(data.index.levels[1]))  # X locations for each state
        bar_width = bar_width  # Width of the bars

        fig, axes = plt.subplots(1, num_metrics, figsize= figsize, sharey=True)

        for i, metric in enumerate(metric_names):
            means = data[metric].groupby(level=1).mean()  # Mean values for each state
            stds = data[metric].groupby(level=1).std()    # Standard deviation for error bars

            axes[i].bar(x, means, width=bar_width, color=[state_colors[state] for state in means.index], 
                        yerr=stds, capsize=5)  # Bar plot with error bars
            axes[i].set_title(metric)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(state_labels[i])
            axes[i].set_ylim(0, 100)  # Set y-axis limit
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)

        # Set common labels
        for ax in axes:
            ax.set_ylabel(y_label)

        # plt.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust top to fit title
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()

    def plot_data_before_final_alignment(self,aligned_data_single_dataset, field, ylabel='Pupil Area', pupil_field=None):
        # Default pupil field for pupil data
        if pupil_field is None and field == 'pupil_data':
            pupil_field = 'area'
            
        # Extract data into long format
        area_data = []
        for trial_id, trial_data in aligned_data_single_dataset.iterrows():
            if pupil_field is not None:
                # Extract pupil data with a specific field (e.g., x_pos, y_pos, area)
                for frame, value in enumerate(trial_data[field][pupil_field]):
                    area_data.append({'trial': trial_id, 'frame': frame, 'value': value})
            elif field == 'neural_data':
                # Calculate mean across cells for each frame
                neural_data_mean = trial_data[field].nanmean(axis=0)  # Mean across cells
                for frame, mean_value in enumerate(neural_data_mean):
                    area_data.append({'trial': trial_id, 'frame': frame, 'value': mean_value})
            else:
                # For velocity or other data types
                for frame, value in enumerate(trial_data[field]):
                    area_data.append({'trial': trial_id, 'frame': frame, 'value': value})
        
        # Convert to DataFrame
        area_df = pd.DataFrame(area_data)

        # Create the plot using Seaborn
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=area_df, x='frame', y='value', hue='trial', alpha=0.7)

        # Add labels and title
        plt.xlabel('Time (frames)')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} Over Time for Each Trial')
        plt.grid(False)

        # Clean up the appearance
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Show the plot
        plt.show()

    def x_axis_sec_aligned(self, stim_frame, length_frames, interval=1, frame_rate=30):
        """
        Convert frame indices to seconds for x-axis ticks.
        
        Args:
            stim_frame (int): Frame number where the stimulus event occurs
            length_frames (int): Total number of frames
            interval (int): Interval for x-ticks (default is 1 second)
            frame_rate (int): Imaging frame rate (default is 30 Hz)
        
        Returns:
            xticks_in (list): Frame indices for x-ticks
            xticks_lab (list): Labels for x-ticks in seconds
        """
        frames_before = stim_frame 
        frames_after = len(np.arange(length_frames - stim_frame)) 
        
        time_before = np.arange(-frames_before, 1) / frame_rate
        time_after = np.arange(1, frames_after + 1) / frame_rate

        time_axis = np.concatenate((time_before, time_after))
         
        frame_indices = np.arange(stim_frame - frames_before, stim_frame + frames_after + 1) 
        
        x_tick_seconds = np.unique(np.floor(time_axis))
        x_tick_seconds = x_tick_seconds[x_tick_seconds % interval == 0]
        x_tick_indices = []

        # Ensure valid indices with proper array indexing
        valid_mask = np.isin(x_tick_seconds, time_axis, assume_unique=True)
        valid_indices = np.where(valid_mask)[0]  # Get array of indices directly

        # Index arrays with valid_indices
        x_tick_seconds = x_tick_seconds[valid_indices]
        x_tick_indices = [np.where(time_axis == sec)[0][0] for sec in x_tick_seconds]

        xticks_in = frame_indices[x_tick_indices] 
        xticks_lab = [str(int(sec)) for sec in x_tick_seconds]
        
        return xticks_in, xticks_lab

    def plot_with_seconds( self, stim_frame,length_frames, frame_rate=30,interval=1):
        """
        Plot data with x-axis in seconds.
        
        Args:
            
            stim_frame (int): Frame number where the stimulus event occurs
            frame_rate (int): Imaging frame rate (default is 30 Hz)
            interval (int): Interval for x-ticks (default is 1 second)  
        """
        length_frames = length_frames
        xticks_in, xticks_lab = self.x_axis_sec_aligned(stim_frame, length_frames, interval=interval, frame_rate=frame_rate)

        plt.xticks(ticks=xticks_in, labels=xticks_lab)
        plt.xlabel('Time (s)')







