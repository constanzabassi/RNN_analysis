import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from .data_plotter import Plotter  # Import Plotter instead

class TrialPlotter:
    def __init__(self):
        self.event_frames = np.array([6., 38., 70., 131., 145.]) #[5, 37, 69, 130, 144]
        self.event_labels = ['Sound 1', 'Sound 2', 'Sound 3', 'Turn', 'Reward']
        self.plotter = Plotter({}, {})
    
    def plot_condition_heatmaps(self, 
                              aligned_imaging: np.ndarray,
                              all_conditions: List,
                              cell_ids: Optional[List[int]] = None,
                              average_type: str = 'cells',
                              save_path: Optional[str] = None,
                              vmin: Optional[float] = None,
                              vmax: Optional[float] = None):
        """
        Create heatmaps of neural activity separated by conditions.
        
        Parameters:
        -----------
        aligned_imaging : np.ndarray
            Shape (trials, neurons, frames)
        all_conditions : list
            Output from TrialDivider containing trial indices and labels
        cell_ids : list, optional
            Specific cell indices to plot
        average_type : str
            'none': no averaging (default)
            'trials': average across trials
            'cells': average across cells
            'both': average across both
        save_path : str, optional
            Path to save the figure
        vmin, vmax : float, optional
            Range for color scaling
        """
        n_conditions = len(all_conditions)
        fig, axs = plt.subplots(n_conditions, 1, 
                               figsize=(6, 3*n_conditions),
                               sharex=True)
        if n_conditions == 1:
            axs = [axs]
        
        for ax, (trials, comb, label) in zip(axs, all_conditions):
            # Select data for current condition
            data = aligned_imaging[trials]
            if cell_ids is not None:
                data = np.squeeze(data[:, cell_ids, :])
            
            # Apply averaging based on average_type
            if average_type == 'trials':
                data = data.mean(axis=0)
                
                # Sort cells by time of peak activity
                peak_times = np.argmax(data, axis=1)
                sort_idx = np.argsort(peak_times)
                data = data[sort_idx]

                ylabel = 'Cells'
            elif average_type == 'cells':
                data = data.mean(axis=1)
                ylabel = 'Trials'
            elif average_type == 'both':
                data = data.mean(axis=(0, 1))[np.newaxis, :]
                ylabel = 'Average'
            else:
                data = data.mean(axis=1)  # Default: average across cells
                ylabel = 'Trials'
            
            # Plot heatmap
            sns.heatmap(data, 
                       ax=ax,
                       cmap='viridis',
                       xticklabels=50,
                       cbar_kws={'label': 'ΔF/F'},
                       vmin=vmin,
                       vmax=vmax)
            
            # Add event lines
            for frame, event_label in zip(self.event_frames, self.event_labels):
                ax.axvline(x=frame, color='white', linestyle=(0, (5, 5)), alpha=0.5)
                ax.text(frame, ax.get_ylim()[1], event_label, 
                       rotation=45, ha='right', va='bottom', color='white')
            
            # Set labels
            ax.set_ylabel(ylabel)
            ax.set_title(f'{label}\n(n={len(trials)} trials, {data.shape[0]} {ylabel.lower()})')

            self.plotter.plot_with_seconds(0, data.shape[1], 30)
            # ax.set_xlabel('Frames')
        
        # plt.xlabel('Frames')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig, axs