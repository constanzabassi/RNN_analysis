import os
import scipy
import numpy as np
# Import DataLoader from helper_functions
from .data_loader import DataLoader

# Define the DataPipeline class with server handling per dataset
class DataPipeline:
    def __init__(self):
        pass  # No need for base_dir

    def load_celltypes(self, animalID, date, server):
        # Use the server specific to this dataset
        path = f"{server}/Connie/ProcessedData/{animalID}/{date}/red_variables/"

        # Load cell types from .mat files
        pyr_str = scipy.io.loadmat(os.path.join(path, 'pyr_cells.mat'))
        pyr = pyr_str['pyr_cells'] - 1  # Convert to Python indices
        pyr = np.transpose(pyr)

        som_str = scipy.io.loadmat(os.path.join(path, 'mcherry_cells.mat'))
        som = som_str['mcherry_cells'] - 1

        pv_str = scipy.io.loadmat(os.path.join(path, 'tdtom_cells.mat'))
        pv = pv_str['tdtom_cells'] - 1

        neuron_groups = {
            'pyr': pyr,
            'som': som,
            'pv': pv
        }

        # Define colors for each group
        colors = {
            'pyr': (0.37, 0.75, 0.49),  # pyr = 0
            'som': (0.17, 0.35, 0.8),   # som = 1
            'pv': (0.82, 0.04, 0.04)    # pv = 2
        }

        # Combine cell types into different indices
        celltype_array = np.zeros(np.shape(np.concatenate((pyr, som, pv)))[0])
        celltype_array[som] = 1
        celltype_array[pv] = 2

        return celltype_array, neuron_groups, colors


    def load_data(self, datasets, save_string='VR', load_celltypes=True):
        data_loaders = []
        celltype_info = {}

        # Iterate through each dataset to load data
        for animalID, date, server in datasets:
            print(f"Loading data for: Animal: {animalID}, Date: {date}, Server: {server}")

            # Initialize DataLoader with the proper arguments
            matlab_file = f"{server}/Connie/ProcessedData/{animalID}/{date}/{save_string}/imaging.mat"
            data_loader = DataLoader(matlab_file, server=server, animalID=animalID, date=date)

            # Store the data loader and celltype info for further analysis
            data_loaders.append(data_loader)
            key = (animalID, date)
            # Optionally load the cell types
            if load_celltypes:
                celltype_array, neuron_groups, colors = self.load_celltypes(animalID, date, server)
                celltype_info[key] = {
                    'celltype_array': celltype_array,
                    'neuron_groups': neuron_groups,
                    'colors': colors
                }
            else:
                celltype_info[key] = None
        
        return data_loaders, celltype_info


    def load_neural_data(self, datasets,save_string = 'VR', load_celltypes=True):
        """
        Load only neural imaging data and optionally cell types
        
        Parameters:
        -----------
        datasets : list of tuples
            List of (animalID, date, server) tuples
        load_celltypes : bool, optional
            Whether to load cell type information
            
        Returns:
        --------
        dict
            Dictionary with keys (animalID, date) containing:
            - 'neural_data': neural activity matrix
            - 'celltype_info': cell type information (if load_celltypes=True)
        """
        neural_data_dict = {}

        for animalID, date, server in datasets:
            print(f"Loading neural data for: Animal: {animalID}, Date: {date}")
            
            # Load neural data
            matlab_file = f"{server}/Connie/ProcessedData/{animalID}/{date}/{save_string}/imaging.mat"
            neural_data = scipy.io.loadmat(matlab_file)['imaging']['data'][0][0]
            
            key = (animalID, date)
            neural_data_dict[key] = {'neural_data': neural_data}
            
            # Optionally load cell types
            if load_celltypes:
                celltype_array, neuron_groups, colors = self.load_celltypes(animalID, date, server)
                neural_data_dict[key].update({
                    'celltype_array': celltype_array,
                    'neuron_groups': neuron_groups,
                    'colors': colors
                })

        return neural_data_dict
