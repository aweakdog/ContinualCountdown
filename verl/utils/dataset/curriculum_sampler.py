from torch.utils.data import Sampler
import numpy as np

class CurriculumSampler(Sampler):
    """A sampler that maintains the order of samples based on their source files."""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = np.arange(len(dataset))
        
        # Group indices by source file to maintain curriculum order
        if hasattr(dataset, 'dataframe') and '_source_file' in dataset.dataframe.columns:
            # Get unique files in the order they were loaded
            unique_files = dataset.dataframe['_source_file'].unique()
            
            # Create ordered indices
            ordered_indices = []
            for file in unique_files:
                file_indices = dataset.dataframe[dataset.dataframe['_source_file'] == file].index.values
                # Shuffle within each file group
                np.random.shuffle(file_indices)
                ordered_indices.extend(file_indices)
            
            self.indices = np.array(ordered_indices)
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)
