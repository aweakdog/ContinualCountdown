from torch.utils.data import Sampler
import numpy as np

class CurriculumSampler(Sampler):
    """A sampler that maintains the order of samples based on rounds, groups, and epochs."""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = np.arange(len(dataset))
        
        # Check if we have curriculum learning information
        if hasattr(dataset, 'dataframe') and all(col in dataset.dataframe.columns for col in ['_group', '_round', '_epoch']):
            ordered_indices = []
            
            # Iterate through rounds
            for round_num in range(dataset.total_rounds):
                # Iterate through operator groups in order
                for group in dataset.operator_groups:
                    # Iterate through epochs for this group
                    for epoch in range(dataset.epochs_per_group):
                        # Get indices for this round, group, and epoch
                        mask = (dataset.dataframe['_round'] == round_num) & \
                               (dataset.dataframe['_group'] == group) & \
                               (dataset.dataframe['_epoch'] == epoch)
                        indices = dataset.dataframe[mask].index.values
                        # Shuffle within this batch
                        np.random.shuffle(indices)
                        ordered_indices.extend(indices)
            
            self.indices = np.array(ordered_indices)
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)
