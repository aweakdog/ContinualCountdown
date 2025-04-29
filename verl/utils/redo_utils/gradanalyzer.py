class VerlGradientAnalyzer:
    """
    Legacy Gradient Analyzer (deprecated):
    This class previously used hooks and buffers for distributed gradient analysis.
    All such logic is now obsolete, as gradient metrics are computed directly on rank 0 after gathering gradients.
    This class is retained only for backward compatibility and should not be used in new code.
    """
    def __init__(self, model, seed=None):
        self.model = model
        if seed is not None:
            import numpy as np
            import torch
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # No hooks or buffers are registered anymore.
        # Use new FSDP-safe metrics instead.
