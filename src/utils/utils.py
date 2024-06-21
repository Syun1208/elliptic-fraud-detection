from typing import Dict, Any
import yaml
import numpy as np
import random


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    
    return data


def resample_testmask(test_mask, p=0.5):
    sample_size = int(np.floor(test_mask.sum()*p))
    # Get indices where value is True
    true_indices = [i for i, val in enumerate(test_mask) if val]

    # Randomly select a subset of these indices
    sampled_indices = random.sample(true_indices, min(sample_size, len(true_indices)))

    # Create new list with False at all indices except the sampled ones
    output_list = [i in sampled_indices for i in range(len(test_mask))]

    return output_list