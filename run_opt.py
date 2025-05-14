## Importing all necessary dependencies
import argparse
import shutil
import yaml
from pathlib import Path
from typing import List, Dict, Any
import datetime
import numpy as np
from ctf4science.data_module import load_dataset, load_validation_dataset, get_validation_prediction_timesteps, parse_pair_ids, get_applicable_plots, get_training_timesteps, get_prediction_timesteps, get_metadata
from ctf4science.eval_module import evaluate_custom, save_results
from ctf4science.visualization_module import Visualization
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from pinn import PINN

# Delete results directory - used for storing batch_results
file_dir = Path(__file__).parent

def main(config_path: str) -> None:
    """
    Main function to run PINN on specified sub-datasets.

    Loads configuration, parses pair_ids, initializes the model, generates predictions,
    evaluates them, and saves results for each sub-dataset under a batch identifier.

    Args:
        config_path (str): Path to the configuration file.
    """

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset name and get list of sub-dataset train/test pair_ids
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])

    # batch_id is from optimize_parameters.py
    model_name = f"{config['model']['name']}_{config['model']['version']}"
    batch_id = f"hyper_opt_{config['model']['batch_id']}"

    # Initialize batch results dictionary for summary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'pairs': []
    }

    # Process each sub-dataset
    for pair_id in pair_ids:
        print('pair_id ----------------------------', pair_id)
        # Load sub-dataset
        train_split = config['model']['train_split']
        train_data, val_data, init_data = load_validation_dataset(dataset_name, pair_id, train_split, transpose=True)
        train_data = np.concatenate(train_data, axis = 1)

        # Load metadata (to provide forecast length)
        training_timesteps = get_training_timesteps(dataset_name, pair_id)
        prediction_timesteps = get_validation_prediction_timesteps(dataset_name, pair_id, train_split)
        prediction_horizon_steps = prediction_timesteps.shape[0]
        delta_t = get_metadata(dataset_name)["delta_t"]
   
        # Initialize the model with the config and train_data
        model = PINN(pair_id, config, train_data, init_data, training_timesteps, prediction_timesteps, delta_t)

        # model = KANctf(config, train_data, init_data, prediction_horizon_steps, pair_id) 
        
        # Generate predictions
        pred_data = model.predict()
       # transpose data back into (Timesteps, Features) format
        pred_data = pred_data.T
        val_data = val_data.T

        # Evaluate predictions using default metrics
        results = evaluate_custom(dataset_name, pair_id, val_data, pred_data)

        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        batch_results['pairs'].append({
            'pair_id': pair_id,
            'metrics': results
        })
       
    # Save aggregated batch results
    results_file = file_dir / f"results_{config['model']['batch_id']}.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(batch_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)