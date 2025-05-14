import argparse
import yaml
from pathlib import Path
import datetime
from ctf4science.data_module import load_dataset, parse_pair_ids, get_training_timesteps, get_prediction_timesteps, get_metadata, get_applicable_plots
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
from pinn import PINN

import matplotlib.pyplot as plt

def main(config_path):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    pair_ids = parse_pair_ids(config["dataset"])
    model_name = "PINN"
    batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    batch_results = {"batch_id": batch_id, "model": model_name, "dataset": dataset_name, "pairs": []}
    
    viz = Visualization()
    applicable_plots = get_applicable_plots(dataset_name)

    for pair_id in pair_ids:
        train_data, init_data = load_dataset(dataset_name, pair_id)
        training_timesteps = get_training_timesteps(dataset_name, pair_id)
        prediction_timesteps = get_prediction_timesteps(dataset_name, pair_id)
        delta_t = get_metadata(dataset_name)["delta_t"]

        model = PINN(pair_id, config, train_data, init_data, training_timesteps, prediction_timesteps, delta_t)
        predictions = model.predict()
        
        results = evaluate(dataset_name, pair_id, predictions)
        results_directory = save_results(dataset_name, model_name, batch_id, pair_id, config, predictions, results)

        batch_results["pairs"].append({"pair_id": pair_id, "metrics": results})

        for plot_type in applicable_plots:
            fig = viz.plot_from_batch(dataset_name, pair_id, results_directory, plot_type=plot_type)
            viz.save_figure_results(fig, dataset_name, model_name, batch_id, pair_id, plot_type, results_directory)
    
    with open(Path(results_directory).parent / "batch_results.yaml", "w") as f:
        yaml.dump(batch_results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)