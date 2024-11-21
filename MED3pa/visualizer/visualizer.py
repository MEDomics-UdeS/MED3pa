import os
import json
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objs as go




# Base Visualization Class
class Visualization:
    def __init__(self, experiment_folder,experiment):
        self.experiment_folder = experiment_folder
        self.experiment = experiment
        self.config_path = os.path.join(experiment_folder, "experiment_config.json")

    def check_experiments(self):
        """Load and validate the experiment config."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        experiment_name = self.config.get("experiment_name")
        if experiment_name not in self.experiment:
            raise ValueError(f"Unsupported experiment type: {experiment_name}")
        print(f"Experiment type validated: {experiment_name}")
    
    def load_data(self, json_file, set="reference"):
        """Load tree profiles based on the set type."""
        # Determine the correct file path based on the set type
        if set == "reference":
            file_path = os.path.join(self.experiment_folder, "reference", json_file)
        elif set == "test":
            file_path = os.path.join(self.experiment_folder, "test", json_file)
        else:
            raise ValueError("The 'set' parameter must be either 'reference' or 'test'.")

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try to open and load the JSON file
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Check if the file is empty (e.g., empty list, empty dictionary, or None)
            if not data:
                raise ValueError(f"The file {file_path} is empty.")

            # Return the data wrapped in a dictionary
            print(f"Data Loaded from {file_path}")
            return data

        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from file: {file_path}")
        except ValueError as ve:
            raise ve  # re-raise if the file is empty or invalid data




    def generate(self, *args, **kwargs):
        """Generate HTML (to be implemented by subclasses)."""
        raise NotImplementedError

    def visualize(self, set='reference', **kwargs):
        """Main method to run the visualization pipeline."""
        self.check_experiments()
        self.generate(set=set, **kwargs)


class TreeVisualization(Visualization):
    def __init__(self, experiment_folder):
        super().__init__(experiment_folder, ["Med3paDetectronExperiment", "Med3paExperiment"])
        self.template_folder = os.path.join(os.path.dirname(__file__), "tree_template")
        
    
    def load_data_tree(self, samp_ratio, dr, set):
        """Retrieve nodes based on user-defined sample ratio and data ratio."""
        
        tree_profiles = self.load_data('profiles.json',set)
        
        try:
            profiles_to_visualize = tree_profiles[str(samp_ratio)][str(dr)]
        except KeyError:
            raise ValueError(f"No nodes found for samp_ratio={samp_ratio} and dr={dr}.")

        print(f"Nodes successfully loaded for samp_ratio={samp_ratio}, dr={dr}")
        return profiles_to_visualize
            
        
    def generate(self, set, samp_ratio, dr):
        """Generate the tree visualization HTML."""
        env = Environment(loader=FileSystemLoader(self.template_folder))
        template = env.get_template('tree.html')

        # Read the profiles for the specified set
        profiles_to_visualize = self.load_data_tree(samp_ratio=samp_ratio, dr=dr, set=set)

        # Get the absolute path to the template folder
        base_path = os.path.abspath(self.template_folder)

        # Render the HTML with the list of nodes and base path
        rendered_html = template.render(
            nodes=profiles_to_visualize,
            base_path=base_path
        )

        # Determine output path
        if set == "reference":
            output_path = os.path.join(self.experiment_folder, f"tree_visualization_reference_{samp_ratio}_{dr}.html")
        elif set == "test":
            output_path = os.path.join(self.experiment_folder, f"tree_visualization_test_{samp_ratio}_{dr}.html")
        else:
            raise ValueError("The 'set' parameter must be either 'reference' or 'test'.")

        # Save the HTML
        with open(output_path, 'w') as f:
            f.write(rendered_html)

        print(f"Tree visualization generated: {output_path}")



class MDRCurveVisualization(Visualization):
    def __init__(self, experiment_folder):
        super().__init__(experiment_folder, ["Med3paDetectronExperiment", "Med3paExperiment"])
        
    
    def load_data_curve(self, set):
        """Retrieve metrics by DR."""
        # Load the data
        metrics_dr = self.load_data('metrics_dr.json', set)
        
        # Ensure indices are sorted from 0 to 100 numerically
        indices = sorted(metrics_dr.keys(), key=int)  # Sort in ascending order (0 to 100)
        
        # Initialize metrics dictionary, excluding 'LogLoss'
        all_metrics = metrics_dr["100"]["metrics"].keys()
        metrics = {metric: [] for metric in all_metrics}

        # Populate metrics while preserving their correspondence with indices
        for index in indices:
            for metric, value in metrics_dr[index]["metrics"].items():
                if metric != "LogLoss":  # Skip LogLoss explicitly
                    metrics[metric].append(value)

        return indices, metrics


    def plot_all_metrics(self, indices, metrics):
        """Plot all metrics dynamically."""
        fig = go.Figure()

        # Plot every metric in the metrics dictionary
        for metric, values in metrics.items():
            fig.add_trace(go.Scatter(
                x=indices,
                y=values,
                mode='lines+markers',
                name=metric
            ))

        fig.update_layout(
            title="Metrics By Declaration Rate Curve",
            xaxis_title="Declaration Rate",
            yaxis_title="Evaluation Metrics",
            template="plotly_white" 
        )

        fig.show()

    def generate(self, set):
        """Generate the curve visualization HTML."""
        indices, metrics = self.load_data_curve(set)
        self.plot_all_metrics(indices, metrics)


