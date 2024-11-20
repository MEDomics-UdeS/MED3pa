import os
import json
from jinja2 import Environment, FileSystemLoader


class Visualizer:
    def __init__(self, experiment_folder):
        self.experiment_folder = experiment_folder
        self.template_folder = os.path.join(os.path.dirname(__file__), "tree_template")
        self.config_path = os.path.join(experiment_folder, "experiment_config.json")
        self.ref_profile_path = os.path.join(experiment_folder, "reference", "profiles.json")
        self.test_profile_path = os.path.join(experiment_folder, "test", "profiles.json")

    def check_experiments(self):
        """Load and validate the experiment config."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        experiment_name = self.config.get("experiment_name")
        if experiment_name not in ["Med3paDetectronExperiment", "Med3paExperiment"]:
            raise ValueError(f"Unsupported experiment type: {experiment_name}")
        print(f"Experiment type validated: {experiment_name}")

    def read_tree_section(self, samp_ratio, dr, set):
        """Retrieve nodes based on user-defined sample ratio and data ratio."""
        if set == "reference":
            tree_path = self.ref_profile_path
        elif set == "test":
            tree_path = self.test_profile_path
        else:
            raise ValueError("The 'set' parameter must be either 'reference' or 'test'.")

        if not os.path.exists(tree_path):
            raise FileNotFoundError(f"Profile tree file not found: {tree_path}")

        with open(tree_path, 'r') as f:
            tree_profiles = json.load(f)

        try:
            profiles_to_visualize = tree_profiles[str(samp_ratio)][str(dr)]
        except KeyError:
            raise ValueError(f"No nodes found for samp_ratio={samp_ratio} and dr={dr}.")

        print(f"Nodes successfully loaded for samp_ratio={samp_ratio}, dr={dr}")
        return profiles_to_visualize

    def generate_tree_html(self, samp_ratio, dr, set):
        """Generate the tree visualization HTML."""
        env = Environment(loader=FileSystemLoader(self.template_folder))
        template = env.get_template('tree.html')

        # Read the profiles for the specified set
        profiles_to_visualize = self.read_tree_section(samp_ratio=samp_ratio, dr=dr, set=set)

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



    def visualize(self, samp_ratio, dr, set):
        """Main method to run the visualization pipeline."""
        self.check_experiments()
        self.generate_tree_html(samp_ratio, dr, set=set)

