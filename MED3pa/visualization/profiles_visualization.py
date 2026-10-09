"""
The profiles_visualization.py module manages visualization methods for the resulting profiles of the MED3pa method.
"""

import os
import re
import shutil
import numpy as np
from copy import deepcopy
from jinja2 import Environment, FileSystemLoader
from typing import Any, List, Optional

from MED3pa.med3pa import Profile
from MED3pa.med3pa.profiles import ProfilesManager
from MED3pa.med3pa.results import Med3paResults


_template_folder = "tree_template"
_save_template_folder = None


def visualize_tree(result: Med3paResults, filename: str = 'profiles', dr: int = 100, samp_ratio: int = 0,
                   data_set: str = 'test', metrics_list: List | None = None, profile_depth: int = 4,
                   open_results: bool = True, port: int = 8000, scaler: Optional[Any] = None,
                   additional_trees: bool = True) -> None:
    """
    Vizualization method for the profiles obtained from the MED3pa method.

    The additional trees of the results (e.g. 'source_apc', 'covariate_shift' and 'outcome_shift', see
    Med3paExperiment._calculate_additional_trees) are saved alongside, in '{filename}_{tree name}.html'.

    Args:
        result (Med3paResults): The results of the experiment to visualize.
        filename (str): The name of the file to be saved. Defaults to 'profiles'.
        dr (int): Declaration rate of predictions for the visualization. Defaults to 100, meaning all predictions.
        samp_ratio (int): The minimum samples ratio in each profile (in percentage). Defaults to 0, meaning no minimal
            samples per profile.
        data_set (str): The name of the data set to visualize. Defaults to 'test', options are 'train', 'valid', 'test'.
        metrics_list (List | None): The list of metrics to visualize. Defaults to None, meaning shown metrics are:
            ['Specificity', 'Sensitivity', 'NPV', 'PPV', 'AUC']
        profile_depth (int): Maximum profile depth. Defaults to 4.
        open_results (bool): Whether to open the results of the experiment in the web browser. Defaults to True.
        port (int): The local port used to host the python server to open the web browser. Defaults to port 8000.
        scaler (Any): To unscale variables, if applicable.
        additional_trees (bool): Whether to also save the additional trees of the results. Defaults to True.
    """
    result_copy = deepcopy(result)
    # Set folder path for the template
    global _save_template_folder, _template_folder
    _save_template_folder = os.path.dirname(filename) + "/" + _template_folder

    assert data_set in ['test'], "Invalid data_set, must be in ['test']"

    if metrics_list is None:
        metrics_list = ['Specificity', 'Sensitivity', 'BalancedAccuracy', 'Auc']

    trees = {filename: result_copy.test_record.profiles_manager}
    if additional_trees:
        for name, additional_tree in result_copy.test_record.get_additional_trees().items():
            trees[f"{filename}_{name}"] = additional_tree['profiles_manager']

    # Save the template with the HTML files
    template_path = os.path.join(os.path.dirname(__file__), _template_folder)
    shutil.copytree(template_path, _save_template_folder, dirs_exist_ok=True)
    for tree_filename, profiles_manager in trees.items():
        rendered_html = _generate_tree_html(profiles_manager=profiles_manager, samp_ratio=samp_ratio, dr=dr,
                                            data_set=data_set, metrics_list=metrics_list,
                                            max_depth=profile_depth, scaler=scaler)
        # Save the HTML file
        with open(tree_filename + '.html', 'w') as f:
            f.write(rendered_html)
        print(f"Tree visualization generated: '{tree_filename}.html'")

    if open_results:  # Open html results file in web browser
        import http.server
        import socketserver
        import threading
        import webbrowser

        def start_server():
            handler = http.server.SimpleHTTPRequestHandler
            with socketserver.TCPServer(("", port), handler) as httpd:
                print(f"Serving at http://localhost:{port}")
                httpd.serve_forever()

        # Start server in background
        thread = threading.Thread(target=start_server, daemon=True)
        thread.start()

        # Open HTML file in default browser
        url = f"http://localhost:{port}/{filename}.html"
        webbrowser.open(url)

        # Keep the script alive
        input("Press Enter to stop the server...\n")
        print("\n\nTo open the HTML file in the web browser:\n"
              "1. Activate your conda environment: conda start <env_name>;\n"
              f"2. Start a local Python server: python -m http.server {port};\n"
              f"3. Open the following URL in your browser: http://localhost:{port}/{filename}.html")


def _generate_tree_html(profiles_manager: ProfilesManager, samp_ratio: int, dr: int, data_set: str,
                        metrics_list: Optional[List] = None, max_depth: Optional[int] = None,
                        scaler: Optional[Any] = None) -> str:
    """
    Generates the tree visualization HTML.

    Args:
        profiles_manager (ProfilesManager): The profiles of the tree to visualize.
        samp_ratio (int): The minimum samples ratio in each profile (in percentage).
        dr (int): Declaration rate of predictions for the visualization.
        data_set (str): The name of the data set to visualize. Defaults to 'test', options are 'train', 'valid', 'test'.
        metrics_list (List | None): The list of metrics to visualize. Defaults to None, meaning no metrics are shown.
        max_depth (int): Maximum profile depth.
        scaler (Any): To unscale variables, if applicable.

    Returns:
        str: The HTML string of the visualization tree.
    """
    global _save_template_folder, _template_folder

    # Get the absolute path of the current file's directory
    template_path = os.path.join(os.path.dirname(__file__), _template_folder)
    env = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template('tree.html')

    # Read profiles for the specified data_set
    profiles_to_visualize = deepcopy(_read_tree_section(profiles_manager=profiles_manager, samp_ratio=samp_ratio,
                                                        dr=dr, data_set=data_set))

    if scaler:
        feature_columns = scaler.get_feature_names_out()
        if isinstance(feature_columns, np.ndarray):
            feature_columns = feature_columns.tolist()

        feature_columns = [
            "_".join(col.strip().split())  # collapse multiple spaces, strip, replace with _
            for col in feature_columns
        ]
        for profile in profiles_to_visualize:
            profile.update_path(_unscale_path(profile.path, scaler))

            # 2. Unscale FeatureResumer explanations
            explanations = profile.node_information.get("explanations")
            if not explanations:
                continue

            for expl in explanations:
                if expl.get("Method_name") == "FeatureResumer":
                    raw = expl.get("result")
                    if not isinstance(raw, str):
                        continue

                    parsed = _parse_feature_resumer_string(raw)
                    if not parsed:
                        continue

                    feature, mean, std, mean_ref, std_ref = parsed

                    mean_u, std_u, mean_ref_u, std_ref_u = _unscale_feature_resumer(
                        feature, mean, std, mean_ref, std_ref, scaler, feature_columns
                    )

                    expl["result"] = _format_feature_resumer(
                        feature, mean_u, std_u, mean_ref_u, std_ref_u
                    )

    # Show only metrics in metrics_list, if specified
    if metrics_list is not None:
        for profile in profiles_to_visualize:
            all_metrics = profile.metrics
            profile.update_metrics_results({key: all_metrics.get(key) for key in metrics_list if key in all_metrics.keys()})

    if max_depth is not None:
        profiles_to_visualize = [profile for profile in profiles_to_visualize if len(profile.path) <= max_depth]

    # Convert Profile objects to dicts before json serialization
    profiles_to_visualize = [profile.to_dict() for profile in profiles_to_visualize]

    # Render the HTML with the list of nodes and base path
    rendered_html = template.render(
        nodes=profiles_to_visualize,
        base_path=_save_template_folder
    )
    return rendered_html


def _unscale_path(path_list, scaler):
    new_path = []
    feature_columns = scaler.get_feature_names_out()
    if isinstance(feature_columns, np.ndarray):
        feature_columns = feature_columns.tolist()

    feature_columns = [
        "_".join(col.strip().split())  # collapse multiple spaces, strip, replace with _
        for col in feature_columns
    ]

    for cond in path_list:
        if cond == "*" or cond.strip() == "":
            new_path.append(cond)
            continue

        column_name, operator, value_str = cond.split(' ')
        original_val = _unscale_value(column_name, value_str, scaler, feature_columns)

        # Rebuild condition with rounded value
        new_cond = f"{column_name} {operator} {original_val:.4f}"
        new_path.append(new_cond)

    return new_path

def _unscale_value(col_name, scaled_value, scaler, feature_columns):
    idx = feature_columns.index(col_name)
    vec = np.zeros(len(feature_columns))
    vec[idx] = scaled_value
    return scaler.inverse_transform([vec])[0][idx]

def _parse_feature_resumer_string(s):
    """
    Parses strings like:
    'Diabetes_mellitus: 0.2500 ± 0.4472  (0.0000 ± 0.0000)'
    Returns: (feature_name, mean, std, mean_ref, std_ref)
    """
    pattern = r"""
        ^\s*
        (?P<feature>[A-Za-z0-9_]+)
        \s*:\s*
        (?P<mean>-?\d+\.?\d*(e[-+]?\d+)?)      # mean
        \s*±\s*
        (?P<std>-?\d+\.?\d*(e[-+]?\d+)?)       # std
        \s*\(\s*
        (?P<mean_ref>-?\d+\.?\d*(e[-+]?\d+)?)  # mean_ref
        \s*±\s*
        (?P<std_ref>-?\d+\.?\d*(e[-+]?\d+)?)   # std_ref
        \s*\)\s*$
    """

    m = re.match(pattern, s, re.VERBOSE)
    if not m:
        return None

    return (
        m.group("feature"),
        float(m.group("mean")),
        float(m.group("std")),
        float(m.group("mean_ref")),
        float(m.group("std_ref")),
    )

def _unscale_feature_resumer(feature, mean, std, mean_ref, std_ref, scaler, feature_columns):
    return (
        _unscale_value(feature, mean, scaler, feature_columns),
        _unscale_std(feature, std, scaler, feature_columns),
        _unscale_value(feature, mean_ref, scaler, feature_columns),
        _unscale_std(feature, std_ref, scaler, feature_columns),
    )

def _unscale_std(col_name, scaled_std, scaler, feature_columns):
    idx = feature_columns.index(col_name)
    return scaler.scale_[idx] * scaled_std


def _format_feature_resumer(feature, mean, std, mean_ref, std_ref):
    return f"{feature}: {mean:.4f} ± {std:.4f}  ({mean_ref:.4f} ± {std_ref:.4f})"


def _read_tree_section(profiles_manager: ProfilesManager, samp_ratio: int, dr: int, data_set:str = "test") -> List:
    """
    Generates the tree visualization HTML.

    Args:
        profiles_manager (ProfilesManager): The profiles of the tree to visualize.
        dr (int): Declaration rate of predictions for the visualization.
        samp_ratio (int): The minimum samples ratio in each profile (in percentage).
        data_set (str): The name of the data set to visualize. Defaults to 'test', options are 'train', 'valid', 'test'.

    Returns:
        List: The profiles to visualize in the HTML file.
    """
    profiles_to_visualize = profiles_manager.profiles_records[samp_ratio][dr]

    # Round condition values
    for profile in profiles_to_visualize:
        path = [re.sub(r'(?<!\w)(\d+\.\d+|\d+)(?!\w)',
                               lambda m: str(round(float(m.group()), 1)), s) for s in profile.path]
        profile.update_path(path=path)

    if dr != 100:  # Get results of DR = 100% to compare each metrics
        original_profiles = _read_tree_section(profiles_manager=profiles_manager, samp_ratio=samp_ratio, dr=100,
                                               data_set=data_set)
        for profile in profiles_to_visualize:
            original_profile_metrics = original_profiles[profile.node_id - 1].metrics
            for metric_name, original_metric_value in original_profile_metrics.items():
                if profile.metrics['metric_name'] is None:
                    profile.metrics[f'diff_{metric_name}'] = None
                else:
                    profile.metrics[f'diff_{metric_name}'] = profile.metrics[metric_name] - original_metric_value

    return profiles_to_visualize
