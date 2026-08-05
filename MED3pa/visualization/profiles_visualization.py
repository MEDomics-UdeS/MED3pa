"""
The profiles_visualization.py module manages visualization methods for the resulting profiles of the MED3pa method.
"""
from __future__ import annotations

import os
import re
import shutil
from jinja2 import Environment, FileSystemLoader
from typing import List, Optional

from MED3pa.med3pa.results import Med3paResults


_template_folder = "tree_template"
_save_template_folder = None


def visualize_tree(result: Med3paResults, filename: str = 'profiles', dr: int = 100, samp_ratio: int = 0,
                   data_set: str = 'test', metrics_list: List | None = None, profile_depth: int = 4,
                   open_results: bool = True, port: int = 8000) -> None:
    """
    Vizualization method for the profiles obtained from the MED3pa method.

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
    """
    # Set folder path for the template
    global _save_template_folder, _template_folder
    _save_template_folder = os.path.dirname(filename) + "/" + _template_folder

    assert data_set in ['test'], "Invalid data_set, must be in ['test']"

    if metrics_list is None:
        metrics_list = ['Specificity', 'Sensitivity', 'NPV', 'PPV', 'AUC']

    rendered_html = _generate_tree_html(result=result, samp_ratio=samp_ratio, dr=dr, data_set=data_set,
                                        metrics_list=metrics_list,
                                        max_depth=profile_depth)

    # Save the template with the HTML file
    template_path = os.path.join(os.path.dirname(__file__), _template_folder)
    shutil.copytree(template_path, _save_template_folder, dirs_exist_ok=True)
    # Save the HTML file
    with open(filename + '.html', 'w') as f:
        f.write(rendered_html)
    print(f"Tree visualization generated: '{filename}.html'")

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


def _generate_tree_html(result: Med3paResults, samp_ratio: int, dr: int, data_set: str,
                        metrics_list: Optional[List] = None, max_depth: Optional[int] = None) -> str:
    """
    Generates the tree visualization HTML.

    Args:
        result (Med3paResults): The results of the experiment to visualize.
        samp_ratio (int): The minimum samples ratio in each profile (in percentage).
        dr (int): Declaration rate of predictions for the visualization.
        data_set (str): The name of the data set to visualize. Defaults to 'test', options are 'train', 'valid', 'test'.
        metrics_list (List | None): The list of metrics to visualize. Defaults to None, meaning no metrics are shown.
        max_depth (int): Maximum profile depth.

    Returns:
        str: The HTML string of the visualization tree.
    """
    global _save_template_folder, _template_folder

    # Get the absolute path of the current file's directory
    template_path = os.path.join(os.path.dirname(__file__), _template_folder)
    env = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template('tree.html')

    # Read profiles for the specified data_set
    profiles_to_visualize = _read_tree_section(result=result, samp_ratio=samp_ratio, dr=dr, data_set=data_set)

    # Show only metrics in metrics_list, if specified
    if metrics_list is not None:
        for profile in profiles_to_visualize:
            all_metrics = profile.metrics
            profile.metrics = {key: all_metrics.get(key) for key in metrics_list if key in all_metrics.keys()}

    if max_depth is not None:
        profiles_to_visualize = [profile for profile in profiles_to_visualize if len(profile.path) <= max_depth]

    # Convert Profile objects to dicts before json serialization
    profiles_to_visualize = [vars(profile) for profile in profiles_to_visualize]

    # Render the HTML with the list of nodes and base path
    rendered_html = template.render(
        nodes=profiles_to_visualize,
        base_path=_save_template_folder
    )
    return rendered_html


def _read_tree_section(result: Med3paResults, samp_ratio: int, dr: int, data_set:str = "test") -> List:
    """
    Generates the tree visualization HTML.

    Args:
        result (Med3paResults): The results of the experiment to visualize.
        dr (int): Declaration rate of predictions for the visualization.
        samp_ratio (int): The minimum samples ratio in each profile (in percentage).
        data_set (str): The name of the data set to visualize. Defaults to 'test', options are 'train', 'valid', 'test'.

    Returns:
        List: The profiles to visualize in the HTML file.
    """
    profiles_to_visualize = result.test_record.profiles_manager.profiles_records[samp_ratio][dr]

    # Round condition values
    for profile in profiles_to_visualize:
        profile.path = [re.sub(r'(?<!\w)(\d+\.\d+|\d+)(?!\w)',
                               lambda m: str(round(float(m.group()), 1)), s) for s in profile.path]

    if dr != 100:  # Get results of DR = 100% to compare each metrics
        original_profiles = _read_tree_section(result=result, samp_ratio=samp_ratio, dr=100, data_set=data_set)
        for profile in profiles_to_visualize:
            original_profile_metrics = original_profiles[profile.node_id - 1].metrics
            for metric_name, original_metric_value in original_profile_metrics.items():
                if profile.metrics['metric_name'] is None:
                    profile.metrics[f'diff_{metric_name}'] = None
                else:
                    profile.metrics[f'diff_{metric_name}'] = profile.metrics[metric_name] - original_metric_value

    return profiles_to_visualize
