"""
The mdr_visualization.py module manages visualization methods for the Metrics by Declaration Rates (MDR) curves.
"""

import matplotlib.pyplot as plt
import os
from typing import List, Optional

from MED3pa.med3pa.results import Med3paResults


def visualize_mdr(result: Med3paResults, filename: str = 'mdr', linewidth: int = 1, metrics: Optional[List[str]] = None,
                  dr: Optional[int] = None, save: bool = True, show: bool = True, save_format: str = 'svg') -> None:
    """
    Visualizes the MDR curves, and saves the plot if save is True.

    Args:
        result (Med3paResults): The results of the experiment to visualize.
        filename (str): The name of the file to be saved. Defaults to 'mdr'.
        linewidth (int): The width of the lines in the plot. Defaults to 1.
        metrics (List[str], optional): List of metrics to add in the plot. Defaults to None, which means all available
        metrics.
        dr (int, optional): Declaration rate applied to predictions. If specified, adds a line on the plot to the
        corresponding declaration rate.
        save (bool): Whether to save the plot. Defaults to True.
        show (bool): Whether to show the plot in the terminal. Defaults to True.
        save_format (str): The format of the saved plot. Defaults to 'svg'.

    """
    mdr_values = result.test_record.metrics_by_dr

    declaration_rates = sorted(map(int, mdr_values.keys()))

    if metrics is None:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1Score', 'Specificity', 'Sensitivity', 'Auc', 'NPV', 'PPV']

    for metric in metrics:
        values = []
        for dr in declaration_rates:
            if metric in mdr_values[dr]['metrics']:
                values.append(mdr_values[dr]['metrics'][metric])
            elif metric in ['Positive%', 'population_percentage', 'min_confidence_level', 'mean_confidence_level']:
                values.append(mdr_values[dr][metric])
            else:
                values.append(None)  # Handle missing values

        plt.plot(declaration_rates, values, label=metric, linewidth=linewidth)

    # If the dr parameter is different from None or 100, add the vertical line
    if dr is not None and dr != 100:
        plt.axvline(x=dr, color='k', linestyle='--', linewidth=linewidth)

    # Plot parameters
    plt.xlabel("Declaration Rate")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.title("Metrics vs Declaration Rate")
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=2)
    if save:
        # Create the directory if it doesn't exist
        os.makedirs(filename, exist_ok=True)
        plt.savefig(f"{filename}.{save_format}", format=save_format)
    if show:
        plt.show()

    plt.close()
