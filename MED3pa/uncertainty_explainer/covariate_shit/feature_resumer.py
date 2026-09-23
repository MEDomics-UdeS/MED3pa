import numpy as np
from typing import Optional, List, Dict

from MED3pa.datasets import MaskedDataset
from MED3pa.uncertainty_explainer.abstract_methods import CovariateShiftDetector


class FeatureResumer(CovariateShiftDetector):
    """
    Computes mean and std for selected features in both:
      - reference dataset (filtered by profile)
      - test dataset x_q

    Returns results formatted as:
        mean_xq ± std_xq   (mean_ref ± std_ref)
    """

    def __init__(self, *,
                 features: List[str] | str,
                 reference_dataset: Optional[MaskedDataset] = None):
        self._reference_dataset = reference_dataset
        features = features if isinstance(features, list) else [features]
        features = [
            "_".join(col.strip().split())  # collapse multiple spaces, strip, replace with _
            for col in features
        ]
        self._features = features
        self._f_model = {}

    def _call__impl(self, x_q: np.ndarray, profile_path: Optional[str] = None):
        if profile_path is None:
            profile_path = ["*"]

        assert self.context_is_set(), (
            "Reference dataset must be set before applying FeatureResumer."
        )

        print(f"Executing FeatureResumer in profile: {profile_path}")

        # --- Reference dataset extraction ---
        x_ref = self._reference_dataset.get_observations()

        # Filter rows belonging to the profile
        ref_idx = np.where(self._reference_dataset.mask_by_profile(path=profile_path))[0]
        if len(ref_idx) == 0:
            return {"error": "No samples found in specified profile."}

        x_ref_profile = x_ref[ref_idx]

        # --- Feature index resolution ---
        feature_indices = []
        for f in self._features:
            if f not in self._reference_dataset.features:
                raise ValueError(f"Feature '{f}' not found in dataset.")
            feature_indices.append(self._reference_dataset.features.index(f))

        # --- Compute statistics ---
        summary: Dict[str, Dict[str, float|str]] = {}

        for f, col_idx in zip(self._features, feature_indices):
            # Reference stats
            ref_vals = x_ref_profile[:, col_idx]
            mean_ref = float(np.mean(ref_vals))
            std_ref = float(np.std(ref_vals, ddof=1))

            # Test stats
            test_vals = x_q[:, col_idx]
            mean_test = float(np.mean(test_vals))
            std_test = float(np.std(test_vals, ddof=1))

            # Save formatted result
            summary[f] = {
                "mean_ref": mean_ref,
                "std_ref": std_ref,
                "mean_test": mean_test,
                "std_test": std_test,
                "formatted": f"{mean_test:.4f} ± {std_test:.4f}  ({mean_ref:.4f} ± {std_ref:.4f})"
            }

        # Build final multi-line string
        lines = []
        for f, stats in summary.items():
            lines.append(f"{f}: {stats['formatted']}")

        final_string = "\n".join(lines)

        return final_string
