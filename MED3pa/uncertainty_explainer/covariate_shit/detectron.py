"""

"""
import copy
import numpy as np
import math
import ray
from checkpointer import checkpoint
from typing import Any, Optional
from scipy.stats import mannwhitneyu

from MED3pa.datasets import MaskedDataset
from MED3pa.uncertainty_explainer.abstract_methods import CovariateShiftDetector


class Detectron(CovariateShiftDetector):
    """

    """
    def __init__(self, *,
                 calibration_rounds:int = 100, ensemble_size:int = 5, significance_level:float = 0.05,
                 min_samples_size:int = 20, min_percentage_in_reference: float = 0.5,
                 test_type:str = "disagreement",
                 reference_dataset: Optional[MaskedDataset] = None, f_model: Optional[Any] = None):
        self._reference_dataset = reference_dataset
        self._f_model = f_model
        self.__calibration_rounds = calibration_rounds
        self.__ensemble_size = ensemble_size
        self.__significance_level = significance_level
        self.__min_samples_size = min_samples_size
        self.__min_percentage_in_reference = min_percentage_in_reference
        self.__test_type = test_type

    # @checkpoint(root_path="checkpoints", verbosity=False)
    def _call__impl(self, x_q:np.ndarray, profile_path:Optional[str] = None):
        if profile_path is None:
            profile_path = ["*"]

        assert self.context_is_set(), ("Reference set and base model must be set before applying the "
                                           "Detectron test.")
        print(f"Executing Detectron test in profile: {profile_path}")

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        N_test = len(x_q)
        x_p = self._reference_dataset.get_observations()

        # filter out only profile in reference
        reference_profile_idx = np.where(self._reference_dataset.mask_by_profile(path=profile_path))[0]
        N_reference = len(reference_profile_idx)
        # The number of samples to include (from reference and test sets) for testing at each round
        N_samples = min(math.floor(self.__min_percentage_in_reference * N_reference), N_test)
        if N_samples < self.__min_samples_size:
            return f"Not enough samples in {'test' if N_test==N_samples else 'reference'}: {N_samples}"

        f_model_ref = ray.put(self._f_model)
        futures = []

        # CDCs tests on reference set
        records = []
        for k in range(self.__calibration_rounds):
            # Sample patients from the reference set
            rng = np.random.default_rng(seed=k)
            sampled_reference_idx = rng.choice(reference_profile_idx, size=N_samples, replace=False)
            train_idx = np.setdiff1d(np.arange(len(x_p)), sampled_reference_idx)

            # set data for experiments
            x_train = x_p[train_idx]
            y_train = self._reference_dataset.get_true_labels()[train_idx]
            x_reference = x_p[sampled_reference_idx]
            x_test = x_q[rng.choice(x_q.shape[0], N_samples, replace=False)]

            for name, x_disagree in [["p", x_reference],
                                     ["q", x_test]]:
                futures.append(_constrained_disagreement.remote(x_agree=x_train, y_agree=y_train, x_disagree=x_disagree,
                                                            f_model=f_model_ref, ensemble_size=self.__ensemble_size))
                records.append({"seed": k, "name": name})

        results = ray.get(futures)

        for i in range(len(records)):
            disagreements, entropies = results[i]
            records[i]["disagreements"] = disagreements
            records[i]["entropies"] = entropies

            # disagreements_ref, entropies_ref = self.__constrained_disagreement(x_agree=x_train, y_agree=y_train,
            #                                                                    x_disagree=x_reference)
            # disagreements_test, entropies_test = self.__constrained_disagreement(x_agree=x_train, y_agree=y_train,
            #                                                                      x_disagree=x_test)
            # for disagreements, entropies, name in [[disagreements_ref, entropies_ref, 'p'],
            #                                        [disagreements_test, entropies_test, 'q']]:
            #     records.append({
            #         "seed": k,
            #         "name": name,
            #         "disagreements": disagreements,
            #         "entropies": entropies
            #     })

        return self.__test_shift(records=records), records

    def __constrained_disagreement(self, x_agree, y_agree, x_disagree):
        y_disagree = 1 - self._f_model.predict(x_disagree)
        n_total = len(x_disagree)

        disagreements = []
        entropies = []

        for i in range(self.__ensemble_size):
            # Generate learning data
            pq_data = np.concatenate([x_agree, x_disagree])
            pq_targets = np.concatenate([y_agree, y_disagree])

            weights = np.concatenate([
                np.ones(len(y_agree)),
                np.ones(len(y_disagree)) / (len(y_disagree) + 1)
            ])

            clf = copy.deepcopy(self._f_model)
            clf.fit(pq_data, pq_targets, sample_weight=weights)

            # Filter data that the cdc disagreed on
            predicted_probs = clf.predict_proba(x_disagree)[:, 1]
            mask = clf.predict(x_disagree) == self._f_model.predict(x_disagree)
            x_disagree = x_disagree[mask]
            y_disagree = y_disagree[mask]
            n_agree = len(x_disagree)

            # Add results
            disagreements.append(1-n_agree/n_total)

            # avoid log(0) issues
            eps = 1e-12
            p = np.clip(predicted_probs, eps, 1 - eps)
            entropy = - (p * np.log(p) + (1 - p) * np.log(1 - p))
            mean_entropy = entropy.mean()
            entropies.append(mean_entropy)

            if len(x_disagree) == 0:
                break

        return disagreements, entropies

    def __test_shift(self, records):
        # Separate reference and test records
        ref_records = [r for r in records if r["name"] == "p"]
        test_records = [r for r in records if r["name"] == "q"]

        ref_stats_dis = np.array([r["disagreements"][-1] for r in ref_records])
        test_stats_dis = np.array([r["disagreements"][-1] for r in test_records])
        test_stat_dis = np.mean(test_stats_dis)
        p_value_dis = np.mean(ref_stats_dis >= test_stat_dis)

        ref_stats_ent = np.array([np.mean(r["entropies"]) for r in ref_records])
        test_stats_ent = np.array([np.mean(r["entropies"]) for r in test_records])
        statistic_ent, p_value_ent = mannwhitneyu(
            test_stats_ent,
            ref_stats_ent,
            alternative="greater"
        )

        min_p_value = min(p_value_dis, p_value_ent)

        if max(p_value_dis, p_value_ent) < self.__significance_level:
            decision = "shift by both"
        elif p_value_dis < self.__significance_level:
            decision = "shift by disagreement"
        elif p_value_ent < self.__significance_level:
            decision = "shift by entropy"
        else:
            decision = "No shift"

        return {
            "decision": decision,
            "p_value": float(min_p_value),
        }

    def __test_shift_specific(self, records):
        # Separate reference and test records
        ref_records = [r for r in records if r["name"] == "p"]
        test_records = [r for r in records if r["name"] == "q"]

        if self.__test_type == "disagreement":
            ref_stats = np.array([r["disagreements"][-1] for r in ref_records])
            test_stats = np.array([r["disagreements"][-1] for r in test_records])
            test_stat = np.mean(test_stats)
            p_value = np.mean(ref_stats >= test_stat)
        elif self.__test_type == "entropy":
            ref_stats = np.array([np.mean(r["entropies"]) for r in ref_records])
            test_stats = np.array([np.mean(r["entropies"]) for r in test_records])
            statistic, p_value = mannwhitneyu(
                test_stats,
                ref_stats,
                alternative="greater"
            )

        return {
            "decision": "shift" if p_value < self.__significance_level else "no shift",
            "p_value": float(p_value),
            "reference_mean": float(np.mean(ref_stats)),
            "test_mean": float(np.mean(test_stats)),
            "reference_std": float(np.std(ref_stats, ddof=1)),
            "test_std": float(np.std(test_stats, ddof=1)),
            "reference_statistics": ref_stats.tolist(),
            "test_statistics": test_stats.tolist(),
        }


@ray.remote
def _constrained_disagreement(x_agree, y_agree, x_disagree, f_model, ensemble_size):
    y_disagree = 1 - f_model.predict(x_disagree)
    n_total = len(x_disagree)

    disagreements = []
    entropies = []

    for i in range(ensemble_size):
        pq_data = np.concatenate([x_agree, x_disagree])
        pq_targets = np.concatenate([y_agree, y_disagree])

        weights = np.concatenate([
            np.ones(len(y_agree)),
            np.ones(len(y_disagree)) / (len(y_disagree) + 1)
        ])

        clf = copy.deepcopy(f_model)
        clf.fit(pq_data, pq_targets, sample_weight=weights)

        predicted_probs = clf.predict_proba(x_disagree)[:, 1]
        mask = clf.predict(x_disagree) == f_model.predict(x_disagree)
        x_disagree = x_disagree[mask]
        y_disagree = y_disagree[mask]
        n_agree = len(x_disagree)

        disagreements.append(1 - n_agree / n_total)

        eps = 1e-12
        p = np.clip(predicted_probs, eps, 1 - eps)
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        entropies.append(float(entropy.mean()))

        if len(x_disagree) == 0:
            break

    return disagreements, entropies