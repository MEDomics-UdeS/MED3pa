import numpy as np
import xgboost as xgb
from MED3pa.datasets.masked import MaskedDataset
from MED3pa.uncertainty_explainer.covariate_shit.detectron import Detectron

from MED3pa.detectron.data.sample_data.uci import train_and_test
from MED3pa.detectron.data.sample_data import uci_heart_numpy

train_and_test(n_seeds=1)
# BASE_MODEL = xgb_trained_on_uci_heart(seed=0)

def create_mock_dataset(observations, true_labels, column_labels=None):
    return MaskedDataset(observations=observations, true_labels=true_labels, column_labels=column_labels)

def test_detectron_with_covariate_shift():
    # Create mock datasets
    reference_observations = np.array([[0.1, 0.2], [0.3, 0.4]])
    reference_true_labels = np.array([0, 1])
    test_observations_shifted = np.array([[0.5, 0.6], [0.7, 0.8]])  # Shifted data
    test_true_labels_shifted = np.array([0, 1])

    reference_dataset = create_mock_dataset(reference_observations, reference_true_labels)
    test_dataset = create_mock_dataset(test_observations_shifted, test_true_labels_shifted)

    # Create a mock model (dummy model for testing purposes)
    class DummyModel:
        def predict(self, x):
            return np.array([0, 1])

        def fit(self, *args, **kwargs):
            pass

        def predict_proba(self, x):
            return np.array([[0.9, 0.1], [0.2, 0.8]])

    f_model = DummyModel()

    # Instantiate Detectron
    detectron = Detectron(reference_dataset=reference_dataset, f_model=f_model)

    # Test with shifted data (covariate shift)
    result = detectron(x_q=test_observations_shifted)
    assert result["decision"] == "shift", f"Expected 'shift' but got {result['decision']}"

def test_detectron_without_covariate_shift():
    # Create mock datasets
    reference_observations = np.array([[0.1, 0.2], [0.3, 0.4]])
    reference_true_labels = np.array([0, 1])
    test_observations_no_shift = np.array([[0.15, 0.25], [0.25, 0.35]])  # Similar to reference data
    test_true_labels_no_shift = np.array([0, 1])

    reference_dataset = create_mock_dataset(reference_observations, reference_true_labels)
    test_dataset = create_mock_dataset(test_observations_no_shift, test_true_labels_no_shift)

    # Create a mock model (dummy model for testing purposes)
    class DummyModel:
        def predict(self, x):
            return np.array([0, 1])

        def fit(self, *args, **kwargs):
            pass

        def predict_proba(self, x):
            return np.array([[0.9, 0.1], [0.2, 0.8]])

    f_model = DummyModel()

    # Instantiate Detectron
    detectron = Detectron(reference_dataset=reference_dataset, f_model=f_model)

    # Test with similar data (no covariate shift)
    result = detectron(x_q=test_observations_no_shift)
    assert result["decision"] == "no shift", f"Expected 'no shift' but got {result['decision']}"

def test_detectron_with_small_reference_dataset():
    # Create mock datasets
    reference_observations = np.array([[0.1, 0.2]])
    reference_true_labels = np.array([0])
    test_observations_shifted = np.array([[0.5, 0.6]])  # Shifted data
    test_true_labels_shifted = np.array([0])

    reference_dataset = create_mock_dataset(reference_observations, reference_true_labels)
    test_dataset = create_mock_dataset(test_observations_shifted, test_true_labels_shifted)

    # Create a mock model (dummy model for testing purposes)
    class DummyModel:
        def predict(self, x):
            return np.array([0])

        def fit(self, *args, **kwargs):
            pass

        def predict_proba(self, x):
            return np.array([[0.9]])

    f_model = DummyModel()

    # Instantiate Detectron
    detectron = Detectron(reference_dataset=reference_dataset, f_model=f_model)

    # Test with shifted data (covariate shift)
    result = detectron(x_q=test_observations_shifted)
    assert "Not enough samples" in result, f"Expected 'Not enough samples' but got {result}"

def test_detectron_with_small_test_dataset():
    # Create mock datasets
    reference_observations = np.array([[0.1, 0.2], [0.3, 0.4]])
    reference_true_labels = np.array([0, 1])
    test_observations_no_shift = np.array([[0.15, 0.25]])  # Similar to reference data
    test_true_labels_no_shift = np.array([0])

    reference_dataset = create_mock_dataset(reference_observations, reference_true_labels)
    test_dataset = create_mock_dataset(test_observations_no_shift, test_true_labels_no_shift)

    # Create a mock model (dummy model for testing purposes)
    class DummyModel:
        def predict(self, x):
            return np.array([0])

        def fit(self, *args, **kwargs):
            pass

        def predict_proba(self, x):
            return np.array([[0.9]])

    f_model = DummyModel()

    # Instantiate Detectron
    detectron = Detectron(reference_dataset=reference_dataset, f_model=f_model)

    # Test with similar data (no covariate shift)
    result = detectron(x_q=test_observations_no_shift)
    assert "Not enough samples" in result, f"Expected 'Not enough samples' but got {result}"

if __name__ == "__main__":
    DATA_NUMPY = uci_heart_numpy()
    N = 40
    count = 0
    n_runs = 4
    all_res = []

    for seed in range(n_runs):
        # collect either p or q data and filter it to size N using random seed
        BASE_MODEL = xgb.XGBClassifier()

        p = DATA_NUMPY['iid_test_data'], DATA_NUMPY['iid_test_labels'][0]
        q = DATA_NUMPY['ood_test_data'], DATA_NUMPY['ood_test_labels'][0]

        BASE_MODEL.fit(p[0], p[1])

        # randomly sample N elements from q
        idx = np.random.RandomState(seed).permutation(len(q[0]))[:N]
        q = q[0][idx, :], q[1][idx]
        # res = detectron_tst(train=train, val=val, q=q)

        p_data = MaskedDataset(p[0], p[1])
        q_data = MaskedDataset(q[0], q[1])

        detect = Detectron(reference_dataset=p_data, f_model=BASE_MODEL, test_type="entropy")

        res, records = detect(x_q=q[0])
        print(res)
        all_res.append(records)

        # for r in res:
        #     r.update({'seed': seed, 'N': N})
        #     for k, v in r.items():
        #         if isinstance(v, np.ndarray):
        #             r[k] = torch.from_numpy(v)
        # print(f'{dataset_name} Rejection rate {N=}: {res[-1]["rejection_rate"]:.2f}')
        # torch.save(res, f'results/{args.run_name}/test_{seed}_{dataset_name}_{N}.pt')

        count += 1
        print(f'Run {count}/{n_runs} complete')
