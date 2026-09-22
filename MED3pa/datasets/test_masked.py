import numpy as np
import pytest

from MED3pa.datasets.masked import MaskedDataset


@pytest.fixture
def sample_dataset():
    observations = np.array([[1, 2], [3, 4]])
    true_labels = np.array([0, 1])
    column_labels = ['feature_1', 'feature_2']
    return MaskedDataset(observations=observations, true_labels=true_labels, column_labels=column_labels)


def test_initialization(sample_dataset):
    assert sample_dataset.get_observations().shape == (2, 2)
    assert np.array_equal(sample_dataset.get_true_labels(), np.array([0, 1]))
    assert sample_dataset.get_column_labels() == ['feature_1', 'feature_2']
    assert len(sample_dataset) == 2


def test_getitem(sample_dataset):
    x, y_hat, y = sample_dataset[0]
    assert np.array_equal(x, np.array([1, 2]))
    assert y_hat is None
    assert y == 0

    x, y_hat, y = sample_dataset[1]
    assert np.array_equal(x, np.array([3, 4]))
    assert y_hat is None
    assert y == 1


def test_refine(sample_dataset):
    mask = np.array([True, False])
    refined_length = len(sample_dataset.refine(mask).get_true_labels())
    assert refined_length == 1
    assert len(sample_dataset) == 1
    x, _, _ = sample_dataset[0]
    assert np.array_equal(x, np.array([1, 2]))


def test_reset_indices(sample_dataset):
    mask = np.array([True, False])
    sample_dataset.refine(mask)
    sample_dataset.reset_indices()
    assert len(sample_dataset) == 2
    x, _, _ = sample_dataset[0]
    assert np.array_equal(x, np.array([1, 2]))


def test_sample_uniform(sample_dataset):
    sampled_set = sample_dataset.sample_uniform(n_samples=1, seed=42)
    assert len(sampled_set) == 1


def test_sample_random(sample_dataset):
    sampled_set = sample_dataset.sample_random(n_samples=1, seed=42)
    assert len(sampled_set) == 1


def test_getters(sample_dataset):
    assert np.array_equal(sample_dataset.get_observations(), np.array([[1, 2], [3, 4]]))
    assert np.array_equal(sample_dataset.get_true_labels(), np.array([0, 1]))
    sample_dataset.set_pseudo_probs_labels(np.array([[0.8], [0.2]]), threshold=0.5)
    assert np.array_equal(sample_dataset.get_pseudo_labels(), np.array([True, False]))


def test_setters(sample_dataset):
    sample_dataset.set_confidence_scores(np.array([0.9, 0.1]))
    assert np.array_equal(sample_dataset.get_confidence_scores(), np.array([0.9, 0.1]))
    sample_dataset.set_pseudo_labels(np.array([False, True]))
    assert np.array_equal(sample_dataset.get_pseudo_labels(), np.array([False, True]))


def test_clone(sample_dataset):
    cloned_set = sample_dataset.clone()
    assert np.array_equal(cloned_set.get_observations(), np.array([[1, 2], [3, 4]]))
    assert np.array_equal(cloned_set.get_true_labels(), np.array([0, 1]))
    sample_dataset.set_pseudo_probs_labels(np.array([[0.8], [0.2]]), threshold=0.5)
    cloned_set = sample_dataset.clone()
    assert np.array_equal(cloned_set.get_pseudo_labels(), np.array([True, False]))


def test_get_info(sample_dataset):
    info = sample_dataset.get_info()
    expected_info = {
        'file_path': None,
        'num_samples': 2,
        'num_observations': 2,
        'has_pseudo_labels': False,
        'has_pseudo_probabilities': False,
        'has_confidence_scores': False,
    }
    assert info == expected_info


def test_summarize(capsys, sample_dataset):
    sample_dataset.summarize()
    captured = capsys.readouterr().out.strip()
    expected_output = (
        "Number of samples: 2\n"
        "Number of observations: 2\n"
        "Has pseudo labels: False\n"
        "Has pseudo probabilities: False\n"
        "Has confidence scores: False"
    )
    assert captured == expected_output


def test_to_dataframe(sample_dataset):
    df = sample_dataset.to_dataframe()
    assert len(df) == 2
    assert list(df.columns) == ['feature_1', 'feature_2', 'true_labels']
    sample_dataset.set_pseudo_probs_labels(np.array([[0.8], [0.2]]), threshold=0.5)
    df = sample_dataset.to_dataframe()
    assert list(df.columns) == ['feature_1', 'feature_2', 'true_labels', 'pseudo_labels']


def test_save_to_csv(tmp_path, sample_dataset):
    csv_file = tmp_path / "test_dataset.csv"
    sample_dataset.save_to_csv(str(csv_file))
    df_read = pd.read_csv(csv_file)
    assert len(df_read) == 2
    assert list(df_read.columns) == ['feature_1', 'feature_2', 'true_labels']