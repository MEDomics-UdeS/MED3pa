"""
this module defines multiple strategies to assess the presence of covariate shift
"""
from det3pa.detectron.record import DetectronRecordsManager
import numpy as np
import scipy.stats as stats
import pandas as pd

def ecdf(x):
    """
    Compute the empirical cumulative distribution function
    :param x: array of 1-D numerical data
    :return: a function that takes a value and returns the probability that a random sample from x is less than or equal to that value
    """
    x = np.sort(x)

    def result(v):
        return np.searchsorted(x, v, side='right') / x.size

    return result

class DetectronStrategy:
    """
    Base class for defining various strategies to evaluate the shifts and discrepancies between calibration and testing datasets.

    Methods:
        execute: Must be implemented by subclasses to execute the strategy.
    """
    @staticmethod
    def execute(calibration_records : DetectronRecordsManager, test_records:DetectronRecordsManager, significance_level):
        pass
    def evaluate(calibration_record: DetectronRecordsManager,
                        test_record: DetectronRecordsManager,
                        alpha=0.05,
                        max_ensemble_size=None):
        pass

class DisagreementStrategy(DetectronStrategy):
    """
    Implements a strategy to detect disagreement based on the empirical cumulative distribution function (ECDF).
    This strategy assesses the first test run only and returns a dictionary containing the calculated p-value, test run results,
    and statistical measures such as the mean and standard deviation of the calibration tests.
    """
    def execute(calibration_records : DetectronRecordsManager, test_records:DetectronRecordsManager, significance_level):
        """
        Executes the disagreement detection strategy using the ECDF approach.

        Args:
            calibration_records (DetectronRecordsManager): Manager storing calibration phase records.
            test_records (DetectronRecordsManager): Manager storing test phase records.
            significance_level (float): The statistical significance level used for tests.

        Returns:
            dict: A dictionary containing the p-value, test statistic, baseline mean, baseline standard deviation,
                  and a shift indicator which is True if a shift is detected at the given significance level.
        """
        cal_counts = calibration_records.counts()
        test_count = test_records.counts()[0]
        cdf = ecdf(cal_counts)
        p_value = cdf(test_count).item()

        test_statistic=test_count.item()
        baseline_mean = cal_counts.mean().item()
        baseline_std = cal_counts.std().item()

        results = {'p_value':p_value, 'test_statistic': test_statistic, 'baseline_mean': baseline_mean, 'baseline_std': baseline_std, 'shift_indicator': (p_value < significance_level)}
        return results

    def evaluate(calibration_record: DetectronRecordsManager,
                        test_record: DetectronRecordsManager,
                        alpha=0.05,
                        max_ensemble_size=None):

        cal_counts = calibration_record.counts(max_ensemble_size=max_ensemble_size)
        test_counts = test_record.counts(max_ensemble_size=max_ensemble_size)
        N = calibration_record.sample_size
        assert N == test_record.sample_size, 'The sample sizes of the calibration and test runs must be the same'

        fpr = (cal_counts <= np.arange(0, N + 2)[:, None]).mean(1)
        tpr = (test_counts <= np.arange(0, N + 2)[:, None]).mean(1)

        quantile = np.quantile(cal_counts, alpha)
        tpr_low = (test_counts < quantile).mean()
        tpr_high = (test_counts <= quantile).mean()

        fpr_low = (cal_counts < quantile).mean()
        fpr_high = (cal_counts <= quantile).mean()

        if fpr_high == fpr_low:
            tpr_at_alpha = tpr_high
        else:  # use linear interpolation if there is no threshold at alpha
            tpr_at_alpha = (tpr_high - tpr_low) / (fpr_high - fpr_low) * (alpha - fpr_low) + tpr_low

        return dict(power=tpr_at_alpha, auc=np.trapz(tpr, fpr), N=N)

class DisagreementStrategy_MW(DetectronStrategy):
    """
    Implements a strategy to detect disagreement based on the Mann-Whitney U test, assessing the dissimilarity of results
    from calibration runs and test runs.
    """
    def execute(calibration_records: DetectronRecordsManager, test_records:DetectronRecordsManager, significance_level):
        """
        Executes the disagreement detection strategy using the Mann-Whitney U test.

        Args:
            calibration_records (DetectronRecordsManager): Manager storing calibration phase records.
            test_records (DetectronRecordsManager): Manager storing test phase records.
            significance_level (float): The statistical significance level used for tests.

        Returns:
            dict: A dictionary containing the calculated p-value, U statistic, z-score quantifying the shift intensity,
                  and a shift indicator based on the significance level.
        """
        # Retrieve count data from both calibration and test records
        cal_counts = calibration_records.rejected_counts()
        test_counts = test_records.rejected_counts()
        
        cal_mean = np.mean(cal_counts)
        cal_std = np.std(cal_counts)
        test_mean = np.mean(test_counts)
        # Combine both groups for ranking
        combined_counts = np.concatenate((cal_counts, test_counts))
        ranks = stats.rankdata(combined_counts)
        
        # Separate the ranks back into two groups
        cal_ranks = ranks[:len(cal_counts)]
        test_ranks = ranks[len(cal_counts):]
        
        # Calculate mean and standard deviation of ranks for both groups
        cal_rank_mean = np.mean(cal_ranks)
        cal_rank_std = np.std(cal_ranks)
        test_rank_mean = np.mean(test_ranks)
        test_rank_std = np.std(test_ranks)

        # Perform the Mann-Whitney U test
        u_statistic, p_value = stats.mannwhitneyu(cal_counts, test_counts, alternative='less')
        z_score = (test_mean - cal_mean) / cal_std

        # Describe the significance of the shift based on the z-score
        significance_description = ""
        if p_value > significance_level:
            significance_description = "no significant shift"
        elif abs(z_score) < 1.0:
            significance_description = "Small"
        elif abs(z_score) < 2.0:
            significance_description = "Moderate"
        elif abs(z_score) < 3.0:
            significance_description = "Large"
        else:
            significance_description = "Very large"
        # Results dictionary including rank statistics
        results = {
            'p_value': p_value,
            'test_statistic': u_statistic,
            'z-score':z_score,
            'shift_indicator': (p_value <= significance_level),
            'shift significance' : significance_description
        }

        return results

class DisagreementStrategy_KS(DetectronStrategy):
    """
    Implements a strategy to detect disagreement based on the Kolmogorov-Smirnov test, assessing the dissimilarity of results
    from calibration runs and test runs.
    """
    def execute(calibration_records: DetectronRecordsManager, test_records:DetectronRecordsManager, significance_level):
        """
        Executes the disagreement detection strategy using the Kolmogorov-Smirnov test.

        Args:
            calibration_records (DetectronRecordsManager): Manager storing calibration phase records.
            test_records (DetectronRecordsManager): Manager storing test phase records.
            significance_level (float): The statistical significance level used for tests.

        Returns:
            dict: A dictionary containing the calculated p-value, KS statistic, and a shift indicator which is True
                  if a shift is detected at the given significance level.
        """
        # Retrieve count data from both calibration and test records
        cal_counts = calibration_records.rejected_counts()
        test_counts = test_records.rejected_counts()
        
        # Perform the Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(cal_counts, test_counts)

        # Calculate statistics for interpretation
        cal_mean = cal_counts.mean()
        cal_std = cal_counts.std()
        test_mean = test_counts.mean()
        test_std = test_counts.std()
        
        z_score = (test_mean - cal_mean) / cal_std
        shift_indicator = (p_value < significance_level) & (test_mean > cal_mean)
        # Describe the significance of the shift based on the z-score
        significance_description = ""
        if shift_indicator is False:
            significance_description = "no significant shift"
        elif abs(z_score) < 1.0:
            significance_description = "Small"
        elif abs(z_score) < 2.0:
            significance_description = "Moderate"
        elif abs(z_score) < 3.0:
            significance_description = "Large"
        else:
            significance_description = "Very Large"
        # Results dictionary including rank statistics
        # Results dictionary including KS test results and distribution statistics
        results = {
            'p_value': p_value,
            'ks_statistic': ks_statistic,
            'z-score':z_score,
            'shift_indicator': (p_value < significance_level) & (test_mean > cal_mean),
            'shift significance' : significance_description
        }

        return results

class DisagreementStrategy_quantile(DetectronStrategy):
    """
    Implements a quantile-based strategy to detect significant shifts between the calibration and test datasets.
    This strategy evaluates the quantile threshold exceeded by the mean of rejected counts from the test records.
    """
    def execute(calibration_records: DetectronRecordsManager, test_records:DetectronRecordsManager, significance_level):
        """
        Executes the disagreement detection strategy based on quantile thresholds.

        Args:
            calibration_records (DetectronRecordsManager): Manager storing calibration phase records.
            test_records (DetectronRecordsManager): Manager storing test phase records.
            significance_level (float): The significance level used to define the quantile threshold.

        Returns:
            dict: A dictionary containing the calculated quantile, the mean of the test rejected counts, the test quantile,
                  and a shift indicator which is True if the test rejected count mean exceeds the calibration quantile threshold.
        """
        # Retrieve count data from both calibration and test records
        cal_rejected_counts = calibration_records.rejected_counts()
        test_rejected_count = test_records.rejected_counts().mean()
        quantile = calibration_records.rejected_count_quantile(1-significance_level, None)
        test_quantile = test_records.rejected_count_quantile(1-significance_level, None)
        shift_indicator = test_rejected_count > quantile
        # Results dictionary including KS test results and distribution statistics
        results = {
            'quantile': quantile,
            'test_mean' : test_rejected_count,
            'test_quantile' : test_quantile,
            'shift_indicator': shift_indicator
        }

        return results
    
class DisagreementStrategy_z_mean(DetectronStrategy):
    """
    Implements a strategy to detect disagreement based on the z-score mean difference between calibration and test datasets.
    This strategy calculates the probability of a shift based on the counts where test rejected counts are compared to calibration rejected counts.
    """
    def execute(calibration_records: DetectronRecordsManager, test_records: DetectronRecordsManager, significance_level=0.05, trim_data=False, proportion_to_cut=0.05):
        """
        Executes the disagreement detection strategy using z-score analysis.

        Args:
            calibration_records (DetectronRecordsManager): Manager storing calibration phase records.
            test_records (DetectronRecordsManager): Manager storing test phase records.
            significance_level (float): The significance level used for statistical testing.
            trim_data (bool): Whether to trim the data using a specified proportion to cut.
            proportion_to_cut (float): The proportion of data to cut from both ends if trimming is enabled.

        Returns:
            dict: A dictionary containing the calculated shift probability, test statistic, baseline mean, baseline standard deviation,
                  and a description of the shift significance.
        """
        cal_counts = np.array(calibration_records.rejected_counts())
        test_counts = np.array(test_records.rejected_counts())

        # Ensure there are enough records to perform bootstrap
        if len(cal_counts) < 2 or len(test_counts) == 0:
            raise ValueError("Not enough records to perform the statistical test.")

        def trim_dataset(data, proportion_to_cut):
            if not 0 <= proportion_to_cut < 0.5:
                raise ValueError("proportion_to_cut must be between 0 and 0.5")
            
            data_sorted = np.sort(data)
            n = len(data)
            trim_count = int(n * proportion_to_cut)
            
            return data_sorted[trim_count:n - trim_count]

        if trim_data:
            # Trim calibration and test data if trimming is enabled
            cal_counts = trim_dataset(cal_counts, proportion_to_cut)
            test_counts = trim_dataset(test_counts, proportion_to_cut)

        # Calculate the baseline mean and standard deviation on trimmed or full data
        baseline_mean = np.mean(cal_counts)
        test_mean = np.mean(test_counts)
        baseline_std = np.std(cal_counts)
        test_std = np.std(test_counts)

        # Calculate the test statistic (mean of test data)
        test_statistic = np.mean(test_counts)

        # Calculate the z-scores for the test data
        z_scores = (test_counts[:, None] - cal_counts) / np.std(cal_counts)

        # Define thresholds for categorizing
        def categorize_z_score(z):
            if z <= 0:
                return 'no shift'
            elif abs(z) < 1:
                return 'small'
            elif abs(z) < 2:
                return 'moderate'
            else:
                return 'large'

        # Categorize each test count based on its z-score
        categories = np.array([categorize_z_score(z) for z in z_scores.flatten()])
        # Calculate the percentage of each category
        category_counts = pd.Series(categories).value_counts(normalize=True) * 100

        # Calculate the one-tailed p-value (test_statistic > baseline_mean)
        p_value = np.sum(cal_counts < test_statistic) / len(cal_counts)

        # Describe the significance of the shift based on the z-score
        significance_description = {
            'no shift': category_counts.get('no shift', 0),
            'small': category_counts.get('small', 0),
            'moderate': category_counts.get('moderate', 0),
            'large': category_counts.get('large', 0)
        }

        results = {
            'shift_probability': p_value,
            'test_statistic': test_statistic,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'significance_description': significance_description,
        }
        return results

    