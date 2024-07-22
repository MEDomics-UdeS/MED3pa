"""
In this module, various strategies to assess the presence of covariate shift are defined. 
Each strategy class, deriving from the original **Disagreement test**, implements a method to evaluate shifts between calibration and testing datasets using different statistical approaches, 
such as empirical cumulative distribution functions (ECDF) and hypothesis tests like the Mann-Whitney U or Kolmogorov-Smirnov tests.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats

from .record import DetectronRecordsManager


class DetectronStrategy:
    """
    Base class for defining various strategies to evaluate the shifts and discrepancies between calibration and testing datasets.

    Methods:
        execute: Must be implemented by subclasses to execute the strategy.
    """
    @staticmethod
    def execute(calibration_records : DetectronRecordsManager, test_records:DetectronRecordsManager):
        pass


class OriginalDisagreementStrategy(DetectronStrategy):
    """
    Implements a strategy to detect disagreement based on the empirical cumulative distribution function (ECDF).
    This strategy assesses the first test run only and returns a dictionary containing the calculated p-value, test run results,
    and statistical measures such as the mean and standard deviation of the calibration tests.
    """
    def execute(calibration_records : DetectronRecordsManager, test_records:DetectronRecordsManager):
        """
        Executes the disagreement detection strategy using the ECDF approach.

        Args:
            calibration_records (DetectronRecordsManager): Manager storing calibration phase records.
            test_records (DetectronRecordsManager): Manager storing test phase records.

        Returns:
            dict: A dictionary containing the p-value, test statistic, baseline mean, baseline standard deviation,
                  and a shift indicator which is True if a shift is detected at the given significance level.
        """
        def ecdf(x):
            """
            Compute the empirical cumulative distribution function.

            Args:
                x (np.ndarray): Array of 1-D numerical data.

            Returns:
                function: A function that takes a value and returns the probability 
                that a random sample from x is less than or equal to that value.
            """
            x = np.sort(x)

            def result(v):
                return np.searchsorted(x, v, side='right') / x.size

            return result

        cal_counts = calibration_records.counts()
        test_count = test_records.counts()[0]
        cdf = ecdf(cal_counts)
        p_value = cdf(test_count).item()

        test_statistic=test_count.item()
        baseline_mean = cal_counts.mean().item()
        baseline_std = cal_counts.std().item()

        results = {
            'p_value':p_value, 
            'test_statistic': test_statistic, 
            'baseline_mean': baseline_mean, 
            'baseline_std': baseline_std
        }
        return results


class MannWhitneyStrategy(DetectronStrategy):
    """
    Implements a strategy to detect disagreement based on the Mann-Whitney U test, assessing the dissimilarity of results
    from calibration runs and test runs.
    """
    def execute(calibration_records: DetectronRecordsManager, test_records:DetectronRecordsManager):
        """
        Executes the disagreement detection strategy using the Mann-Whitney U test.

        Args:
            calibration_records (DetectronRecordsManager): Manager storing calibration phase records.
            test_records (DetectronRecordsManager): Manager storing test phase records.

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
                
        # Perform the Mann-Whitney U test
        u_statistic, p_value = stats.mannwhitneyu(cal_counts, test_counts, alternative='less')
        
        # Calculate the z-scores for the test data
        z_scores = (test_counts[:, None] - cal_counts) / np.std(cal_counts)

        # Define thresholds for categorizing
        def categorize_z_score(z):
            if z <= 0:
                return 'no significant shift'
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

        # Describe the significance of the shift based on the z-score
        significance_description = {
            'unsignificant shift': category_counts.get('no significant shift', 0),
            'small': category_counts.get('small', 0),
            'moderate': category_counts.get('moderate', 0),
            'large': category_counts.get('large', 0)
        }

        results = {
            'p_value': p_value,
            'u_statistic': u_statistic,
            'significance_description' : significance_description
        }

        return results


class KolmogorovSmirnovStrategy(DetectronStrategy):
    """
    Implements a strategy to detect disagreement based on the Kolmogorov-Smirnov test, assessing the dissimilarity of results
    from calibration runs and test runs.
    """
    def execute(calibration_records: DetectronRecordsManager, test_records:DetectronRecordsManager):
        """
        Executes the disagreement detection strategy using the Kolmogorov-Smirnov test.

        Args:
            calibration_records (DetectronRecordsManager): Manager storing calibration phase records.
            test_records (DetectronRecordsManager): Manager storing test phase records.

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
        # Describe the significance of the shift based on the z-score
        significance_description = ""
        if z_score <= 0:
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
            'shift significance' : significance_description
        }

        return results


class EnhancedDisagreementStrategy(DetectronStrategy):
    """
    Implements a strategy to detect disagreement based on the z-score mean difference between calibration and test datasets.
    This strategy calculates the probability of a shift based on the counts where test rejected counts are compared to calibration rejected counts.
    """
    def execute(calibration_records: DetectronRecordsManager, test_records: DetectronRecordsManager, trim_data=True, proportion_to_cut=0.05):
        """
        Executes the disagreement detection strategy using z-score analysis.

        Args:
            calibration_records (DetectronRecordsManager): Manager storing calibration phase records.
            test_records (DetectronRecordsManager): Manager storing test phase records.
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
                return 'no significant shift'
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
        p_value = np.mean(cal_counts < test_statistic)

        # Describe the significance of the shift based on the z-score
        significance_description = {
            'unsignificant shift': category_counts.get('no significant shift', 0),
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

