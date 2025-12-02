"""
Data Examiner for automatic data profiling and characterization.

This module analyzes data characteristics without requiring LLM calls,
providing a statistical profile that informs method selection.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Union, Optional, List, Tuple
import warnings


class DataExaminer:
    """
    Examines data and creates comprehensive profile for analysis planning.
    
    The DataExaminer performs purely statistical analysis to understand
    data characteristics, detect patterns, and identify potential issues.
    
    Parameters
    ----------
    data : array_like or DataFrame
        Data to examine
    verbose : bool, optional
        Whether to print examination progress (default: False)
    
    Attributes
    ----------
    data : np.ndarray
        Cleaned data array
    profile : dict
        Comprehensive data profile
    
    Examples
    --------
    >>> examiner = DataExaminer(data)
    >>> profile = examiner.examine()
    >>> print(profile['data_type'])
    'continuous'
    """
    
    def __init__(self, data: Union[np.ndarray, pd.Series, pd.DataFrame, list], 
                 verbose: bool = False):
        """Initialize the data examiner."""
        self.verbose = verbose
        self.original_data = data
        self.data = self._prepare_data(data)
        self.profile = {}
        
    def _prepare_data(self, data: Union[np.ndarray, pd.Series, pd.DataFrame, list]) -> np.ndarray:
        """Convert data to numpy array and handle basic cleaning."""
        if isinstance(data, pd.DataFrame):
            if data.shape[1] == 1:
                data = data.iloc[:, 0].values
            else:
                # For multivariate data, store separately
                self.multivariate = True
                return data.values
        elif isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)
        else:
            data = np.asarray(data)
        
        self.multivariate = False
        return data.flatten()
    
    def examine(self) -> Dict:
        """
        Perform comprehensive data examination.
        
        Returns
        -------
        profile : dict
            Complete data profile with all characteristics
        """
        if self.verbose:
            print("Examining data...")
        
        # Basic statistics
        self.profile['basic_stats'] = self._compute_basic_stats()
        
        # Data type detection
        self.profile['data_type'] = self._detect_data_type()
        
        # Distribution characteristics
        self.profile['distribution'] = self._analyze_distribution()
        
        # Quality checks
        self.profile['quality'] = self._check_data_quality()
        
        # Outlier detection
        self.profile['outliers'] = self._detect_outliers()
        
        # Pattern detection
        self.profile['patterns'] = self._detect_patterns()
        
        # Recommendations
        self.profile['recommendations'] = self._generate_recommendations()
        
        # Flatten for easy access
        self.profile.update(self._flatten_for_prompts())
        
        if self.verbose:
            print("Examination complete!")
        
        return self.profile
    
    def _compute_basic_stats(self) -> Dict:
        """Compute basic statistical measures."""
        clean_data = self.data[~np.isnan(self.data)]
        
        if len(clean_data) == 0:
            return {'error': 'No valid data'}
        
        stats_dict = {
            'n': len(self.data),
            'n_valid': len(clean_data),
            'n_missing': len(self.data) - len(clean_data),
            'mean': float(np.mean(clean_data)),
            'median': float(np.median(clean_data)),
            'std': float(np.std(clean_data, ddof=1)) if len(clean_data) > 1 else 0.0,
            'variance': float(np.var(clean_data, ddof=1)) if len(clean_data) > 1 else 0.0,
            'min': float(np.min(clean_data)),
            'max': float(np.max(clean_data)),
            'range': float(np.max(clean_data) - np.min(clean_data)),
            'q1': float(np.percentile(clean_data, 25)),
            'q3': float(np.percentile(clean_data, 75)),
            'iqr': float(np.percentile(clean_data, 75) - np.percentile(clean_data, 25)),
        }
        
        # Moments
        if len(clean_data) > 3:
            stats_dict['skewness'] = float(stats.skew(clean_data))
            stats_dict['kurtosis'] = float(stats.kurtosis(clean_data))
        else:
            stats_dict['skewness'] = None
            stats_dict['kurtosis'] = None
        
        # Coefficient of variation
        if stats_dict['mean'] != 0:
            stats_dict['cv'] = stats_dict['std'] / abs(stats_dict['mean'])
        else:
            stats_dict['cv'] = None
        
        return stats_dict
    
    def _detect_data_type(self) -> str:
        """Detect the type of data (discrete, continuous, binary, etc.)."""
        clean_data = self.data[~np.isnan(self.data)]
        
        if len(clean_data) == 0:
            return 'unknown'
        
        unique_count = len(np.unique(clean_data))
        n = len(clean_data)
        
        # Check if all values are integers
        all_integers = np.allclose(clean_data, np.round(clean_data))
        
        # Check if binary
        if unique_count == 2:
            return 'binary'
        
        # Check if categorical (few unique values relative to sample size)
        if unique_count < min(20, n * 0.05) and all_integers:
            return 'categorical'
        
        # Check if discrete count data
        if all_integers and np.all(clean_data >= 0):
            return 'discrete_count'
        
        # Check if discrete
        if all_integers:
            return 'discrete'
        
        # Check if continuous positive (common for survival analysis)
        if np.all(clean_data > 0):
            return 'continuous_positive'
        
        # Default to continuous
        return 'continuous'
    
    def _analyze_distribution(self) -> Dict:
        """Analyze distribution characteristics."""
        clean_data = self.data[~np.isnan(self.data)]
        
        if len(clean_data) < 3:
            return {'error': 'Insufficient data for distribution analysis'}
        
        dist_info = {
            'unique_count': len(np.unique(clean_data)),
            'unique_ratio': len(np.unique(clean_data)) / len(clean_data),
        }
        
        # Check for specific properties
        dist_info['has_zeros'] = np.any(clean_data == 0)
        dist_info['has_negatives'] = np.any(clean_data < 0)
        dist_info['all_positive'] = np.all(clean_data > 0)
        dist_info['all_nonnegative'] = np.all(clean_data >= 0)
        
        # Overdispersion check (variance > mean, relevant for count data)
        mean = np.mean(clean_data)
        var = np.var(clean_data, ddof=1)
        if mean > 0:
            dist_info['dispersion_ratio'] = var / mean
            dist_info['overdispersed'] = var > mean * 1.1  # 10% threshold
        else:
            dist_info['dispersion_ratio'] = None
            dist_info['overdispersed'] = None
        
        # Normality tests
        if len(clean_data) >= 8:
            try:
                _, p_shapiro = stats.shapiro(clean_data[:5000])  # Limit for performance
                dist_info['normality_p_value'] = float(p_shapiro)
                dist_info['appears_normal'] = p_shapiro > 0.05
            except:
                dist_info['normality_p_value'] = None
                dist_info['appears_normal'] = None
        else:
            dist_info['normality_p_value'] = None
            dist_info['appears_normal'] = None
        
        # Symmetry
        skew = self.profile.get('basic_stats', {}).get('skewness', 0)
        if skew is not None:
            dist_info['symmetric'] = abs(skew) < 0.5
            dist_info['right_skewed'] = skew > 0.5
            dist_info['left_skewed'] = skew < -0.5
        
        return dist_info
    
    def _check_data_quality(self) -> Dict:
        """Check data quality issues."""
        quality = {
            'sample_size_adequate': len(self.data) >= 30,
            'missing_percentage': (len(self.data) - len(self.data[~np.isnan(self.data)])) / len(self.data) * 100,
            'has_missing': np.any(np.isnan(self.data)),
        }
        
        clean_data = self.data[~np.isnan(self.data)]
        
        # Check for constant data
        quality['is_constant'] = len(np.unique(clean_data)) == 1 if len(clean_data) > 0 else True
        
        # Check for extreme variance
        if len(clean_data) > 0:
            cv = self.profile.get('basic_stats', {}).get('cv', 0)
            quality['high_variance'] = cv is not None and cv > 1.0
        
        return quality
    
    def _detect_outliers(self) -> Dict:
        """Detect outliers using multiple methods."""
        clean_data = self.data[~np.isnan(self.data)]
        
        if len(clean_data) < 4:
            return {'count': 0, 'indices': [], 'method': 'none'}
        
        # IQR method
        q1 = np.percentile(clean_data, 25)
        q3 = np.percentile(clean_data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (clean_data < lower_bound) | (clean_data > upper_bound)
        outlier_count = np.sum(outlier_mask)
        
        return {
            'count': int(outlier_count),
            'percentage': float(outlier_count / len(clean_data) * 100),
            'method': 'IQR',
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
        }
    
    def _detect_patterns(self) -> Dict:
        """Detect patterns in data."""
        clean_data = self.data[~np.isnan(self.data)]
        
        patterns = {}
        
        # Check for multimodality (simple check using histogram)
        if len(clean_data) >= 50:
            hist, _ = np.histogram(clean_data, bins='auto')
            # Count peaks
            peaks = 0
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                    peaks += 1
            patterns['potential_multimodal'] = peaks > 1
            patterns['peak_count'] = peaks
        
        # Check for clustering (high concentration around mean)
        if len(clean_data) > 0:
            mean = np.mean(clean_data)
            std = np.std(clean_data)
            within_1std = np.sum(np.abs(clean_data - mean) <= std) / len(clean_data)
            patterns['concentration_1std'] = float(within_1std)
            patterns['highly_concentrated'] = within_1std > 0.85
        
        return patterns
    
    def _generate_recommendations(self) -> List[str]:
        """Generate preliminary recommendations based on data characteristics."""
        recommendations = []
        
        data_type = self.profile.get('data_type', 'unknown')
        basic_stats = self.profile.get('basic_stats', {})
        distribution = self.profile.get('distribution', {})
        quality = self.profile.get('quality', {})
        
        # Sample size recommendations
        if not quality.get('sample_size_adequate', False):
            recommendations.append("Small sample size may limit analysis reliability")
        
        # Data type specific recommendations
        if data_type == 'discrete_count':
            if distribution.get('overdispersed', False):
                recommendations.append("Consider Negative Binomial for overdispersed count data")
            else:
                recommendations.append("Poisson or Negative Binomial may be appropriate")
        
        if data_type == 'continuous_positive':
            recommendations.append("Consider exponential, gamma, or log-normal distributions")
            recommendations.append("Survival analysis methods may be appropriate")
        
        if data_type == 'continuous':
            if distribution.get('appears_normal', False):
                recommendations.append("Data appears normally distributed - parametric tests suitable")
            else:
                recommendations.append("Consider transformation or non-parametric methods")
        
        # Outlier recommendations
        outliers = self.profile.get('outliers', {})
        if outliers.get('percentage', 0) > 5:
            recommendations.append("Significant outliers detected - consider robust methods")
        
        # Variance recommendations
        if distribution.get('dispersion_ratio'):
            ratio = distribution['dispersion_ratio']
            if ratio > 2:
                recommendations.append("High dispersion - consider models that account for overdispersion")
        
        return recommendations
    
    def _flatten_for_prompts(self) -> Dict:
        """Flatten profile for easy use in prompts."""
        basic = self.profile.get('basic_stats', {})
        dist = self.profile.get('distribution', {})
        
        return {
            'n': basic.get('n', 0),
            'mean': basic.get('mean', 0),
            'median': basic.get('median', 0),
            'std': basic.get('std', 0),
            'variance': basic.get('variance', 0),
            'min': basic.get('min', 0),
            'max': basic.get('max', 0),
            'range': basic.get('range', 0),
            'skewness': basic.get('skewness', 0),
            'kurtosis': basic.get('kurtosis', 0),
            'has_zeros': dist.get('has_zeros', False),
            'has_negatives': dist.get('has_negatives', False),
            'unique_count': dist.get('unique_count', 0),
            'missing_count': basic.get('n_missing', 0),
        }
    
    def summary(self) -> str:
        """Generate human-readable summary of data examination."""
        if not self.profile:
            return "Data not yet examined. Call examine() first."
        
        basic = self.profile.get('basic_stats', {})
        data_type = self.profile.get('data_type', 'unknown')
        dist = self.profile.get('distribution', {})
        quality = self.profile.get('quality', {})
        outliers = self.profile.get('outliers', {})
        recommendations = self.profile.get('recommendations', [])
        
        summary = f"""
Data Examination Summary
========================

Basic Statistics:
  Sample size: {basic.get('n', 0)} ({basic.get('n_valid', 0)} valid)
  Mean: {basic.get('mean', 0):.4f}
  Median: {basic.get('median', 0):.4f}
  Std Dev: {basic.get('std', 0):.4f}
  Range: [{basic.get('min', 0):.4f}, {basic.get('max', 0):.4f}]

Data Type: {data_type}

Distribution Characteristics:
  Unique values: {dist.get('unique_count', 0)}
  Skewness: {basic.get('skewness', 'N/A')}
  Kurtosis: {basic.get('kurtosis', 'N/A')}
  Has zeros: {dist.get('has_zeros', False)}
  Has negatives: {dist.get('has_negatives', False)}
  Overdispersed: {dist.get('overdispersed', 'N/A')}

Data Quality:
  Missing: {quality.get('missing_percentage', 0):.1f}%
  Outliers: {outliers.get('count', 0)} ({outliers.get('percentage', 0):.1f}%)
  Sample adequate: {quality.get('sample_size_adequate', False)}

Recommendations:
"""
        for rec in recommendations:
            summary += f"  - {rec}\n"
        
        return summary

