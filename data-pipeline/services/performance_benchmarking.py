import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy import stats
import logging
from .cohort_analysis import WalletMetrics

logger = logging.getLogger(__name__)

class PerformanceBenchmarking:
    """Performance benchmarking and validation against control groups"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_random_wallet_sample(self, wallet_universe: List[WalletMetrics],
                                   sample_size: int = 10000) -> List[WalletMetrics]:
        """Create random wallet sample for benchmark comparison"""

        self.logger.info(f"Creating random sample of {sample_size} wallets from universe of {len(wallet_universe)}")

        if len(wallet_universe) < sample_size:
            self.logger.warning(f"Universe size ({len(wallet_universe)}) smaller than requested sample size ({sample_size})")
            sample_size = len(wallet_universe)

        # Random sampling without replacement
        random_indices = np.random.choice(len(wallet_universe), size=sample_size, replace=False)
        random_sample = [wallet_universe[i] for i in random_indices]

        return random_sample

    def create_volume_matched_sample(self, smart_money_cohort: List[WalletMetrics],
                                   wallet_universe: List[WalletMetrics]) -> List[WalletMetrics]:
        """Create volume-matched sample with similar trading volumes but no smart money filtering"""

        # Calculate volume distribution of smart money cohort
        smart_money_volumes = [w.avg_daily_volume_eth for w in smart_money_cohort]
        volume_percentiles = np.percentile(smart_money_volumes, [25, 50, 75])

        self.logger.info(f"Smart money volume percentiles: 25th={volume_percentiles[0]:.4f}, "
                        f"50th={volume_percentiles[1]:.4f}, 75th={volume_percentiles[2]:.4f}")

        # Filter universe to similar volume ranges
        matched_candidates = []
        for wallet in wallet_universe:
            if (volume_percentiles[0] <= wallet.avg_daily_volume_eth <= volume_percentiles[2]):
                matched_candidates.append(wallet)

        # Sample to match cohort size
        target_size = min(len(smart_money_cohort) * 2, len(matched_candidates))
        if len(matched_candidates) > target_size:
            random_indices = np.random.choice(len(matched_candidates), size=target_size, replace=False)
            volume_matched_sample = [matched_candidates[i] for i in random_indices]
        else:
            volume_matched_sample = matched_candidates

        self.logger.info(f"Created volume-matched sample of {len(volume_matched_sample)} wallets")
        return volume_matched_sample

    def create_time_matched_sample(self, smart_money_cohort: List[WalletMetrics],
                                  wallet_universe: List[WalletMetrics]) -> List[WalletMetrics]:
        """Create time-matched sample of wallets active during same periods"""

        # Get smart money activity period
        smart_money_start_dates = pd.to_datetime([w.first_trade for w in smart_money_cohort])
        smart_money_end_dates = pd.to_datetime([w.last_trade for w in smart_money_cohort])

        cohort_start = smart_money_start_dates.min()
        cohort_end = smart_money_end_dates.max()

        self.logger.info(f"Smart money active period: {cohort_start} to {cohort_end}")

        # Filter universe for overlapping activity
        time_matched_candidates = []
        for wallet in wallet_universe:
            wallet_start = pd.to_datetime(wallet.first_trade)
            wallet_end = pd.to_datetime(wallet.last_trade)

            # Check for temporal overlap
            if (wallet_start <= cohort_end and wallet_end >= cohort_start):
                time_matched_candidates.append(wallet)

        # Sample to reasonable size
        target_size = min(len(smart_money_cohort) * 3, len(time_matched_candidates))
        if len(time_matched_candidates) > target_size:
            random_indices = np.random.choice(len(time_matched_candidates), size=target_size, replace=False)
            time_matched_sample = [time_matched_candidates[i] for i in random_indices]
        else:
            time_matched_sample = time_matched_candidates

        self.logger.info(f"Created time-matched sample of {len(time_matched_sample)} wallets")
        return time_matched_sample

    def calculate_aggregate_performance_metrics(self, wallet_group: List[WalletMetrics]) -> Dict[str, float]:
        """Calculate aggregate performance metrics for a wallet group"""

        if not wallet_group:
            return {}

        metrics = {
            'mean_total_return': np.mean([w.total_return for w in wallet_group]),
            'median_total_return': np.median([w.total_return for w in wallet_group]),
            'mean_sharpe_ratio': np.mean([w.sharpe_ratio for w in wallet_group]),
            'median_sharpe_ratio': np.median([w.sharpe_ratio for w in wallet_group]),
            'mean_win_rate': np.mean([w.win_rate for w in wallet_group]),
            'median_win_rate': np.median([w.win_rate for w in wallet_group]),
            'mean_volatility': np.mean([w.volatility for w in wallet_group]),
            'median_volatility': np.median([w.volatility for w in wallet_group]),
            'mean_max_drawdown': np.mean([w.max_drawdown for w in wallet_group]),
            'median_max_drawdown': np.median([w.max_drawdown for w in wallet_group]),
            'group_size': len(wallet_group)
        }

        return metrics

    def statistical_significance_testing(self, smart_money_cohort: List[WalletMetrics],
                                       benchmark_sample: List[WalletMetrics]) -> Dict[str, Any]:
        """Implement statistical significance testing for outperformance"""

        # Extract performance arrays
        cohort_returns = [w.total_return for w in smart_money_cohort]
        benchmark_returns = [w.total_return for w in benchmark_sample]

        cohort_sharpe = [w.sharpe_ratio for w in smart_money_cohort]
        benchmark_sharpe = [w.sharpe_ratio for w in benchmark_sample]

        cohort_win_rates = [w.win_rate for w in smart_money_cohort]
        benchmark_win_rates = [w.win_rate for w in benchmark_sample]

        # Two-sample t-tests
        return_t_stat, return_p_value = stats.ttest_ind(cohort_returns, benchmark_returns)
        sharpe_t_stat, sharpe_p_value = stats.ttest_ind(cohort_sharpe, benchmark_sharpe)
        win_rate_t_stat, win_rate_p_value = stats.ttest_ind(cohort_win_rates, benchmark_win_rates)

        # Mann-Whitney U tests (non-parametric)
        return_u_stat, return_u_p = stats.mannwhitneyu(cohort_returns, benchmark_returns, alternative='greater')
        sharpe_u_stat, sharpe_u_p = stats.mannwhitneyu(cohort_sharpe, benchmark_sharpe, alternative='greater')
        win_rate_u_stat, win_rate_u_p = stats.mannwhitneyu(cohort_win_rates, benchmark_win_rates, alternative='greater')

        # Effect sizes (Cohen's d)
        def cohens_d(group1, group2):
            pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) +
                                 (len(group2) - 1) * np.var(group2, ddof=1)) /
                                (len(group1) + len(group2) - 2))
            return (np.mean(group1) - np.mean(group2)) / pooled_std

        return_effect_size = cohens_d(cohort_returns, benchmark_returns)
        sharpe_effect_size = cohens_d(cohort_sharpe, benchmark_sharpe)
        win_rate_effect_size = cohens_d(cohort_win_rates, benchmark_win_rates)

        significance_results = {
            'total_return': {
                't_test': {
                    'statistic': return_t_stat,
                    'p_value': return_p_value,
                    'significant': return_p_value < 0.01
                },
                'mann_whitney': {
                    'statistic': return_u_stat,
                    'p_value': return_u_p,
                    'significant': return_u_p < 0.01
                },
                'effect_size': return_effect_size,
                'cohort_mean': np.mean(cohort_returns),
                'benchmark_mean': np.mean(benchmark_returns),
                'outperformance': np.mean(cohort_returns) - np.mean(benchmark_returns)
            },
            'sharpe_ratio': {
                't_test': {
                    'statistic': sharpe_t_stat,
                    'p_value': sharpe_p_value,
                    'significant': sharpe_p_value < 0.01
                },
                'mann_whitney': {
                    'statistic': sharpe_u_stat,
                    'p_value': sharpe_u_p,
                    'significant': sharpe_u_p < 0.01
                },
                'effect_size': sharpe_effect_size,
                'cohort_mean': np.mean(cohort_sharpe),
                'benchmark_mean': np.mean(benchmark_sharpe),
                'outperformance': np.mean(cohort_sharpe) - np.mean(benchmark_sharpe)
            },
            'win_rate': {
                't_test': {
                    'statistic': win_rate_t_stat,
                    'p_value': win_rate_p_value,
                    'significant': win_rate_p_value < 0.01
                },
                'mann_whitney': {
                    'statistic': win_rate_u_stat,
                    'p_value': win_rate_u_p,
                    'significant': win_rate_u_p < 0.01
                },
                'effect_size': win_rate_effect_size,
                'cohort_mean': np.mean(cohort_win_rates),
                'benchmark_mean': np.mean(benchmark_win_rates),
                'outperformance': np.mean(cohort_win_rates) - np.mean(benchmark_win_rates)
            }
        }

        return significance_results

    def risk_adjusted_performance_comparison(self, smart_money_cohort: List[WalletMetrics],
                                           benchmark_sample: List[WalletMetrics]) -> Dict[str, Any]:
        """Add risk-adjusted performance comparison analysis"""

        # Risk-adjusted metrics for cohort
        cohort_risk_adjusted = self._calculate_risk_adjusted_metrics(smart_money_cohort)
        benchmark_risk_adjusted = self._calculate_risk_adjusted_metrics(benchmark_sample)

        # Compare risk-adjusted performance
        comparison = {
            'cohort_metrics': cohort_risk_adjusted,
            'benchmark_metrics': benchmark_risk_adjusted,
            'relative_performance': {
                'sharpe_ratio_improvement': (
                    cohort_risk_adjusted['mean_sharpe'] - benchmark_risk_adjusted['mean_sharpe']
                ) / benchmark_risk_adjusted['mean_sharpe'] * 100,
                'return_per_volatility_improvement': (
                    cohort_risk_adjusted['return_volatility_ratio'] -
                    benchmark_risk_adjusted['return_volatility_ratio']
                ) / benchmark_risk_adjusted['return_volatility_ratio'] * 100,
                'max_drawdown_improvement': (
                    benchmark_risk_adjusted['mean_max_drawdown'] -
                    cohort_risk_adjusted['mean_max_drawdown']
                ) / abs(benchmark_risk_adjusted['mean_max_drawdown']) * 100
            }
        }

        return comparison

    def _calculate_risk_adjusted_metrics(self, wallet_group: List[WalletMetrics]) -> Dict[str, float]:
        """Calculate comprehensive risk-adjusted metrics for a wallet group"""

        returns = [w.total_return for w in wallet_group]
        sharpe_ratios = [w.sharpe_ratio for w in wallet_group]
        volatilities = [w.volatility for w in wallet_group]
        max_drawdowns = [w.max_drawdown for w in wallet_group]

        metrics = {
            'mean_return': np.mean(returns),
            'mean_sharpe': np.mean(sharpe_ratios),
            'mean_volatility': np.mean(volatilities),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'return_volatility_ratio': np.mean(returns) / np.mean(volatilities),
            'downside_deviation': np.std([r for r in returns if r < 0]),
            'upside_capture': np.mean([r for r in returns if r > 0]),
            'risk_adjusted_return': np.mean(returns) / (np.mean(volatilities) + abs(np.mean(max_drawdowns)))
        }

        return metrics

    def generate_confidence_intervals(self, smart_money_cohort: List[WalletMetrics],
                                    benchmark_sample: List[WalletMetrics],
                                    confidence_level: float = 0.99) -> Dict[str, Any]:
        """Generate confidence intervals for performance differences"""

        alpha = 1 - confidence_level

        # Calculate differences for key metrics
        cohort_returns = np.array([w.total_return for w in smart_money_cohort])
        benchmark_returns = np.array([w.total_return for w in benchmark_sample])

        cohort_sharpe = np.array([w.sharpe_ratio for w in smart_money_cohort])
        benchmark_sharpe = np.array([w.sharpe_ratio for w in benchmark_sample])

        # Bootstrap confidence intervals
        def bootstrap_ci(sample1, sample2, n_bootstrap=1000):
            differences = []
            for _ in range(n_bootstrap):
                resample1 = np.random.choice(sample1, size=len(sample1), replace=True)
                resample2 = np.random.choice(sample2, size=len(sample2), replace=True)
                differences.append(np.mean(resample1) - np.mean(resample2))

            lower = np.percentile(differences, (alpha/2) * 100)
            upper = np.percentile(differences, (1 - alpha/2) * 100)
            return lower, upper, np.mean(differences)

        return_ci_lower, return_ci_upper, return_diff = bootstrap_ci(cohort_returns, benchmark_returns)
        sharpe_ci_lower, sharpe_ci_upper, sharpe_diff = bootstrap_ci(cohort_sharpe, benchmark_sharpe)

        confidence_intervals = {
            'confidence_level': confidence_level,
            'total_return_difference': {
                'point_estimate': return_diff,
                'confidence_interval': [return_ci_lower, return_ci_upper],
                'significant': return_ci_lower > 0  # CI doesn't include 0
            },
            'sharpe_ratio_difference': {
                'point_estimate': sharpe_diff,
                'confidence_interval': [sharpe_ci_lower, sharpe_ci_upper],
                'significant': sharpe_ci_lower > 0  # CI doesn't include 0
            }
        }

        return confidence_intervals

    def validate_performance_consistency(self, smart_money_cohort: List[WalletMetrics]) -> Dict[str, Any]:
        """Validate cohort performance consistency across time periods"""

        # Group wallets by entry periods (quarterly)
        wallet_periods = {}
        for wallet in smart_money_cohort:
            entry_date = pd.to_datetime(wallet.first_trade)
            quarter_key = f"{entry_date.year}Q{entry_date.quarter}"

            if quarter_key not in wallet_periods:
                wallet_periods[quarter_key] = []
            wallet_periods[quarter_key].append(wallet)

        # Calculate performance by period
        period_performance = {}
        for period, wallets in wallet_periods.items():
            if len(wallets) >= 10:  # Minimum sample size for meaningful analysis
                period_performance[period] = {
                    'count': len(wallets),
                    'mean_return': np.mean([w.total_return for w in wallets]),
                    'mean_sharpe': np.mean([w.sharpe_ratio for w in wallets]),
                    'mean_win_rate': np.mean([w.win_rate for w in wallets])
                }

        # Test for consistency across periods
        if len(period_performance) >= 2:
            period_returns = [perf['mean_return'] for perf in period_performance.values()]
            period_sharpes = [perf['mean_sharpe'] for perf in period_performance.values()]

            # Coefficient of variation (lower = more consistent)
            return_cv = np.std(period_returns) / np.mean(period_returns)
            sharpe_cv = np.std(period_sharpes) / np.mean(period_sharpes)

            consistency_analysis = {
                'period_breakdown': period_performance,
                'consistency_metrics': {
                    'return_coefficient_variation': return_cv,
                    'sharpe_coefficient_variation': sharpe_cv,
                    'periods_analyzed': len(period_performance),
                    'is_consistent': return_cv < 0.5 and sharpe_cv < 0.5  # Threshold for consistency
                },
                'summary': {
                    'min_period_return': min(period_returns),
                    'max_period_return': max(period_returns),
                    'min_period_sharpe': min(period_sharpes),
                    'max_period_sharpe': max(period_sharpes)
                }
            }
        else:
            consistency_analysis = {
                'period_breakdown': period_performance,
                'warning': 'Insufficient periods for consistency analysis',
                'periods_analyzed': len(period_performance)
            }

        return consistency_analysis

    def comprehensive_benchmarking_report(self, smart_money_cohort: List[WalletMetrics],
                                        wallet_universe: List[WalletMetrics]) -> Dict[str, Any]:
        """Generate comprehensive benchmarking report"""

        self.logger.info("Generating comprehensive benchmarking report")

        # Create benchmark samples
        random_sample = self.create_random_wallet_sample(wallet_universe)
        volume_matched_sample = self.create_volume_matched_sample(smart_money_cohort, wallet_universe)
        time_matched_sample = self.create_time_matched_sample(smart_money_cohort, wallet_universe)

        # Performance comparisons
        benchmarks = {
            'random_sample': random_sample,
            'volume_matched': volume_matched_sample,
            'time_matched': time_matched_sample
        }

        benchmark_results = {}
        for benchmark_name, benchmark_sample in benchmarks.items():
            self.logger.info(f"Analyzing performance vs {benchmark_name}")

            benchmark_results[benchmark_name] = {
                'aggregate_metrics': self.calculate_aggregate_performance_metrics(benchmark_sample),
                'significance_tests': self.statistical_significance_testing(smart_money_cohort, benchmark_sample),
                'risk_adjusted_comparison': self.risk_adjusted_performance_comparison(smart_money_cohort, benchmark_sample),
                'confidence_intervals': self.generate_confidence_intervals(smart_money_cohort, benchmark_sample)
            }

        # Cohort consistency analysis
        consistency_analysis = self.validate_performance_consistency(smart_money_cohort)

        # Overall summary
        cohort_metrics = self.calculate_aggregate_performance_metrics(smart_money_cohort)

        comprehensive_report = {
            'executive_summary': {
                'cohort_size': len(smart_money_cohort),
                'cohort_performance': cohort_metrics,
                'benchmarks_analyzed': list(benchmarks.keys()),
                'significant_outperformance': all(
                    benchmark_results[benchmark]['significance_tests']['total_return']['t_test']['significant']
                    for benchmark in benchmarks.keys()
                )
            },
            'benchmark_comparisons': benchmark_results,
            'temporal_consistency': consistency_analysis,
            'methodology': {
                'random_sample_size': len(random_sample),
                'volume_matched_size': len(volume_matched_sample),
                'time_matched_size': len(time_matched_sample),
                'significance_threshold': 0.01,
                'confidence_level': 0.99
            }
        }

        return comprehensive_report