import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

@dataclass
class WalletMetrics:
    """Wallet performance and activity metrics"""
    wallet_address: str
    total_return: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    volatility: float
    total_trades: int
    trading_days: int
    avg_daily_volume_eth: float
    unique_tokens_traded: int
    first_trade: str
    last_trade: str
    volume_per_gas: float
    mev_damage_ratio: float
    performance_consistency: float
    portfolio_concentration: float
    avg_position_size: float
    trade_frequency: float
    gas_efficiency: float

class CohortStatisticalAnalysis:
    """Comprehensive statistical analysis of wallet cohort"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_cohort_statistics(self, wallet_cohort: List[WalletMetrics]) -> Dict[str, Any]:
        """Generate comprehensive statistical analysis of final cohort"""

        self.logger.info(f"Generating statistics for cohort of {len(wallet_cohort)} wallets")

        # Extract metrics arrays
        sharpe_ratios = [w.sharpe_ratio for w in wallet_cohort]
        total_returns = [w.total_return for w in wallet_cohort]
        win_rates = [w.win_rate for w in wallet_cohort]
        volatilities = [w.volatility for w in wallet_cohort]
        total_trades = [w.total_trades for w in wallet_cohort]
        avg_volumes = [w.avg_daily_volume_eth for w in wallet_cohort]
        unique_tokens = [w.unique_tokens_traded for w in wallet_cohort]
        volume_per_gas = [w.volume_per_gas for w in wallet_cohort]
        mev_ratios = [w.mev_damage_ratio for w in wallet_cohort]
        consistency_scores = [w.performance_consistency for w in wallet_cohort]

        stats = {
            'cohort_size': len(wallet_cohort),
            'performance': {
                'mean_sharpe': np.mean(sharpe_ratios),
                'std_sharpe': np.std(sharpe_ratios),
                'median_return': np.median(total_returns),
                'mean_return': np.mean(total_returns),
                'win_rate_distribution': {
                    'q25': np.percentile(win_rates, 25),
                    'q50': np.percentile(win_rates, 50),
                    'q75': np.percentile(win_rates, 75),
                    'min': np.min(win_rates),
                    'max': np.max(win_rates)
                },
                'volatility_range': {
                    'min': np.min(volatilities),
                    'max': np.max(volatilities),
                    'mean': np.mean(volatilities),
                    'median': np.median(volatilities)
                }
            },
            'activity': {
                'median_trades': np.median(total_trades),
                'mean_trades': np.mean(total_trades),
                'avg_volume_eth': np.mean(avg_volumes),
                'median_volume_eth': np.median(avg_volumes),
                'token_diversity': {
                    'mean': np.mean(unique_tokens),
                    'median': np.median(unique_tokens),
                    'std': np.std(unique_tokens)
                }
            },
            'quality': {
                'gas_efficiency': {
                    'mean': np.mean(volume_per_gas),
                    'median': np.median(volume_per_gas),
                    'std': np.std(volume_per_gas)
                },
                'mev_impact': {
                    'mean': np.mean(mev_ratios),
                    'median': np.median(mev_ratios),
                    'std': np.std(mev_ratios)
                },
                'consistency_score': {
                    'mean': np.mean(consistency_scores),
                    'median': np.median(consistency_scores),
                    'std': np.std(consistency_scores)
                }
            }
        }

        return stats

    def calculate_percentile_analysis(self, wallet_cohort: List[WalletMetrics]) -> Dict[str, Any]:
        """Generate percentile analysis and outlier identification"""

        metrics = {
            'sharpe_ratio': [w.sharpe_ratio for w in wallet_cohort],
            'total_return': [w.total_return for w in wallet_cohort],
            'win_rate': [w.win_rate for w in wallet_cohort],
            'volatility': [w.volatility for w in wallet_cohort],
            'volume_per_gas': [w.volume_per_gas for w in wallet_cohort]
        }

        percentile_analysis = {}

        for metric_name, values in metrics.items():
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            percentile_values = np.percentile(values, percentiles)

            # Outlier detection using IQR method
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = [v for v in values if v < lower_bound or v > upper_bound]

            percentile_analysis[metric_name] = {
                'percentiles': dict(zip(percentiles, percentile_values)),
                'outliers': {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(values) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'values': outliers[:10]  # Show first 10 outliers
                }
            }

        return percentile_analysis

    def create_correlation_analysis(self, wallet_cohort: List[WalletMetrics]) -> Dict[str, Any]:
        """Create correlation analysis between different quality metrics"""

        # Build correlation matrix
        metrics_data = []
        for wallet in wallet_cohort:
            metrics_data.append([
                wallet.sharpe_ratio,
                wallet.total_return,
                wallet.win_rate,
                wallet.volatility,
                wallet.volume_per_gas,
                wallet.performance_consistency,
                wallet.unique_tokens_traded,
                wallet.avg_daily_volume_eth
            ])

        df = pd.DataFrame(metrics_data, columns=[
            'sharpe_ratio', 'total_return', 'win_rate', 'volatility',
            'volume_per_gas', 'performance_consistency', 'unique_tokens_traded',
            'avg_daily_volume_eth'
        ])

        correlation_matrix = df.corr()

        # Find strongest correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Strong correlation threshold
                    strong_correlations.append({
                        'metric1': correlation_matrix.columns[i],
                        'metric2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'Strong' if abs(corr_value) > 0.7 else 'Moderate'
                    })

        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'summary': {
                'total_metric_pairs': len(correlation_matrix.columns) * (len(correlation_matrix.columns) - 1) // 2,
                'strong_correlations_count': len(strong_correlations)
            }
        }

    def statistical_significance_testing(self, wallet_cohort: List[WalletMetrics]) -> Dict[str, Any]:
        """Implement statistical significance testing for cohort characteristics"""

        # Test if performance metrics follow normal distribution
        sharpe_ratios = [w.sharpe_ratio for w in wallet_cohort]
        total_returns = [w.total_return for w in wallet_cohort]
        win_rates = [w.win_rate for w in wallet_cohort]

        # Shapiro-Wilk test for normality
        normality_tests = {}
        for metric_name, values in [('sharpe_ratio', sharpe_ratios),
                                   ('total_return', total_returns),
                                   ('win_rate', win_rates)]:
            stat, p_value = stats.shapiro(values[:5000])  # Limit for Shapiro-Wilk
            normality_tests[metric_name] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }

        # Test for significant differences from market baseline
        # Using one-sample t-test against market benchmarks
        market_benchmarks = {
            'sharpe_ratio': 0.3,  # Typical market Sharpe ratio
            'total_return': 0.1,  # 10% annual return baseline
            'win_rate': 0.5       # 50% win rate baseline
        }

        significance_tests = {}
        for metric_name, values in [('sharpe_ratio', sharpe_ratios),
                                   ('total_return', total_returns),
                                   ('win_rate', win_rates)]:
            benchmark = market_benchmarks[metric_name]
            t_stat, p_value = stats.ttest_1samp(values, benchmark)

            significance_tests[metric_name] = {
                'benchmark_value': benchmark,
                'cohort_mean': np.mean(values),
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.01,  # 99% confidence
                'effect_size': (np.mean(values) - benchmark) / np.std(values)
            }

        return {
            'normality_tests': normality_tests,
            'significance_tests': significance_tests,
            'summary': {
                'cohort_size': len(wallet_cohort),
                'significant_metrics': [
                    metric for metric, test in significance_tests.items()
                    if test['is_significant']
                ]
            }
        }

    def temporal_stability_analysis(self, wallet_cohort: List[WalletMetrics]) -> Dict[str, Any]:
        """Add temporal stability analysis for cohort composition"""

        # Analyze trading period distribution
        first_trades = pd.to_datetime([w.first_trade for w in wallet_cohort])
        last_trades = pd.to_datetime([w.last_trade for w in wallet_cohort])

        # Calculate activity periods
        activity_periods = (last_trades - first_trades).dt.days

        # Monthly cohort entry analysis
        monthly_entries = first_trades.groupby(first_trades.dt.to_period('M')).count()

        # Temporal stability metrics
        temporal_analysis = {
            'activity_period_stats': {
                'mean_days': np.mean(activity_periods),
                'median_days': np.median(activity_periods),
                'std_days': np.std(activity_periods),
                'min_days': np.min(activity_periods),
                'max_days': np.max(activity_periods)
            },
            'cohort_entry_distribution': {
                'monthly_counts': monthly_entries.to_dict(),
                'entry_spread_months': len(monthly_entries),
                'most_active_month': monthly_entries.idxmax(),
                'least_active_month': monthly_entries.idxmin()
            },
            'temporal_consistency': {
                'entry_coefficient_variation': np.std(monthly_entries) / np.mean(monthly_entries),
                'temporal_balance': len(monthly_entries) >= 3  # At least 3 months spread
            }
        }

        return temporal_analysis

    def generate_visualization_data(self, wallet_cohort: List[WalletMetrics]) -> Dict[str, Any]:
        """Generate visualization data for cohort characteristic distributions"""

        visualization_data = {
            'histograms': {},
            'box_plots': {},
            'scatter_plots': {},
            'summary_stats': {}
        }

        metrics = {
            'sharpe_ratio': [w.sharpe_ratio for w in wallet_cohort],
            'total_return': [w.total_return for w in wallet_cohort],
            'win_rate': [w.win_rate for w in wallet_cohort],
            'volatility': [w.volatility for w in wallet_cohort],
            'volume_per_gas': [w.volume_per_gas for w in wallet_cohort],
            'unique_tokens_traded': [w.unique_tokens_traded for w in wallet_cohort]
        }

        for metric_name, values in metrics.items():
            # Histogram data (bins and counts)
            hist, bin_edges = np.histogram(values, bins=20)
            visualization_data['histograms'][metric_name] = {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }

            # Box plot data
            visualization_data['box_plots'][metric_name] = {
                'q1': np.percentile(values, 25),
                'median': np.percentile(values, 50),
                'q3': np.percentile(values, 75),
                'whisker_low': np.percentile(values, 5),
                'whisker_high': np.percentile(values, 95),
                'outliers': [v for v in values if v < np.percentile(values, 5) or v > np.percentile(values, 95)]
            }

            # Summary statistics
            visualization_data['summary_stats'][metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'skewness': stats.skew(values),
                'kurtosis': stats.kurtosis(values)
            }

        # Scatter plot data for key relationships
        visualization_data['scatter_plots'] = {
            'sharpe_vs_return': {
                'x': [w.sharpe_ratio for w in wallet_cohort],
                'y': [w.total_return for w in wallet_cohort]
            },
            'volume_vs_tokens': {
                'x': [w.avg_daily_volume_eth for w in wallet_cohort],
                'y': [w.unique_tokens_traded for w in wallet_cohort]
            },
            'consistency_vs_sharpe': {
                'x': [w.performance_consistency for w in wallet_cohort],
                'y': [w.sharpe_ratio for w in wallet_cohort]
            }
        }

        return visualization_data