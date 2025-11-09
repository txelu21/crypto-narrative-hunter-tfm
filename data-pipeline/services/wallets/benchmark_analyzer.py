"""
Benchmark comparison and relative performance analysis module.

This module provides functionality to compare wallet performance against
various benchmarks including market indices, peer groups, and narrative
categories. Calculates alpha, beta, and relative performance metrics.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from decimal import Decimal
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class BenchmarkAnalyzer:
    """
    Analyzes wallet performance relative to benchmarks and peer groups.

    This class provides methods for calculating alpha, beta, relative
    performance rankings, and peer group comparisons across different
    market conditions and time periods.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the BenchmarkAnalyzer.

        Args:
            risk_free_rate: Annual risk-free rate for calculations (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = risk_free_rate / 365

    def calculate_benchmark_returns(
        self,
        benchmark_prices: pd.Series
    ) -> pd.Series:
        """
        Calculate benchmark returns from price series.

        Args:
            benchmark_prices: Series of benchmark prices indexed by date

        Returns:
            Series of daily benchmark returns
        """
        if benchmark_prices.empty:
            return pd.Series(dtype=float)

        # Calculate daily returns
        returns = benchmark_prices.pct_change().fillna(0)

        return returns

    def calculate_alpha_beta(
        self,
        wallet_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate Jensen's alpha and beta for a wallet.

        Alpha measures risk-adjusted excess return over the benchmark.
        Beta measures the wallet's sensitivity to benchmark movements.

        Args:
            wallet_returns: Series of wallet daily returns
            benchmark_returns: Series of benchmark daily returns
            risk_free_rate: Annual risk-free rate (uses instance default if None)

        Returns:
            Dictionary containing alpha and beta values
        """
        if wallet_returns.empty or benchmark_returns.empty:
            return {'alpha': 0.0, 'beta': 0.0, 'r_squared': 0.0}

        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        # Align series by index
        aligned_data = pd.DataFrame({
            'wallet': wallet_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(aligned_data) < 30:  # Need sufficient data
            return {'alpha': 0.0, 'beta': 0.0, 'r_squared': 0.0}

        wallet_aligned = aligned_data['wallet']
        benchmark_aligned = aligned_data['benchmark']

        # Calculate excess returns
        daily_rf = risk_free_rate / 365
        wallet_excess = wallet_aligned - daily_rf
        benchmark_excess = benchmark_aligned - daily_rf

        # Calculate beta using covariance
        if benchmark_excess.var() > 0:
            beta = np.cov(wallet_excess, benchmark_excess)[0, 1] / benchmark_excess.var()
        else:
            beta = 0.0

        # Calculate alpha using CAPM
        # Alpha = (Rp - Rf) - Beta * (Rm - Rf)
        avg_wallet_excess = wallet_excess.mean() * 365  # Annualize
        avg_benchmark_excess = benchmark_excess.mean() * 365  # Annualize

        alpha = avg_wallet_excess - (beta * avg_benchmark_excess)

        # Calculate R-squared (correlation coefficient squared)
        if len(wallet_aligned) > 1:
            correlation = np.corrcoef(wallet_aligned, benchmark_aligned)[0, 1]
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0.0
        else:
            r_squared = 0.0

        return {
            'alpha': float(alpha),
            'beta': float(beta),
            'r_squared': float(r_squared)
        }

    def calculate_tracking_error(
        self,
        wallet_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate tracking error (volatility of excess returns).

        Args:
            wallet_returns: Series of wallet daily returns
            benchmark_returns: Series of benchmark daily returns

        Returns:
            Annualized tracking error
        """
        if wallet_returns.empty or benchmark_returns.empty:
            return 0.0

        # Align series and calculate excess returns
        aligned_data = pd.DataFrame({
            'wallet': wallet_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(aligned_data) < 2:
            return 0.0

        excess_returns = aligned_data['wallet'] - aligned_data['benchmark']

        # Calculate annualized tracking error
        tracking_error = excess_returns.std() * np.sqrt(365)

        return float(tracking_error)

    def calculate_information_ratio(
        self,
        wallet_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate information ratio (excess return / tracking error).

        Args:
            wallet_returns: Series of wallet daily returns
            benchmark_returns: Series of benchmark daily returns

        Returns:
            Information ratio
        """
        if wallet_returns.empty or benchmark_returns.empty:
            return 0.0

        # Align series
        aligned_data = pd.DataFrame({
            'wallet': wallet_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(aligned_data) < 2:
            return 0.0

        wallet_aligned = aligned_data['wallet']
        benchmark_aligned = aligned_data['benchmark']

        # Calculate excess return
        excess_return = (wallet_aligned.mean() - benchmark_aligned.mean()) * 365

        # Calculate tracking error
        tracking_error = self.calculate_tracking_error(wallet_returns, benchmark_returns)

        if tracking_error == 0:
            return 0.0

        return float(excess_return / tracking_error)

    def create_benchmark_comparison(
        self,
        wallet_returns: pd.Series,
        benchmarks: Dict[str, pd.Series]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare wallet performance against multiple benchmarks.

        Args:
            wallet_returns: Series of wallet daily returns
            benchmarks: Dictionary mapping benchmark names to return series

        Returns:
            Dictionary containing comparison metrics for each benchmark
        """
        comparisons = {}

        for benchmark_name, benchmark_returns in benchmarks.items():
            if benchmark_returns.empty:
                continue

            # Calculate alpha/beta
            alpha_beta = self.calculate_alpha_beta(wallet_returns, benchmark_returns)

            # Calculate tracking error and information ratio
            tracking_error = self.calculate_tracking_error(wallet_returns, benchmark_returns)
            info_ratio = self.calculate_information_ratio(wallet_returns, benchmark_returns)

            # Calculate relative performance
            aligned_data = pd.DataFrame({
                'wallet': wallet_returns,
                'benchmark': benchmark_returns
            }).dropna()

            if not aligned_data.empty:
                cumulative_wallet = (1 + aligned_data['wallet']).cumprod().iloc[-1] - 1
                cumulative_benchmark = (1 + aligned_data['benchmark']).cumprod().iloc[-1] - 1
                relative_return = cumulative_wallet - cumulative_benchmark
            else:
                relative_return = 0.0

            comparisons[benchmark_name] = {
                'alpha': alpha_beta['alpha'],
                'beta': alpha_beta['beta'],
                'r_squared': alpha_beta['r_squared'],
                'tracking_error': tracking_error,
                'information_ratio': info_ratio,
                'relative_return': float(relative_return),
                'correlation': float(np.corrcoef(
                    aligned_data['wallet'], aligned_data['benchmark']
                )[0, 1]) if len(aligned_data) > 1 else 0.0
            }

        return comparisons

    def calculate_peer_group_rankings(
        self,
        wallet_returns: pd.Series,
        peer_returns: Dict[str, pd.Series],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate wallet rankings within peer group.

        Args:
            wallet_returns: Series of wallet daily returns
            peer_returns: Dictionary mapping peer wallet addresses to return series
            metrics: List of metrics to rank (default: standard performance metrics)

        Returns:
            Dictionary containing rankings and percentiles
        """
        if metrics is None:
            metrics = ['total_return', 'sharpe_ratio', 'volatility', 'max_drawdown']

        # Calculate metrics for all wallets
        all_metrics = {}

        # Add target wallet
        all_metrics['target'] = self._calculate_wallet_metrics(wallet_returns)

        # Add peer wallets
        for peer_id, peer_ret in peer_returns.items():
            if not peer_ret.empty:
                all_metrics[peer_id] = self._calculate_wallet_metrics(peer_ret)

        if len(all_metrics) < 2:
            return {'rankings': {}, 'percentiles': {}, 'peer_count': 0}

        # Calculate rankings for each metric
        rankings = {}
        percentiles = {}

        for metric in metrics:
            if metric not in all_metrics['target']:
                continue

            # Extract metric values
            metric_values = []
            wallet_ids = []

            for wallet_id, wallet_metrics in all_metrics.items():
                if metric in wallet_metrics and not np.isnan(wallet_metrics[metric]):
                    metric_values.append(wallet_metrics[metric])
                    wallet_ids.append(wallet_id)

            if len(metric_values) < 2:
                continue

            # Sort by metric (higher is better for most metrics, except volatility and max_drawdown)
            reverse_sort = metric not in ['volatility', 'max_drawdown']
            sorted_pairs = sorted(
                zip(metric_values, wallet_ids),
                key=lambda x: x[0],
                reverse=reverse_sort
            )

            # Find target wallet ranking
            target_rank = None
            for rank, (value, wallet_id) in enumerate(sorted_pairs, 1):
                if wallet_id == 'target':
                    target_rank = rank
                    break

            if target_rank is not None:
                rankings[metric] = target_rank
                percentiles[metric] = (len(sorted_pairs) - target_rank + 1) / len(sorted_pairs) * 100

        return {
            'rankings': rankings,
            'percentiles': percentiles,
            'peer_count': len(all_metrics) - 1,
            'total_wallets': len(all_metrics)
        }

    def _calculate_wallet_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate standard performance metrics for a wallet.

        Args:
            returns: Series of daily returns

        Returns:
            Dictionary of performance metrics
        """
        if returns.empty:
            return {}

        # Total return
        total_return = (1 + returns).prod() - 1

        # Annualized return
        if len(returns) > 0:
            days = len(returns)
            annualized_return = ((1 + total_return) ** (365 / days)) - 1
        else:
            annualized_return = 0

        # Volatility
        volatility = returns.std() * np.sqrt(365)

        # Sharpe ratio
        excess_returns = returns - self.daily_rf_rate
        if excess_returns.std() > 0:
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(365)
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown)
        }

    def create_narrative_benchmarks(
        self,
        wallet_returns: pd.Series,
        narrative_returns: Dict[str, pd.Series],
        wallet_narrative: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare wallet against narrative-specific benchmarks.

        Args:
            wallet_returns: Series of wallet daily returns
            narrative_returns: Dictionary mapping narrative names to return series
            wallet_narrative: Primary narrative category for the wallet

        Returns:
            Dictionary containing narrative comparison results
        """
        narrative_comparison = {}

        for narrative_name, narrative_ret in narrative_returns.items():
            if narrative_ret.empty:
                continue

            # Calculate basic comparison metrics
            comparison = self.create_benchmark_comparison(
                wallet_returns,
                {narrative_name: narrative_ret}
            )

            if narrative_name in comparison:
                narrative_comparison[narrative_name] = comparison[narrative_name]

                # Add narrative-specific context
                narrative_comparison[narrative_name]['is_primary_narrative'] = (
                    narrative_name == wallet_narrative
                )

        # Find best and worst performing narratives
        if narrative_comparison:
            relative_returns = {
                name: metrics['relative_return']
                for name, metrics in narrative_comparison.items()
            }

            best_narrative = max(relative_returns, key=relative_returns.get)
            worst_narrative = min(relative_returns, key=relative_returns.get)

            narrative_comparison['_summary'] = {
                'best_relative_narrative': best_narrative,
                'worst_relative_narrative': worst_narrative,
                'narrative_count': len(narrative_comparison) - 1  # Exclude _summary
            }

        return narrative_comparison

    def calculate_market_regime_performance(
        self,
        wallet_returns: pd.Series,
        benchmark_returns: pd.Series,
        regime_threshold: float = 0.02
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance across different market regimes.

        Args:
            wallet_returns: Series of wallet daily returns
            benchmark_returns: Series of benchmark daily returns
            regime_threshold: Threshold for defining bull/bear markets

        Returns:
            Dictionary containing performance by market regime
        """
        # Align data
        aligned_data = pd.DataFrame({
            'wallet': wallet_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(aligned_data) < 30:
            return {}

        # Define market regimes based on benchmark performance
        rolling_benchmark = aligned_data['benchmark'].rolling(window=20).mean()

        regimes = pd.Series(index=aligned_data.index, dtype=str)
        regimes[rolling_benchmark > regime_threshold] = 'bull'
        regimes[rolling_benchmark < -regime_threshold] = 'bear'
        regimes[(rolling_benchmark >= -regime_threshold) & (rolling_benchmark <= regime_threshold)] = 'sideways'

        regime_performance = {}

        for regime in ['bull', 'bear', 'sideways']:
            regime_mask = regimes == regime

            if regime_mask.sum() < 5:  # Need minimum observations
                continue

            regime_wallet_returns = aligned_data.loc[regime_mask, 'wallet']
            regime_benchmark_returns = aligned_data.loc[regime_mask, 'benchmark']

            # Calculate metrics for this regime
            regime_metrics = self._calculate_wallet_metrics(regime_wallet_returns)
            benchmark_metrics = self._calculate_wallet_metrics(regime_benchmark_returns)

            # Calculate alpha/beta for this regime
            alpha_beta = self.calculate_alpha_beta(
                regime_wallet_returns,
                regime_benchmark_returns
            )

            regime_performance[regime] = {
                **regime_metrics,
                'alpha': alpha_beta['alpha'],
                'beta': alpha_beta['beta'],
                'excess_return': regime_metrics['total_return'] - benchmark_metrics['total_return'],
                'days_in_regime': regime_mask.sum(),
                'win_rate': (regime_wallet_returns > 0).mean()
            }

        return regime_performance

    def calculate_rolling_beta(
        self,
        wallet_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """
        Calculate rolling beta over time.

        Args:
            wallet_returns: Series of wallet daily returns
            benchmark_returns: Series of benchmark daily returns
            window: Rolling window size in days

        Returns:
            Series of rolling beta values
        """
        # Align data
        aligned_data = pd.DataFrame({
            'wallet': wallet_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(aligned_data) < window:
            return pd.Series(dtype=float)

        # Calculate rolling beta
        rolling_betas = []
        dates = []

        for i in range(window - 1, len(aligned_data)):
            window_data = aligned_data.iloc[i - window + 1:i + 1]

            wallet_window = window_data['wallet']
            benchmark_window = window_data['benchmark']

            # Calculate beta for this window
            if benchmark_window.var() > 0:
                beta = np.cov(wallet_window, benchmark_window)[0, 1] / benchmark_window.var()
            else:
                beta = 0.0

            rolling_betas.append(beta)
            dates.append(aligned_data.index[i])

        return pd.Series(rolling_betas, index=dates, name='rolling_beta')

    def generate_benchmark_report(
        self,
        wallet_returns: pd.Series,
        benchmarks: Dict[str, pd.Series],
        peer_returns: Optional[Dict[str, pd.Series]] = None,
        narrative_returns: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive benchmark comparison report.

        Args:
            wallet_returns: Series of wallet daily returns
            benchmarks: Dictionary of benchmark return series
            peer_returns: Optional peer group return series
            narrative_returns: Optional narrative benchmark return series

        Returns:
            Comprehensive benchmark analysis report
        """
        report = {
            'wallet_metrics': self._calculate_wallet_metrics(wallet_returns),
            'benchmark_comparisons': {},
            'peer_rankings': {},
            'narrative_analysis': {},
            'market_regime_performance': {}
        }

        # Benchmark comparisons
        if benchmarks:
            report['benchmark_comparisons'] = self.create_benchmark_comparison(
                wallet_returns, benchmarks
            )

        # Peer group analysis
        if peer_returns:
            report['peer_rankings'] = self.calculate_peer_group_rankings(
                wallet_returns, peer_returns
            )

        # Narrative analysis
        if narrative_returns:
            report['narrative_analysis'] = self.create_narrative_benchmarks(
                wallet_returns, narrative_returns
            )

        # Market regime analysis (using first benchmark if available)
        if benchmarks:
            first_benchmark = next(iter(benchmarks.values()))
            report['market_regime_performance'] = self.calculate_market_regime_performance(
                wallet_returns, first_benchmark
            )

        # Calculate overall performance score
        report['performance_score'] = self._calculate_performance_score(report)

        return report

    def _calculate_performance_score(self, report: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate overall performance score based on multiple factors.

        Args:
            report: Benchmark analysis report

        Returns:
            Dictionary containing performance scores
        """
        scores = {}

        # Absolute performance score (0-100)
        wallet_metrics = report.get('wallet_metrics', {})
        sharpe_ratio = wallet_metrics.get('sharpe_ratio', 0)
        total_return = wallet_metrics.get('total_return', 0)

        # Score based on Sharpe ratio (capped at 3.0)
        sharpe_score = min(100, max(0, (sharpe_ratio + 1) / 4 * 100))

        # Score based on total return (capped at 100% return)
        return_score = min(100, max(0, (total_return + 0.5) / 1.5 * 100))

        scores['absolute_score'] = (sharpe_score + return_score) / 2

        # Relative performance score
        peer_rankings = report.get('peer_rankings', {})
        if peer_rankings.get('percentiles'):
            avg_percentile = np.mean(list(peer_rankings['percentiles'].values()))
            scores['relative_score'] = avg_percentile
        else:
            scores['relative_score'] = 50  # Neutral if no peer data

        # Benchmark alpha score
        benchmark_comparisons = report.get('benchmark_comparisons', {})
        if benchmark_comparisons:
            alphas = [comp.get('alpha', 0) for comp in benchmark_comparisons.values()]
            avg_alpha = np.mean(alphas)
            # Convert alpha to 0-100 score (alpha of 0.1 = 100 points)
            alpha_score = min(100, max(0, avg_alpha / 0.1 * 100 + 50))
            scores['alpha_score'] = alpha_score
        else:
            scores['alpha_score'] = 50

        # Overall score (weighted average)
        scores['overall_score'] = (
            scores['absolute_score'] * 0.4 +
            scores['relative_score'] * 0.3 +
            scores['alpha_score'] * 0.3
        )

        return scores