"""
Portfolio Concentration Calculator
Category 3: 6 features for portfolio diversification analysis

Features:
1. portfolio_hhi - Herfindahl-Hirschman Index (0-10000)
2. portfolio_gini - Gini coefficient (0-1)
3. top3_concentration_pct - % of portfolio value in top 3 tokens
4. num_tokens_avg - Average number of tokens held
5. num_tokens_std - Standard deviation of number of tokens
6. portfolio_turnover - Portfolio composition change rate
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConcentrationMetrics:
    """Container for wallet concentration metrics"""
    wallet_address: str
    portfolio_hhi: float
    portfolio_gini: float
    top3_concentration_pct: float
    num_tokens_avg: float
    num_tokens_std: float
    portfolio_turnover: float


class ConcentrationCalculator:
    """Calculate portfolio concentration and diversification metrics"""

    def __init__(self, balances_df: pd.DataFrame):
        """
        Initialize calculator with balance data

        Args:
            balances_df: Daily balance snapshots
        """
        self.balances_df = balances_df

        # Prepare data
        self._prepare_data()

    def _prepare_data(self):
        """Prepare and validate input data"""
        # Convert dates
        if 'snapshot_date' in self.balances_df.columns:
            self.balances_df['snapshot_date'] = pd.to_datetime(self.balances_df['snapshot_date'])

        # Use balance_formatted as the balance column
        # Note: We don't have USD values, so we'll use token amounts
        if 'balance_formatted' in self.balances_df.columns:
            self.balances_df['balance'] = pd.to_numeric(
                self.balances_df['balance_formatted'],
                errors='coerce'
            ).fillna(0)
        else:
            self.balances_df['balance'] = 0

    def calculate_all_metrics(self, wallet_address: str) -> Optional[ConcentrationMetrics]:
        """
        Calculate all concentration metrics for a wallet

        Args:
            wallet_address: Ethereum wallet address

        Returns:
            ConcentrationMetrics object or None if insufficient data
        """
        try:
            # Get wallet balances
            wallet_balances = self.balances_df[
                self.balances_df['wallet_address'].str.lower() == wallet_address.lower()
            ].copy()

            if len(wallet_balances) == 0:
                logger.warning(f"No balance data for wallet {wallet_address}")
                return None

            # Calculate each metric
            hhi = self._calculate_hhi(wallet_balances)
            gini = self._calculate_gini(wallet_balances)
            top3_pct = self._calculate_top3_concentration(wallet_balances)
            num_tokens_avg, num_tokens_std = self._calculate_token_count_stats(wallet_balances)
            turnover = self._calculate_portfolio_turnover(wallet_balances)

            return ConcentrationMetrics(
                wallet_address=wallet_address,
                portfolio_hhi=hhi,
                portfolio_gini=gini,
                top3_concentration_pct=top3_pct,
                num_tokens_avg=num_tokens_avg,
                num_tokens_std=num_tokens_std,
                portfolio_turnover=turnover
            )

        except Exception as e:
            logger.error(f"Error calculating concentration metrics for {wallet_address}: {e}")
            return None

    def _calculate_hhi(self, balances: pd.DataFrame) -> float:
        """
        Calculate Herfindahl-Hirschman Index (HHI)

        HHI = sum of squared market shares (in percentage points)
        - 10,000 = perfect concentration (one token = 100%)
        - <1,500 = diversified
        - 1,500-2,500 = moderate concentration
        - >2,500 = highly concentrated
        """
        if len(balances) == 0:
            return 0.0

        # Calculate average HHI across all snapshots
        hhis = []

        for date, day_balances in balances.groupby('snapshot_date'):
            # Get total portfolio value for this day
            total_value = day_balances['balance'].sum()

            if total_value == 0:
                continue

            # Calculate each token's percentage share
            shares = (day_balances['balance'] / total_value * 100) ** 2
            hhi = shares.sum()
            hhis.append(hhi)

        if len(hhis) == 0:
            return 0.0

        avg_hhi = np.mean(hhis)
        return float(avg_hhi)

    def _calculate_gini(self, balances: pd.DataFrame) -> float:
        """
        Calculate Gini coefficient

        Measures inequality in portfolio distribution
        - 0 = perfect equality (all tokens same value)
        - 1 = perfect inequality (one token has everything)
        """
        if len(balances) == 0:
            return 0.0

        # Calculate average Gini across all snapshots
        ginis = []

        for date, day_balances in balances.groupby('snapshot_date'):
            values = day_balances['balance'].values
            values = values[values > 0]  # Remove zeros

            if len(values) == 0:
                continue

            # Sort values
            sorted_values = np.sort(values)
            n = len(sorted_values)

            # Calculate Gini coefficient
            # Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
            cumsum = np.cumsum(sorted_values)
            total = cumsum[-1]

            if total == 0:
                continue

            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * total) - (n + 1) / n
            ginis.append(gini)

        if len(ginis) == 0:
            return 0.0

        avg_gini = np.mean(ginis)
        return float(avg_gini)

    def _calculate_top3_concentration(self, balances: pd.DataFrame) -> float:
        """
        Calculate percentage of portfolio value in top 3 tokens

        Higher % = more concentrated in few tokens
        """
        if len(balances) == 0:
            return 0.0

        # Calculate average top3 concentration across all snapshots
        top3_pcts = []

        for date, day_balances in balances.groupby('snapshot_date'):
            total_value = day_balances['balance'].sum()

            if total_value == 0:
                continue

            # Sort by balance and take top 3
            top3_value = day_balances.nlargest(3, 'balance')['balance'].sum()
            top3_pct = (top3_value / total_value) * 100
            top3_pcts.append(top3_pct)

        if len(top3_pcts) == 0:
            return 0.0

        avg_top3 = np.mean(top3_pcts)
        return float(avg_top3)

    def _calculate_token_count_stats(self, balances: pd.DataFrame) -> tuple:
        """
        Calculate average and standard deviation of token count

        Returns:
            (avg_tokens, std_tokens)
        """
        if len(balances) == 0:
            return 0.0, 0.0

        # Count unique tokens per day
        daily_counts = balances.groupby('snapshot_date')['token_address'].nunique()

        if len(daily_counts) == 0:
            return 0.0, 0.0

        avg_tokens = daily_counts.mean()
        std_tokens = daily_counts.std() if len(daily_counts) > 1 else 0.0

        return float(avg_tokens), float(std_tokens)

    def _calculate_portfolio_turnover(self, balances: pd.DataFrame) -> float:
        """
        Calculate portfolio turnover rate

        Measures how much the portfolio composition changes day-to-day
        Higher value = more active rebalancing
        """
        if len(balances) == 0:
            return 0.0

        # Get unique dates sorted
        dates = sorted(balances['snapshot_date'].unique())

        if len(dates) < 2:
            return 0.0

        # Calculate turnover between consecutive days
        turnovers = []

        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]

            prev_tokens = set(balances[balances['snapshot_date'] == prev_date]['token_address'])
            curr_tokens = set(balances[balances['snapshot_date'] == curr_date]['token_address'])

            # Turnover = (tokens added + tokens removed) / average portfolio size
            tokens_added = len(curr_tokens - prev_tokens)
            tokens_removed = len(prev_tokens - curr_tokens)
            avg_size = (len(prev_tokens) + len(curr_tokens)) / 2

            if avg_size > 0:
                turnover = (tokens_added + tokens_removed) / avg_size
                turnovers.append(turnover)

        if len(turnovers) == 0:
            return 0.0

        avg_turnover = np.mean(turnovers)
        return float(avg_turnover)


def calculate_concentration_metrics_batch(
    wallet_addresses: list,
    balances_df: pd.DataFrame,
    batch_size: int = 100
) -> pd.DataFrame:
    """
    Calculate concentration metrics for multiple wallets

    Args:
        wallet_addresses: List of wallet addresses
        balances_df: Balances DataFrame
        batch_size: Number of wallets to process at once

    Returns:
        DataFrame with concentration metrics for all wallets
    """
    calculator = ConcentrationCalculator(balances_df=balances_df)

    results = []
    total = len(wallet_addresses)

    for i, wallet in enumerate(wallet_addresses):
        if i % 100 == 0:
            logger.info(f"Processing wallet {i+1}/{total}")

        metrics = calculator.calculate_all_metrics(wallet)

        if metrics:
            results.append({
                'wallet_address': metrics.wallet_address,
                'portfolio_hhi': metrics.portfolio_hhi,
                'portfolio_gini': metrics.portfolio_gini,
                'top3_concentration_pct': metrics.top3_concentration_pct,
                'num_tokens_avg': metrics.num_tokens_avg,
                'num_tokens_std': metrics.num_tokens_std,
                'portfolio_turnover': metrics.portfolio_turnover
            })

    logger.info(f"Calculated concentration metrics for {len(results)}/{total} wallets")

    return pd.DataFrame(results)
