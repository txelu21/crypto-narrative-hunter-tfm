"""
Accumulation/Distribution Detector
Category 5: 6 features for accumulation/distribution phase analysis

Features:
1. accumulation_phase_days - Number of days net accumulating
2. distribution_phase_days - Number of days net distributing
3. accumulation_intensity - Average accumulation rate
4. distribution_intensity - Average distribution rate
5. balance_volatility - Portfolio size volatility
6. trend_direction - Overall trend: accumulating (+1) or distributing (-1)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AccumulationMetrics:
    """Container for wallet accumulation/distribution metrics"""
    wallet_address: str
    accumulation_phase_days: float
    distribution_phase_days: float
    accumulation_intensity: float
    distribution_intensity: float
    balance_volatility: float
    trend_direction: float


class AccumulationDetector:
    """Detect accumulation and distribution phases in wallet behavior"""

    def __init__(self, balances_df: pd.DataFrame):
        """
        Initialize detector with balance data

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

        # Ensure balance column
        if 'balance_formatted' in self.balances_df.columns:
            self.balances_df['balance'] = pd.to_numeric(
                self.balances_df['balance_formatted'],
                errors='coerce'
            ).fillna(0)
        else:
            self.balances_df['balance'] = 0

    def calculate_all_metrics(self, wallet_address: str) -> Optional[AccumulationMetrics]:
        """
        Calculate all accumulation/distribution metrics for a wallet

        Args:
            wallet_address: Ethereum wallet address

        Returns:
            AccumulationMetrics object or None if insufficient data
        """
        try:
            # Get wallet balances
            wallet_balances = self.balances_df[
                self.balances_df['wallet_address'].str.lower() == wallet_address.lower()
            ].copy()

            if len(wallet_balances) == 0:
                logger.warning(f"No balance data for wallet {wallet_address}")
                return None

            # Calculate daily portfolio values
            daily_values = wallet_balances.groupby('snapshot_date')['balance'].sum().sort_index()

            if len(daily_values) < 2:
                # Need at least 2 days to detect trends
                return None

            # Calculate each metric
            accum_days, distrib_days = self._calculate_phase_days(daily_values)
            accum_intensity = self._calculate_accumulation_intensity(daily_values)
            distrib_intensity = self._calculate_distribution_intensity(daily_values)
            volatility = self._calculate_balance_volatility(daily_values)
            trend = self._calculate_trend_direction(daily_values)

            return AccumulationMetrics(
                wallet_address=wallet_address,
                accumulation_phase_days=accum_days,
                distribution_phase_days=distrib_days,
                accumulation_intensity=accum_intensity,
                distribution_intensity=distrib_intensity,
                balance_volatility=volatility,
                trend_direction=trend
            )

        except Exception as e:
            logger.error(f"Error calculating accumulation metrics for {wallet_address}: {e}")
            return None

    def _calculate_phase_days(self, daily_values: pd.Series) -> tuple:
        """
        Calculate number of accumulation vs distribution days

        Returns:
            (accumulation_days, distribution_days)
        """
        if len(daily_values) < 2:
            return 0.0, 0.0

        # Calculate daily changes
        daily_changes = daily_values.diff()

        # Count accumulation days (positive change)
        accumulation_days = (daily_changes > 0).sum()

        # Count distribution days (negative change)
        distribution_days = (daily_changes < 0).sum()

        return float(accumulation_days), float(distribution_days)

    def _calculate_accumulation_intensity(self, daily_values: pd.Series) -> float:
        """
        Calculate accumulation intensity

        Average rate of growth on accumulation days
        """
        if len(daily_values) < 2:
            return 0.0

        # Calculate daily changes
        daily_changes = daily_values.diff()

        # Get accumulation days only
        accumulation_changes = daily_changes[daily_changes > 0]

        if len(accumulation_changes) == 0:
            return 0.0

        # Calculate average accumulation rate
        # Use percentage change for normalization
        prev_values = daily_values.shift(1)[daily_changes > 0]
        pct_changes = (accumulation_changes / prev_values) * 100

        # Remove infinities and NaNs
        pct_changes = pct_changes.replace([np.inf, -np.inf], np.nan).dropna()

        if len(pct_changes) == 0:
            return 0.0

        avg_intensity = pct_changes.mean()
        return float(avg_intensity)

    def _calculate_distribution_intensity(self, daily_values: pd.Series) -> float:
        """
        Calculate distribution intensity

        Average rate of decline on distribution days
        """
        if len(daily_values) < 2:
            return 0.0

        # Calculate daily changes
        daily_changes = daily_values.diff()

        # Get distribution days only
        distribution_changes = daily_changes[daily_changes < 0]

        if len(distribution_changes) == 0:
            return 0.0

        # Calculate average distribution rate (absolute value)
        # Use percentage change for normalization
        prev_values = daily_values.shift(1)[daily_changes < 0]
        pct_changes = abs((distribution_changes / prev_values) * 100)

        # Remove infinities and NaNs
        pct_changes = pct_changes.replace([np.inf, -np.inf], np.nan).dropna()

        if len(pct_changes) == 0:
            return 0.0

        avg_intensity = pct_changes.mean()
        return float(avg_intensity)

    def _calculate_balance_volatility(self, daily_values: pd.Series) -> float:
        """
        Calculate portfolio balance volatility

        Standard deviation of daily percentage changes
        """
        if len(daily_values) < 2:
            return 0.0

        # Calculate daily percentage changes
        pct_changes = daily_values.pct_change() * 100

        # Remove infinities and NaNs
        pct_changes = pct_changes.replace([np.inf, -np.inf], np.nan).dropna()

        if len(pct_changes) == 0:
            return 0.0

        volatility = pct_changes.std()
        return float(volatility)

    def _calculate_trend_direction(self, daily_values: pd.Series) -> float:
        """
        Calculate overall trend direction

        Returns:
            +1.0 = net accumulation (growing portfolio)
            0.0 = neutral (flat)
            -1.0 = net distribution (shrinking portfolio)
        """
        if len(daily_values) < 2:
            return 0.0

        # Compare first and last values
        first_value = daily_values.iloc[0]
        last_value = daily_values.iloc[-1]

        if first_value == 0:
            return 0.0

        # Calculate overall change percentage
        overall_change = ((last_value - first_value) / first_value) * 100

        # Normalize to -1 to +1 scale
        # Using tanh to squash to range
        trend = np.tanh(overall_change / 100)

        return float(trend)


def calculate_accumulation_metrics_batch(
    wallet_addresses: list,
    balances_df: pd.DataFrame,
    batch_size: int = 100
) -> pd.DataFrame:
    """
    Calculate accumulation/distribution metrics for multiple wallets

    Args:
        wallet_addresses: List of wallet addresses
        balances_df: Balances DataFrame
        batch_size: Number of wallets to process at once

    Returns:
        DataFrame with accumulation metrics for all wallets
    """
    detector = AccumulationDetector(balances_df=balances_df)

    results = []
    total = len(wallet_addresses)

    for i, wallet in enumerate(wallet_addresses):
        if i % 100 == 0:
            logger.info(f"Processing wallet {i+1}/{total}")

        metrics = detector.calculate_all_metrics(wallet)

        if metrics:
            results.append({
                'wallet_address': metrics.wallet_address,
                'accumulation_phase_days': metrics.accumulation_phase_days,
                'distribution_phase_days': metrics.distribution_phase_days,
                'accumulation_intensity': metrics.accumulation_intensity,
                'distribution_intensity': metrics.distribution_intensity,
                'balance_volatility': metrics.balance_volatility,
                'trend_direction': metrics.trend_direction
            })

    logger.info(f"Calculated accumulation metrics for {len(results)}/{total} wallets")

    return pd.DataFrame(results)
