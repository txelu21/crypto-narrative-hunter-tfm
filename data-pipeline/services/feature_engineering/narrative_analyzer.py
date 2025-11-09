"""
Narrative Exposure Analyzer
Category 4: 6 features for narrative preference analysis

Features:
1. narrative_diversity_score - Shannon entropy of narrative exposure
2. primary_narrative_pct - % of portfolio in primary narrative
3. defi_exposure_pct - % exposure to DeFi tokens
4. ai_exposure_pct - % exposure to AI tokens
5. meme_exposure_pct - % exposure to meme tokens
6. stablecoin_usage_ratio - Ratio of stablecoin usage
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class NarrativeMetrics:
    """Container for wallet narrative exposure metrics"""
    wallet_address: str
    narrative_diversity_score: float
    primary_narrative_pct: float
    defi_exposure_pct: float
    ai_exposure_pct: float
    meme_exposure_pct: float
    stablecoin_usage_ratio: float


class NarrativeAnalyzer:
    """Analyze wallet narrative exposure and preferences"""

    def __init__(
        self,
        balances_df: pd.DataFrame,
        tokens_df: pd.DataFrame
    ):
        """
        Initialize analyzer with balance and token data

        Args:
            balances_df: Daily balance snapshots
            tokens_df: Token metadata with narrative categories
        """
        self.balances_df = balances_df
        self.tokens_df = tokens_df

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

        # Create token narrative lookup
        if 'token_address' in self.tokens_df.columns and 'narrative_category' in self.tokens_df.columns:
            self.token_narratives = dict(zip(
                self.tokens_df['token_address'].str.lower(),
                self.tokens_df['narrative_category']
            ))
        else:
            self.token_narratives = {}

    def calculate_all_metrics(self, wallet_address: str) -> Optional[NarrativeMetrics]:
        """
        Calculate all narrative metrics for a wallet

        Args:
            wallet_address: Ethereum wallet address

        Returns:
            NarrativeMetrics object or None if insufficient data
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
            diversity = self._calculate_narrative_diversity(wallet_balances)
            primary_pct = self._calculate_primary_narrative_pct(wallet_balances)
            defi_pct = self._calculate_category_exposure(wallet_balances, 'DeFi')
            ai_pct = self._calculate_category_exposure(wallet_balances, 'AI')
            meme_pct = self._calculate_category_exposure(wallet_balances, 'Meme')
            stablecoin_ratio = self._calculate_stablecoin_usage(wallet_balances)

            return NarrativeMetrics(
                wallet_address=wallet_address,
                narrative_diversity_score=diversity,
                primary_narrative_pct=primary_pct,
                defi_exposure_pct=defi_pct,
                ai_exposure_pct=ai_pct,
                meme_exposure_pct=meme_pct,
                stablecoin_usage_ratio=stablecoin_ratio
            )

        except Exception as e:
            logger.error(f"Error calculating narrative metrics for {wallet_address}: {e}")
            return None

    def _calculate_narrative_diversity(self, balances: pd.DataFrame) -> float:
        """
        Calculate narrative diversity using Shannon entropy

        Higher score = more diverse narrative exposure
        Lower score = concentrated in one narrative
        """
        if len(balances) == 0:
            return 0.0

        # Get average narrative exposure across all days
        narrative_exposures = []

        for date, day_balances in balances.groupby('snapshot_date'):
            # Map tokens to narratives
            day_balances['narrative'] = day_balances['token_address'].str.lower().map(
                self.token_narratives
            ).fillna('Other')

            # Calculate total value per narrative
            narrative_totals = day_balances.groupby('narrative')['balance'].sum()
            total_balance = narrative_totals.sum()

            if total_balance > 0:
                # Calculate entropy
                probabilities = narrative_totals / total_balance
                entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                narrative_exposures.append(entropy)

        if len(narrative_exposures) == 0:
            return 0.0

        avg_diversity = np.mean(narrative_exposures)
        return float(avg_diversity)

    def _calculate_primary_narrative_pct(self, balances: pd.DataFrame) -> float:
        """
        Calculate percentage in primary (dominant) narrative

        Higher % = more concentrated in one narrative
        """
        if len(balances) == 0:
            return 0.0

        primary_pcts = []

        for date, day_balances in balances.groupby('snapshot_date'):
            # Map tokens to narratives
            day_balances['narrative'] = day_balances['token_address'].str.lower().map(
                self.token_narratives
            ).fillna('Other')

            # Calculate total value per narrative
            narrative_totals = day_balances.groupby('narrative')['balance'].sum()
            total_balance = narrative_totals.sum()

            if total_balance > 0:
                # Get primary narrative percentage
                primary_value = narrative_totals.max()
                primary_pct = (primary_value / total_balance) * 100
                primary_pcts.append(primary_pct)

        if len(primary_pcts) == 0:
            return 0.0

        avg_primary_pct = np.mean(primary_pcts)
        return float(avg_primary_pct)

    def _calculate_category_exposure(
        self,
        balances: pd.DataFrame,
        category: str
    ) -> float:
        """
        Calculate exposure to specific category (DeFi, AI, Meme, etc.)

        Args:
            balances: Wallet balances
            category: Category name to calculate

        Returns:
            Percentage of portfolio in this category
        """
        if len(balances) == 0:
            return 0.0

        category_pcts = []

        for date, day_balances in balances.groupby('snapshot_date'):
            # Map tokens to narratives
            day_balances['narrative'] = day_balances['token_address'].str.lower().map(
                self.token_narratives
            ).fillna('Other')

            # Calculate category exposure
            category_balance = day_balances[
                day_balances['narrative'] == category
            ]['balance'].sum()

            total_balance = day_balances['balance'].sum()

            if total_balance > 0:
                category_pct = (category_balance / total_balance) * 100
                category_pcts.append(category_pct)

        if len(category_pcts) == 0:
            return 0.0

        avg_category_pct = np.mean(category_pcts)
        return float(avg_category_pct)

    def _calculate_stablecoin_usage(self, balances: pd.DataFrame) -> float:
        """
        Calculate stablecoin usage ratio

        Ratio of days with stablecoins to total days
        """
        if len(balances) == 0:
            return 0.0

        days_with_stablecoins = 0
        total_days = 0

        for date, day_balances in balances.groupby('snapshot_date'):
            total_days += 1

            # Map tokens to narratives
            day_balances['narrative'] = day_balances['token_address'].str.lower().map(
                self.token_narratives
            ).fillna('Other')

            # Check if any stablecoins held
            stablecoin_balance = day_balances[
                day_balances['narrative'] == 'Stablecoin'
            ]['balance'].sum()

            if stablecoin_balance > 0:
                days_with_stablecoins += 1

        if total_days == 0:
            return 0.0

        stablecoin_ratio = days_with_stablecoins / total_days
        return float(stablecoin_ratio)


def calculate_narrative_metrics_batch(
    wallet_addresses: list,
    balances_df: pd.DataFrame,
    tokens_df: pd.DataFrame,
    batch_size: int = 100
) -> pd.DataFrame:
    """
    Calculate narrative metrics for multiple wallets

    Args:
        wallet_addresses: List of wallet addresses
        balances_df: Balances DataFrame
        tokens_df: Tokens DataFrame with categories
        batch_size: Number of wallets to process at once

    Returns:
        DataFrame with narrative metrics for all wallets
    """
    analyzer = NarrativeAnalyzer(
        balances_df=balances_df,
        tokens_df=tokens_df
    )

    results = []
    total = len(wallet_addresses)

    for i, wallet in enumerate(wallet_addresses):
        if i % 100 == 0:
            logger.info(f"Processing wallet {i+1}/{total}")

        metrics = analyzer.calculate_all_metrics(wallet)

        if metrics:
            results.append({
                'wallet_address': metrics.wallet_address,
                'narrative_diversity_score': metrics.narrative_diversity_score,
                'primary_narrative_pct': metrics.primary_narrative_pct,
                'defi_exposure_pct': metrics.defi_exposure_pct,
                'ai_exposure_pct': metrics.ai_exposure_pct,
                'meme_exposure_pct': metrics.meme_exposure_pct,
                'stablecoin_usage_ratio': metrics.stablecoin_usage_ratio
            })

    logger.info(f"Calculated narrative metrics for {len(results)}/{total} wallets")

    return pd.DataFrame(results)
