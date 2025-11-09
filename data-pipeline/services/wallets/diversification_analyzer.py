"""
Portfolio diversification analysis module for wallet assessment.

This module provides functionality to analyze portfolio diversification
metrics including concentration indices, token correlations, sector
allocations, and rebalancing patterns.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from decimal import Decimal
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class DiversificationAnalyzer:
    """
    Analyzes portfolio diversification and concentration metrics.

    This class provides methods for calculating diversification indices,
    correlation analysis, sector allocations, and rebalancing patterns
    to assess portfolio risk management strategies.
    """

    def __init__(self):
        """Initialize the DiversificationAnalyzer."""
        self.correlation_cache = {}

    def calculate_herfindahl_hirschman_index(
        self,
        portfolio_weights: Dict[str, float]
    ) -> float:
        """
        Calculate the Herfindahl-Hirschman Index for portfolio concentration.

        The HHI ranges from 0 to 1, where:
        - Values close to 0 indicate high diversification
        - Values close to 1 indicate high concentration

        Args:
            portfolio_weights: Dictionary mapping token symbols to portfolio weights

        Returns:
            HHI value between 0 and 1
        """
        if not portfolio_weights:
            return 0.0

        # Ensure weights sum to 1
        total_weight = sum(portfolio_weights.values())
        if total_weight == 0:
            return 0.0

        normalized_weights = {
            token: weight / total_weight
            for token, weight in portfolio_weights.items()
        }

        # Calculate HHI as sum of squared weights
        hhi = sum(weight ** 2 for weight in normalized_weights.values())

        return hhi

    def calculate_effective_tokens(
        self,
        portfolio_weights: Dict[str, float]
    ) -> float:
        """
        Calculate the effective number of tokens in the portfolio.

        The effective number is 1/HHI and represents the number of
        equally-weighted tokens that would give the same HHI.

        Args:
            portfolio_weights: Dictionary mapping token symbols to portfolio weights

        Returns:
            Effective number of tokens
        """
        hhi = self.calculate_herfindahl_hirschman_index(portfolio_weights)

        if hhi == 0:
            return 0.0

        return 1.0 / hhi

    def analyze_token_diversification(
        self,
        transactions: List[Dict],
        narrative_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze token diversification across narrative categories.

        Args:
            transactions: List of transaction dictionaries
            narrative_mapping: Optional mapping of tokens to narrative categories

        Returns:
            Dictionary containing diversification metrics by narrative
        """
        if not transactions:
            return {
                'unique_tokens': 0,
                'unique_narratives': 0,
                'tokens_per_narrative': {},
                'narrative_weights': {},
                'narrative_hhi': 0.0
            }

        # Track token positions
        token_positions = defaultdict(float)
        narrative_positions = defaultdict(lambda: defaultdict(float))

        for txn in transactions:
            token = txn['token_symbol']

            if txn['type'] == 'buy':
                token_positions[token] += txn['amount'] * txn.get('price', 0)
            elif txn['type'] == 'sell':
                token_positions[token] -= txn['amount'] * txn.get('price', 0)

            # Track by narrative if mapping provided
            if narrative_mapping and token in narrative_mapping:
                narrative = narrative_mapping[token]
                if txn['type'] == 'buy':
                    narrative_positions[narrative][token] += txn['amount'] * txn.get('price', 0)
                elif txn['type'] == 'sell':
                    narrative_positions[narrative][token] -= txn['amount'] * txn.get('price', 0)

        # Calculate narrative weights
        total_value = sum(max(0, value) for value in token_positions.values())
        narrative_weights = {}
        tokens_per_narrative = {}

        if narrative_mapping:
            for narrative, tokens in narrative_positions.items():
                narrative_value = sum(max(0, value) for value in tokens.values())
                if total_value > 0:
                    narrative_weights[narrative] = narrative_value / total_value

                # Count unique tokens in narrative
                active_tokens = [t for t, v in tokens.items() if v > 0]
                tokens_per_narrative[narrative] = len(active_tokens)

        # Calculate narrative HHI
        narrative_hhi = self.calculate_herfindahl_hirschman_index(narrative_weights)

        return {
            'unique_tokens': len([t for t, v in token_positions.items() if v > 0]),
            'unique_narratives': len(narrative_weights),
            'tokens_per_narrative': tokens_per_narrative,
            'narrative_weights': narrative_weights,
            'narrative_hhi': narrative_hhi,
            'effective_narratives': 1.0 / narrative_hhi if narrative_hhi > 0 else 0
        }

    def calculate_correlation_matrix(
        self,
        price_data: Dict[str, pd.Series],
        tokens: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between token holdings.

        Args:
            price_data: Dictionary mapping token symbols to price series
            tokens: Optional list of tokens to analyze (defaults to all)

        Returns:
            Correlation matrix as DataFrame
        """
        if not price_data:
            return pd.DataFrame()

        # Use specified tokens or all available
        if tokens:
            selected_tokens = [t for t in tokens if t in price_data]
        else:
            selected_tokens = list(price_data.keys())

        if len(selected_tokens) < 2:
            return pd.DataFrame()

        # Create price DataFrame
        price_df = pd.DataFrame({
            token: price_data[token]
            for token in selected_tokens
        })

        # Calculate returns
        returns = price_df.pct_change().dropna()

        # Calculate correlation matrix
        correlation_matrix = returns.corr()

        return correlation_matrix

    def analyze_portfolio_correlation(
        self,
        portfolio_weights: Dict[str, float],
        correlation_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Analyze portfolio-level correlation metrics.

        Args:
            portfolio_weights: Dictionary mapping token symbols to weights
            correlation_matrix: Correlation matrix between tokens

        Returns:
            Dictionary containing portfolio correlation metrics
        """
        if correlation_matrix.empty or not portfolio_weights:
            return {
                'average_correlation': 0.0,
                'portfolio_variance': 0.0,
                'diversification_ratio': 1.0
            }

        # Filter weights for tokens in correlation matrix
        available_tokens = set(correlation_matrix.columns)
        filtered_weights = {
            token: weight
            for token, weight in portfolio_weights.items()
            if token in available_tokens
        }

        if not filtered_weights:
            return {
                'average_correlation': 0.0,
                'portfolio_variance': 0.0,
                'diversification_ratio': 1.0
            }

        # Normalize weights
        total_weight = sum(filtered_weights.values())
        weights = np.array([
            filtered_weights.get(token, 0) / total_weight
            for token in correlation_matrix.columns
        ])

        # Calculate average pairwise correlation
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        avg_correlation = (correlation_matrix.values * mask).sum() / mask.sum()

        # Calculate portfolio variance
        cov_matrix = correlation_matrix.values
        portfolio_variance = weights @ cov_matrix @ weights.T

        # Calculate diversification ratio
        # DR = weighted average volatility / portfolio volatility
        individual_vols = np.sqrt(np.diag(cov_matrix))
        weighted_avg_vol = weights @ individual_vols
        portfolio_vol = np.sqrt(portfolio_variance)

        if portfolio_vol > 0:
            diversification_ratio = weighted_avg_vol / portfolio_vol
        else:
            diversification_ratio = 1.0

        return {
            'average_correlation': float(avg_correlation),
            'portfolio_variance': float(portfolio_variance),
            'diversification_ratio': float(diversification_ratio)
        }

    def calculate_maximum_position_size(
        self,
        portfolio_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate maximum position size metrics.

        Args:
            portfolio_weights: Dictionary mapping token symbols to weights

        Returns:
            Dictionary containing position size metrics
        """
        if not portfolio_weights:
            return {
                'max_position_size': 0.0,
                'max_position_token': None,
                'top3_concentration': 0.0,
                'top5_concentration': 0.0
            }

        # Normalize weights
        total_weight = sum(portfolio_weights.values())
        if total_weight == 0:
            return {
                'max_position_size': 0.0,
                'max_position_token': None,
                'top3_concentration': 0.0,
                'top5_concentration': 0.0
            }

        normalized_weights = {
            token: weight / total_weight * 100
            for token, weight in portfolio_weights.items()
        }

        # Sort by weight
        sorted_positions = sorted(
            normalized_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Calculate metrics
        max_position = sorted_positions[0] if sorted_positions else (None, 0)
        top3_concentration = sum(weight for _, weight in sorted_positions[:3])
        top5_concentration = sum(weight for _, weight in sorted_positions[:5])

        return {
            'max_position_size': max_position[1],
            'max_position_token': max_position[0],
            'top3_concentration': top3_concentration,
            'top5_concentration': top5_concentration,
            'position_count': len(sorted_positions)
        }

    def analyze_sector_allocation(
        self,
        portfolio_weights: Dict[str, float],
        narrative_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Analyze portfolio allocation across sectors/narratives.

        Args:
            portfolio_weights: Dictionary mapping token symbols to weights
            narrative_mapping: Mapping of tokens to narrative categories

        Returns:
            Dictionary containing sector allocation metrics
        """
        if not portfolio_weights or not narrative_mapping:
            return {
                'sector_weights': {},
                'sector_count': 0,
                'dominant_sector': None,
                'sector_hhi': 0.0
            }

        # Calculate sector weights
        sector_weights = defaultdict(float)
        total_weight = sum(portfolio_weights.values())

        if total_weight == 0:
            return {
                'sector_weights': {},
                'sector_count': 0,
                'dominant_sector': None,
                'sector_hhi': 0.0
            }

        for token, weight in portfolio_weights.items():
            if token in narrative_mapping:
                sector = narrative_mapping[token]
                sector_weights[sector] += weight / total_weight

        # Sort sectors by weight
        sorted_sectors = sorted(
            sector_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Calculate sector HHI
        sector_hhi = self.calculate_herfindahl_hirschman_index(dict(sector_weights))

        return {
            'sector_weights': dict(sector_weights),
            'sector_count': len(sector_weights),
            'dominant_sector': sorted_sectors[0][0] if sorted_sectors else None,
            'dominant_sector_weight': sorted_sectors[0][1] if sorted_sectors else 0,
            'sector_hhi': sector_hhi,
            'effective_sectors': 1.0 / sector_hhi if sector_hhi > 0 else 0
        }

    def track_rebalancing_frequency(
        self,
        transactions: List[Dict],
        window_days: int = 30
    ) -> Dict[str, Any]:
        """
        Track portfolio rebalancing frequency and patterns.

        Args:
            transactions: List of transaction dictionaries
            window_days: Time window for rebalancing detection

        Returns:
            Dictionary containing rebalancing metrics
        """
        if not transactions:
            return {
                'rebalancing_events': 0,
                'avg_days_between_rebalances': 0,
                'tokens_rebalanced': set(),
                'rebalancing_intensity': 0.0
            }

        # Sort transactions by timestamp
        sorted_txns = sorted(transactions, key=lambda x: x['timestamp'])

        # Group transactions by time window
        rebalancing_windows = defaultdict(lambda: {'buys': [], 'sells': []})

        for txn in sorted_txns:
            window_key = txn['timestamp'].date()

            if txn['type'] == 'buy':
                rebalancing_windows[window_key]['buys'].append(txn)
            elif txn['type'] == 'sell':
                rebalancing_windows[window_key]['sells'].append(txn)

        # Identify rebalancing events (days with both buys and sells)
        rebalancing_events = []
        tokens_rebalanced = set()

        for date, activity in rebalancing_windows.items():
            if activity['buys'] and activity['sells']:
                # This is a potential rebalancing event
                rebalancing_events.append(date)

                # Track tokens involved
                for txn in activity['buys'] + activity['sells']:
                    tokens_rebalanced.add(txn['token_symbol'])

        # Calculate average time between rebalances
        if len(rebalancing_events) > 1:
            time_deltas = [
                (rebalancing_events[i] - rebalancing_events[i-1]).days
                for i in range(1, len(rebalancing_events))
            ]
            avg_days_between = np.mean(time_deltas)
        else:
            avg_days_between = 0

        # Calculate rebalancing intensity (turnover rate)
        total_volume = sum(
            txn['amount'] * txn.get('price', 0)
            for txn in sorted_txns
        )

        if sorted_txns:
            time_span = (sorted_txns[-1]['timestamp'] - sorted_txns[0]['timestamp']).days
            if time_span > 0:
                daily_turnover = total_volume / time_span
                rebalancing_intensity = daily_turnover
            else:
                rebalancing_intensity = 0
        else:
            rebalancing_intensity = 0

        return {
            'rebalancing_events': len(rebalancing_events),
            'avg_days_between_rebalances': avg_days_between,
            'tokens_rebalanced': list(tokens_rebalanced),
            'rebalancing_intensity': rebalancing_intensity,
            'first_rebalance': rebalancing_events[0] if rebalancing_events else None,
            'last_rebalance': rebalancing_events[-1] if rebalancing_events else None
        }

    def calculate_diversification_score(
        self,
        portfolio_weights: Dict[str, float],
        correlation_matrix: Optional[pd.DataFrame] = None,
        narrative_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Calculate overall portfolio diversification score.

        Combines multiple metrics into a single score from 0 to 100.

        Args:
            portfolio_weights: Dictionary mapping token symbols to weights
            correlation_matrix: Optional correlation matrix between tokens
            narrative_mapping: Optional mapping of tokens to narratives

        Returns:
            Dictionary containing diversification score and components
        """
        scores = {}

        # 1. Concentration score (based on HHI)
        hhi = self.calculate_herfindahl_hirschman_index(portfolio_weights)
        # Lower HHI is better, so invert for score
        concentration_score = (1 - hhi) * 100
        scores['concentration'] = concentration_score

        # 2. Token count score
        token_count = len([w for w in portfolio_weights.values() if w > 0])
        # Normalize to 0-100, with diminishing returns after 20 tokens
        token_score = min(100, (token_count / 20) * 100)
        scores['token_diversity'] = token_score

        # 3. Correlation score (if provided)
        if correlation_matrix is not None and not correlation_matrix.empty:
            correlation_metrics = self.analyze_portfolio_correlation(
                portfolio_weights,
                correlation_matrix
            )
            # Lower average correlation is better
            avg_corr = correlation_metrics['average_correlation']
            correlation_score = (1 - min(1, max(0, avg_corr))) * 100
            scores['correlation'] = correlation_score
        else:
            scores['correlation'] = 50  # Neutral if no data

        # 4. Narrative diversification (if provided)
        if narrative_mapping:
            narrative_analysis = self.analyze_token_diversification(
                [],  # Empty transactions, just using weights
                narrative_mapping
            )

            # Calculate narrative weights from portfolio weights
            narrative_weights = defaultdict(float)
            for token, weight in portfolio_weights.items():
                if token in narrative_mapping:
                    narrative = narrative_mapping[token]
                    narrative_weights[narrative] += weight

            narrative_hhi = self.calculate_herfindahl_hirschman_index(dict(narrative_weights))
            narrative_score = (1 - narrative_hhi) * 100
            scores['narrative_diversity'] = narrative_score
        else:
            scores['narrative_diversity'] = 50  # Neutral if no data

        # 5. Position sizing score
        position_metrics = self.calculate_maximum_position_size(portfolio_weights)
        max_position = position_metrics['max_position_size']
        # Penalize large positions (>25% gets lower score)
        position_score = max(0, 100 - (max_position - 25) * 4) if max_position > 25 else 100
        scores['position_sizing'] = position_score

        # Calculate weighted overall score
        weights = {
            'concentration': 0.25,
            'token_diversity': 0.20,
            'correlation': 0.20,
            'narrative_diversity': 0.20,
            'position_sizing': 0.15
        }

        overall_score = sum(
            scores[metric] * weight
            for metric, weight in weights.items()
        )

        return {
            'overall_score': overall_score,
            'component_scores': scores,
            'rating': self._get_diversification_rating(overall_score)
        }

    def _get_diversification_rating(self, score: float) -> str:
        """
        Get qualitative rating based on diversification score.

        Args:
            score: Diversification score (0-100)

        Returns:
            Rating string
        """
        if score >= 80:
            return 'Excellent'
        elif score >= 70:
            return 'Good'
        elif score >= 60:
            return 'Moderate'
        elif score >= 50:
            return 'Fair'
        else:
            return 'Poor'

    def generate_diversification_summary(
        self,
        portfolio_weights: Dict[str, float],
        transactions: List[Dict],
        price_data: Optional[Dict[str, pd.Series]] = None,
        narrative_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive diversification analysis summary.

        Args:
            portfolio_weights: Dictionary mapping token symbols to weights
            transactions: List of transaction dictionaries
            price_data: Optional price data for correlation analysis
            narrative_mapping: Optional narrative categorization

        Returns:
            Dictionary containing complete diversification analysis
        """
        summary = {}

        # Basic concentration metrics
        summary['concentration'] = {
            'hhi': self.calculate_herfindahl_hirschman_index(portfolio_weights),
            'effective_tokens': self.calculate_effective_tokens(portfolio_weights)
        }

        # Position sizing
        summary['position_sizing'] = self.calculate_maximum_position_size(portfolio_weights)

        # Token and narrative diversification
        summary['token_analysis'] = self.analyze_token_diversification(
            transactions,
            narrative_mapping
        )

        # Correlation analysis if price data available
        if price_data:
            tokens = list(portfolio_weights.keys())
            correlation_matrix = self.calculate_correlation_matrix(price_data, tokens)
            summary['correlation'] = self.analyze_portfolio_correlation(
                portfolio_weights,
                correlation_matrix
            )
        else:
            summary['correlation'] = None

        # Sector allocation if narrative mapping available
        if narrative_mapping:
            summary['sector_allocation'] = self.analyze_sector_allocation(
                portfolio_weights,
                narrative_mapping
            )
        else:
            summary['sector_allocation'] = None

        # Rebalancing patterns
        summary['rebalancing'] = self.track_rebalancing_frequency(transactions)

        # Overall diversification score
        correlation_matrix = self.calculate_correlation_matrix(price_data) if price_data else None
        summary['diversification_score'] = self.calculate_diversification_score(
            portfolio_weights,
            correlation_matrix,
            narrative_mapping
        )

        return summary