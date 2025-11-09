"""
Balance Analysis Module

Analyzes daily balance snapshots to identify accumulation/distribution patterns,
portfolio composition changes, and smart money behavioral signals.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import asyncpg


@dataclass
class AccumulationMetrics:
    """Metrics for token accumulation/distribution analysis."""
    wallet_address: str
    token_address: str
    token_symbol: str
    start_date: datetime
    end_date: datetime

    # Balance changes
    initial_balance: Decimal
    final_balance: Decimal
    total_change: Decimal
    pct_change: float

    # Accumulation pattern
    is_accumulating: bool  # Net positive change
    accumulation_rate: float  # Tokens per day
    accumulation_velocity: float  # Change in rate over time

    # Consistency metrics
    days_with_balance: int
    days_with_increase: int
    days_with_decrease: int
    consistency_score: float  # 0-1, how consistently they accumulated

    # Timing metrics
    first_acquisition_day: int  # Day of period when first acquired
    peak_balance_day: int  # Day when balance was highest

    # Conviction signals
    held_through_volatility: bool
    add_on_dips: int  # Number of times added during price drops


class BalanceAnalyzer:
    """Analyzes balance snapshots for behavioral patterns."""

    def __init__(self, db_pool: asyncpg.Pool):
        """
        Initialize balance analyzer.

        Args:
            db_pool: Database connection pool
        """
        self.db_pool = db_pool

    async def get_wallet_balances(
        self,
        wallet_address: str,
        start_date: datetime,
        end_date: datetime,
        token_address: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get balance time series for a wallet.

        Args:
            wallet_address: Wallet address
            start_date: Start date
            end_date: End date
            token_address: Optional token filter

        Returns:
            DataFrame with columns: date, token_address, token_symbol, balance_formatted
        """
        query = """
            SELECT
                snapshot_date,
                token_address,
                token_symbol,
                balance_formatted
            FROM wallet_token_balances
            WHERE wallet_address = $1
              AND snapshot_date >= $2
              AND snapshot_date <= $3
        """
        params = [wallet_address.lower(), start_date.date(), end_date.date()]

        if token_address:
            query += " AND token_address = $4"
            params.append(token_address.lower())

        query += " ORDER BY snapshot_date, token_address"

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=['snapshot_date', 'token_address', 'token_symbol', 'balance_formatted'])
        df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])

        return df

    async def calculate_accumulation_metrics(
        self,
        wallet_address: str,
        token_address: str,
        start_date: datetime,
        end_date: datetime,
        price_data: Optional[pd.DataFrame] = None
    ) -> Optional[AccumulationMetrics]:
        """
        Calculate accumulation/distribution metrics for a wallet-token pair.

        Args:
            wallet_address: Wallet address
            token_address: Token contract address
            start_date: Analysis start date
            end_date: Analysis end date
            price_data: Optional DataFrame with columns [date, price] for timing analysis

        Returns:
            AccumulationMetrics or None if insufficient data
        """
        # Get balance time series
        df = await self.get_wallet_balances(
            wallet_address,
            start_date,
            end_date,
            token_address
        )

        if df.empty or len(df) < 2:
            return None

        # Sort by date
        df = df.sort_values('snapshot_date')

        # Extract balance series
        balances = df['balance_formatted'].values
        dates = df['snapshot_date'].values
        token_symbol = df['token_symbol'].iloc[0]

        # Basic metrics
        initial_balance = Decimal(str(balances[0]))
        final_balance = Decimal(str(balances[-1]))
        total_change = final_balance - initial_balance
        pct_change = float((total_change / initial_balance * 100) if initial_balance > 0 else 0)

        # Accumulation pattern
        is_accumulating = total_change > 0

        # Calculate daily changes
        daily_changes = np.diff(balances)
        num_days = len(balances)
        accumulation_rate = float(total_change) / num_days if num_days > 0 else 0

        # Accumulation velocity (change in rate over time)
        # Split period in half and compare rates
        mid_point = len(daily_changes) // 2
        first_half_rate = np.mean(daily_changes[:mid_point]) if mid_point > 0 else 0
        second_half_rate = np.mean(daily_changes[mid_point:]) if mid_point < len(daily_changes) else 0
        accumulation_velocity = float(second_half_rate - first_half_rate)

        # Consistency metrics
        days_with_increase = int(np.sum(daily_changes > 0))
        days_with_decrease = int(np.sum(daily_changes < 0))
        days_with_balance = num_days

        # Consistency score: how often changes align with overall trend
        if is_accumulating:
            consistency_score = days_with_increase / max(days_with_increase + days_with_decrease, 1)
        else:
            consistency_score = days_with_decrease / max(days_with_increase + days_with_decrease, 1)

        # Timing metrics
        first_acquisition_day = 0  # First day with non-zero balance
        peak_balance_day = int(np.argmax(balances))

        # Conviction signals
        # Held through volatility: did they maintain position when balance could have been sold?
        held_through_volatility = peak_balance_day < (len(balances) - 1) and final_balance > 0

        # Add on dips: if we have price data, count times they added during price drops
        add_on_dips = 0
        if price_data is not None and len(price_data) > 1:
            add_on_dips = self._calculate_add_on_dips(daily_changes, dates, price_data)

        return AccumulationMetrics(
            wallet_address=wallet_address,
            token_address=token_address,
            token_symbol=token_symbol,
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance,
            final_balance=final_balance,
            total_change=total_change,
            pct_change=pct_change,
            is_accumulating=is_accumulating,
            accumulation_rate=accumulation_rate,
            accumulation_velocity=accumulation_velocity,
            days_with_balance=days_with_balance,
            days_with_increase=days_with_increase,
            days_with_decrease=days_with_decrease,
            consistency_score=consistency_score,
            first_acquisition_day=first_acquisition_day,
            peak_balance_day=peak_balance_day,
            held_through_volatility=held_through_volatility,
            add_on_dips=add_on_dips
        )

    def _calculate_add_on_dips(
        self,
        daily_changes: np.ndarray,
        dates: np.ndarray,
        price_data: pd.DataFrame
    ) -> int:
        """
        Calculate number of times wallet added to position during price dips.

        Args:
            daily_changes: Array of daily balance changes
            dates: Array of dates
            price_data: DataFrame with [date, price]

        Returns:
            Count of "buy the dip" events
        """
        # Merge price data with balance changes
        price_dict = dict(zip(price_data['date'], price_data['price']))

        add_on_dips = 0
        for i, date in enumerate(dates[1:], start=1):
            if daily_changes[i-1] > 0:  # Balance increased
                # Check if price dropped from previous day
                current_price = price_dict.get(pd.Timestamp(date))
                prev_price = price_dict.get(pd.Timestamp(dates[i-1]))

                if current_price and prev_price and current_price < prev_price * 0.95:  # 5%+ drop
                    add_on_dips += 1

        return add_on_dips

    async def calculate_portfolio_composition(
        self,
        wallet_address: str,
        snapshot_date: datetime
    ) -> Dict[str, float]:
        """
        Calculate portfolio composition (% allocation per token) at a specific date.

        Note: This requires USD valuation, which needs price data.
        For now, returns token count-based composition.

        Args:
            wallet_address: Wallet address
            snapshot_date: Date of snapshot

        Returns:
            Dict mapping token_address to percentage
        """
        query = """
            SELECT
                token_address,
                token_symbol,
                balance_formatted
            FROM wallet_token_balances
            WHERE wallet_address = $1
              AND snapshot_date = $2
              AND balance_formatted > 0
        """

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, wallet_address.lower(), snapshot_date.date())

        if not rows:
            return {}

        # Simple equal-weight for now (needs price data for proper weighting)
        total_tokens = len(rows)
        return {row['token_address']: 1.0 / total_tokens for row in rows}

    async def calculate_hhi_concentration(
        self,
        wallet_address: str,
        snapshot_date: datetime
    ) -> float:
        """
        Calculate Herfindahl-Hirschman Index (HHI) for portfolio concentration.

        HHI = sum of squared market shares (0 = perfectly diversified, 10000 = single holding)

        Args:
            wallet_address: Wallet address
            snapshot_date: Date of snapshot

        Returns:
            HHI value (0-10000)
        """
        composition = await self.calculate_portfolio_composition(wallet_address, snapshot_date)

        if not composition:
            return 0.0

        # HHI = sum of squared percentages (in basis points)
        hhi = sum((pct * 100) ** 2 for pct in composition.values())
        return hhi

    async def get_narrative_allocation(
        self,
        wallet_address: str,
        snapshot_date: datetime
    ) -> Dict[str, float]:
        """
        Calculate portfolio allocation by narrative category.

        Args:
            wallet_address: Wallet address
            snapshot_date: Date of snapshot

        Returns:
            Dict mapping narrative to percentage
        """
        query = """
            SELECT
                t.narrative,
                COUNT(*) as token_count
            FROM wallet_token_balances wtb
            JOIN tokens t ON wtb.token_address = t.contract_address
            WHERE wtb.wallet_address = $1
              AND wtb.snapshot_date = $2
              AND wtb.balance_formatted > 0
            GROUP BY t.narrative
        """

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, wallet_address.lower(), snapshot_date.date())

        if not rows:
            return {}

        total = sum(row['token_count'] for row in rows)
        return {
            row['narrative']: row['token_count'] / total
            for row in rows
        }

    async def identify_wallet_archetype(
        self,
        wallet_address: str,
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """
        Classify wallet into behavioral archetype based on balance patterns.

        Archetypes:
        - "diamond_hands": Accumulates and holds through volatility
        - "momentum_trader": Rapid position changes, follows trends
        - "early_accumulator": Builds position early in period
        - "late_entrant": Enters positions late in period
        - "rotator": Frequently changes token allocations
        - "concentrated": Holds few tokens with high concentration
        - "diversified": Holds many tokens with low concentration

        Args:
            wallet_address: Wallet address
            start_date: Analysis start date
            end_date: Analysis end date

        Returns:
            Archetype label
        """
        # Get all balance data for wallet
        df = await self.get_wallet_balances(wallet_address, start_date, end_date)

        if df.empty:
            return "unknown"

        # Calculate metrics for classification
        unique_tokens = df['token_address'].nunique()
        total_days = (end_date - start_date).days + 1

        # Portfolio turnover: how often do holdings change?
        daily_token_counts = df.groupby('snapshot_date')['token_address'].nunique()
        avg_tokens_per_day = daily_token_counts.mean()
        token_volatility = daily_token_counts.std()

        # Concentration
        avg_hhi = 0
        for date in df['snapshot_date'].unique():
            hhi = await self.calculate_hhi_concentration(wallet_address, pd.Timestamp(date).to_pydatetime())
            avg_hhi += hhi
        avg_hhi /= len(df['snapshot_date'].unique())

        # Classification logic
        if avg_hhi > 5000:
            return "concentrated"
        elif unique_tokens > 15:
            return "diversified"
        elif token_volatility > avg_tokens_per_day * 0.3:
            return "rotator"
        else:
            return "holder"

    async def generate_wallet_balance_report(
        self,
        wallet_address: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Generate comprehensive balance analysis report for a wallet.

        Args:
            wallet_address: Wallet address
            start_date: Report start date
            end_date: Report end date

        Returns:
            Dictionary with analysis results
        """
        # Get all balances
        df = await self.get_wallet_balances(wallet_address, start_date, end_date)

        if df.empty:
            return {"error": "No balance data found"}

        # Calculate metrics for each token
        token_metrics = []
        for token_address in df['token_address'].unique():
            metrics = await self.calculate_accumulation_metrics(
                wallet_address,
                token_address,
                start_date,
                end_date
            )
            if metrics:
                token_metrics.append(metrics)

        # Identify archetype
        archetype = await self.identify_wallet_archetype(wallet_address, start_date, end_date)

        # Portfolio statistics
        unique_tokens = len(token_metrics)
        accumulating_tokens = sum(1 for m in token_metrics if m.is_accumulating)
        distributing_tokens = unique_tokens - accumulating_tokens

        return {
            "wallet_address": wallet_address,
            "period": {
                "start": start_date,
                "end": end_date,
                "days": (end_date - start_date).days + 1
            },
            "archetype": archetype,
            "portfolio_summary": {
                "unique_tokens": unique_tokens,
                "accumulating_tokens": accumulating_tokens,
                "distributing_tokens": distributing_tokens,
                "accumulation_bias": accumulating_tokens / max(unique_tokens, 1)
            },
            "token_metrics": [
                {
                    "token": m.token_symbol,
                    "address": m.token_address,
                    "pct_change": round(m.pct_change, 2),
                    "is_accumulating": m.is_accumulating,
                    "consistency_score": round(m.consistency_score, 2),
                    "held_through_volatility": m.held_through_volatility
                }
                for m in sorted(token_metrics, key=lambda x: abs(x.pct_change), reverse=True)[:10]
            ]
        }
