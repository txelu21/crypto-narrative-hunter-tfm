"""
Gas Efficiency and Cost Analysis Module

This module provides comprehensive gas cost analysis, efficiency metrics,
and MEV impact assessment for wallet transactions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class TransactionType(Enum):
    """Transaction type classification for gas analysis"""
    SWAP = "swap"
    TRANSFER = "transfer"
    LIQUIDITY_ADD = "liquidity_add"
    LIQUIDITY_REMOVE = "liquidity_remove"
    BRIDGE = "bridge"
    OTHER = "other"


@dataclass
class GasMetrics:
    """Comprehensive gas efficiency metrics for a wallet"""
    # Total gas costs
    total_gas_cost_eth: Decimal
    total_gas_cost_usd: Decimal
    total_transactions: int

    # Efficiency metrics
    gas_per_transaction_avg: Decimal
    gas_price_avg_gwei: Decimal
    volume_per_gas_eth: Decimal
    volume_per_gas_usd: Decimal

    # Cost breakdown
    gas_cost_by_type: Dict[str, Decimal]
    gas_efficiency_by_type: Dict[str, Decimal]

    # Optimization metrics
    gas_price_optimization_score: float
    failed_transaction_rate: float
    network_congestion_correlation: float

    # MEV impact analysis
    estimated_mev_loss_eth: Decimal
    estimated_mev_loss_usd: Decimal
    sandwich_attack_count: int
    frontrun_impact_count: int

    # Time-based analysis
    gas_cost_trend: List[Tuple[datetime, Decimal]]
    peak_hour_premium: float
    weekend_vs_weekday_ratio: float

    # Benchmarking
    peer_group_efficiency_percentile: Optional[float]
    gas_efficiency_rank: Optional[int]


@dataclass
class TransactionGasData:
    """Gas data for a single transaction"""
    transaction_hash: str
    timestamp: datetime
    gas_limit: int
    gas_used: int
    gas_price_gwei: Decimal
    gas_cost_eth: Decimal
    gas_cost_usd: Decimal
    transaction_type: TransactionType
    volume_eth: Decimal
    volume_usd: Decimal
    failed: bool
    network_utilization: Optional[float]
    estimated_mev_impact: Optional[Decimal]


@dataclass
class MEVImpactAnalysis:
    """MEV (Maximal Extractable Value) impact analysis"""
    total_detected_attacks: int
    sandwich_attacks: List[Dict[str, Any]]
    frontrun_incidents: List[Dict[str, Any]]
    estimated_total_loss_eth: Decimal
    estimated_total_loss_usd: Decimal
    protection_rate: float
    most_vulnerable_tokens: List[str]
    time_patterns: Dict[str, Any]


class GasEfficiencyAnalyzer:
    """
    Comprehensive gas efficiency analyzer

    Analyzes transaction costs, optimization opportunities, and MEV impact
    for wallet performance assessment.
    """

    def __init__(self, eth_gas_oracle_data: Optional[Dict] = None):
        """
        Initialize gas analyzer

        Args:
            eth_gas_oracle_data: Optional gas price oracle data for benchmarking
        """
        self.gas_oracle_data = eth_gas_oracle_data or {}

    def analyze_wallet_gas_efficiency(
        self,
        transactions: List[TransactionGasData],
        network_conditions: Optional[pd.DataFrame] = None,
        peer_group_data: Optional[List[GasMetrics]] = None
    ) -> GasMetrics:
        """
        Perform comprehensive gas efficiency analysis

        Args:
            transactions: List of transaction gas data
            network_conditions: Optional network congestion data
            peer_group_data: Optional peer group gas metrics for comparison

        Returns:
            GasMetrics with complete gas efficiency analysis
        """
        if not transactions:
            return self._create_empty_gas_metrics()

        # Calculate basic gas metrics
        basic_metrics = self._calculate_basic_gas_metrics(transactions)

        # Analyze gas costs by transaction type
        type_analysis = self._analyze_gas_by_transaction_type(transactions)

        # Calculate optimization metrics
        optimization_metrics = self._calculate_optimization_metrics(
            transactions, network_conditions
        )

        # Perform MEV impact analysis
        mev_analysis = self._analyze_mev_impact(transactions)

        # Calculate time-based patterns
        time_patterns = self._analyze_time_based_patterns(transactions)

        # Benchmarking against peers
        benchmark_metrics = self._calculate_benchmark_metrics(
            basic_metrics, peer_group_data
        )

        return GasMetrics(
            **basic_metrics,
            **type_analysis,
            **optimization_metrics,
            **mev_analysis,
            **time_patterns,
            **benchmark_metrics
        )

    def _calculate_basic_gas_metrics(
        self,
        transactions: List[TransactionGasData]
    ) -> Dict[str, Any]:
        """Calculate basic gas cost and efficiency metrics"""

        total_gas_cost_eth = sum(tx.gas_cost_eth for tx in transactions)
        total_gas_cost_usd = sum(tx.gas_cost_usd for tx in transactions)
        total_transactions = len(transactions)

        # Average gas metrics
        avg_gas_per_tx = sum(tx.gas_used for tx in transactions) / total_transactions
        avg_gas_price = sum(tx.gas_price_gwei for tx in transactions) / total_transactions

        # Volume efficiency
        total_volume_eth = sum(tx.volume_eth for tx in transactions)
        total_volume_usd = sum(tx.volume_usd for tx in transactions)

        volume_per_gas_eth = (
            total_volume_eth / total_gas_cost_eth
            if total_gas_cost_eth > 0 else Decimal(0)
        )
        volume_per_gas_usd = (
            total_volume_usd / total_gas_cost_usd
            if total_gas_cost_usd > 0 else Decimal(0)
        )

        return {
            'total_gas_cost_eth': total_gas_cost_eth,
            'total_gas_cost_usd': total_gas_cost_usd,
            'total_transactions': total_transactions,
            'gas_per_transaction_avg': Decimal(str(avg_gas_per_tx)),
            'gas_price_avg_gwei': Decimal(str(avg_gas_price)),
            'volume_per_gas_eth': volume_per_gas_eth,
            'volume_per_gas_usd': volume_per_gas_usd
        }

    def _analyze_gas_by_transaction_type(
        self,
        transactions: List[TransactionGasData]
    ) -> Dict[str, Any]:
        """Analyze gas costs and efficiency by transaction type"""

        # Group transactions by type
        type_groups = {}
        for tx in transactions:
            tx_type = tx.transaction_type.value
            if tx_type not in type_groups:
                type_groups[tx_type] = []
            type_groups[tx_type].append(tx)

        gas_cost_by_type = {}
        gas_efficiency_by_type = {}

        for tx_type, tx_list in type_groups.items():
            total_cost = sum(tx.gas_cost_eth for tx in tx_list)
            total_volume = sum(tx.volume_eth for tx in tx_list)

            gas_cost_by_type[tx_type] = total_cost
            gas_efficiency_by_type[tx_type] = (
                float(total_volume / total_cost) if total_cost > 0 else 0.0
            )

        return {
            'gas_cost_by_type': gas_cost_by_type,
            'gas_efficiency_by_type': gas_efficiency_by_type
        }

    def _calculate_optimization_metrics(
        self,
        transactions: List[TransactionGasData],
        network_conditions: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Calculate gas price optimization and network correlation metrics"""

        # Gas price optimization score
        optimization_score = self._calculate_gas_price_optimization_score(transactions)

        # Failed transaction rate
        failed_txs = [tx for tx in transactions if tx.failed]
        failed_rate = len(failed_txs) / len(transactions)

        # Network congestion correlation
        network_correlation = self._calculate_network_congestion_correlation(
            transactions, network_conditions
        )

        return {
            'gas_price_optimization_score': optimization_score,
            'failed_transaction_rate': failed_rate,
            'network_congestion_correlation': network_correlation
        }

    def _calculate_gas_price_optimization_score(
        self,
        transactions: List[TransactionGasData]
    ) -> float:
        """
        Calculate gas price optimization score

        Compares actual gas prices paid vs optimal prices for network conditions
        """
        if not transactions:
            return 0.0

        optimization_scores = []

        for tx in transactions:
            # Get network conditions at transaction time
            optimal_gas_price = self._get_optimal_gas_price(
                tx.timestamp, tx.transaction_type
            )

            if optimal_gas_price > 0:
                # Score: 1.0 = perfect optimization, 0.0 = very poor
                actual_price = float(tx.gas_price_gwei)
                if actual_price <= optimal_gas_price:
                    score = 1.0
                else:
                    # Penalty for overpaying
                    overpay_ratio = actual_price / optimal_gas_price
                    score = max(0.0, 2.0 - overpay_ratio)
                optimization_scores.append(score)

        return np.mean(optimization_scores) if optimization_scores else 0.0

    def _get_optimal_gas_price(
        self,
        timestamp: datetime,
        tx_type: TransactionType
    ) -> float:
        """Get optimal gas price for given timestamp and transaction type"""
        # This would integrate with gas price oracles
        # For now, return a simplified estimate
        base_price = 20.0  # Base gas price in Gwei

        # Adjust for transaction type urgency
        type_multipliers = {
            TransactionType.SWAP: 1.1,      # Slightly higher for swaps
            TransactionType.TRANSFER: 0.9,   # Lower for simple transfers
            TransactionType.BRIDGE: 1.2,    # Higher for bridges
            TransactionType.LIQUIDITY_ADD: 1.0,
            TransactionType.LIQUIDITY_REMOVE: 1.0,
            TransactionType.OTHER: 1.0
        }

        return base_price * type_multipliers.get(tx_type, 1.0)

    def _calculate_network_congestion_correlation(
        self,
        transactions: List[TransactionGasData],
        network_conditions: Optional[pd.DataFrame]
    ) -> float:
        """Calculate correlation between gas prices and network congestion"""
        if network_conditions is None or network_conditions.empty or len(transactions) < 5:
            return 0.0

        gas_prices = [float(tx.gas_price_gwei) for tx in transactions]
        utilizations = []

        for tx in transactions:
            # Find closest network utilization data
            closest_util = self._get_network_utilization_at_time(
                tx.timestamp, network_conditions
            )
            if closest_util is not None:
                utilizations.append(closest_util)

        if len(utilizations) < 5:
            return 0.0

        # Calculate correlation
        correlation = np.corrcoef(gas_prices[:len(utilizations)], utilizations)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0

    def _get_network_utilization_at_time(
        self,
        timestamp: datetime,
        network_conditions: pd.DataFrame
    ) -> Optional[float]:
        """Get network utilization at specific timestamp"""
        if network_conditions.empty:
            return None

        # Find closest timestamp in network data
        time_diffs = abs(network_conditions.index - timestamp)
        closest_idx = time_diffs.idxmin()

        if time_diffs[closest_idx] < timedelta(minutes=15):  # Within 15 minutes
            return float(network_conditions.loc[closest_idx, 'utilization'])

        return None

    def _analyze_mev_impact(
        self,
        transactions: List[TransactionGasData]
    ) -> Dict[str, Any]:
        """Analyze MEV (Maximal Extractable Value) impact"""

        total_mev_loss_eth = sum(
            tx.estimated_mev_impact or Decimal(0) for tx in transactions
        )

        # Convert to USD (simplified - would use historical prices)
        avg_eth_price = Decimal(2000)  # Placeholder
        total_mev_loss_usd = total_mev_loss_eth * avg_eth_price

        # Count different types of MEV attacks
        sandwich_count = sum(
            1 for tx in transactions
            if tx.estimated_mev_impact and tx.estimated_mev_impact > Decimal('0.001')
        )

        frontrun_count = sum(
            1 for tx in transactions
            if tx.estimated_mev_impact and tx.estimated_mev_impact > Decimal('0.0005')
        )

        return {
            'estimated_mev_loss_eth': total_mev_loss_eth,
            'estimated_mev_loss_usd': total_mev_loss_usd,
            'sandwich_attack_count': sandwich_count,
            'frontrun_impact_count': frontrun_count
        }

    def _analyze_time_based_patterns(
        self,
        transactions: List[TransactionGasData]
    ) -> Dict[str, Any]:
        """Analyze time-based gas cost patterns"""

        # Gas cost trend over time
        gas_trend = [(tx.timestamp, tx.gas_cost_eth) for tx in transactions]
        gas_trend.sort(key=lambda x: x[0])

        # Peak hour analysis
        peak_hour_premium = self._calculate_peak_hour_premium(transactions)

        # Weekend vs weekday comparison
        weekend_weekday_ratio = self._calculate_weekend_weekday_ratio(transactions)

        return {
            'gas_cost_trend': gas_trend,
            'peak_hour_premium': peak_hour_premium,
            'weekend_vs_weekday_ratio': weekend_weekday_ratio
        }

    def _calculate_peak_hour_premium(
        self,
        transactions: List[TransactionGasData]
    ) -> float:
        """Calculate gas price premium during peak hours (10am-4pm UTC)"""

        peak_hour_prices = []
        off_peak_prices = []

        for tx in transactions:
            hour = tx.timestamp.hour
            if 10 <= hour <= 16:  # Peak hours
                peak_hour_prices.append(float(tx.gas_price_gwei))
            else:
                off_peak_prices.append(float(tx.gas_price_gwei))

        if not peak_hour_prices or not off_peak_prices:
            return 0.0

        peak_avg = np.mean(peak_hour_prices)
        off_peak_avg = np.mean(off_peak_prices)

        return (peak_avg / off_peak_avg) - 1.0 if off_peak_avg > 0 else 0.0

    def _calculate_weekend_weekday_ratio(
        self,
        transactions: List[TransactionGasData]
    ) -> float:
        """Calculate gas price ratio between weekends and weekdays"""

        weekend_prices = []
        weekday_prices = []

        for tx in transactions:
            if tx.timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
                weekend_prices.append(float(tx.gas_price_gwei))
            else:
                weekday_prices.append(float(tx.gas_price_gwei))

        if not weekend_prices or not weekday_prices:
            return 1.0

        weekend_avg = np.mean(weekend_prices)
        weekday_avg = np.mean(weekday_prices)

        return weekend_avg / weekday_avg if weekday_avg > 0 else 1.0

    def _calculate_benchmark_metrics(
        self,
        basic_metrics: Dict[str, Any],
        peer_group_data: Optional[List[GasMetrics]]
    ) -> Dict[str, Any]:
        """Calculate benchmarking metrics against peer group"""

        if not peer_group_data:
            return {
                'peer_group_efficiency_percentile': None,
                'gas_efficiency_rank': None
            }

        wallet_efficiency = float(basic_metrics['volume_per_gas_eth'])
        peer_efficiencies = [float(peer.volume_per_gas_eth) for peer in peer_group_data]

        # Calculate percentile
        better_count = sum(1 for eff in peer_efficiencies if eff < wallet_efficiency)
        percentile = (better_count / len(peer_efficiencies)) * 100

        # Calculate rank
        all_efficiencies = peer_efficiencies + [wallet_efficiency]
        all_efficiencies.sort(reverse=True)
        rank = all_efficiencies.index(wallet_efficiency) + 1

        return {
            'peer_group_efficiency_percentile': percentile,
            'gas_efficiency_rank': rank
        }

    def analyze_mev_protection_opportunities(
        self,
        transactions: List[TransactionGasData]
    ) -> MEVImpactAnalysis:
        """
        Analyze MEV protection opportunities and patterns

        Identifies when and how the wallet could better protect against MEV
        """
        sandwich_attacks = []
        frontrun_incidents = []
        total_loss_eth = Decimal(0)

        # Pattern analysis
        time_patterns = self._analyze_mev_time_patterns(transactions)
        vulnerable_tokens = self._identify_vulnerable_tokens(transactions)

        # Calculate protection rate
        protected_txs = sum(
            1 for tx in transactions
            if not tx.estimated_mev_impact or tx.estimated_mev_impact == 0
        )
        protection_rate = protected_txs / len(transactions) if transactions else 0.0

        for tx in transactions:
            if tx.estimated_mev_impact and tx.estimated_mev_impact > 0:
                total_loss_eth += tx.estimated_mev_impact

                # Classify MEV type based on impact size and patterns
                if tx.estimated_mev_impact > Decimal('0.001'):
                    sandwich_attacks.append({
                        'transaction_hash': tx.transaction_hash,
                        'timestamp': tx.timestamp,
                        'loss_eth': tx.estimated_mev_impact,
                        'token_involved': self._extract_token_from_tx(tx)
                    })
                else:
                    frontrun_incidents.append({
                        'transaction_hash': tx.transaction_hash,
                        'timestamp': tx.timestamp,
                        'loss_eth': tx.estimated_mev_impact
                    })

        return MEVImpactAnalysis(
            total_detected_attacks=len(sandwich_attacks) + len(frontrun_incidents),
            sandwich_attacks=sandwich_attacks,
            frontrun_incidents=frontrun_incidents,
            estimated_total_loss_eth=total_loss_eth,
            estimated_total_loss_usd=total_loss_eth * Decimal(2000),  # Simplified
            protection_rate=protection_rate,
            most_vulnerable_tokens=vulnerable_tokens,
            time_patterns=time_patterns
        )

    def _analyze_mev_time_patterns(
        self,
        transactions: List[TransactionGasData]
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in MEV attacks"""

        mev_by_hour = {hour: 0 for hour in range(24)}
        mev_by_day = {day: 0 for day in range(7)}

        for tx in transactions:
            if tx.estimated_mev_impact and tx.estimated_mev_impact > 0:
                hour = tx.timestamp.hour
                day = tx.timestamp.weekday()

                mev_by_hour[hour] += float(tx.estimated_mev_impact)
                mev_by_day[day] += float(tx.estimated_mev_impact)

        return {
            'hourly_mev_impact': mev_by_hour,
            'daily_mev_impact': mev_by_day,
            'peak_mev_hour': max(mev_by_hour.items(), key=lambda x: x[1])[0],
            'safest_hour': min(mev_by_hour.items(), key=lambda x: x[1])[0]
        }

    def _identify_vulnerable_tokens(
        self,
        transactions: List[TransactionGasData]
    ) -> List[str]:
        """Identify tokens most vulnerable to MEV attacks"""

        token_mev_impact = {}

        for tx in transactions:
            if tx.estimated_mev_impact and tx.estimated_mev_impact > 0:
                token = self._extract_token_from_tx(tx)
                if token not in token_mev_impact:
                    token_mev_impact[token] = Decimal(0)
                token_mev_impact[token] += tx.estimated_mev_impact

        # Sort by impact and return top vulnerable tokens
        sorted_tokens = sorted(
            token_mev_impact.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [token for token, _ in sorted_tokens[:5]]

    def _extract_token_from_tx(self, tx: TransactionGasData) -> str:
        """Extract token symbol from transaction data"""
        # This would parse transaction data to identify token
        # For now, return a placeholder
        return "UNKNOWN"

    def generate_gas_optimization_report(
        self,
        gas_metrics: GasMetrics,
        mev_analysis: MEVImpactAnalysis
    ) -> Dict[str, Any]:
        """Generate comprehensive gas optimization report"""

        return {
            'executive_summary': {
                'total_gas_cost_usd': float(gas_metrics.total_gas_cost_usd),
                'optimization_score': gas_metrics.gas_price_optimization_score,
                'efficiency_percentile': gas_metrics.peer_group_efficiency_percentile,
                'mev_loss_usd': float(mev_analysis.estimated_total_loss_usd),
                'potential_savings_usd': self._calculate_potential_savings(gas_metrics)
            },
            'efficiency_breakdown': {
                'volume_per_gas': float(gas_metrics.volume_per_gas_usd),
                'gas_cost_by_type': {
                    k: float(v) for k, v in gas_metrics.gas_cost_by_type.items()
                },
                'efficiency_by_type': gas_metrics.gas_efficiency_by_type
            },
            'optimization_opportunities': {
                'gas_price_optimization': {
                    'current_score': gas_metrics.gas_price_optimization_score,
                    'improvement_potential': 1.0 - gas_metrics.gas_price_optimization_score
                },
                'timing_optimization': {
                    'peak_hour_premium': gas_metrics.peak_hour_premium,
                    'weekend_savings': 1.0 - gas_metrics.weekend_vs_weekday_ratio
                },
                'mev_protection': {
                    'current_protection_rate': mev_analysis.protection_rate,
                    'vulnerable_tokens': mev_analysis.most_vulnerable_tokens,
                    'safest_trading_hour': mev_analysis.time_patterns.get('safest_hour')
                }
            },
            'recommendations': self._generate_optimization_recommendations(
                gas_metrics, mev_analysis
            )
        }

    def _calculate_potential_savings(self, gas_metrics: GasMetrics) -> float:
        """Calculate potential gas cost savings"""
        current_cost = float(gas_metrics.total_gas_cost_usd)
        optimization_potential = 1.0 - gas_metrics.gas_price_optimization_score
        return current_cost * optimization_potential * 0.3  # Assume 30% savings potential

    def _generate_optimization_recommendations(
        self,
        gas_metrics: GasMetrics,
        mev_analysis: MEVImpactAnalysis
    ) -> List[str]:
        """Generate actionable optimization recommendations"""
        recommendations = []

        # Gas price optimization
        if gas_metrics.gas_price_optimization_score < 0.7:
            recommendations.append(
                "Consider using dynamic gas price strategies and monitoring network congestion"
            )

        # Timing optimization
        if gas_metrics.peak_hour_premium > 0.2:
            recommendations.append(
                "Avoid trading during peak hours (10am-4pm UTC) to save on gas costs"
            )

        # MEV protection
        if mev_analysis.protection_rate < 0.8:
            recommendations.append(
                "Consider using MEV protection services or private mempools for large trades"
            )

        if mev_analysis.most_vulnerable_tokens:
            recommendations.append(
                f"Be extra cautious when trading {', '.join(mev_analysis.most_vulnerable_tokens[:3])} "
                "due to high MEV vulnerability"
            )

        # Transaction batching
        if gas_metrics.total_transactions > 100 and gas_metrics.gas_per_transaction_avg < 100000:
            recommendations.append(
                "Consider batching small transactions to improve gas efficiency"
            )

        return recommendations

    def _create_empty_gas_metrics(self) -> GasMetrics:
        """Create empty gas metrics for edge cases"""
        return GasMetrics(
            total_gas_cost_eth=Decimal(0),
            total_gas_cost_usd=Decimal(0),
            total_transactions=0,
            gas_per_transaction_avg=Decimal(0),
            gas_price_avg_gwei=Decimal(0),
            volume_per_gas_eth=Decimal(0),
            volume_per_gas_usd=Decimal(0),
            gas_cost_by_type={},
            gas_efficiency_by_type={},
            gas_price_optimization_score=0.0,
            failed_transaction_rate=0.0,
            network_congestion_correlation=0.0,
            estimated_mev_loss_eth=Decimal(0),
            estimated_mev_loss_usd=Decimal(0),
            sandwich_attack_count=0,
            frontrun_impact_count=0,
            gas_cost_trend=[],
            peak_hour_premium=0.0,
            weekend_vs_weekday_ratio=1.0,
            peer_group_efficiency_percentile=None,
            gas_efficiency_rank=None
        )