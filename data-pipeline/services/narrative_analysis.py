import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging
from .cohort_analysis import WalletMetrics

logger = logging.getLogger(__name__)

@dataclass
class TokenInfo:
    """Token information with narrative categorization"""
    token_address: str
    symbol: str
    name: str
    narrative_category: str
    market_cap_rank: int
    avg_daily_volume_usd: float
    liquidity_tier: int

@dataclass
class Trade:
    """Individual trade information"""
    wallet_address: str
    token_address: str
    volume_usd: float
    timestamp: str
    dex_name: str

class NarrativeRepresentationAnalysis:
    """Analyze narrative and sector representation across wallet cohort"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_narrative_representation(self, wallet_cohort: List[WalletMetrics],
                                       wallet_trades: Dict[str, List[Trade]],
                                       token_universe: Dict[str, TokenInfo]) -> Dict[str, Any]:
        """Analyze trading coverage across narrative categories"""

        self.logger.info(f"Analyzing narrative representation for {len(wallet_cohort)} wallets")

        # Calculate trading volume by narrative
        narrative_volumes = {}
        narrative_trade_counts = {}
        wallet_narrative_exposure = {}

        for wallet in wallet_cohort:
            wallet_address = wallet.wallet_address
            if wallet_address not in wallet_trades:
                continue

            wallet_narrative_volumes = {}
            total_wallet_volume = 0

            for trade in wallet_trades[wallet_address]:
                if trade.token_address in token_universe:
                    token_info = token_universe[trade.token_address]
                    narrative = token_info.narrative_category

                    # Aggregate by narrative
                    narrative_volumes[narrative] = narrative_volumes.get(narrative, 0) + trade.volume_usd
                    narrative_trade_counts[narrative] = narrative_trade_counts.get(narrative, 0) + 1

                    # Track wallet-level exposure
                    wallet_narrative_volumes[narrative] = wallet_narrative_volumes.get(narrative, 0) + trade.volume_usd
                    total_wallet_volume += trade.volume_usd

            # Calculate wallet's narrative allocation
            if total_wallet_volume > 0:
                wallet_narrative_exposure[wallet_address] = {
                    narrative: volume / total_wallet_volume
                    for narrative, volume in wallet_narrative_volumes.items()
                }

        # Calculate representation metrics
        total_volume = sum(narrative_volumes.values())
        total_trades = sum(narrative_trade_counts.values())

        if total_volume == 0:
            self.logger.warning("No trading volume found for analysis")
            return {}

        representation = {}
        for narrative, volume in narrative_volumes.items():
            wallets_trading_narrative = self._count_wallets_trading_narrative(
                wallet_cohort, wallet_trades, token_universe, narrative
            )

            representation[narrative] = {
                'volume_share': volume / total_volume,
                'trade_count_share': narrative_trade_counts.get(narrative, 0) / total_trades,
                'wallet_count': wallets_trading_narrative,
                'wallet_participation_rate': wallets_trading_narrative / len(wallet_cohort),
                'avg_allocation_per_wallet': volume / wallets_trading_narrative if wallets_trading_narrative > 0 else 0,
                'total_volume_usd': volume
            }

        # Target representation analysis
        target_ranges = {
            'DeFi': (0.35, 0.45),
            'Infrastructure': (0.20, 0.30),
            'Gaming': (0.10, 0.20),
            'AI': (0.05, 0.15),
            'Other': (0.00, 0.15)
        }

        target_compliance = {}
        for narrative, (min_target, max_target) in target_ranges.items():
            actual_share = representation.get(narrative, {}).get('volume_share', 0)
            target_compliance[narrative] = {
                'target_range': [min_target, max_target],
                'actual_share': actual_share,
                'within_target': min_target <= actual_share <= max_target,
                'deviation': actual_share - ((min_target + max_target) / 2)
            }

        return {
            'narrative_representation': representation,
            'target_compliance': target_compliance,
            'wallet_narrative_exposure': wallet_narrative_exposure,
            'summary': {
                'total_volume_analyzed': total_volume,
                'total_trades_analyzed': total_trades,
                'narratives_covered': len(representation),
                'target_compliance_rate': sum(1 for comp in target_compliance.values() if comp['within_target']) / len(target_compliance)
            }
        }

    def _count_wallets_trading_narrative(self, wallet_cohort: List[WalletMetrics],
                                       wallet_trades: Dict[str, List[Trade]],
                                       token_universe: Dict[str, TokenInfo],
                                       narrative: str) -> int:
        """Count wallets that have traded tokens in a specific narrative"""

        count = 0
        for wallet in wallet_cohort:
            wallet_address = wallet.wallet_address
            if wallet_address not in wallet_trades:
                continue

            for trade in wallet_trades[wallet_address]:
                if (trade.token_address in token_universe and
                    token_universe[trade.token_address].narrative_category == narrative):
                    count += 1
                    break  # Count wallet only once per narrative

        return count

    def calculate_sector_allocation_metrics(self, wallet_cohort: List[WalletMetrics],
                                          wallet_trades: Dict[str, List[Trade]],
                                          token_universe: Dict[str, TokenInfo]) -> Dict[str, Any]:
        """Calculate sector allocation and diversification metrics"""

        diversification_metrics = {}

        for wallet in wallet_cohort:
            wallet_address = wallet.wallet_address
            if wallet_address not in wallet_trades:
                continue

            # Calculate narrative allocation for this wallet
            narrative_volumes = {}
            total_volume = 0

            for trade in wallet_trades[wallet_address]:
                if trade.token_address in token_universe:
                    narrative = token_universe[trade.token_address].narrative_category
                    narrative_volumes[narrative] = narrative_volumes.get(narrative, 0) + trade.volume_usd
                    total_volume += trade.volume_usd

            if total_volume > 0:
                # Calculate allocation percentages
                allocations = {narrative: volume / total_volume
                             for narrative, volume in narrative_volumes.items()}

                # Diversification metrics
                herfindahl_index = sum(allocation ** 2 for allocation in allocations.values())
                narrative_count = len(allocations)
                max_allocation = max(allocations.values()) if allocations else 0

                diversification_metrics[wallet_address] = {
                    'narrative_allocations': allocations,
                    'diversification_score': 1 - herfindahl_index,  # Higher = more diversified
                    'narrative_count': narrative_count,
                    'max_single_allocation': max_allocation,
                    'is_well_diversified': herfindahl_index < 0.5 and narrative_count >= 3
                }

        # Aggregate diversification analysis
        if diversification_metrics:
            diversification_scores = [metrics['diversification_score']
                                    for metrics in diversification_metrics.values()]
            narrative_counts = [metrics['narrative_count']
                              for metrics in diversification_metrics.values()]
            max_allocations = [metrics['max_single_allocation']
                             for metrics in diversification_metrics.values()]

            aggregate_analysis = {
                'mean_diversification_score': np.mean(diversification_scores),
                'median_diversification_score': np.median(diversification_scores),
                'mean_narrative_count': np.mean(narrative_counts),
                'median_narrative_count': np.median(narrative_counts),
                'mean_max_allocation': np.mean(max_allocations),
                'well_diversified_percentage': sum(1 for metrics in diversification_metrics.values()
                                                 if metrics['is_well_diversified']) / len(diversification_metrics) * 100
            }
        else:
            aggregate_analysis = {}

        return {
            'individual_metrics': diversification_metrics,
            'aggregate_analysis': aggregate_analysis
        }

    def validate_balanced_representation(self, narrative_representation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate balanced representation across DeFi ecosystem segments"""

        # Define balance criteria
        balance_criteria = {
            'volume_distribution': {
                'description': 'No single narrative should dominate (>50%) total volume',
                'threshold': 0.5,
                'metric': 'volume_share'
            },
            'participation_rate': {
                'description': 'Major narratives should have >20% wallet participation',
                'threshold': 0.2,
                'metric': 'wallet_participation_rate'
            },
            'minimum_coverage': {
                'description': 'Each major narrative should have >5% volume share',
                'threshold': 0.05,
                'metric': 'volume_share'
            }
        }

        validation_results = {}
        major_narratives = ['DeFi', 'Infrastructure', 'Gaming', 'AI']

        for criterion_name, criterion in balance_criteria.items():
            validation_results[criterion_name] = {
                'description': criterion['description'],
                'threshold': criterion['threshold'],
                'results': {},
                'overall_pass': True
            }

            for narrative in major_narratives:
                if narrative in narrative_representation:
                    metric_value = narrative_representation[narrative].get(criterion['metric'], 0)

                    if criterion_name == 'volume_distribution':
                        # For volume distribution, value should be BELOW threshold (not dominating)
                        passes = metric_value <= criterion['threshold']
                    else:
                        # For other criteria, value should be ABOVE threshold
                        passes = metric_value >= criterion['threshold']

                    validation_results[criterion_name]['results'][narrative] = {
                        'value': metric_value,
                        'passes': passes
                    }

                    if not passes:
                        validation_results[criterion_name]['overall_pass'] = False

        # Overall balance score
        total_criteria = len(balance_criteria)
        passed_criteria = sum(1 for result in validation_results.values() if result['overall_pass'])
        balance_score = passed_criteria / total_criteria

        return {
            'balance_criteria': validation_results,
            'overall_balance_score': balance_score,
            'is_well_balanced': balance_score >= 0.8,  # 80% of criteria must pass
            'summary': {
                'criteria_evaluated': total_criteria,
                'criteria_passed': passed_criteria,
                'balance_percentage': balance_score * 100
            }
        }

    def geographic_temporal_pattern_analysis(self, wallet_cohort: List[WalletMetrics],
                                           wallet_trades: Dict[str, List[Trade]]) -> Dict[str, Any]:
        """Add geographic and temporal trading pattern analysis"""

        # Temporal pattern analysis
        temporal_patterns = {}

        # Group trades by time periods
        hourly_volumes = {}
        daily_volumes = {}
        monthly_volumes = {}

        for wallet in wallet_cohort:
            wallet_address = wallet.wallet_address
            if wallet_address not in wallet_trades:
                continue

            for trade in wallet_trades[wallet_address]:
                timestamp = pd.to_datetime(trade.timestamp)

                # Hour of day pattern
                hour = timestamp.hour
                hourly_volumes[hour] = hourly_volumes.get(hour, 0) + trade.volume_usd

                # Day of week pattern
                day_of_week = timestamp.dayofweek  # 0=Monday
                daily_volumes[day_of_week] = daily_volumes.get(day_of_week, 0) + trade.volume_usd

                # Monthly pattern
                month = timestamp.month
                monthly_volumes[month] = monthly_volumes.get(month, 0) + trade.volume_usd

        # Analyze patterns
        total_hourly = sum(hourly_volumes.values()) if hourly_volumes else 1
        total_daily = sum(daily_volumes.values()) if daily_volumes else 1
        total_monthly = sum(monthly_volumes.values()) if monthly_volumes else 1

        temporal_patterns = {
            'hourly_distribution': {
                hour: volume / total_hourly for hour, volume in hourly_volumes.items()
            },
            'daily_distribution': {
                day: volume / total_daily for day, volume in daily_volumes.items()
            },
            'monthly_distribution': {
                month: volume / total_monthly for month, volume in monthly_volumes.items()
            },
            'peak_trading_hour': max(hourly_volumes, key=hourly_volumes.get) if hourly_volumes else None,
            'peak_trading_day': max(daily_volumes, key=daily_volumes.get) if daily_volumes else None,
            'peak_trading_month': max(monthly_volumes, key=monthly_volumes.get) if monthly_volumes else None
        }

        # DEX distribution analysis (proxy for geographic patterns)
        dex_volumes = {}
        for wallet in wallet_cohort:
            wallet_address = wallet.wallet_address
            if wallet_address not in wallet_trades:
                continue

            for trade in wallet_trades[wallet_address]:
                dex_volumes[trade.dex_name] = dex_volumes.get(trade.dex_name, 0) + trade.volume_usd

        total_dex_volume = sum(dex_volumes.values()) if dex_volumes else 1
        dex_distribution = {
            dex: volume / total_dex_volume for dex, volume in dex_volumes.items()
        }

        return {
            'temporal_patterns': temporal_patterns,
            'dex_distribution': dex_distribution,
            'analysis_summary': {
                'temporal_concentration': max(temporal_patterns['hourly_distribution'].values()) if temporal_patterns['hourly_distribution'] else 0,
                'dex_concentration': max(dex_distribution.values()) if dex_distribution else 0,
                'trading_diversity': len(dex_distribution)
            }
        }

    def create_narrative_specialization_index(self, wallet_cohort: List[WalletMetrics],
                                            wallet_trades: Dict[str, List[Trade]],
                                            token_universe: Dict[str, TokenInfo]) -> Dict[str, Any]:
        """Create narrative specialization index for wallet categorization"""

        specialization_index = {}

        for wallet in wallet_cohort:
            wallet_address = wallet.wallet_address
            if wallet_address not in wallet_trades:
                continue

            # Calculate narrative volumes for this wallet
            narrative_volumes = {}
            total_volume = 0

            for trade in wallet_trades[wallet_address]:
                if trade.token_address in token_universe:
                    narrative = token_universe[trade.token_address].narrative_category
                    narrative_volumes[narrative] = narrative_volumes.get(narrative, 0) + trade.volume_usd
                    total_volume += trade.volume_usd

            if total_volume > 0:
                # Calculate specialization metrics
                allocations = {narrative: volume / total_volume
                             for narrative, volume in narrative_volumes.items()}

                # Primary narrative (highest allocation)
                primary_narrative = max(allocations, key=allocations.get)
                primary_allocation = allocations[primary_narrative]

                # Specialization score (higher = more specialized)
                herfindahl_index = sum(allocation ** 2 for allocation in allocations.values())

                # Classification
                if primary_allocation >= 0.7:
                    specialization_type = 'Highly Specialized'
                elif primary_allocation >= 0.5:
                    specialization_type = 'Moderately Specialized'
                elif len(allocations) >= 3 and primary_allocation < 0.4:
                    specialization_type = 'Generalist'
                else:
                    specialization_type = 'Focused'

                specialization_index[wallet_address] = {
                    'primary_narrative': primary_narrative,
                    'primary_allocation': primary_allocation,
                    'specialization_score': herfindahl_index,
                    'specialization_type': specialization_type,
                    'narrative_count': len(allocations),
                    'allocations': allocations
                }

        # Aggregate specialization analysis
        if specialization_index:
            specialization_types = [data['specialization_type']
                                  for data in specialization_index.values()]
            specialization_type_counts = pd.Series(specialization_types).value_counts().to_dict()

            primary_narratives = [data['primary_narrative']
                                for data in specialization_index.values()]
            primary_narrative_counts = pd.Series(primary_narratives).value_counts().to_dict()

            aggregate_analysis = {
                'specialization_type_distribution': specialization_type_counts,
                'primary_narrative_distribution': primary_narrative_counts,
                'mean_specialization_score': np.mean([data['specialization_score']
                                                    for data in specialization_index.values()]),
                'mean_narrative_count': np.mean([data['narrative_count']
                                               for data in specialization_index.values()])
            }
        else:
            aggregate_analysis = {}

        return {
            'individual_specialization': specialization_index,
            'aggregate_analysis': aggregate_analysis
        }

    def generate_sector_exposure_heatmaps(self, narrative_representation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sector exposure heatmaps for cohort visualization"""

        # Create exposure matrix data
        narratives = list(narrative_representation.keys())
        metrics = ['volume_share', 'wallet_participation_rate', 'trade_count_share']

        heatmap_data = {}
        for metric in metrics:
            heatmap_data[metric] = {
                narrative: narrative_representation[narrative].get(metric, 0)
                for narrative in narratives
            }

        # Risk-return matrix for narratives
        if all('avg_allocation_per_wallet' in narrative_representation[narrative] for narrative in narratives):
            risk_return_matrix = {
                narrative: {
                    'allocation': narrative_representation[narrative]['avg_allocation_per_wallet'],
                    'participation': narrative_representation[narrative]['wallet_participation_rate']
                }
                for narrative in narratives
            }
        else:
            risk_return_matrix = {}

        # Concentration analysis
        volume_shares = [narrative_representation[narrative].get('volume_share', 0)
                        for narrative in narratives]
        gini_coefficient = self._calculate_gini_coefficient(volume_shares)

        return {
            'exposure_matrices': heatmap_data,
            'risk_return_positioning': risk_return_matrix,
            'concentration_analysis': {
                'gini_coefficient': gini_coefficient,
                'concentration_level': 'High' if gini_coefficient > 0.7 else 'Medium' if gini_coefficient > 0.4 else 'Low',
                'dominant_narrative': max(narratives, key=lambda n: narrative_representation[n].get('volume_share', 0)),
                'narrative_count': len(narratives)
            }
        }

    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for concentration measurement"""
        if not values or sum(values) == 0:
            return 0

        sorted_values = sorted(values)
        n = len(sorted_values)
        cumulative_values = np.cumsum(sorted_values)
        total = cumulative_values[-1]

        if total == 0:
            return 0

        gini = (2 * sum((i + 1) * value for i, value in enumerate(sorted_values))) / (n * total) - (n + 1) / n
        return gini

    def comprehensive_narrative_analysis_report(self, wallet_cohort: List[WalletMetrics],
                                               wallet_trades: Dict[str, List[Trade]],
                                               token_universe: Dict[str, TokenInfo]) -> Dict[str, Any]:
        """Generate comprehensive narrative and sector analysis report"""

        self.logger.info("Generating comprehensive narrative analysis report")

        # Main narrative representation analysis
        narrative_representation = self.analyze_narrative_representation(
            wallet_cohort, wallet_trades, token_universe
        )

        # Sector allocation and diversification
        sector_allocation = self.calculate_sector_allocation_metrics(
            wallet_cohort, wallet_trades, token_universe
        )

        # Balance validation
        balance_validation = self.validate_balanced_representation(
            narrative_representation['narrative_representation']
        )

        # Geographic and temporal patterns
        temporal_geographic = self.geographic_temporal_pattern_analysis(
            wallet_cohort, wallet_trades
        )

        # Specialization index
        specialization_analysis = self.create_narrative_specialization_index(
            wallet_cohort, wallet_trades, token_universe
        )

        # Sector exposure heatmaps
        exposure_heatmaps = self.generate_sector_exposure_heatmaps(
            narrative_representation['narrative_representation']
        )

        comprehensive_report = {
            'executive_summary': {
                'total_wallets_analyzed': len(wallet_cohort),
                'narratives_covered': len(narrative_representation['narrative_representation']),
                'balance_score': balance_validation['overall_balance_score'],
                'is_well_balanced': balance_validation['is_well_balanced'],
                'target_compliance_rate': narrative_representation['summary']['target_compliance_rate']
            },
            'narrative_representation': narrative_representation,
            'sector_allocation': sector_allocation,
            'balance_validation': balance_validation,
            'temporal_geographic_patterns': temporal_geographic,
            'specialization_analysis': specialization_analysis,
            'exposure_visualization': exposure_heatmaps,
            'recommendations': self._generate_narrative_recommendations(
                narrative_representation, balance_validation, specialization_analysis
            )
        }

        return comprehensive_report

    def _generate_narrative_recommendations(self, narrative_representation: Dict[str, Any],
                                          balance_validation: Dict[str, Any],
                                          specialization_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on narrative analysis"""

        recommendations = []

        # Check balance issues
        if not balance_validation['is_well_balanced']:
            recommendations.append(
                "Consider rebalancing cohort to achieve better narrative representation"
            )

        # Check target compliance
        target_compliance = narrative_representation.get('target_compliance', {})
        for narrative, compliance in target_compliance.items():
            if not compliance['within_target']:
                if compliance['actual_share'] < compliance['target_range'][0]:
                    recommendations.append(
                        f"Increase representation in {narrative} narrative "
                        f"(current: {compliance['actual_share']:.1%}, "
                        f"target: {compliance['target_range'][0]:.1%}-{compliance['target_range'][1]:.1%})"
                    )
                elif compliance['actual_share'] > compliance['target_range'][1]:
                    recommendations.append(
                        f"Reduce over-representation in {narrative} narrative "
                        f"(current: {compliance['actual_share']:.1%}, "
                        f"target: {compliance['target_range'][0]:.1%}-{compliance['target_range'][1]:.1%})"
                    )

        # Specialization recommendations
        if specialization_analysis.get('aggregate_analysis'):
            specialization_dist = specialization_analysis['aggregate_analysis'].get('specialization_type_distribution', {})
            if specialization_dist.get('Highly Specialized', 0) > len(specialization_analysis['individual_specialization']) * 0.4:
                recommendations.append(
                    "High specialization detected - consider adding more generalist wallets for balance"
                )

        if not recommendations:
            recommendations.append("Narrative representation appears well-balanced and within target ranges")

        return recommendations