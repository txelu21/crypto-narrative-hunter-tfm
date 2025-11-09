import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import logging
from .cohort_analysis import WalletMetrics

logger = logging.getLogger(__name__)

class RiskProfileSegmentation:
    """Risk profile segmentation and clustering analysis"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def segment_risk_profiles(self, wallet_cohort: List[WalletMetrics],
                            n_clusters: int = 4) -> Dict[str, Any]:
        """Segment wallets by risk profile and trading sophistication"""

        self.logger.info(f"Segmenting {len(wallet_cohort)} wallets into {n_clusters} risk profiles")

        # Feature matrix for clustering
        features = np.array([
            [w.volatility, w.max_drawdown, w.portfolio_concentration,
             w.avg_position_size, w.trade_frequency, w.gas_efficiency]
            for w in wallet_cohort
        ])

        # Handle missing or invalid data
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        risk_clusters = kmeans.fit_predict(features_scaled)

        # Calculate silhouette score for cluster quality
        silhouette_avg = silhouette_score(features_scaled, risk_clusters)

        # Analyze cluster characteristics
        cluster_characteristics = self._analyze_cluster_characteristics(
            wallet_cohort, risk_clusters, features, n_clusters
        )

        # Assign meaningful labels based on cluster characteristics
        cluster_labels = self._assign_cluster_labels(cluster_characteristics)

        # Calculate cluster sizes
        cluster_sizes = [np.sum(risk_clusters == i) for i in range(n_clusters)]

        return {
            'clusters': risk_clusters.tolist(),
            'cluster_labels': cluster_labels,
            'cluster_characteristics': cluster_characteristics,
            'cluster_sizes': cluster_sizes,
            'silhouette_score': silhouette_avg,
            'clustering_quality': 'Good' if silhouette_avg > 0.5 else 'Fair' if silhouette_avg > 0.3 else 'Poor',
            'scaler_params': {
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist()
            },
            'centroids': kmeans.cluster_centers_.tolist()
        }

    def _analyze_cluster_characteristics(self, wallet_cohort: List[WalletMetrics],
                                       clusters: np.ndarray, features: np.ndarray,
                                       n_clusters: int) -> Dict[int, Dict[str, Any]]:
        """Analyze characteristics of each cluster"""

        feature_names = ['volatility', 'max_drawdown', 'portfolio_concentration',
                        'avg_position_size', 'trade_frequency', 'gas_efficiency']

        cluster_characteristics = {}

        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_wallets = [wallet for i, wallet in enumerate(wallet_cohort) if cluster_mask[i]]
            cluster_features = features[cluster_mask]

            if len(cluster_wallets) == 0:
                continue

            # Performance metrics
            returns = [w.total_return for w in cluster_wallets]
            sharpe_ratios = [w.sharpe_ratio for w in cluster_wallets]
            win_rates = [w.win_rate for w in cluster_wallets]

            # Risk metrics
            volatilities = [w.volatility for w in cluster_wallets]
            max_drawdowns = [w.max_drawdown for w in cluster_wallets]
            concentrations = [w.portfolio_concentration for w in cluster_wallets]

            # Activity metrics
            trade_frequencies = [w.trade_frequency for w in cluster_wallets]
            volumes = [w.avg_daily_volume_eth for w in cluster_wallets]

            cluster_characteristics[cluster_id] = {
                'size': len(cluster_wallets),
                'performance': {
                    'mean_return': np.mean(returns),
                    'median_return': np.median(returns),
                    'mean_sharpe': np.mean(sharpe_ratios),
                    'median_sharpe': np.median(sharpe_ratios),
                    'mean_win_rate': np.mean(win_rates)
                },
                'risk_profile': {
                    'mean_volatility': np.mean(volatilities),
                    'mean_max_drawdown': np.mean(max_drawdowns),
                    'mean_concentration': np.mean(concentrations),
                    'volatility_range': [np.min(volatilities), np.max(volatilities)]
                },
                'activity_profile': {
                    'mean_trade_frequency': np.mean(trade_frequencies),
                    'mean_volume': np.mean(volumes),
                    'median_volume': np.median(volumes)
                },
                'feature_means': {
                    feature_names[i]: np.mean(cluster_features[:, i])
                    for i in range(len(feature_names))
                }
            }

        return cluster_characteristics

    def _assign_cluster_labels(self, cluster_characteristics: Dict[int, Dict[str, Any]]) -> Dict[int, str]:
        """Assign meaningful labels to clusters based on their characteristics"""

        labels = {}

        # Sort clusters by risk level (volatility * concentration)
        risk_scores = {}
        for cluster_id, chars in cluster_characteristics.items():
            risk_score = (chars['risk_profile']['mean_volatility'] *
                         chars['risk_profile']['mean_concentration'])
            risk_scores[cluster_id] = risk_score

        sorted_clusters = sorted(risk_scores.items(), key=lambda x: x[1])

        # Assign labels based on risk ranking
        label_mapping = ['Conservative', 'Balanced', 'Aggressive', 'Ultra-High-Risk']

        for i, (cluster_id, _) in enumerate(sorted_clusters):
            if i < len(label_mapping):
                labels[cluster_id] = label_mapping[i]
            else:
                labels[cluster_id] = f'Cluster_{cluster_id}'

        return labels

    def create_sophistication_tiers(self, wallet_cohort: List[WalletMetrics]) -> Dict[str, Any]:
        """Create sophistication tiers based on trading complexity"""

        self.logger.info("Creating sophistication tiers for wallet cohort")

        # Calculate sophistication score for each wallet
        sophistication_scores = []
        sophistication_details = {}

        for wallet in wallet_cohort:
            # Sophistication components
            performance_component = self._normalize_performance_score(wallet)
            diversification_component = self._calculate_diversification_score(wallet)
            risk_management_component = self._calculate_risk_management_score(wallet)
            efficiency_component = self._calculate_efficiency_score(wallet)

            # Weighted sophistication score
            weights = {'performance': 0.3, 'diversification': 0.25,
                      'risk_management': 0.25, 'efficiency': 0.2}

            sophistication_score = (
                weights['performance'] * performance_component +
                weights['diversification'] * diversification_component +
                weights['risk_management'] * risk_management_component +
                weights['efficiency'] * efficiency_component
            )

            sophistication_scores.append(sophistication_score)
            sophistication_details[wallet.wallet_address] = {
                'total_score': sophistication_score,
                'components': {
                    'performance': performance_component,
                    'diversification': diversification_component,
                    'risk_management': risk_management_component,
                    'efficiency': efficiency_component
                }
            }

        # Define tier thresholds based on percentiles
        percentile_95 = np.percentile(sophistication_scores, 95)
        percentile_80 = np.percentile(sophistication_scores, 80)
        percentile_60 = np.percentile(sophistication_scores, 60)
        percentile_40 = np.percentile(sophistication_scores, 40)

        # Assign tiers
        tier_assignments = {}
        tier_counts = {'Expert': 0, 'Advanced': 0, 'Intermediate': 0, 'Developing': 0}

        for wallet in wallet_cohort:
            score = sophistication_details[wallet.wallet_address]['total_score']

            if score >= percentile_95:
                tier = 'Expert'
            elif score >= percentile_80:
                tier = 'Advanced'
            elif score >= percentile_60:
                tier = 'Intermediate'
            elif score >= percentile_40:
                tier = 'Developing'
            else:
                tier = 'Developing'  # Bottom tier

            tier_assignments[wallet.wallet_address] = tier
            tier_counts[tier] += 1

        # Tier characteristics
        tier_characteristics = self._calculate_tier_characteristics(
            wallet_cohort, tier_assignments, sophistication_details
        )

        return {
            'tier_assignments': tier_assignments,
            'tier_counts': tier_counts,
            'tier_characteristics': tier_characteristics,
            'sophistication_details': sophistication_details,
            'thresholds': {
                'expert': percentile_95,
                'advanced': percentile_80,
                'intermediate': percentile_60,
                'developing': percentile_40
            },
            'distribution': {
                'mean_score': np.mean(sophistication_scores),
                'median_score': np.median(sophistication_scores),
                'std_score': np.std(sophistication_scores)
            }
        }

    def _normalize_performance_score(self, wallet: WalletMetrics) -> float:
        """Calculate normalized performance score (0-1)"""
        # Combine multiple performance metrics
        sharpe_normalized = min(wallet.sharpe_ratio / 3.0, 1.0)  # Cap at 3.0 Sharpe
        return_normalized = min(max(wallet.total_return, -1.0) + 1.0, 1.0)  # Cap at 100% return
        win_rate_score = wallet.win_rate

        return (sharpe_normalized + return_normalized + win_rate_score) / 3.0

    def _calculate_diversification_score(self, wallet: WalletMetrics) -> float:
        """Calculate diversification score based on token variety and concentration"""
        # Token diversity component
        token_diversity = min(wallet.unique_tokens_traded / 50.0, 1.0)  # Cap at 50 tokens

        # Portfolio concentration component (lower is better for diversification)
        concentration_score = max(0, 1.0 - wallet.portfolio_concentration)

        return (token_diversity + concentration_score) / 2.0

    def _calculate_risk_management_score(self, wallet: WalletMetrics) -> float:
        """Calculate risk management score based on drawdown control and consistency"""
        # Max drawdown management (smaller drawdown is better)
        drawdown_score = max(0, 1.0 + wallet.max_drawdown)  # max_drawdown is negative

        # Performance consistency
        consistency_score = wallet.performance_consistency

        return (drawdown_score + consistency_score) / 2.0

    def _calculate_efficiency_score(self, wallet: WalletMetrics) -> float:
        """Calculate efficiency score based on gas usage and execution"""
        # Gas efficiency (volume per gas)
        gas_efficiency_normalized = min(wallet.gas_efficiency / 1000.0, 1.0)  # Normalize

        # MEV damage resistance (lower is better)
        mev_resistance = max(0, 1.0 - wallet.mev_damage_ratio)

        return (gas_efficiency_normalized + mev_resistance) / 2.0

    def _calculate_tier_characteristics(self, wallet_cohort: List[WalletMetrics],
                                      tier_assignments: Dict[str, str],
                                      sophistication_details: Dict[str, Dict]) -> Dict[str, Dict[str, Any]]:
        """Calculate characteristics for each sophistication tier"""

        tier_characteristics = {}

        for tier in ['Expert', 'Advanced', 'Intermediate', 'Developing']:
            tier_wallets = [wallet for wallet in wallet_cohort
                           if tier_assignments.get(wallet.wallet_address) == tier]

            if not tier_wallets:
                continue

            # Performance metrics
            returns = [w.total_return for w in tier_wallets]
            sharpe_ratios = [w.sharpe_ratio for w in tier_wallets]
            win_rates = [w.win_rate for w in tier_wallets]

            # Risk metrics
            volatilities = [w.volatility for w in tier_wallets]
            max_drawdowns = [w.max_drawdown for w in tier_wallets]

            # Activity metrics
            volumes = [w.avg_daily_volume_eth for w in tier_wallets]
            token_counts = [w.unique_tokens_traded for w in tier_wallets]

            # Sophistication scores
            sophistication_scores = [
                sophistication_details[w.wallet_address]['total_score']
                for w in tier_wallets
            ]

            tier_characteristics[tier] = {
                'wallet_count': len(tier_wallets),
                'percentage_of_cohort': len(tier_wallets) / len(wallet_cohort) * 100,
                'performance': {
                    'mean_return': np.mean(returns),
                    'mean_sharpe': np.mean(sharpe_ratios),
                    'mean_win_rate': np.mean(win_rates)
                },
                'risk_profile': {
                    'mean_volatility': np.mean(volatilities),
                    'mean_max_drawdown': np.mean(max_drawdowns)
                },
                'activity': {
                    'mean_volume': np.mean(volumes),
                    'mean_token_count': np.mean(token_counts)
                },
                'sophistication': {
                    'mean_score': np.mean(sophistication_scores),
                    'min_score': np.min(sophistication_scores),
                    'max_score': np.max(sophistication_scores)
                }
            }

        return tier_characteristics

    def risk_tolerance_segmentation(self, wallet_cohort: List[WalletMetrics]) -> Dict[str, Any]:
        """Add risk tolerance segmentation using drawdown and volatility"""

        risk_tolerance_assignments = {}
        tolerance_metrics = {}

        for wallet in wallet_cohort:
            # Risk tolerance factors
            volatility_tolerance = self._categorize_volatility_tolerance(wallet.volatility)
            drawdown_tolerance = self._categorize_drawdown_tolerance(wallet.max_drawdown)
            position_size_discipline = self._assess_position_sizing(wallet.avg_position_size, wallet.avg_daily_volume_eth)

            # Combined risk tolerance
            risk_tolerance = self._determine_overall_risk_tolerance(
                volatility_tolerance, drawdown_tolerance, position_size_discipline
            )

            risk_tolerance_assignments[wallet.wallet_address] = risk_tolerance
            tolerance_metrics[wallet.wallet_address] = {
                'volatility_tolerance': volatility_tolerance,
                'drawdown_tolerance': drawdown_tolerance,
                'position_size_discipline': position_size_discipline,
                'overall_tolerance': risk_tolerance
            }

        # Aggregate analysis
        tolerance_distribution = pd.Series(list(risk_tolerance_assignments.values())).value_counts().to_dict()

        # Risk tolerance characteristics
        tolerance_characteristics = {}
        for tolerance_level in ['High', 'Medium', 'Low']:
            tolerance_wallets = [
                wallet for wallet in wallet_cohort
                if risk_tolerance_assignments.get(wallet.wallet_address) == tolerance_level
            ]

            if tolerance_wallets:
                tolerance_characteristics[tolerance_level] = {
                    'count': len(tolerance_wallets),
                    'percentage': len(tolerance_wallets) / len(wallet_cohort) * 100,
                    'avg_volatility': np.mean([w.volatility for w in tolerance_wallets]),
                    'avg_max_drawdown': np.mean([w.max_drawdown for w in tolerance_wallets]),
                    'avg_return': np.mean([w.total_return for w in tolerance_wallets]),
                    'avg_sharpe': np.mean([w.sharpe_ratio for w in tolerance_wallets])
                }

        return {
            'risk_tolerance_assignments': risk_tolerance_assignments,
            'tolerance_distribution': tolerance_distribution,
            'tolerance_characteristics': tolerance_characteristics,
            'individual_metrics': tolerance_metrics
        }

    def _categorize_volatility_tolerance(self, volatility: float) -> str:
        """Categorize volatility tolerance"""
        if volatility > 0.6:
            return 'High'
        elif volatility > 0.3:
            return 'Medium'
        else:
            return 'Low'

    def _categorize_drawdown_tolerance(self, max_drawdown: float) -> str:
        """Categorize drawdown tolerance"""
        # max_drawdown is negative, so more negative = higher tolerance
        if max_drawdown < -0.3:
            return 'High'
        elif max_drawdown < -0.15:
            return 'Medium'
        else:
            return 'Low'

    def _assess_position_sizing(self, avg_position_size: float, avg_volume: float) -> float:
        """Assess position sizing discipline (0-1 score)"""
        if avg_volume == 0:
            return 0.5  # Neutral score for no data

        position_ratio = avg_position_size / avg_volume
        # Good position sizing is typically 2-10% of portfolio
        if 0.02 <= position_ratio <= 0.10:
            return 1.0  # Excellent discipline
        elif 0.01 <= position_ratio <= 0.20:
            return 0.75  # Good discipline
        elif position_ratio <= 0.30:
            return 0.5  # Fair discipline
        else:
            return 0.25  # Poor discipline

    def _determine_overall_risk_tolerance(self, volatility_tolerance: str,
                                        drawdown_tolerance: str,
                                        position_size_discipline: float) -> str:
        """Determine overall risk tolerance"""
        tolerance_scores = {'High': 3, 'Medium': 2, 'Low': 1}

        vol_score = tolerance_scores[volatility_tolerance]
        drawdown_score = tolerance_scores[drawdown_tolerance]
        discipline_penalty = 1 if position_size_discipline < 0.5 else 0

        combined_score = (vol_score + drawdown_score) / 2 - discipline_penalty

        if combined_score >= 2.5:
            return 'High'
        elif combined_score >= 1.5:
            return 'Medium'
        else:
            return 'Low'

    def generate_trading_style_classification(self, wallet_cohort: List[WalletMetrics]) -> Dict[str, Any]:
        """Generate trading style classification (conservative, aggressive, balanced)"""

        style_assignments = {}
        style_metrics = {}

        for wallet in wallet_cohort:
            # Trading style factors
            risk_aggressiveness = self._calculate_risk_aggressiveness(wallet)
            trading_frequency_style = self._classify_trading_frequency(wallet.trade_frequency)
            diversification_style = self._classify_diversification_approach(wallet.unique_tokens_traded, wallet.portfolio_concentration)

            # Determine overall style
            trading_style = self._determine_trading_style(
                risk_aggressiveness, trading_frequency_style, diversification_style
            )

            style_assignments[wallet.wallet_address] = trading_style
            style_metrics[wallet.wallet_address] = {
                'risk_aggressiveness': risk_aggressiveness,
                'frequency_style': trading_frequency_style,
                'diversification_style': diversification_style,
                'overall_style': trading_style
            }

        # Style distribution and characteristics
        style_distribution = pd.Series(list(style_assignments.values())).value_counts().to_dict()

        style_characteristics = {}
        for style in ['Conservative', 'Balanced', 'Aggressive']:
            style_wallets = [
                wallet for wallet in wallet_cohort
                if style_assignments.get(wallet.wallet_address) == style
            ]

            if style_wallets:
                style_characteristics[style] = {
                    'count': len(style_wallets),
                    'percentage': len(style_wallets) / len(wallet_cohort) * 100,
                    'avg_return': np.mean([w.total_return for w in style_wallets]),
                    'avg_volatility': np.mean([w.volatility for w in style_wallets]),
                    'avg_sharpe': np.mean([w.sharpe_ratio for w in style_wallets]),
                    'avg_trade_frequency': np.mean([w.trade_frequency for w in style_wallets]),
                    'avg_token_diversity': np.mean([w.unique_tokens_traded for w in style_wallets])
                }

        return {
            'style_assignments': style_assignments,
            'style_distribution': style_distribution,
            'style_characteristics': style_characteristics,
            'individual_metrics': style_metrics
        }

    def _calculate_risk_aggressiveness(self, wallet: WalletMetrics) -> float:
        """Calculate risk aggressiveness score (0-1)"""
        volatility_score = min(wallet.volatility / 0.8, 1.0)  # Normalize by 80% volatility
        drawdown_score = min(abs(wallet.max_drawdown) / 0.5, 1.0)  # Normalize by 50% drawdown
        concentration_score = wallet.portfolio_concentration

        return (volatility_score + drawdown_score + concentration_score) / 3.0

    def _classify_trading_frequency(self, trade_frequency: float) -> str:
        """Classify trading frequency style"""
        if trade_frequency > 5:  # More than 5 trades per day on average
            return 'High-Frequency'
        elif trade_frequency > 1:  # More than 1 trade per day
            return 'Active'
        else:
            return 'Low-Frequency'

    def _classify_diversification_approach(self, unique_tokens: int, concentration: float) -> str:
        """Classify diversification approach"""
        if unique_tokens > 30 and concentration < 0.3:
            return 'Highly-Diversified'
        elif unique_tokens > 15 and concentration < 0.5:
            return 'Moderately-Diversified'
        else:
            return 'Concentrated'

    def _determine_trading_style(self, risk_aggressiveness: float,
                               frequency_style: str, diversification_style: str) -> str:
        """Determine overall trading style"""
        # Score components
        risk_score = risk_aggressiveness
        frequency_score = {'High-Frequency': 1.0, 'Active': 0.6, 'Low-Frequency': 0.2}[frequency_style]
        diversification_score = {'Concentrated': 1.0, 'Moderately-Diversified': 0.5, 'Highly-Diversified': 0.0}[diversification_style]

        # Weighted style score
        style_score = (risk_score * 0.5 + frequency_score * 0.3 + diversification_score * 0.2)

        if style_score > 0.7:
            return 'Aggressive'
        elif style_score > 0.4:
            return 'Balanced'
        else:
            return 'Conservative'

    def create_cohort_segments(self, wallet_cohort: List[WalletMetrics]) -> Dict[str, Any]:
        """Create cohort segments for targeted analysis"""

        # Multiple segmentation approaches
        risk_profiling = self.segment_risk_profiles(wallet_cohort)
        sophistication_tiers = self.create_sophistication_tiers(wallet_cohort)
        risk_tolerance = self.risk_tolerance_segmentation(wallet_cohort)
        trading_styles = self.generate_trading_style_classification(wallet_cohort)

        # Create combined segments
        combined_segments = {}
        for wallet in wallet_cohort:
            wallet_address = wallet.wallet_address

            # Get cluster assignment
            wallet_index = next(i for i, w in enumerate(wallet_cohort) if w.wallet_address == wallet_address)
            risk_cluster = risk_profiling['clusters'][wallet_index]
            risk_label = risk_profiling['cluster_labels'][risk_cluster]

            segment_key = (
                risk_label,
                sophistication_tiers['tier_assignments'].get(wallet_address, 'Unknown'),
                risk_tolerance['risk_tolerance_assignments'].get(wallet_address, 'Unknown'),
                trading_styles['style_assignments'].get(wallet_address, 'Unknown')
            )

            if segment_key not in combined_segments:
                combined_segments[segment_key] = []
            combined_segments[segment_key].append(wallet)

        # Analyze segment characteristics
        segment_analysis = {}
        for segment_key, segment_wallets in combined_segments.items():
            if len(segment_wallets) >= 5:  # Minimum viable segment size
                risk_profile, sophistication, tolerance, style = segment_key

                segment_analysis[f"{risk_profile}_{sophistication}_{tolerance}_{style}"] = {
                    'segment_definition': {
                        'risk_profile': risk_profile,
                        'sophistication_tier': sophistication,
                        'risk_tolerance': tolerance,
                        'trading_style': style
                    },
                    'size': len(segment_wallets),
                    'percentage': len(segment_wallets) / len(wallet_cohort) * 100,
                    'performance': {
                        'mean_return': np.mean([w.total_return for w in segment_wallets]),
                        'mean_sharpe': np.mean([w.sharpe_ratio for w in segment_wallets]),
                        'mean_win_rate': np.mean([w.win_rate for w in segment_wallets])
                    },
                    'characteristics': {
                        'mean_volatility': np.mean([w.volatility for w in segment_wallets]),
                        'mean_volume': np.mean([w.avg_daily_volume_eth for w in segment_wallets]),
                        'mean_token_diversity': np.mean([w.unique_tokens_traded for w in segment_wallets])
                    }
                }

        return {
            'individual_segmentations': {
                'risk_profiling': risk_profiling,
                'sophistication_tiers': sophistication_tiers,
                'risk_tolerance': risk_tolerance,
                'trading_styles': trading_styles
            },
            'combined_segments': segment_analysis,
            'summary': {
                'total_segments': len(segment_analysis),
                'largest_segment': max(segment_analysis.items(), key=lambda x: x[1]['size']) if segment_analysis else None,
                'segment_coverage': sum(segment['size'] for segment in segment_analysis.values()) / len(wallet_cohort) * 100
            }
        }

    def validate_segment_homogeneity(self, cohort_segments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate segment homogeneity and separation"""

        validation_results = {}

        for segment_name, segment_data in cohort_segments['combined_segments'].items():
            if segment_data['size'] < 10:  # Skip small segments
                continue

            # Homogeneity metrics (within-segment consistency)
            performance_cv = self._calculate_coefficient_variation([
                segment_data['performance']['mean_return'],
                segment_data['performance']['mean_sharpe'],
                segment_data['performance']['mean_win_rate']
            ])

            characteristics_cv = self._calculate_coefficient_variation([
                segment_data['characteristics']['mean_volatility'],
                segment_data['characteristics']['mean_volume'],
                segment_data['characteristics']['mean_token_diversity']
            ])

            # Overall homogeneity score
            homogeneity_score = 1 - ((performance_cv + characteristics_cv) / 2)

            validation_results[segment_name] = {
                'size': segment_data['size'],
                'homogeneity_score': homogeneity_score,
                'performance_consistency': 1 - performance_cv,
                'characteristics_consistency': 1 - characteristics_cv,
                'is_homogeneous': homogeneity_score > 0.7,
                'quality_rating': (
                    'High' if homogeneity_score > 0.8 else
                    'Medium' if homogeneity_score > 0.6 else
                    'Low'
                )
            }

        # Overall validation summary
        if validation_results:
            mean_homogeneity = np.mean([v['homogeneity_score'] for v in validation_results.values()])
            high_quality_segments = sum(1 for v in validation_results.values() if v['quality_rating'] == 'High')

            summary = {
                'mean_homogeneity_score': mean_homogeneity,
                'high_quality_segments': high_quality_segments,
                'total_validated_segments': len(validation_results),
                'overall_quality': (
                    'Excellent' if mean_homogeneity > 0.8 else
                    'Good' if mean_homogeneity > 0.6 else
                    'Fair'
                )
            }
        else:
            summary = {'validation_status': 'Insufficient data for validation'}

        return {
            'segment_validation': validation_results,
            'summary': summary
        }

    def _calculate_coefficient_variation(self, values: List[float]) -> float:
        """Calculate coefficient of variation for consistency measurement"""
        if not values or np.mean(values) == 0:
            return 0
        return np.std(values) / np.mean(values)

    def comprehensive_risk_profiling_report(self, wallet_cohort: List[WalletMetrics]) -> Dict[str, Any]:
        """Generate comprehensive risk profiling and segmentation report"""

        self.logger.info("Generating comprehensive risk profiling report")

        # Create all segmentations
        cohort_segments = self.create_cohort_segments(wallet_cohort)

        # Validate segment quality
        segment_validation = self.validate_segment_homogeneity(cohort_segments)

        # Generate executive summary
        executive_summary = {
            'total_wallets': len(wallet_cohort),
            'risk_clusters': len(cohort_segments['individual_segmentations']['risk_profiling']['cluster_labels']),
            'sophistication_tiers': len(cohort_segments['individual_segmentations']['sophistication_tiers']['tier_counts']),
            'combined_segments': len(cohort_segments['combined_segments']),
            'clustering_quality': cohort_segments['individual_segmentations']['risk_profiling']['clustering_quality'],
            'segment_coverage': cohort_segments['summary']['segment_coverage'],
            'validation_quality': segment_validation['summary'].get('overall_quality', 'Unknown')
        }

        comprehensive_report = {
            'executive_summary': executive_summary,
            'risk_profiling': cohort_segments['individual_segmentations']['risk_profiling'],
            'sophistication_analysis': cohort_segments['individual_segmentations']['sophistication_tiers'],
            'risk_tolerance_analysis': cohort_segments['individual_segmentations']['risk_tolerance'],
            'trading_style_analysis': cohort_segments['individual_segmentations']['trading_styles'],
            'combined_segments': cohort_segments['combined_segments'],
            'segment_validation': segment_validation,
            'recommendations': self._generate_risk_profiling_recommendations(
                cohort_segments, segment_validation
            )
        }

        return comprehensive_report

    def _generate_risk_profiling_recommendations(self, cohort_segments: Dict[str, Any],
                                               segment_validation: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on risk profiling analysis"""

        recommendations = []

        # Check clustering quality
        clustering_quality = cohort_segments['individual_segmentations']['risk_profiling']['clustering_quality']
        if clustering_quality == 'Poor':
            recommendations.append("Consider adjusting clustering parameters or feature selection for better risk segmentation")

        # Check segment balance
        segment_sizes = [seg['size'] for seg in cohort_segments['combined_segments'].values()]
        if segment_sizes and max(segment_sizes) > len([w for w in cohort_segments]) * 0.6:
            recommendations.append("Large segment imbalance detected - consider refining segmentation criteria")

        # Check sophistication distribution
        sophistication_tiers = cohort_segments['individual_segmentations']['sophistication_tiers']['tier_counts']
        if sophistication_tiers.get('Expert', 0) < len([w for w in cohort_segments]) * 0.05:
            recommendations.append("Low expert-tier representation - consider expanding criteria for expert identification")

        # Validation quality
        if segment_validation['summary'].get('overall_quality') == 'Fair':
            recommendations.append("Segment homogeneity could be improved - review segmentation methodology")

        if not recommendations:
            recommendations.append("Risk profiling and segmentation analysis shows good quality and balance")

        return recommendations