"""
Sybil Detection and Cluster Analysis System

This module implements algorithms to detect coordinated wallet clusters (Sybil attacks)
using transaction patterns, funding correlations, temporal analysis, and network clustering.
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from decimal import Decimal
import logging
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class FundingConnection:
    """Represents a funding connection between wallets."""
    source_wallet: str
    destination_wallet: str
    timestamp: datetime
    amount_eth: Decimal
    transaction_hash: str
    hop_count: int = 1  # Number of hops from original source


@dataclass
class TradingPatternSimilarity:
    """Represents similarity in trading patterns between wallets."""
    wallet_a: str
    wallet_b: str
    correlation_score: float
    common_tokens: Set[str]
    temporal_overlap: float
    pattern_similarity: float


@dataclass
class SybilCluster:
    """Represents a detected Sybil cluster."""
    cluster_id: str
    wallet_addresses: Set[str]
    evidence_score: float
    detection_methods: List[str]
    funding_connections: List[FundingConnection]
    pattern_similarities: List[TradingPatternSimilarity]
    risk_level: str  # "high", "medium", "low"
    manual_review_required: bool = False


@dataclass
class SybilDetectionResult:
    """Result of sybil detection analysis."""
    wallet_address: str
    is_sybil: bool
    sybil_score: float
    cluster_id: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    detection_timestamp: datetime = field(default_factory=datetime.now)


class SybilDetector:
    """
    Comprehensive Sybil detection system for identifying coordinated wallet clusters.

    This class implements multiple detection algorithms including funding correlation,
    trading pattern analysis, temporal correlation, and network clustering.
    """

    def __init__(self,
                 funding_correlation_threshold: float = 0.7,
                 trading_correlation_threshold: float = 0.8,
                 temporal_correlation_threshold: float = 0.6,
                 min_cluster_size: int = 2,
                 max_hops_funding_trace: int = 3):
        """
        Initialize the Sybil detector with configurable thresholds.

        Args:
            funding_correlation_threshold: Minimum correlation for funding connections
            trading_correlation_threshold: Minimum correlation for trading patterns
            temporal_correlation_threshold: Minimum correlation for temporal patterns
            min_cluster_size: Minimum size for a cluster to be considered Sybil
            max_hops_funding_trace: Maximum hops to trace funding connections
        """
        self.funding_threshold = funding_correlation_threshold
        self.trading_threshold = trading_correlation_threshold
        self.temporal_threshold = temporal_correlation_threshold
        self.min_cluster_size = min_cluster_size
        self.max_hops = max_hops_funding_trace

        # Detection state
        self.funding_graph = nx.DiGraph()
        self.wallet_patterns = {}
        self.detected_clusters = []
        self.detection_stats = {
            'wallets_analyzed': 0,
            'clusters_detected': 0,
            'sybil_wallets_flagged': 0
        }

        logger.info(f"SybilDetector initialized with thresholds: funding={funding_correlation_threshold}, "
                   f"trading={trading_correlation_threshold}, temporal={temporal_correlation_threshold}")

    def analyze_wallet_for_sybil(self, wallet_address: str, wallet_data: Dict[str, Any]) -> SybilDetectionResult:
        """
        Analyze a single wallet for Sybil behavior.

        Args:
            wallet_address: The wallet address to analyze
            wallet_data: Complete wallet data including trades, funding, etc.

        Returns:
            SybilDetectionResult with detection findings
        """
        self.detection_stats['wallets_analyzed'] += 1

        try:
            # Extract wallet patterns for comparison
            self._extract_wallet_patterns(wallet_address, wallet_data)

            # Check if wallet is already part of a detected cluster
            existing_cluster = self._find_existing_cluster(wallet_address)
            if existing_cluster:
                return SybilDetectionResult(
                    wallet_address=wallet_address,
                    is_sybil=True,
                    sybil_score=existing_cluster.evidence_score,
                    cluster_id=existing_cluster.cluster_id,
                    evidence={'existing_cluster': True}
                )

            # Run detection algorithms
            funding_evidence = self._analyze_funding_connections(wallet_address, wallet_data)
            pattern_evidence = self._analyze_trading_patterns(wallet_address)
            temporal_evidence = self._analyze_temporal_correlations(wallet_address)
            network_evidence = self._analyze_network_position(wallet_address)

            # Combine evidence scores
            combined_score = self._calculate_combined_sybil_score(
                funding_evidence, pattern_evidence, temporal_evidence, network_evidence
            )

            is_sybil = combined_score >= 0.7  # High confidence threshold

            if is_sybil:
                self.detection_stats['sybil_wallets_flagged'] += 1

            return SybilDetectionResult(
                wallet_address=wallet_address,
                is_sybil=is_sybil,
                sybil_score=combined_score,
                evidence={
                    'funding': funding_evidence,
                    'patterns': pattern_evidence,
                    'temporal': temporal_evidence,
                    'network': network_evidence
                }
            )

        except Exception as e:
            logger.error(f"Error analyzing wallet {wallet_address} for Sybil behavior: {e}")
            return SybilDetectionResult(
                wallet_address=wallet_address,
                is_sybil=False,
                sybil_score=0.0,
                evidence={'error': str(e)}
            )

    def detect_sybil_clusters(self, wallet_data_list: List[Dict[str, Any]]) -> List[SybilCluster]:
        """
        Detect Sybil clusters across a batch of wallets.

        Args:
            wallet_data_list: List of wallet data dictionaries

        Returns:
            List of detected SybilCluster objects
        """
        logger.info(f"Starting Sybil cluster detection on {len(wallet_data_list)} wallets")

        # Extract patterns for all wallets
        for wallet_data in wallet_data_list:
            wallet_address = wallet_data['wallet_address']
            self._extract_wallet_patterns(wallet_address, wallet_data)
            self._update_funding_graph(wallet_address, wallet_data)

        # Run clustering algorithms
        funding_clusters = self._cluster_by_funding_sources()
        pattern_clusters = self._cluster_by_trading_patterns()
        temporal_clusters = self._cluster_by_temporal_correlation()
        network_clusters = self._cluster_by_network_analysis()

        # Combine and validate clusters
        combined_clusters = self._combine_cluster_evidence(
            funding_clusters, pattern_clusters, temporal_clusters, network_clusters
        )

        # Filter clusters by minimum size and evidence strength
        validated_clusters = self._validate_clusters(combined_clusters)

        self.detected_clusters = validated_clusters
        self.detection_stats['clusters_detected'] = len(validated_clusters)

        logger.info(f"Detected {len(validated_clusters)} Sybil clusters")
        return validated_clusters

    def _extract_wallet_patterns(self, wallet_address: str, wallet_data: Dict[str, Any]) -> None:
        """Extract trading patterns and features for a wallet."""
        trades = wallet_data.get('trades', [])

        if not trades:
            return

        # Token trading patterns
        token_counts = Counter(trade.token_address for trade in trades)
        total_trades = len(trades)

        # Temporal patterns
        trade_times = [trade.timestamp for trade in trades]
        hourly_distribution = [0] * 24
        for trade_time in trade_times:
            hourly_distribution[trade_time.hour] += 1

        # Volume patterns
        buy_volumes = [float(trade.amount) for trade in trades if trade.is_buy]
        sell_volumes = [float(trade.amount) for trade in trades if not trade.is_buy]

        # Gas patterns
        gas_prices = [float(trade.gas_price_gwei) for trade in trades]

        # Calculate pattern fingerprint
        pattern_vector = self._create_pattern_vector(
            token_counts, hourly_distribution, buy_volumes, sell_volumes, gas_prices
        )

        self.wallet_patterns[wallet_address] = {
            'token_distribution': dict(token_counts),
            'hourly_distribution': hourly_distribution,
            'volume_stats': {
                'buy_avg': np.mean(buy_volumes) if buy_volumes else 0,
                'sell_avg': np.mean(sell_volumes) if sell_volumes else 0,
                'buy_std': np.std(buy_volumes) if buy_volumes else 0,
                'sell_std': np.std(sell_volumes) if sell_volumes else 0
            },
            'gas_stats': {
                'avg_gas_price': np.mean(gas_prices) if gas_prices else 0,
                'std_gas_price': np.std(gas_prices) if gas_prices else 0
            },
            'pattern_vector': pattern_vector,
            'total_trades': total_trades
        }

    def _create_pattern_vector(self, token_counts: Counter, hourly_dist: List[int],
                              buy_vols: List[float], sell_vols: List[float],
                              gas_prices: List[float]) -> np.ndarray:
        """Create a normalized pattern vector for similarity comparison."""
        features = []

        # Top 10 token frequencies (normalized)
        total_trades = sum(token_counts.values())
        top_tokens = token_counts.most_common(10)
        token_features = [count / total_trades for _, count in top_tokens]
        token_features.extend([0] * (10 - len(token_features)))  # Pad to 10
        features.extend(token_features)

        # Hourly distribution (normalized)
        total_hourly = sum(hourly_dist)
        hourly_features = [count / total_hourly if total_hourly > 0 else 0 for count in hourly_dist]
        features.extend(hourly_features)

        # Volume statistics
        features.extend([
            np.mean(buy_vols) if buy_vols else 0,
            np.std(buy_vols) if buy_vols else 0,
            np.mean(sell_vols) if sell_vols else 0,
            np.std(sell_vols) if sell_vols else 0
        ])

        # Gas statistics
        features.extend([
            np.mean(gas_prices) if gas_prices else 0,
            np.std(gas_prices) if gas_prices else 0
        ])

        return np.array(features)

    def _analyze_funding_connections(self, wallet_address: str, wallet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze funding source correlations for a wallet."""
        funding_data = wallet_data.get('funding_sources', [])

        if not funding_data:
            return {'score': 0.0, 'connections': [], 'evidence': 'no_funding_data'}

        # Find wallets that share funding sources
        shared_sources = []
        for other_wallet, other_patterns in self.wallet_patterns.items():
            if other_wallet == wallet_address:
                continue

            # Compare funding sources (simplified - would need actual implementation)
            # This is a placeholder for the actual funding correlation logic
            correlation = self._calculate_funding_correlation(wallet_address, other_wallet, funding_data)

            if correlation > self.funding_threshold:
                shared_sources.append({
                    'wallet': other_wallet,
                    'correlation': correlation
                })

        funding_score = min(len(shared_sources) * 0.3, 1.0)  # Score based on number of correlations

        return {
            'score': funding_score,
            'connections': shared_sources,
            'evidence': 'funding_correlation' if shared_sources else 'no_correlations'
        }

    def _analyze_trading_patterns(self, wallet_address: str) -> Dict[str, Any]:
        """Analyze trading pattern similarities with other wallets."""
        if wallet_address not in self.wallet_patterns:
            return {'score': 0.0, 'similarities': [], 'evidence': 'no_pattern_data'}

        wallet_pattern = self.wallet_patterns[wallet_address]
        wallet_vector = wallet_pattern['pattern_vector']

        similar_wallets = []
        for other_wallet, other_pattern in self.wallet_patterns.items():
            if other_wallet == wallet_address:
                continue

            other_vector = other_pattern['pattern_vector']

            # Calculate cosine similarity between pattern vectors
            similarity = cosine_similarity([wallet_vector], [other_vector])[0][0]

            if similarity > self.trading_threshold:
                similar_wallets.append({
                    'wallet': other_wallet,
                    'similarity': float(similarity)
                })

        pattern_score = min(len(similar_wallets) * 0.4, 1.0)

        return {
            'score': pattern_score,
            'similarities': similar_wallets,
            'evidence': 'pattern_similarity' if similar_wallets else 'no_similar_patterns'
        }

    def _analyze_temporal_correlations(self, wallet_address: str) -> Dict[str, Any]:
        """Analyze temporal trading correlations with other wallets."""
        if wallet_address not in self.wallet_patterns:
            return {'score': 0.0, 'correlations': [], 'evidence': 'no_temporal_data'}

        wallet_hourly = self.wallet_patterns[wallet_address]['hourly_distribution']

        temporal_correlations = []
        for other_wallet, other_pattern in self.wallet_patterns.items():
            if other_wallet == wallet_address:
                continue

            other_hourly = other_pattern['hourly_distribution']

            # Calculate Pearson correlation for hourly distributions
            if sum(wallet_hourly) > 0 and sum(other_hourly) > 0:
                correlation = np.corrcoef(wallet_hourly, other_hourly)[0, 1]

                if not np.isnan(correlation) and correlation > self.temporal_threshold:
                    temporal_correlations.append({
                        'wallet': other_wallet,
                        'correlation': float(correlation)
                    })

        temporal_score = min(len(temporal_correlations) * 0.3, 1.0)

        return {
            'score': temporal_score,
            'correlations': temporal_correlations,
            'evidence': 'temporal_correlation' if temporal_correlations else 'no_temporal_correlation'
        }

    def _analyze_network_position(self, wallet_address: str) -> Dict[str, Any]:
        """Analyze network position and centrality for Sybil detection."""
        if not self.funding_graph.has_node(wallet_address):
            return {'score': 0.0, 'metrics': {}, 'evidence': 'not_in_network'}

        try:
            # Calculate network centrality metrics
            betweenness = nx.betweenness_centrality(self.funding_graph)[wallet_address]
            closeness = nx.closeness_centrality(self.funding_graph)[wallet_address]
            degree = self.funding_graph.degree(wallet_address)

            # High centrality might indicate hub behavior (potential Sybil controller)
            network_score = (betweenness * 0.4 + closeness * 0.3 + min(degree / 10, 1.0) * 0.3)

            return {
                'score': network_score,
                'metrics': {
                    'betweenness_centrality': betweenness,
                    'closeness_centrality': closeness,
                    'degree': degree
                },
                'evidence': 'network_analysis'
            }

        except Exception as e:
            logger.warning(f"Network analysis failed for {wallet_address}: {e}")
            return {'score': 0.0, 'metrics': {}, 'evidence': 'network_analysis_failed'}

    def _calculate_combined_sybil_score(self, funding: Dict, patterns: Dict,
                                       temporal: Dict, network: Dict) -> float:
        """Combine evidence scores into a final Sybil probability score."""
        weights = {
            'funding': 0.3,
            'patterns': 0.3,
            'temporal': 0.2,
            'network': 0.2
        }

        combined_score = (
            funding['score'] * weights['funding'] +
            patterns['score'] * weights['patterns'] +
            temporal['score'] * weights['temporal'] +
            network['score'] * weights['network']
        )

        return min(combined_score, 1.0)

    def _calculate_funding_correlation(self, wallet_a: str, wallet_b: str, funding_data: List) -> float:
        """Calculate funding source correlation between two wallets."""
        # Placeholder implementation - would need actual funding source data
        # This would compare funding transaction patterns, source addresses, etc.
        return 0.0

    def _update_funding_graph(self, wallet_address: str, wallet_data: Dict[str, Any]) -> None:
        """Update the funding relationship graph."""
        funding_sources = wallet_data.get('funding_sources', [])

        self.funding_graph.add_node(wallet_address)

        for source in funding_sources:
            source_address = source.get('source_address')
            if source_address:
                self.funding_graph.add_edge(source_address, wallet_address,
                                          weight=float(source.get('amount', 0)))

    def _cluster_by_funding_sources(self) -> List[Set[str]]:
        """Cluster wallets by funding source similarity."""
        # Use community detection on funding graph
        clusters = []
        try:
            communities = nx.community.greedy_modularity_communities(self.funding_graph.to_undirected())
            clusters = [set(community) for community in communities if len(community) >= self.min_cluster_size]
        except Exception as e:
            logger.warning(f"Funding clustering failed: {e}")

        return clusters

    def _cluster_by_trading_patterns(self) -> List[Set[str]]:
        """Cluster wallets by trading pattern similarity."""
        if len(self.wallet_patterns) < 2:
            return []

        # Create similarity matrix
        wallets = list(self.wallet_patterns.keys())
        vectors = [self.wallet_patterns[w]['pattern_vector'] for w in wallets]

        # Use DBSCAN clustering
        scaler = StandardScaler()
        scaled_vectors = scaler.fit_transform(vectors)

        clustering = DBSCAN(eps=0.5, min_samples=self.min_cluster_size)
        labels = clustering.fit_predict(scaled_vectors)

        # Group wallets by cluster label
        clusters = defaultdict(set)
        for wallet, label in zip(wallets, labels):
            if label != -1:  # Ignore noise points
                clusters[label].add(wallet)

        return [cluster for cluster in clusters.values() if len(cluster) >= self.min_cluster_size]

    def _cluster_by_temporal_correlation(self) -> List[Set[str]]:
        """Cluster wallets by temporal trading patterns."""
        if len(self.wallet_patterns) < 2:
            return []

        wallets = list(self.wallet_patterns.keys())
        temporal_matrix = []

        for wallet in wallets:
            temporal_matrix.append(self.wallet_patterns[wallet]['hourly_distribution'])

        # Calculate correlation matrix
        temporal_matrix = np.array(temporal_matrix)
        correlation_matrix = np.corrcoef(temporal_matrix)

        # Find highly correlated pairs
        clusters = []
        processed = set()

        for i, wallet_a in enumerate(wallets):
            if wallet_a in processed:
                continue

            cluster = {wallet_a}
            for j, wallet_b in enumerate(wallets):
                if i != j and wallet_b not in processed:
                    if correlation_matrix[i, j] > self.temporal_threshold:
                        cluster.add(wallet_b)

            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)
                processed.update(cluster)

        return clusters

    def _cluster_by_network_analysis(self) -> List[Set[str]]:
        """Cluster wallets using network analysis of funding relationships."""
        clusters = []

        # Find strongly connected components
        try:
            components = list(nx.strongly_connected_components(self.funding_graph))
            clusters = [comp for comp in components if len(comp) >= self.min_cluster_size]
        except Exception as e:
            logger.warning(f"Network clustering failed: {e}")

        return clusters

    def _combine_cluster_evidence(self, funding_clusters: List[Set[str]],
                                 pattern_clusters: List[Set[str]],
                                 temporal_clusters: List[Set[str]],
                                 network_clusters: List[Set[str]]) -> List[SybilCluster]:
        """Combine evidence from different clustering methods."""
        all_clusters = []
        cluster_id_counter = 0

        # Process each type of cluster
        cluster_types = [
            ('funding', funding_clusters),
            ('pattern', pattern_clusters),
            ('temporal', temporal_clusters),
            ('network', network_clusters)
        ]

        for cluster_type, clusters in cluster_types:
            for cluster_wallets in clusters:
                cluster_id = f"sybil_cluster_{cluster_id_counter:04d}"
                cluster_id_counter += 1

                # Calculate evidence score based on cluster size and type
                base_score = 0.5
                size_bonus = min((len(cluster_wallets) - self.min_cluster_size) * 0.1, 0.3)
                type_weights = {'funding': 0.3, 'pattern': 0.3, 'temporal': 0.2, 'network': 0.2}
                evidence_score = base_score + size_bonus + type_weights.get(cluster_type, 0.1)

                sybil_cluster = SybilCluster(
                    cluster_id=cluster_id,
                    wallet_addresses=cluster_wallets,
                    evidence_score=min(evidence_score, 1.0),
                    detection_methods=[cluster_type],
                    funding_connections=[],
                    pattern_similarities=[],
                    risk_level=self._assess_risk_level(evidence_score, len(cluster_wallets))
                )

                all_clusters.append(sybil_cluster)

        return all_clusters

    def _validate_clusters(self, clusters: List[SybilCluster]) -> List[SybilCluster]:
        """Validate and merge overlapping clusters."""
        # Merge clusters with significant wallet overlap
        merged_clusters = []
        processed_wallets = set()

        for cluster in sorted(clusters, key=lambda c: c.evidence_score, reverse=True):
            # Check for overlap with existing clusters
            overlap_found = False
            cluster_wallets = cluster.wallet_addresses

            for merged_cluster in merged_clusters:
                overlap = len(cluster_wallets & merged_cluster.wallet_addresses)
                overlap_ratio = overlap / len(cluster_wallets)

                if overlap_ratio > 0.5:  # 50% overlap threshold
                    # Merge clusters
                    merged_cluster.wallet_addresses.update(cluster_wallets)
                    merged_cluster.detection_methods.extend(cluster.detection_methods)
                    merged_cluster.evidence_score = max(merged_cluster.evidence_score, cluster.evidence_score)
                    overlap_found = True
                    break

            if not overlap_found and not (cluster_wallets & processed_wallets):
                # Add as new cluster
                merged_clusters.append(cluster)
                processed_wallets.update(cluster_wallets)

        # Final validation - require minimum evidence score
        validated_clusters = [
            cluster for cluster in merged_clusters
            if cluster.evidence_score >= 0.6 and len(cluster.wallet_addresses) >= self.min_cluster_size
        ]

        return validated_clusters

    def _assess_risk_level(self, evidence_score: float, cluster_size: int) -> str:
        """Assess risk level based on evidence score and cluster size."""
        if evidence_score >= 0.8 or cluster_size >= 10:
            return "high"
        elif evidence_score >= 0.6 or cluster_size >= 5:
            return "medium"
        else:
            return "low"

    def _find_existing_cluster(self, wallet_address: str) -> Optional[SybilCluster]:
        """Find if wallet is already part of a detected cluster."""
        for cluster in self.detected_clusters:
            if wallet_address in cluster.wallet_addresses:
                return cluster
        return None

    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get Sybil detection statistics."""
        return {
            **self.detection_stats,
            'clusters_by_risk': {
                'high': len([c for c in self.detected_clusters if c.risk_level == 'high']),
                'medium': len([c for c in self.detected_clusters if c.risk_level == 'medium']),
                'low': len([c for c in self.detected_clusters if c.risk_level == 'low'])
            },
            'average_cluster_size': np.mean([len(c.wallet_addresses) for c in self.detected_clusters]) if self.detected_clusters else 0
        }