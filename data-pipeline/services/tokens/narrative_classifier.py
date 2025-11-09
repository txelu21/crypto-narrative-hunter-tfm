"""
Narrative categorization system for crypto tokens.

Implements rule-based classification with confidence scoring:
- Keyword matching for token names and symbols
- Contract address pattern recognition for known protocols
- Confidence scoring system for automated classifications
- Support for multiple narrative categories
- Edge case handling for multi-category tokens
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from data_collection.common.logging_setup import get_logger


class NarrativeCategory(Enum):
    """Enumeration of supported narrative categories"""
    DEFI = "DeFi"
    LAYER2 = "Layer2"
    GAMING = "Gaming"
    AI = "AI"
    RWA = "RWA"
    LIQUID_STAKING = "LiquidStaking"
    PRIVACY = "Privacy"
    INFRASTRUCTURE = "Infrastructure"
    MEME = "Meme"
    STABLECOIN = "Stablecoin"
    OTHER = "Other"


@dataclass
class ClassificationResult:
    """Result of narrative classification with confidence score"""
    category: NarrativeCategory
    confidence: float
    matched_keywords: List[str]
    matched_patterns: List[str]
    reasoning: str


class NarrativeClassifier:
    """
    Rule-based narrative classification system for crypto tokens.

    Categorizes tokens based on:
    - Token name and symbol keyword matching
    - Known protocol contract address patterns
    - Confidence scoring for classification reliability
    """

    def __init__(self):
        self.logger = get_logger("narrative_classifier")
        self._initialize_classification_rules()

    def _initialize_classification_rules(self) -> None:
        """Initialize classification keywords and patterns"""

        # DeFi keywords - decentralized finance protocols, DEXs, lending, yield farming
        self.defi_keywords = {
            'primary': ['defi', 'swap', 'exchange', 'dex', 'lending', 'yield', 'liquidity',
                       'farm', 'vault', 'compound', 'aave', 'maker', 'curve', 'balancer',
                       'sushi', 'pancake', 'uniswap', 'finance', 'protocol', 'dao'],
            'secondary': ['stake', 'staking', 'pool', 'lp', 'reward', 'governance', 'vote',
                         'treasury', 'synthetic', 'derivative', 'leverage', 'margin']
        }

        # Gaming keywords - GameFi, NFT gaming, virtual worlds, play-to-earn
        self.gaming_keywords = {
            'primary': ['game', 'gaming', 'gamefi', 'play', 'metaverse', 'virtual', 'world',
                       'land', 'avatar', 'character', 'item', 'weapon', 'collectible',
                       'sandbox', 'decentraland', 'axie', 'crypto', 'kitty'],
            'secondary': ['nft', 'earn', 'battle', 'quest', 'adventure', 'rpg', 'mmo',
                         'breeding', 'racing', 'sport', 'card', 'pet', 'monster']
        }

        # AI keywords - artificial intelligence, machine learning, data processing
        self.ai_keywords = {
            'primary': ['ai', 'artificial', 'intelligence', 'machine', 'learning', 'neural',
                       'network', 'data', 'compute', 'processing', 'algorithm', 'model',
                       'prediction', 'automation', 'bot'],
            'secondary': ['deep', 'cognitive', 'smart', 'analytics', 'insight', 'vision',
                         'language', 'nlp', 'ml', 'tensorflow', 'pytorch']
        }

        # Infrastructure keywords - Layer 2, bridges, oracles, development tools
        self.infrastructure_keywords = {
            'primary': ['layer', 'bridge', 'oracle', 'validator', 'node', 'network',
                       'infrastructure', 'scaling', 'sidechain', 'rollup', 'consensus',
                       'blockchain', 'protocol', 'chain'],
            'secondary': ['interoperability', 'cross', 'relay', 'rpc', 'api', 'sdk',
                         'developer', 'tool', 'framework', 'middleware', 'service']
        }

        # Meme keywords - community-driven tokens, meme coins, social tokens
        self.meme_keywords = {
            'primary': ['meme', 'doge', 'shib', 'inu', 'moon', 'diamond', 'ape', 'pepe',
                       'wojak', 'chad', 'bonk', 'floki', 'safe', 'baby', 'mini'],
            'secondary': ['community', 'social', 'viral', 'trending', 'pump', 'hodl',
                         'rocket', 'lambo', 'gem', 'x1000', 'moon', 'mars']
        }

        # Stablecoin keywords - USD-pegged, algorithmic stables, reserve-backed
        self.stablecoin_keywords = {
            'primary': ['stable', 'usd', 'usdt', 'usdc', 'dai', 'busd', 'ust', 'frax',
                       'algorithmic', 'pegged', 'backed', 'reserve', 'collateral'],
            'secondary': ['dollar', 'fiat', 'basket', 'treasury', 'mint', 'burn',
                         'redemption', 'arbitrage', 'parity']
        }

        # Layer 2 & Scaling keywords - L2 solutions, rollups, sidechains
        self.layer2_keywords = {
            'primary': ['layer2', 'l2', 'rollup', 'optimism', 'arbitrum', 'zksync', 'zkrollup',
                       'optimistic', 'zkevm', 'polygon', 'matic', 'scroll', 'starknet',
                       'base', 'linea', 'metis', 'boba', 'loopring'],
            'secondary': ['scaling', 'sidechain', 'plasma', 'state channel', 'validity proof',
                         'fraud proof', 'sequencer', 'prover', 'zkp', 'zero knowledge']
        }

        # Real World Assets (RWA) keywords - tokenized real-world assets
        self.rwa_keywords = {
            'primary': ['rwa', 'real world', 'tokenized', 'asset backed', 'bond', 'treasury',
                       'tbill', 't-bill', 'securities', 'credit', 'commodities', 'gold',
                       'property', 'real estate', 'realestate', 'centrifuge', 'maple',
                       'goldfinch', 'backed'],
            'secondary': ['yield', 'income', 'institutional', 'compliance', 'regulated',
                         'tradfi', 'traditional finance', 'securitization', 'debt',
                         'equity', 'fund', 'investment']
        }

        # Liquid Staking & Restaking keywords - LSD tokens, staking derivatives
        self.liquid_staking_keywords = {
            'primary': ['steth', 'lido', 'reth', 'rocket', 'liquid staking', 'staked eth',
                       'staked', 'cbeth', 'wsteth', 'restaking', 'eigenlayer', 'eigen',
                       'liquid restaking', 'lrt', 'validator', 'beacon'],
            'secondary': ['staking derivative', 'eth2', 'consensus', 'withdraw', 'unstake',
                         'slashing', 'rewards', 'yield bearing', 'wrapped staked']
        }

        # Privacy & Security keywords - privacy coins, zero-knowledge, security protocols
        self.privacy_keywords = {
            'primary': ['privacy', 'private', 'anonymous', 'zkp', 'zero knowledge', 'zk',
                       'confidential', 'secret', 'tornado', 'aztec', 'railgun', 'monero',
                       'zcash', 'encryption', 'encrypted', 'stealth'],
            'secondary': ['anonymity', 'untraceable', 'ring signature', 'mixer', 'mixing',
                         'shielded', 'dark', 'secure', 'security', 'audit']
        }

        # Known protocol address patterns (first 6 characters after 0x)
        self.protocol_patterns = {
            # DeFi protocols
            '0xa0b86a': NarrativeCategory.DEFI,  # Compound tokens
            '0x1f9840': NarrativeCategory.DEFI,  # Uniswap
            '0x6b175': NarrativeCategory.DEFI,  # DAI
            '0x7fc66': NarrativeCategory.DEFI,  # Aave tokens
            '0xc02a': NarrativeCategory.DEFI,   # WETH

            # Gaming protocols
            '0xf5b0': NarrativeCategory.GAMING,  # Example gaming protocol
            '0x037a': NarrativeCategory.GAMING,  # Example NFT gaming

            # Infrastructure
            '0x4200': NarrativeCategory.INFRASTRUCTURE,  # Layer 2 tokens
            '0x1337': NarrativeCategory.INFRASTRUCTURE,  # Bridge tokens

            # Stablecoins
            '0xa0b8': NarrativeCategory.STABLECOIN,  # USDC pattern example
            '0xdac1': NarrativeCategory.STABLECOIN,  # DAI pattern
        }

    def classify_token(self, token_address: str, symbol: str, name: str,
                      description: Optional[str] = None) -> ClassificationResult:
        """
        Classify a token into narrative categories with confidence scoring.

        Args:
            token_address: Token contract address
            symbol: Token symbol (e.g., "UNI")
            name: Token name (e.g., "Uniswap")
            description: Optional token description

        Returns:
            ClassificationResult with category, confidence, and reasoning
        """
        # Normalize inputs for matching
        symbol_lower = symbol.lower() if symbol else ""
        name_lower = name.lower() if name else ""
        desc_lower = description.lower() if description else ""

        # Combine text for analysis
        combined_text = f"{symbol_lower} {name_lower} {desc_lower}".strip()

        # Check protocol address patterns first (highest confidence)
        pattern_result = self._check_protocol_patterns(token_address)
        if pattern_result:
            return pattern_result

        # Perform keyword-based classification
        keyword_results = self._classify_by_keywords(combined_text, symbol_lower, name_lower)

        # Handle edge cases and conflicts
        final_result = self._resolve_classification_conflicts(keyword_results)

        self.logger.log_operation(
            operation="classify_token",
            params={
                "token_address": token_address[:10] + "...",
                "symbol": symbol,
                "category": final_result.category.value,
                "confidence": final_result.confidence
            },
            status="completed",
            message=f"Token classified as {final_result.category.value} with {final_result.confidence:.1f}% confidence"
        )

        return final_result

    def _check_protocol_patterns(self, token_address: str) -> Optional[ClassificationResult]:
        """Check if token address matches known protocol patterns"""
        if not token_address or len(token_address) < 8:
            return None

        # Check first 6 characters after 0x
        address_prefix = token_address[:8].lower()

        for pattern, category in self.protocol_patterns.items():
            if address_prefix.startswith(pattern.lower()):
                return ClassificationResult(
                    category=category,
                    confidence=95.0,
                    matched_keywords=[],
                    matched_patterns=[pattern],
                    reasoning=f"Matched known protocol pattern {pattern}"
                )

        return None

    def _classify_by_keywords(self, combined_text: str, symbol: str, name: str) -> List[ClassificationResult]:
        """Classify token based on keyword matching"""
        results = []

        # Check each category
        categories_to_check = [
            (NarrativeCategory.DEFI, self.defi_keywords),
            (NarrativeCategory.LAYER2, self.layer2_keywords),
            (NarrativeCategory.GAMING, self.gaming_keywords),
            (NarrativeCategory.AI, self.ai_keywords),
            (NarrativeCategory.RWA, self.rwa_keywords),
            (NarrativeCategory.LIQUID_STAKING, self.liquid_staking_keywords),
            (NarrativeCategory.PRIVACY, self.privacy_keywords),
            (NarrativeCategory.INFRASTRUCTURE, self.infrastructure_keywords),
            (NarrativeCategory.MEME, self.meme_keywords),
            (NarrativeCategory.STABLECOIN, self.stablecoin_keywords)
        ]

        for category, keywords in categories_to_check:
            result = self._calculate_keyword_confidence(category, keywords, combined_text, symbol, name)
            if result.confidence > 10.0:  # Only include if above threshold
                results.append(result)

        return results

    def _calculate_keyword_confidence(self, category: NarrativeCategory, keywords: Dict[str, List[str]],
                                    combined_text: str, symbol: str, name: str) -> ClassificationResult:
        """Calculate confidence score for a specific category"""
        matched_keywords = []
        confidence = 0.0

        # Check primary keywords (higher weight)
        for keyword in keywords['primary']:
            if self._keyword_matches(keyword, combined_text, symbol, name):
                matched_keywords.append(keyword)
                confidence += 20.0 if keyword in symbol else 15.0

        # Check secondary keywords (lower weight)
        for keyword in keywords['secondary']:
            if self._keyword_matches(keyword, combined_text, symbol, name):
                matched_keywords.append(keyword)
                confidence += 8.0 if keyword in symbol else 5.0

        # Cap confidence at 90% for keyword-only matches
        confidence = min(confidence, 90.0)

        reasoning = f"Matched keywords: {matched_keywords}" if matched_keywords else "No keyword matches"

        return ClassificationResult(
            category=category,
            confidence=confidence,
            matched_keywords=matched_keywords,
            matched_patterns=[],
            reasoning=reasoning
        )

    def _keyword_matches(self, keyword: str, combined_text: str, symbol: str, name: str) -> bool:
        """Check if keyword matches in token symbol or name with word boundaries"""
        # Exact match in symbol gets priority
        if keyword in symbol:
            return True

        # Word boundary match in name or combined text
        pattern = r'\b' + re.escape(keyword) + r'\b'
        return bool(re.search(pattern, combined_text, re.IGNORECASE))

    def _resolve_classification_conflicts(self, results: List[ClassificationResult]) -> ClassificationResult:
        """Resolve conflicts when multiple categories match"""
        if not results:
            return ClassificationResult(
                category=NarrativeCategory.OTHER,
                confidence=0.0,
                matched_keywords=[],
                matched_patterns=[],
                reasoning="No classification matches found"
            )

        # Sort by confidence (highest first)
        results.sort(key=lambda x: x.confidence, reverse=True)

        # If highest confidence is significantly higher, use it
        if len(results) == 1 or results[0].confidence - results[1].confidence > 20.0:
            return results[0]

        # Handle ties or close matches
        return self._handle_classification_ties(results)

    def _handle_classification_ties(self, results: List[ClassificationResult]) -> ClassificationResult:
        """Handle cases where multiple categories have similar confidence"""
        # Priority order for tie-breaking (most specific to least specific)
        priority_order = [
            NarrativeCategory.STABLECOIN,         # Most specific - pegged assets
            NarrativeCategory.LIQUID_STAKING,     # Very specific - staking derivatives
            NarrativeCategory.RWA,                # Specific - real-world backed
            NarrativeCategory.LAYER2,             # Specific - scaling solutions
            NarrativeCategory.PRIVACY,            # Specific - privacy-focused
            NarrativeCategory.DEFI,               # Broad - DeFi protocols
            NarrativeCategory.GAMING,             # Broad - gaming & metaverse
            NarrativeCategory.AI,                 # Broad - AI & data
            NarrativeCategory.INFRASTRUCTURE,     # Very broad - core infrastructure
            NarrativeCategory.MEME,               # Can overlap with many
            NarrativeCategory.OTHER               # Least specific - catch-all
        ]

        # Find highest priority category among top results
        top_confidence = results[0].confidence
        top_results = [r for r in results if abs(r.confidence - top_confidence) < 10.0]

        for priority_cat in priority_order:
            for result in top_results:
                if result.category == priority_cat:
                    # Adjust confidence down for tie-breaking
                    result.confidence = max(result.confidence - 5.0, 50.0)
                    result.reasoning = f"Tie-broken by priority: {result.reasoning}"
                    return result

        # Fallback to first result
        return results[0]

    def get_category_statistics(self, tokens: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Get classification statistics for a list of tokens.

        Args:
            tokens: List of token dictionaries with 'address', 'symbol', 'name'

        Returns:
            Dictionary with category counts
        """
        stats = {cat.value: 0 for cat in NarrativeCategory}

        for token in tokens:
            result = self.classify_token(
                token.get('address', ''),
                token.get('symbol', ''),
                token.get('name', ''),
                token.get('description', '')
            )
            stats[result.category.value] += 1

        return stats

    def batch_classify_tokens(self, tokens: List[Dict[str, str]]) -> List[Tuple[Dict[str, str], ClassificationResult]]:
        """
        Classify multiple tokens in batch.

        Args:
            tokens: List of token dictionaries

        Returns:
            List of (token, classification_result) tuples
        """
        results = []

        for token in tokens:
            classification = self.classify_token(
                token.get('address', ''),
                token.get('symbol', ''),
                token.get('name', ''),
                token.get('description', '')
            )
            results.append((token, classification))

        return results

    def is_high_confidence_classification(self, result: ClassificationResult) -> bool:
        """Check if classification result has high confidence (>80%)"""
        return result.confidence > 80.0

    def is_manual_review_required(self, result: ClassificationResult) -> bool:
        """Check if classification requires manual review (<50% confidence)"""
        return result.confidence < 50.0