#!/usr/bin/env python3
"""
Quick test script to verify new narrative categories work correctly.
Tests the 4 new categories: Layer2, RWA, LiquidStaking, Privacy
"""

from services.tokens.narrative_classifier import NarrativeClassifier, NarrativeCategory


def test_new_categories():
    """Test that new narrative categories classify correctly"""
    classifier = NarrativeClassifier()

    # Test cases for each new category
    test_tokens = [
        # Layer 2 tokens
        {
            "address": "0x123...",
            "symbol": "ARB",
            "name": "Arbitrum",
            "expected": NarrativeCategory.LAYER2
        },
        {
            "address": "0x234...",
            "symbol": "OP",
            "name": "Optimism",
            "expected": NarrativeCategory.LAYER2
        },
        {
            "address": "0x345...",
            "symbol": "MATIC",
            "name": "Polygon",
            "expected": NarrativeCategory.LAYER2
        },
        # RWA tokens
        {
            "address": "0x456...",
            "symbol": "RWA",
            "name": "Real World Asset Token",
            "expected": NarrativeCategory.RWA
        },
        {
            "address": "0x567...",
            "symbol": "ONDO",
            "name": "Ondo Finance Treasury",
            "expected": NarrativeCategory.RWA
        },
        {
            "address": "0x678...",
            "symbol": "CFG",
            "name": "Centrifuge",
            "expected": NarrativeCategory.RWA
        },
        # Liquid Staking tokens
        {
            "address": "0xae7ab96520de3a18e5e111b5eaab095312d7fe84",
            "symbol": "stETH",
            "name": "Lido Staked Ether",
            "expected": NarrativeCategory.LIQUID_STAKING
        },
        {
            "address": "0x789...",
            "symbol": "rETH",
            "name": "Rocket Pool ETH",
            "expected": NarrativeCategory.LIQUID_STAKING
        },
        {
            "address": "0x890...",
            "symbol": "cbETH",
            "name": "Coinbase Staked ETH",
            "expected": NarrativeCategory.LIQUID_STAKING
        },
        # Privacy tokens
        {
            "address": "0x901...",
            "symbol": "RAIL",
            "name": "Railgun Privacy Protocol",
            "expected": NarrativeCategory.PRIVACY
        },
        {
            "address": "0x012...",
            "symbol": "AZTEC",
            "name": "Aztec Network",
            "expected": NarrativeCategory.PRIVACY
        },
        {
            "address": "0x123abc...",
            "symbol": "SECRET",
            "name": "Secret Network",
            "expected": NarrativeCategory.PRIVACY
        }
    ]

    print("Testing New Narrative Categories")
    print("=" * 60)

    passed = 0
    failed = 0

    for token in test_tokens:
        result = classifier.classify_token(
            token["address"],
            token["symbol"],
            token["name"]
        )

        status = "✅ PASS" if result.category == token["expected"] else "❌ FAIL"

        if result.category == token["expected"]:
            passed += 1
        else:
            failed += 1

        print(f"\n{status}")
        print(f"  Token: {token['symbol']} - {token['name']}")
        print(f"  Expected: {token['expected'].value}")
        print(f"  Got: {result.category.value} ({result.confidence:.1f}% confidence)")
        print(f"  Matched: {result.matched_keywords[:3]}")  # Show first 3 keywords

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_tokens)} tests")
    print("=" * 60)

    # Test that all categories are available
    print("\nAll Available Categories:")
    for cat in NarrativeCategory:
        print(f"  - {cat.value}")

    return passed == len(test_tokens)


if __name__ == "__main__":
    success = test_new_categories()
    exit(0 if success else 1)
