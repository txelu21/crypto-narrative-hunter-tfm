"""Tests for transaction decoder."""

import pytest
from decimal import Decimal
from services.transactions.decoder import SwapDecoder, UNISWAP_V2_SWAP_SIGNATURE, WETH_ADDRESS


@pytest.fixture
def decoder():
    """Create decoder instance."""
    return SwapDecoder()


def test_identify_dex_uniswap_v2(decoder):
    """Test DEX identification for Uniswap V2."""
    router = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
    assert decoder._identify_dex(router) == "uniswap_v2"


def test_identify_dex_uniswap_v3(decoder):
    """Test DEX identification for Uniswap V3."""
    router = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
    assert decoder._identify_dex(router) == "uniswap_v3"


def test_identify_dex_unknown(decoder):
    """Test DEX identification for unknown router."""
    unknown = "0x0000000000000000000000000000000000000000"
    assert decoder._identify_dex(unknown) is None


def test_decode_uniswap_v2_swap(decoder):
    """Test decoding Uniswap V2 swap."""
    tx_hash = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
    receipt = {
        "transactionHash": tx_hash,
        "blockNumber": 15000000,
        "gasUsed": 150000,
        "effectiveGasPrice": 50000000000,  # 50 gwei
        "status": 1,
        "to": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"  # Uniswap V2 Router
    }

    # Mock swap log
    logs = [{
        "address": "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc",  # USDC-WETH pool
        "topics": [
            bytes.fromhex(UNISWAP_V2_SWAP_SIGNATURE[2:]),
            bytes.fromhex("0000000000000000000000007a250d5630b4cf539739df2c5dacb4c659f2488d"),
            bytes.fromhex("0000000000000000000000001234567890123456789012345678901234567890")
        ],
        "data": (
            # amount0In (0)
            "0000000000000000000000000000000000000000000000000000000000000000"
            # amount1In (1 WETH = 1e18)
            "0000000000000000000000000000000000000000000000000de0b6b3a7640000"
            # amount0Out (2000 USDC = 2000e6)
            "0000000000000000000000000000000000000000000000000000000077359400"
            # amount1Out (0)
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
    }]

    wallet_address = "0x1234567890123456789012345678901234567890"
    swaps = decoder.decode_transaction(tx_hash, receipt, logs, wallet_address)

    assert len(swaps) == 1
    swap = swaps[0]
    assert swap.tx_hash == tx_hash
    assert swap.dex_name == "uniswap_v2"
    assert swap.transaction_status == "success"
    assert swap.wallet_address == wallet_address.lower()


def test_decode_uniswap_v3_swap(decoder):
    """Test decoding Uniswap V3 swap."""
    from services.transactions.decoder import UNISWAP_V3_SWAP_SIGNATURE

    tx_hash = "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
    receipt = {
        "transactionHash": tx_hash,
        "blockNumber": 16000000,
        "gasUsed": 180000,
        "effectiveGasPrice": 60000000000,  # 60 gwei
        "status": 1,
        "to": "0xE592427A0AEce92De3Edee1F18E0157C05861564"  # Uniswap V3 Router
    }

    # Mock V3 swap log with signed integers
    # amount0 = -1000000 (negative = out)
    # amount1 = 500000000000000000 (positive = in)
    logs = [{
        "address": "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8",  # USDC-WETH V3 pool
        "topics": [
            bytes.fromhex(UNISWAP_V3_SWAP_SIGNATURE[2:]),
            bytes.fromhex("0000000000000000000000007a250d5630b4cf539739df2c5dacb4c659f2488d"),
            bytes.fromhex("0000000000000000000000001234567890123456789012345678901234567890")
        ],
        "data": (
            # amount0 (-1000000, two's complement for negative) - 32 bytes
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffff0bdc0"
            # amount1 (500000000000000000) - 32 bytes
            "00000000000000000000000000000000000000000000000006f05b59d3b20000"
            # sqrtPriceX96 - 32 bytes
            "0000000000000000000000000000000001a56e76f6d1e6b7e8e9ea00000000000"
            # liquidity - 32 bytes
            "0000000000000000000000000000000000000000000000000000000075bcd15"
            # tick - 32 bytes (signed)
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff8c42"
        )
    }]

    wallet_address = "0x1234567890123456789012345678901234567890"
    swaps = decoder.decode_transaction(tx_hash, receipt, logs, wallet_address)

    assert len(swaps) == 1
    swap = swaps[0]
    assert swap.tx_hash == tx_hash
    assert swap.dex_name == "uniswap_v3"
    assert swap.transaction_status == "success"


def test_decode_failed_transaction(decoder):
    """Test decoding failed transaction."""
    tx_hash = "0xfailed1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
    receipt = {
        "transactionHash": tx_hash,
        "blockNumber": 15000000,
        "gasUsed": 50000,
        "effectiveGasPrice": 50000000000,
        "status": 0,  # Failed
        "to": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
    }

    logs = []
    wallet_address = "0x1234567890123456789012345678901234567890"

    swaps = decoder.decode_transaction(tx_hash, receipt, logs, wallet_address)

    # Failed transactions with no swap logs should return empty list
    assert len(swaps) == 0


def test_calculate_slippage(decoder):
    """Test slippage calculation."""
    # No slippage
    slippage = decoder._calculate_slippage(Decimal("1.0"), Decimal("1.0"))
    assert slippage == Decimal("0.00")

    # Positive slippage (gain)
    slippage = decoder._calculate_slippage(Decimal("1.0"), Decimal("1.05"))
    assert slippage == Decimal("5.00")

    # Negative slippage (loss)
    slippage = decoder._calculate_slippage(Decimal("1.0"), Decimal("0.95"))
    assert slippage == Decimal("-5.00")

    # Zero input
    slippage = decoder._calculate_slippage(Decimal("0"), Decimal("1.0"))
    assert slippage == Decimal("0")


def test_normalize_to_eth_weth(decoder):
    """Test ETH normalization for WETH."""
    amount = 1000000000000000000  # 1 WETH
    eth_value = decoder._normalize_to_eth(
        WETH_ADDRESS,
        amount,
        "0x0000000000000000000000000000000000000000",
        15000000
    )

    assert eth_value == Decimal("1.0")


def test_parse_v2_swap_direction(decoder):
    """Test parsing V2 swap direction."""
    pool_address = "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc"

    # Swap token0 for token1
    token_in, amount_in, token_out, amount_out = decoder._parse_v2_swap_direction(
        pool_address,
        amount0_in=1000000,  # 1 USDC in
        amount1_in=0,
        amount0_out=0,
        amount1_out=500000000000000000,  # 0.5 WETH out
        block_number=15000000
    )

    assert amount_in == 1000000
    assert amount_out == 500000000000000000


def test_decode_from_logs_multiple_dex(decoder):
    """Test decoding from logs with multiple DEX protocols."""
    from services.transactions.decoder import CURVE_EXCHANGE_SIGNATURE

    tx_hash = "0xmulti1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
    receipt = {
        "transactionHash": tx_hash,
        "blockNumber": 15000000,
        "gasUsed": 200000,
        "effectiveGasPrice": 50000000000,
        "status": 1,
        "to": "0x0000000000000000000000000000000000000000"  # Unknown router
    }

    # Mix of V2 and Curve swaps
    logs = [
        {
            "address": "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc",
            "topics": [bytes.fromhex(UNISWAP_V2_SWAP_SIGNATURE[2:])],
            "data": "00" * 128
        },
        {
            "address": "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7",
            "topics": [
                bytes.fromhex(CURVE_EXCHANGE_SIGNATURE[2:]),
                bytes.fromhex("0000000000000000000000001234567890123456789012345678901234567890")
            ],
            "data": "00" * 128
        }
    ]

    wallet_address = "0x1234567890123456789012345678901234567890"
    swaps = decoder._decode_from_logs(tx_hash, receipt, logs, wallet_address)

    # Should attempt to decode both
    assert isinstance(swaps, list)