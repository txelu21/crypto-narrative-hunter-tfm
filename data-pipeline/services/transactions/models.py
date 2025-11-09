"""Pydantic models for transaction data."""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from web3 import Web3


class RawLog(BaseModel):
    """Raw log entry from eth_getLogs."""

    address: str
    topics: list[str]
    data: str
    block_number: int = Field(alias="blockNumber")
    transaction_hash: str = Field(alias="transactionHash")
    transaction_index: int = Field(alias="transactionIndex")
    block_hash: str = Field(alias="blockHash")
    log_index: int = Field(alias="logIndex")
    removed: bool = False

    class Config:
        populate_by_name = True


class TransactionReceipt(BaseModel):
    """Transaction receipt from eth_getTransactionReceipt."""

    transaction_hash: str = Field(alias="transactionHash")
    block_number: int = Field(alias="blockNumber")
    block_hash: str = Field(alias="blockHash")
    transaction_index: int = Field(alias="transactionIndex")
    from_address: str = Field(alias="from")
    to_address: Optional[str] = Field(None, alias="to")
    cumulative_gas_used: int = Field(alias="cumulativeGasUsed")
    gas_used: int = Field(alias="gasUsed")
    effective_gas_price: int = Field(alias="effectiveGasPrice")
    status: int
    logs: list[dict]

    class Config:
        populate_by_name = True

    @field_validator("from_address", "to_address")
    @classmethod
    def validate_address(cls, v: Optional[str]) -> Optional[str]:
        """Validate Ethereum address format."""
        if v is None:
            return None
        if not v.startswith("0x") or len(v) != 42:
            raise ValueError(f"Invalid Ethereum address: {v}")
        return v.lower()


class Transaction(BaseModel):
    """Full transaction details from eth_getTransactionByHash."""

    hash: str
    nonce: int
    block_hash: str = Field(alias="blockHash")
    block_number: int = Field(alias="blockNumber")
    transaction_index: int = Field(alias="transactionIndex")
    from_address: str = Field(alias="from")
    to_address: Optional[str] = Field(None, alias="to")
    value: str
    gas: int
    gas_price: str = Field(alias="gasPrice")
    input: str
    v: str
    r: str
    s: str

    class Config:
        populate_by_name = True

    @field_validator("from_address", "to_address")
    @classmethod
    def validate_address(cls, v: Optional[str]) -> Optional[str]:
        """Validate Ethereum address format."""
        if v is None:
            return None
        if not v.startswith("0x") or len(v) != 42:
            raise ValueError(f"Invalid Ethereum address: {v}")
        return v.lower()


class SwapTransaction(BaseModel):
    """Decoded swap transaction with pricing."""

    tx_hash: str
    block_number: int
    timestamp: datetime
    wallet_address: str

    # DEX and pool information
    dex_name: str
    pool_address: str

    # Swap details
    token_in: str
    amount_in: Decimal
    token_out: str
    amount_out: Decimal

    # Gas and execution
    gas_used: int
    gas_price_gwei: Decimal
    transaction_status: str = Field(pattern="^(success|failed)$")

    # Value calculations
    eth_value_in: Optional[Decimal] = None
    eth_value_out: Optional[Decimal] = None
    usd_value_in: Optional[Decimal] = None
    usd_value_out: Optional[Decimal] = None

    # MEV and quality flags
    mev_type: Optional[str] = None
    mev_damage_eth: Optional[Decimal] = None
    slippage_percentage: Optional[Decimal] = None

    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("wallet_address", "pool_address", "token_in", "token_out")
    @classmethod
    def validate_eth_address(cls, v: str) -> str:
        """Ensure valid checksummed Ethereum address."""
        if not Web3.is_checksum_address(v):
            try:
                v = Web3.to_checksum_address(v)
            except ValueError:
                raise ValueError(f"Invalid Ethereum address: {v}")
        return v

    @field_validator("tx_hash")
    @classmethod
    def validate_tx_hash(cls, v: str) -> str:
        """Validate transaction hash format."""
        if not v.startswith("0x") or len(v) != 66:
            raise ValueError(f"Invalid transaction hash: {v}")
        return v.lower()

    @field_validator("dex_name")
    @classmethod
    def validate_dex_name(cls, v: str) -> str:
        """Validate DEX name is recognized."""
        valid_dexs = {"uniswap_v2", "uniswap_v3", "curve", "sushiswap", "balancer"}
        if v.lower() not in valid_dexs:
            raise ValueError(f"Unrecognized DEX: {v}. Valid options: {valid_dexs}")
        return v.lower()


class DecodedSwap(BaseModel):
    """Decoded swap from transaction logs."""

    tx_hash: str
    block_number: int
    wallet_address: str

    # DEX and pool information
    dex_name: str
    pool_address: str

    # Swap details
    token_in: str
    amount_in: str  # String to preserve precision
    token_out: str
    amount_out: str  # String to preserve precision

    # Gas and execution
    gas_used: int
    gas_price_gwei: str
    transaction_status: str

    # Value calculations (as strings to preserve precision)
    eth_value_in: str
    eth_value_out: str

    # Quality metrics
    slippage_percentage: Optional[str] = None

    @field_validator("wallet_address", "pool_address", "token_in", "token_out")
    @classmethod
    def validate_eth_address(cls, v: str) -> str:
        """Ensure valid Ethereum address."""
        if not v.startswith("0x") or len(v) != 42:
            raise ValueError(f"Invalid Ethereum address: {v}")
        return v.lower()


class PriceImpact(BaseModel):
    """Price impact analysis for a swap."""

    tx_hash: str
    pool_address: str
    expected_output: Decimal
    actual_output: Decimal
    price_impact_percentage: Decimal
    is_suspicious: bool = False


class FailedTransaction(BaseModel):
    """Failed transaction with analysis."""

    tx_hash: str
    block_number: int
    timestamp: datetime
    wallet_address: str

    # Failure information
    status: str = "failed"
    revert_reason: Optional[str] = None

    # Gas costs
    gas_used: int
    gas_price_gwei: Decimal
    gas_cost_eth: Decimal

    # MEV impact
    mev_type: Optional[str] = None
    mev_damage_eth: Optional[Decimal] = None

    # Context
    intended_dex: Optional[str] = None
    attempted_token_in: Optional[str] = None
    attempted_token_out: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("wallet_address", "attempted_token_in", "attempted_token_out")
    @classmethod
    def validate_eth_address(cls, v: Optional[str]) -> Optional[str]:
        """Ensure valid checksummed Ethereum address."""
        if v is None:
            return None
        if not Web3.is_checksum_address(v):
            try:
                v = Web3.to_checksum_address(v)
            except ValueError:
                raise ValueError(f"Invalid Ethereum address: {v}")
        return v

    @field_validator("mev_type")
    @classmethod
    def validate_mev_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate MEV type."""
        if v is None:
            return None
        valid_types = {"sandwich", "front_run", "back_run", "liquidation"}
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid MEV type: {v}. Valid: {valid_types}")
        return v.lower()