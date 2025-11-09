"""
Failed transaction analysis module.

Analyzes failed transactions to determine:
- Failure reasons and revert messages
- Gas costs and wasted expenditure
- MEV attack patterns (sandwich, front-running, back-running)
- MEV damage estimation
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from web3 import Web3
from web3.types import TxReceipt, LogReceipt
from eth_abi import decode

from .models import FailedTransaction

logger = logging.getLogger(__name__)

# Known MEV bot addresses (partial list)
KNOWN_MEV_BOTS = {
    "0x000000000035b5e5ad9019092c665357240f594e",  # Flash bots relay
    "0x00000000003b3cc22af3ae1eac0440bcee416b40",  # MEV bot
    # Add more known MEV bots
}


class FailedTransactionAnalyzer:
    """Analyzes failed transactions and MEV impacts."""

    def __init__(self, web3: Optional[Web3] = None):
        """
        Initialize analyzer with Web3 instance.

        Args:
            web3: Web3 instance for on-chain calls. If None, creates default instance.
        """
        self.web3 = web3 or Web3()
        self._block_tx_cache: Dict[int, List[Dict[str, Any]]] = {}

    def analyze_failed_transaction(
        self,
        tx_hash: str,
        receipt: TxReceipt,
        wallet_address: str,
        timestamp: datetime
    ) -> Optional[FailedTransaction]:
        """
        Analyze a failed transaction.

        Args:
            tx_hash: Transaction hash
            receipt: Transaction receipt
            wallet_address: Wallet address
            timestamp: Transaction timestamp

        Returns:
            FailedTransaction object or None if transaction succeeded
        """
        if receipt.get("status", 1) == 1:
            return None  # Transaction succeeded

        # Calculate gas costs
        gas_used = receipt["gasUsed"]
        gas_price_gwei = Decimal(receipt["effectiveGasPrice"]) / Decimal(1e9)
        gas_cost_eth = Decimal(gas_used) * Decimal(receipt["effectiveGasPrice"]) / Decimal(1e18)

        # Extract revert reason
        revert_reason = self._extract_revert_reason(tx_hash, receipt)

        # Detect MEV type and estimate damage
        mev_type, mev_damage = self._detect_mev_attack(tx_hash, receipt)

        # Extract intended swap details
        intended_dex, token_in, token_out = self._extract_intended_swap(receipt)

        failed_tx = FailedTransaction(
            tx_hash=tx_hash,
            block_number=receipt["blockNumber"],
            timestamp=timestamp,
            wallet_address=wallet_address,
            status="failed",
            revert_reason=revert_reason,
            gas_used=gas_used,
            gas_price_gwei=gas_price_gwei,
            gas_cost_eth=gas_cost_eth,
            mev_type=mev_type,
            mev_damage_eth=mev_damage,
            intended_dex=intended_dex,
            attempted_token_in=token_in,
            attempted_token_out=token_out
        )

        return failed_tx

    def _extract_revert_reason(self, tx_hash: str, receipt: TxReceipt) -> Optional[str]:
        """
        Extract revert reason from failed transaction.

        Args:
            tx_hash: Transaction hash
            receipt: Transaction receipt

        Returns:
            Revert reason string or None
        """
        try:
            # Try to get revert reason via eth_call replay
            # This requires re-executing the transaction
            # For now, return None (would need full implementation with Web3)
            return None
        except Exception as e:
            logger.debug(f"Could not extract revert reason for {tx_hash}: {e}")
            return None

    def _detect_mev_attack(
        self,
        tx_hash: str,
        receipt: TxReceipt
    ) -> Tuple[Optional[str], Optional[Decimal]]:
        """
        Detect MEV attack type and estimate damage.

        Args:
            tx_hash: Transaction hash
            receipt: Transaction receipt

        Returns:
            Tuple of (mev_type, estimated_damage_eth)
        """
        block_number = receipt["blockNumber"]
        tx_index = receipt["transactionIndex"]

        # Get all transactions in the block
        block_txs = self._get_block_transactions(block_number)

        # Check for sandwich attack
        if self._is_sandwich_attack(tx_index, block_txs, receipt):
            damage = self._estimate_sandwich_damage(tx_hash, tx_index, block_txs)
            return "sandwich", damage

        # Check for front-running
        if self._is_front_run(tx_index, block_txs, receipt):
            damage = self._estimate_frontrun_damage(tx_hash, tx_index, block_txs)
            return "front_run", damage

        # Check for back-running
        if self._is_back_run(tx_index, block_txs, receipt):
            return "back_run", None

        return None, None

    def _get_block_transactions(self, block_number: int) -> List[Dict[str, Any]]:
        """
        Get all transactions in a block.

        Args:
            block_number: Block number

        Returns:
            List of transaction dicts
        """
        # Check cache
        if block_number in self._block_tx_cache:
            return self._block_tx_cache[block_number]

        # TODO: Implement actual block transaction fetching
        # For now, return empty list
        txs: List[Dict[str, Any]] = []

        self._block_tx_cache[block_number] = txs
        return txs

    def _is_sandwich_attack(
        self,
        tx_index: int,
        block_txs: List[Dict[str, Any]],
        receipt: TxReceipt
    ) -> bool:
        """
        Check if transaction was sandwich attacked.

        Sandwich attack pattern:
        - MEV bot front-runs (tx before victim)
        - Victim transaction (current tx)
        - MEV bot back-runs (tx after victim)

        Args:
            tx_index: Transaction index in block
            block_txs: All transactions in block
            receipt: Transaction receipt

        Returns:
            True if sandwich attack detected
        """
        if not block_txs or len(block_txs) < 3:
            return False

        # Find current transaction in list
        current_idx = None
        for idx, tx in enumerate(block_txs):
            if tx.get("transactionIndex") == tx_index:
                current_idx = idx
                break

        if current_idx is None or current_idx == 0 or current_idx >= len(block_txs) - 1:
            return False

        prev_tx = block_txs[current_idx - 1]
        next_tx = block_txs[current_idx + 1]

        # Check if same sender (MEV bot)
        if prev_tx.get("from", "").lower() == next_tx.get("from", "").lower():
            # Check if it's a known MEV bot
            sender = prev_tx.get("from", "").lower()
            if sender in KNOWN_MEV_BOTS:
                return True

            # Check if both transactions have higher gas price
            victim_gas = receipt.get("effectiveGasPrice", 0)
            if (prev_tx.get("gasPrice", 0) >= victim_gas and
                next_tx.get("gasPrice", 0) >= victim_gas):
                return True

        return False

    def _is_front_run(
        self,
        tx_index: int,
        block_txs: List[Dict[str, Any]],
        receipt: TxReceipt
    ) -> bool:
        """
        Check if transaction was front-run.

        Front-running pattern:
        - Similar transaction with higher gas before victim

        Args:
            tx_index: Transaction index in block
            block_txs: All transactions in block
            receipt: Transaction receipt

        Returns:
            True if front-running detected
        """
        if not block_txs or tx_index == 0:
            return False

        victim_gas = receipt.get("effectiveGasPrice", 0)
        victim_to = receipt.get("to", "").lower()

        # Check previous transactions (up to 5 transactions before)
        # Need to find transactions in the block_txs list
        for prev_tx in block_txs:
            prev_idx = prev_tx.get("transactionIndex", -1)

            # Only check transactions before the victim
            if prev_idx >= tx_index or prev_idx < max(0, tx_index - 5):
                continue

            # Same target contract with higher gas
            if (prev_tx.get("to", "").lower() == victim_to and
                prev_tx.get("gasPrice", 0) > victim_gas):
                return True

        return False

    def _is_back_run(
        self,
        tx_index: int,
        block_txs: List[Dict[str, Any]],
        receipt: TxReceipt
    ) -> bool:
        """
        Check if transaction was back-run.

        Back-running pattern:
        - Immediate follow-up transaction capturing arbitrage

        Args:
            tx_index: Transaction index in block
            block_txs: All transactions in block
            receipt: Transaction receipt

        Returns:
            True if back-running detected
        """
        if not block_txs or len(block_txs) <= 1:
            return False

        # Find current transaction in list
        current_idx = None
        for idx, tx in enumerate(block_txs):
            if tx.get("transactionIndex") == tx_index:
                current_idx = idx
                break

        if current_idx is None or current_idx >= len(block_txs) - 1:
            return False

        next_tx = block_txs[current_idx + 1]

        # Check if next transaction is to same DEX/pool
        victim_to = receipt.get("to", "").lower()
        if next_tx.get("to", "").lower() == victim_to:
            # Check if higher gas (bot trying to execute first)
            if next_tx.get("gasPrice", 0) > receipt.get("effectiveGasPrice", 0):
                return True

        return False

    def _estimate_sandwich_damage(
        self,
        tx_hash: str,
        tx_index: int,
        block_txs: List[Dict[str, Any]]
    ) -> Optional[Decimal]:
        """
        Estimate sandwich attack damage.

        Args:
            tx_hash: Transaction hash
            tx_index: Transaction index
            block_txs: All transactions in block

        Returns:
            Estimated damage in ETH
        """
        # TODO: Implement damage estimation by analyzing price impact
        # Would require:
        # 1. Parse intended swap amounts from victim tx
        # 2. Parse MEV bot's front-run amounts
        # 3. Calculate price impact difference
        # 4. Estimate value lost

        return None

    def _estimate_frontrun_damage(
        self,
        tx_hash: str,
        tx_index: int,
        block_txs: List[Dict[str, Any]]
    ) -> Optional[Decimal]:
        """
        Estimate front-running damage.

        Args:
            tx_hash: Transaction hash
            tx_index: Transaction index
            block_txs: All transactions in block

        Returns:
            Estimated damage in ETH
        """
        # TODO: Implement front-run damage estimation
        return None

    def _extract_intended_swap(
        self,
        receipt: TxReceipt
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract intended swap details from failed transaction.

        Args:
            receipt: Transaction receipt

        Returns:
            Tuple of (dex_name, token_in, token_out)
        """
        # TODO: Implement swap detail extraction from input data
        # Would require ABI decoding of the transaction input

        return None, None, None

    def categorize_failure_reason(self, revert_reason: Optional[str]) -> str:
        """
        Categorize failure reason into buckets.

        Args:
            revert_reason: Revert reason string

        Returns:
            Failure category
        """
        if not revert_reason:
            return "unknown"

        revert_lower = revert_reason.lower()

        # Common failure patterns (order matters - check specific before general)
        if "balance" in revert_lower:
            return "insufficient_balance"
        elif "allowance" in revert_lower or "approval" in revert_lower:
            return "insufficient_allowance"
        elif "liquidity" in revert_lower:
            return "insufficient_liquidity"
        elif "slippage" in revert_lower or "insufficient" in revert_lower:
            return "slippage"
        elif "expired" in revert_lower or "deadline" in revert_lower:
            return "deadline_expired"
        elif "reentrancy" in revert_lower:
            return "reentrancy_guard"
        else:
            return "other"

    def calculate_mev_statistics(
        self,
        failed_transactions: List[FailedTransaction]
    ) -> Dict[str, Any]:
        """
        Calculate MEV statistics across failed transactions.

        Args:
            failed_transactions: List of failed transactions

        Returns:
            Dictionary of MEV statistics
        """
        total_failed = len(failed_transactions)
        mev_affected = [tx for tx in failed_transactions if tx.mev_type is not None]

        mev_by_type: Dict[str, int] = {}
        total_gas_wasted = Decimal(0)
        total_mev_damage = Decimal(0)

        for tx in failed_transactions:
            total_gas_wasted += tx.gas_cost_eth

            if tx.mev_type:
                mev_by_type[tx.mev_type] = mev_by_type.get(tx.mev_type, 0) + 1

                if tx.mev_damage_eth:
                    total_mev_damage += tx.mev_damage_eth

        return {
            "total_failed": total_failed,
            "mev_affected_count": len(mev_affected),
            "mev_affected_percentage": (len(mev_affected) / total_failed * 100) if total_failed > 0 else 0,
            "mev_by_type": mev_by_type,
            "total_gas_wasted_eth": float(total_gas_wasted),
            "total_mev_damage_eth": float(total_mev_damage),
            "average_gas_cost_eth": float(total_gas_wasted / total_failed) if total_failed > 0 else 0
        }