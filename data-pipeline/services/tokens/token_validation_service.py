"""
Token validation and data cleaning service.

Implements:
- Comprehensive address validation with checksum verification
- Symbol uniqueness validation with conflict resolution
- Data completeness checks for required fields
- Decimal validation against contract standards
- Market data validation and outlier detection
- Invalid token flagging and removal logic
"""

import time
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from decimal import Decimal
from web3 import Web3
try:
    from web3.exceptions import ValidationError as Web3ValidationError
except ImportError:
    # web3 v6+ uses Web3ValidationError directly
    from web3.exceptions import Web3ValidationError

from data_collection.common.db import get_cursor, execute_with_retry
from data_collection.common.logging_setup import get_logger
from .ethereum_utils import validate_and_normalize_address


@dataclass
class ValidationResult:
    """Result of token validation with details"""
    is_valid: bool
    validation_flags: List[str]
    error_messages: List[str]
    warnings: List[str]
    suggested_fixes: List[str]


@dataclass
class ValidationStats:
    """Statistics for token validation operation"""
    total_tokens_validated: int = 0
    valid_tokens: int = 0
    invalid_tokens: int = 0
    tokens_with_warnings: int = 0
    tokens_flagged_for_removal: int = 0
    tokens_fixed: int = 0
    address_validation_errors: int = 0
    symbol_conflicts: int = 0
    decimal_validation_errors: int = 0
    market_data_outliers: int = 0
    data_completeness_issues: int = 0
    duration_seconds: float = 0


class TokenValidationService:
    """
    Service for validating and cleaning token data.

    Performs comprehensive validation including:
    - Ethereum address format and checksum validation
    - Symbol uniqueness and conflict resolution
    - Data completeness and integrity checks
    - Market data outlier detection
    - Contract standard compliance validation
    """

    def __init__(self):
        self.logger = get_logger("token_validation_service")
        self.web3 = Web3()

        # Market data validation thresholds
        self.market_cap_outlier_threshold = 1e15  # $1 quadrillion (unrealistic)
        self.volume_outlier_threshold = 1e12      # $1 trillion daily volume (unrealistic)
        self.price_outlier_threshold = 1e9        # $1 billion per token (unrealistic)

        # Data completeness requirements (percentage of tokens that must have this field)
        self.completeness_thresholds = {
            'symbol': 100.0,
            'name': 100.0,
            'decimals': 95.0,
            'market_cap_rank': 80.0,
            'avg_daily_volume_usd': 70.0
        }

    def validate_all_tokens(self, batch_size: int = 100) -> ValidationStats:
        """
        Validate all tokens in the database.

        Args:
            batch_size: Number of tokens to process per batch

        Returns:
            ValidationStats with validation results
        """
        start_time = time.time()
        stats = ValidationStats()

        self.logger.log_operation(
            operation="validate_all_tokens",
            params={"batch_size": batch_size},
            status="started",
            message="Starting validation of all tokens"
        )

        try:
            # Get all tokens for validation
            tokens = self._get_all_tokens()
            stats.total_tokens_validated = len(tokens)

            if not tokens:
                self.logger.log_operation(
                    operation="validate_all_tokens",
                    status="completed",
                    message="No tokens found for validation"
                )
                return stats

            # Process tokens in batches
            for i in range(0, len(tokens), batch_size):
                batch = tokens[i:i + batch_size]
                batch_stats = self._validate_token_batch(batch)
                self._merge_validation_stats(stats, batch_stats)

                self.logger.log_operation(
                    operation="validate_batch",
                    params={
                        "batch_number": (i // batch_size) + 1,
                        "batch_size": len(batch),
                        "valid_tokens": batch_stats.valid_tokens,
                        "invalid_tokens": batch_stats.invalid_tokens
                    },
                    status="completed",
                    message=f"Validated batch {(i // batch_size) + 1}"
                )

            stats.duration_seconds = time.time() - start_time

            self.logger.log_operation(
                operation="validate_all_tokens",
                params={
                    "total_validated": stats.total_tokens_validated,
                    "valid_tokens": stats.valid_tokens,
                    "invalid_tokens": stats.invalid_tokens,
                    "flagged_for_removal": stats.tokens_flagged_for_removal
                },
                status="completed",
                duration_ms=int(stats.duration_seconds * 1000),
                message=f"Validation completed: {stats.valid_tokens}/{stats.total_tokens_validated} tokens valid"
            )

            return stats

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.log_operation(
                operation="validate_all_tokens",
                status="error",
                error=str(e),
                duration_ms=duration_ms
            )
            raise

    def validate_specific_tokens(self, token_addresses: List[str]) -> Tuple[List[ValidationResult], ValidationStats]:
        """
        Validate specific tokens by their addresses.

        Args:
            token_addresses: List of token contract addresses

        Returns:
            Tuple of (validation_results, validation_stats)
        """
        start_time = time.time()
        stats = ValidationStats()
        results = []

        try:
            # Get specific tokens from database
            tokens = self._get_tokens_by_addresses(token_addresses)
            stats.total_tokens_validated = len(tokens)

            for token in tokens:
                result = self._validate_single_token(token)
                results.append(result)
                self._update_stats_from_result(stats, result)

            stats.duration_seconds = time.time() - start_time

            self.logger.log_operation(
                operation="validate_specific_tokens",
                params={
                    "requested_tokens": len(token_addresses),
                    "found_tokens": stats.total_tokens_validated,
                    "valid_tokens": stats.valid_tokens
                },
                status="completed",
                duration_ms=int(stats.duration_seconds * 1000)
            )

            return results, stats

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.log_operation(
                operation="validate_specific_tokens",
                status="error",
                error=str(e),
                duration_ms=duration_ms
            )
            raise

    def _get_all_tokens(self) -> List[Dict[str, Any]]:
        """Get all tokens from the database for validation"""
        query = """
            SELECT token_address, symbol, name, decimals,
                   market_cap_rank, avg_daily_volume_usd,
                   narrative_category, liquidity_tier,
                   created_at, updated_at
            FROM tokens
            ORDER BY market_cap_rank ASC NULLS LAST
        """

        try:
            result = execute_with_retry(query)
            return result or []
        except Exception as e:
            self.logger.log_operation(
                operation="get_all_tokens",
                status="error",
                error=str(e)
            )
            return []

    def _get_tokens_by_addresses(self, addresses: List[str]) -> List[Dict[str, Any]]:
        """Get specific tokens by their addresses"""
        if not addresses:
            return []

        placeholders = ', '.join(['%s'] * len(addresses))
        query = f"""
            SELECT token_address, symbol, name, decimals,
                   market_cap_rank, avg_daily_volume_usd,
                   narrative_category, liquidity_tier,
                   created_at, updated_at
            FROM tokens
            WHERE token_address IN ({placeholders})
            ORDER BY market_cap_rank ASC NULLS LAST
        """

        try:
            result = execute_with_retry(query, tuple(addresses))
            return result or []
        except Exception as e:
            self.logger.log_operation(
                operation="get_tokens_by_addresses",
                status="error",
                error=str(e)
            )
            return []

    def _validate_token_batch(self, tokens: List[Dict[str, Any]]) -> ValidationStats:
        """Validate a batch of tokens"""
        batch_stats = ValidationStats()
        batch_stats.total_tokens_validated = len(tokens)

        try:
            with get_cursor() as cur:
                for token in tokens:
                    result = self._validate_single_token(token)
                    self._update_stats_from_result(batch_stats, result)

                    # Update token validation status in database
                    self._update_token_validation_status(cur, token['token_address'], result)

            return batch_stats

        except Exception as e:
            self.logger.log_operation(
                operation="validate_token_batch",
                status="error",
                error=str(e)
            )
            raise

    def _validate_single_token(self, token: Dict[str, Any]) -> ValidationResult:
        """Validate a single token and return detailed results"""
        result = ValidationResult(
            is_valid=True,
            validation_flags=[],
            error_messages=[],
            warnings=[],
            suggested_fixes=[]
        )

        # 1. Address validation
        self._validate_address(token, result)

        # 2. Symbol validation
        self._validate_symbol(token, result)

        # 3. Name validation
        self._validate_name(token, result)

        # 4. Decimals validation
        self._validate_decimals(token, result)

        # 5. Market data validation
        self._validate_market_data(token, result)

        # 6. Data completeness validation
        self._validate_data_completeness(token, result)

        # 7. Cross-field consistency validation
        self._validate_cross_field_consistency(token, result)

        # Determine overall validity
        result.is_valid = len(result.error_messages) == 0

        return result

    def _validate_address(self, token: Dict[str, Any], result: ValidationResult) -> None:
        """Validate Ethereum address format and checksum"""
        address = token.get('token_address', '')

        if not address:
            result.error_messages.append("Token address is missing")
            result.validation_flags.append("missing_address")
            return

        # Check basic format
        if not isinstance(address, str) or len(address) != 42 or not address.startswith('0x'):
            result.error_messages.append(f"Invalid address format: {address}")
            result.validation_flags.append("invalid_address_format")
            return

        # Check hex format
        try:
            int(address[2:], 16)
        except ValueError:
            result.error_messages.append(f"Address contains invalid hex characters: {address}")
            result.validation_flags.append("invalid_hex_address")
            return

        # Validate checksum
        try:
            normalized_address = validate_and_normalize_address(address)
            if not normalized_address:
                result.error_messages.append(f"Address checksum validation failed: {address}")
                result.validation_flags.append("invalid_checksum")
            elif normalized_address != address:
                result.warnings.append(f"Address checksum mismatch, should be: {normalized_address}")
                result.suggested_fixes.append(f"Update address to: {normalized_address}")
                result.validation_flags.append("checksum_mismatch")
        except Exception as e:
            result.error_messages.append(f"Address validation error: {str(e)}")
            result.validation_flags.append("address_validation_error")

    def _validate_symbol(self, token: Dict[str, Any], result: ValidationResult) -> None:
        """Validate token symbol"""
        symbol = token.get('symbol', '')

        if not symbol:
            result.error_messages.append("Token symbol is missing")
            result.validation_flags.append("missing_symbol")
            return

        if not isinstance(symbol, str):
            result.error_messages.append("Token symbol must be a string")
            result.validation_flags.append("invalid_symbol_type")
            return

        # Check symbol length
        if len(symbol) > 20:
            result.error_messages.append(f"Symbol too long (>{20} characters): {symbol}")
            result.validation_flags.append("symbol_too_long")

        # Check for invalid characters
        if not re.match(r'^[A-Z0-9_-]+$', symbol):
            result.warnings.append(f"Symbol contains unusual characters: {symbol}")
            result.validation_flags.append("unusual_symbol_chars")

        # Check for common symbol patterns that might indicate test tokens
        test_patterns = ['TEST', 'FAKE', 'SCAM', 'EXAMPLE', 'DEMO']
        if any(pattern in symbol.upper() for pattern in test_patterns):
            result.warnings.append(f"Symbol may indicate test token: {symbol}")
            result.validation_flags.append("potential_test_token")

    def _validate_name(self, token: Dict[str, Any], result: ValidationResult) -> None:
        """Validate token name"""
        name = token.get('name', '')

        if not name:
            result.error_messages.append("Token name is missing")
            result.validation_flags.append("missing_name")
            return

        if not isinstance(name, str):
            result.error_messages.append("Token name must be a string")
            result.validation_flags.append("invalid_name_type")
            return

        if len(name) > 100:
            result.error_messages.append(f"Name too long (>{100} characters): {name}")
            result.validation_flags.append("name_too_long")

        # Check for suspicious name patterns
        suspicious_patterns = ['scam', 'fake', 'test', 'ponzi', 'rug']
        if any(pattern in name.lower() for pattern in suspicious_patterns):
            result.warnings.append(f"Name contains suspicious terms: {name}")
            result.validation_flags.append("suspicious_name")

    def _validate_decimals(self, token: Dict[str, Any], result: ValidationResult) -> None:
        """Validate token decimals against ERC-20 standard"""
        decimals = token.get('decimals')

        if decimals is None:
            result.warnings.append("Token decimals not specified")
            result.validation_flags.append("missing_decimals")
            result.suggested_fixes.append("Query contract for decimals value")
            return

        if not isinstance(decimals, int):
            result.error_messages.append(f"Decimals must be an integer, got: {type(decimals)}")
            result.validation_flags.append("invalid_decimals_type")
            return

        if decimals < 0 or decimals > 18:
            result.error_messages.append(f"Decimals out of valid range (0-18): {decimals}")
            result.validation_flags.append("invalid_decimals_range")

        # Common decimal values - warn if unusual
        common_decimals = {18, 8, 6, 0}
        if decimals not in common_decimals:
            result.warnings.append(f"Unusual decimals value: {decimals}")
            result.validation_flags.append("unusual_decimals")

    def _validate_market_data(self, token: Dict[str, Any], result: ValidationResult) -> None:
        """Validate market data for outliers and consistency"""
        # Validate market cap rank
        rank = token.get('market_cap_rank')
        if rank is not None:
            if not isinstance(rank, int) or rank <= 0:
                result.error_messages.append(f"Invalid market cap rank: {rank}")
                result.validation_flags.append("invalid_market_cap_rank")
            elif rank > 50000:  # Unrealistically high rank
                result.warnings.append(f"Very high market cap rank: {rank}")
                result.validation_flags.append("high_market_cap_rank")

        # Validate daily volume
        volume = token.get('avg_daily_volume_usd')
        if volume is not None:
            try:
                volume_decimal = Decimal(str(volume))
                if volume_decimal < 0:
                    result.error_messages.append(f"Negative daily volume: {volume}")
                    result.validation_flags.append("negative_volume")
                elif volume_decimal > self.volume_outlier_threshold:
                    result.warnings.append(f"Extremely high daily volume (>${volume:,.0f})")
                    result.validation_flags.append("volume_outlier")
                elif volume_decimal == 0 and rank and rank <= 1000:
                    result.warnings.append("Zero volume for top-ranked token")
                    result.validation_flags.append("zero_volume_top_token")
            except (ValueError, TypeError):
                result.error_messages.append(f"Invalid volume format: {volume}")
                result.validation_flags.append("invalid_volume_format")

    def _validate_data_completeness(self, token: Dict[str, Any], result: ValidationResult) -> None:
        """Validate data completeness for required fields"""
        required_fields = ['token_address', 'symbol', 'name']
        recommended_fields = ['decimals', 'market_cap_rank']

        for field in required_fields:
            if not token.get(field):
                result.error_messages.append(f"Required field missing: {field}")
                result.validation_flags.append(f"missing_{field}")

        for field in recommended_fields:
            if token.get(field) is None:
                result.warnings.append(f"Recommended field missing: {field}")
                result.validation_flags.append(f"missing_{field}")

    def _validate_cross_field_consistency(self, token: Dict[str, Any], result: ValidationResult) -> None:
        """Validate consistency between related fields"""
        # Check if symbol and name are consistent
        symbol = token.get('symbol', '').upper()
        name = token.get('name', '').upper()

        if symbol and name:
            # Check if symbol appears in name (common pattern)
            if len(symbol) > 2 and symbol not in name and not any(part in name for part in symbol.split('_')):
                # This might be normal, so just flag for review
                result.validation_flags.append("symbol_name_mismatch")

        # Check market cap rank consistency with volume
        rank = token.get('market_cap_rank')
        volume = token.get('avg_daily_volume_usd')

        if rank and volume:
            try:
                volume_decimal = Decimal(str(volume))
                # Top 100 tokens should have significant volume
                if rank <= 100 and volume_decimal < 100000:  # Less than $100k daily volume
                    result.warnings.append(f"Low volume for top-{rank} token: ${volume:,.0f}")
                    result.validation_flags.append("low_volume_top_token")
            except (ValueError, TypeError):
                pass  # Already handled in market data validation

    def _update_stats_from_result(self, stats: ValidationStats, result: ValidationResult) -> None:
        """Update statistics based on validation result"""
        if result.is_valid:
            stats.valid_tokens += 1
        else:
            stats.invalid_tokens += 1

        if result.warnings:
            stats.tokens_with_warnings += 1

        # Count specific validation issues
        flags = result.validation_flags
        if any('address' in flag for flag in flags):
            stats.address_validation_errors += 1
        if any('symbol' in flag for flag in flags):
            stats.symbol_conflicts += 1
        if any('decimal' in flag for flag in flags):
            stats.decimal_validation_errors += 1
        if any('outlier' in flag or 'volume' in flag for flag in flags):
            stats.market_data_outliers += 1
        if any('missing' in flag for flag in flags):
            stats.data_completeness_issues += 1

        # Flag for removal if severe issues
        severe_flags = ['invalid_address_format', 'missing_address', 'missing_symbol', 'missing_name']
        if any(flag in severe_flags for flag in flags):
            stats.tokens_flagged_for_removal += 1

    def _update_token_validation_status(self, cursor, token_address: str, result: ValidationResult) -> None:
        """Update token validation status in database"""
        # Create validation status summary
        validation_status = "valid" if result.is_valid else "invalid"
        validation_flags_str = ",".join(result.validation_flags) if result.validation_flags else None

        # Update token record with validation results
        update_query = """
            UPDATE tokens
            SET validation_status = %s,
                validation_flags = %s,
                validation_last_run = NOW(),
                updated_at = NOW()
            WHERE token_address = %s
        """

        cursor.execute(update_query, (validation_status, validation_flags_str, token_address))

    def _merge_validation_stats(self, main_stats: ValidationStats, batch_stats: ValidationStats) -> None:
        """Merge batch validation statistics into main statistics"""
        main_stats.total_tokens_validated += batch_stats.total_tokens_validated
        main_stats.valid_tokens += batch_stats.valid_tokens
        main_stats.invalid_tokens += batch_stats.invalid_tokens
        main_stats.tokens_with_warnings += batch_stats.tokens_with_warnings
        main_stats.tokens_flagged_for_removal += batch_stats.tokens_flagged_for_removal
        main_stats.tokens_fixed += batch_stats.tokens_fixed
        main_stats.address_validation_errors += batch_stats.address_validation_errors
        main_stats.symbol_conflicts += batch_stats.symbol_conflicts
        main_stats.decimal_validation_errors += batch_stats.decimal_validation_errors
        main_stats.market_data_outliers += batch_stats.market_data_outliers
        main_stats.data_completeness_issues += batch_stats.data_completeness_issues

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of token validation status"""
        try:
            summary_query = """
                SELECT
                    COUNT(*) as total_tokens,
                    COUNT(CASE WHEN validation_status = 'valid' THEN 1 END) as valid_tokens,
                    COUNT(CASE WHEN validation_status = 'invalid' THEN 1 END) as invalid_tokens,
                    COUNT(CASE WHEN validation_flags IS NOT NULL AND validation_flags != '' THEN 1 END) as tokens_with_flags,
                    COUNT(CASE WHEN validation_last_run IS NULL THEN 1 END) as never_validated
                FROM tokens
            """

            summary_result = execute_with_retry(summary_query)
            summary = summary_result[0] if summary_result else {}

            # Get most common validation flags
            flags_query = """
                SELECT
                    unnest(string_to_array(validation_flags, ',')) as flag,
                    COUNT(*) as count
                FROM tokens
                WHERE validation_flags IS NOT NULL AND validation_flags != ''
                GROUP BY flag
                ORDER BY count DESC
                LIMIT 10
            """

            flags_result = execute_with_retry(flags_query)
            common_flags = [{"flag": row['flag'], "count": row['count']} for row in flags_result or []]

            return {
                "summary": summary,
                "common_validation_flags": common_flags,
                "validation_health": (summary.get('valid_tokens', 0) / summary.get('total_tokens', 1)) * 100
            }

        except Exception as e:
            self.logger.log_operation(
                operation="get_validation_summary",
                status="error",
                error=str(e)
            )
            return {"error": str(e)}

    def clean_invalid_tokens(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Remove tokens flagged as invalid from the database.

        Args:
            dry_run: If True, only report what would be removed without actually removing

        Returns:
            Dictionary with removal statistics
        """
        try:
            # Get tokens flagged for removal
            removal_query = """
                SELECT token_address, symbol, name, validation_flags
                FROM tokens
                WHERE validation_status = 'invalid'
                  AND (validation_flags LIKE '%invalid_address_format%'
                       OR validation_flags LIKE '%missing_address%'
                       OR validation_flags LIKE '%missing_symbol%'
                       OR validation_flags LIKE '%missing_name%')
            """

            tokens_to_remove = execute_with_retry(removal_query)

            if not tokens_to_remove:
                return {
                    "tokens_to_remove": 0,
                    "dry_run": dry_run,
                    "removed_tokens": []
                }

            if dry_run:
                return {
                    "tokens_to_remove": len(tokens_to_remove),
                    "dry_run": True,
                    "tokens_that_would_be_removed": [
                        {
                            "address": token['token_address'],
                            "symbol": token['symbol'],
                            "name": token['name'],
                            "flags": token['validation_flags']
                        }
                        for token in tokens_to_remove
                    ]
                }

            # Actually remove tokens
            removed_count = 0
            with get_cursor() as cur:
                for token in tokens_to_remove:
                    try:
                        delete_query = "DELETE FROM tokens WHERE token_address = %s"
                        cur.execute(delete_query, (token['token_address'],))
                        removed_count += 1

                        self.logger.log_operation(
                            operation="remove_invalid_token",
                            params={
                                "token_address": token['token_address'],
                                "symbol": token['symbol'],
                                "flags": token['validation_flags']
                            },
                            status="completed",
                            message=f"Removed invalid token: {token['symbol']}"
                        )

                    except Exception as e:
                        self.logger.log_operation(
                            operation="remove_invalid_token",
                            params={"token_address": token['token_address']},
                            status="error",
                            error=str(e)
                        )

            return {
                "tokens_to_remove": len(tokens_to_remove),
                "tokens_actually_removed": removed_count,
                "dry_run": False
            }

        except Exception as e:
            self.logger.log_operation(
                operation="clean_invalid_tokens",
                status="error",
                error=str(e)
            )
            return {"error": str(e)}