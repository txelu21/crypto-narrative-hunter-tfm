"""
Error handling and edge case management module for wallet performance analysis.

This module provides comprehensive error handling, data validation, and edge case
management for all wallet analysis components. Includes graceful degradation,
data quality checks, and robust error recovery mechanisms.
"""

import logging
import traceback
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import pandas as pd
import numpy as np
from decimal import Decimal, InvalidOperation
from functools import wraps
import warnings

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class InsufficientDataError(Exception):
    """Custom exception for insufficient data scenarios."""
    pass


class CalculationError(Exception):
    """Custom exception for calculation-related errors."""
    pass


class ErrorHandler:
    """
    Comprehensive error handling and validation for wallet analysis.

    This class provides methods for data validation, error recovery,
    edge case handling, and graceful degradation when encountering
    problematic data or calculation scenarios.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the ErrorHandler.

        Args:
            strict_mode: If True, raises exceptions for validation errors.
                        If False, logs warnings and attempts graceful degradation.
        """
        self.strict_mode = strict_mode
        self.validation_results = {}
        self.error_count = 0

    def validate_wallet_address(self, address: str) -> bool:
        """
        Validate Ethereum wallet address format.

        Args:
            address: Wallet address to validate

        Returns:
            True if valid, False otherwise

        Raises:
            ValidationError: If address is invalid and strict_mode is True
        """
        if not address:
            error_msg = "Wallet address cannot be empty"
            return self._handle_validation_error("wallet_address", error_msg)

        if not isinstance(address, str):
            error_msg = f"Wallet address must be string, got {type(address)}"
            return self._handle_validation_error("wallet_address", error_msg)

        # Remove '0x' prefix if present
        clean_address = address.lower()
        if clean_address.startswith('0x'):
            clean_address = clean_address[2:]

        # Check length (should be 40 characters)
        if len(clean_address) != 40:
            error_msg = f"Invalid address length: {len(clean_address)}, expected 40"
            return self._handle_validation_error("wallet_address", error_msg)

        # Check if all characters are hexadecimal
        try:
            int(clean_address, 16)
        except ValueError:
            error_msg = f"Address contains invalid characters: {address}"
            return self._handle_validation_error("wallet_address", error_msg)

        return True

    def validate_transaction_data(self, transactions: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate and sanitize transaction data.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Tuple of (is_valid, sanitized_transactions)
        """
        if not transactions:
            if self.strict_mode:
                raise InsufficientDataError("No transaction data provided")
            logger.warning("No transaction data provided")
            return False, []

        sanitized_transactions = []
        invalid_count = 0

        required_fields = ['timestamp', 'token_symbol', 'type', 'amount']

        for i, txn in enumerate(transactions):
            try:
                # Check required fields
                for field in required_fields:
                    if field not in txn or txn[field] is None:
                        raise ValidationError(f"Missing required field: {field}")

                # Validate and sanitize transaction data
                sanitized_txn = self._sanitize_transaction(txn)
                sanitized_transactions.append(sanitized_txn)

            except Exception as e:
                invalid_count += 1
                logger.warning(f"Invalid transaction at index {i}: {e}")

                if self.strict_mode:
                    raise ValidationError(f"Transaction validation failed at index {i}: {e}")

        if invalid_count > 0:
            logger.warning(f"Removed {invalid_count} invalid transactions out of {len(transactions)}")

        if not sanitized_transactions:
            error_msg = "No valid transactions after sanitization"
            if self.strict_mode:
                raise InsufficientDataError(error_msg)
            logger.error(error_msg)
            return False, []

        return True, sanitized_transactions

    def _sanitize_transaction(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize individual transaction data.

        Args:
            txn: Transaction dictionary

        Returns:
            Sanitized transaction dictionary
        """
        sanitized = {}

        # Timestamp validation
        if isinstance(txn['timestamp'], str):
            try:
                sanitized['timestamp'] = pd.to_datetime(txn['timestamp'])
            except:
                raise ValidationError(f"Invalid timestamp format: {txn['timestamp']}")
        elif isinstance(txn['timestamp'], (datetime, pd.Timestamp)):
            sanitized['timestamp'] = txn['timestamp']
        else:
            raise ValidationError(f"Invalid timestamp type: {type(txn['timestamp'])}")

        # Token symbol validation
        token_symbol = str(txn['token_symbol']).upper().strip()
        if not token_symbol or len(token_symbol) > 20:
            raise ValidationError(f"Invalid token symbol: {txn['token_symbol']}")
        sanitized['token_symbol'] = token_symbol

        # Transaction type validation
        txn_type = str(txn['type']).lower().strip()
        if txn_type not in ['buy', 'sell', 'transfer_in', 'transfer_out']:
            raise ValidationError(f"Invalid transaction type: {txn['type']}")
        sanitized['type'] = txn_type

        # Amount validation
        try:
            amount = float(txn['amount'])
            if amount <= 0:
                raise ValidationError(f"Amount must be positive: {amount}")
            if amount > 1e18:  # Reasonable upper bound
                raise ValidationError(f"Amount too large: {amount}")
            sanitized['amount'] = amount
        except (ValueError, TypeError):
            raise ValidationError(f"Invalid amount: {txn['amount']}")

        # Optional price validation
        if 'price' in txn and txn['price'] is not None:
            try:
                price = float(txn['price'])
                if price <= 0:
                    logger.warning(f"Non-positive price: {price}, setting to None")
                    sanitized['price'] = None
                elif price > 1e12:  # Reasonable upper bound
                    logger.warning(f"Extremely high price: {price}, capping")
                    sanitized['price'] = 1e12
                else:
                    sanitized['price'] = price
            except (ValueError, TypeError):
                logger.warning(f"Invalid price: {txn['price']}, setting to None")
                sanitized['price'] = None

        # Copy other fields
        for key, value in txn.items():
            if key not in sanitized:
                sanitized[key] = value

        return sanitized

    def validate_performance_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize performance metrics.

        Args:
            metrics: Dictionary of performance metrics

        Returns:
            Sanitized metrics dictionary
        """
        sanitized = {}

        # Define validation rules for each metric
        validation_rules = {
            'total_return': {'min': -0.99, 'max': 100.0, 'type': float},
            'win_rate': {'min': 0.0, 'max': 1.0, 'type': float},
            'sharpe_ratio': {'min': -10.0, 'max': 10.0, 'type': float},
            'volatility': {'min': 0.0, 'max': 10.0, 'type': float},
            'max_drawdown': {'min': -1.0, 'max': 0.0, 'type': float},
            'total_trades': {'min': 0, 'max': 1000000, 'type': int},
            'diversification_score': {'min': 0.0, 'max': 100.0, 'type': float}
        }

        for metric_name, value in metrics.items():
            try:
                if metric_name in validation_rules:
                    sanitized[metric_name] = self._validate_metric(
                        metric_name, value, validation_rules[metric_name]
                    )
                else:
                    # Pass through unknown metrics with basic sanitization
                    sanitized[metric_name] = self._sanitize_numeric_value(value)

            except Exception as e:
                logger.warning(f"Error validating metric {metric_name}: {e}")
                if not self.strict_mode:
                    sanitized[metric_name] = 0.0  # Default value
                else:
                    raise

        return sanitized

    def _validate_metric(self, name: str, value: Any, rules: Dict[str, Any]) -> Union[int, float]:
        """
        Validate individual metric against rules.

        Args:
            name: Metric name
            value: Metric value
            rules: Validation rules

        Returns:
            Validated and sanitized value
        """
        if value is None:
            if self.strict_mode:
                raise ValidationError(f"Metric {name} cannot be None")
            return 0.0

        # Type conversion
        try:
            if rules['type'] == int:
                converted_value = int(value)
            else:
                converted_value = float(value)
        except (ValueError, TypeError):
            error_msg = f"Cannot convert {name} to {rules['type'].__name__}: {value}"
            if self.strict_mode:
                raise ValidationError(error_msg)
            logger.warning(error_msg)
            return 0.0

        # Handle NaN and infinity
        if isinstance(converted_value, float):
            if np.isnan(converted_value) or np.isinf(converted_value):
                error_msg = f"Invalid value for {name}: {converted_value}"
                if self.strict_mode:
                    raise ValidationError(error_msg)
                logger.warning(error_msg)
                return 0.0

        # Range validation
        if converted_value < rules['min']:
            error_msg = f"{name} below minimum: {converted_value} < {rules['min']}"
            if self.strict_mode:
                raise ValidationError(error_msg)
            logger.warning(f"{error_msg}, clamping to minimum")
            return rules['min']

        if converted_value > rules['max']:
            error_msg = f"{name} above maximum: {converted_value} > {rules['max']}"
            if self.strict_mode:
                raise ValidationError(error_msg)
            logger.warning(f"{error_msg}, clamping to maximum")
            return rules['max']

        return converted_value

    def _sanitize_numeric_value(self, value: Any) -> float:
        """
        Sanitize a numeric value with basic error handling.

        Args:
            value: Value to sanitize

        Returns:
            Sanitized numeric value
        """
        if value is None:
            return 0.0

        try:
            numeric_value = float(value)
            if np.isnan(numeric_value) or np.isinf(numeric_value):
                return 0.0
            return numeric_value
        except (ValueError, TypeError):
            return 0.0

    def handle_insufficient_data(
        self,
        data_type: str,
        minimum_required: int,
        actual_count: int,
        fallback_value: Any = None
    ) -> Any:
        """
        Handle insufficient data scenarios.

        Args:
            data_type: Type of data (e.g., "transactions", "returns")
            minimum_required: Minimum required data points
            actual_count: Actual number of data points
            fallback_value: Value to return if insufficient data

        Returns:
            Fallback value or raises exception in strict mode
        """
        error_msg = f"Insufficient {data_type}: {actual_count} < {minimum_required} required"

        if self.strict_mode:
            raise InsufficientDataError(error_msg)

        logger.warning(error_msg)
        return fallback_value

    def safe_division(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Perform safe division with zero-division handling.

        Args:
            numerator: Numerator value
            denominator: Denominator value
            default: Default value if division by zero

        Returns:
            Division result or default value
        """
        try:
            if denominator == 0:
                if self.strict_mode:
                    raise CalculationError("Division by zero")
                return default

            result = numerator / denominator

            if np.isnan(result) or np.isinf(result):
                if self.strict_mode:
                    raise CalculationError(f"Invalid division result: {result}")
                return default

            return result

        except Exception as e:
            if self.strict_mode:
                raise CalculationError(f"Division error: {e}")
            logger.warning(f"Division error: {e}, returning default: {default}")
            return default

    def safe_calculation(self, func: Callable, *args, default: Any = None, **kwargs) -> Any:
        """
        Execute a calculation function with error handling.

        Args:
            func: Function to execute
            *args: Function arguments
            default: Default value if function fails
            **kwargs: Function keyword arguments

        Returns:
            Function result or default value
        """
        try:
            result = func(*args, **kwargs)

            # Check for invalid results
            if isinstance(result, (int, float)):
                if np.isnan(result) or np.isinf(result):
                    raise CalculationError(f"Invalid calculation result: {result}")

            return result

        except Exception as e:
            error_msg = f"Calculation error in {func.__name__}: {e}"

            if self.strict_mode:
                raise CalculationError(error_msg)

            logger.warning(error_msg)
            return default

    def validate_price_data(self, price_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Validate and clean price data series.

        Args:
            price_data: Dictionary mapping symbols to price series

        Returns:
            Validated price data dictionary
        """
        validated_data = {}

        for symbol, prices in price_data.items():
            try:
                if not isinstance(prices, pd.Series):
                    logger.warning(f"Price data for {symbol} is not a pandas Series")
                    continue

                if len(prices) == 0:
                    logger.warning(f"Empty price data for {symbol}")
                    continue

                # Remove NaN and infinite values
                clean_prices = prices.replace([np.inf, -np.inf], np.nan).dropna()

                if len(clean_prices) == 0:
                    logger.warning(f"No valid price data for {symbol} after cleaning")
                    continue

                # Check for reasonable price values
                if (clean_prices <= 0).any():
                    logger.warning(f"Non-positive prices found for {symbol}, removing")
                    clean_prices = clean_prices[clean_prices > 0]

                if len(clean_prices) == 0:
                    continue

                # Check for extreme price movements (more than 1000x change)
                price_ratios = clean_prices / clean_prices.shift(1)
                extreme_moves = (price_ratios > 1000) | (price_ratios < 0.001)

                if extreme_moves.any():
                    logger.warning(f"Extreme price movements detected for {symbol}")
                    if not self.strict_mode:
                        # Remove extreme outliers
                        clean_prices = clean_prices[~extreme_moves.fillna(False)]

                if len(clean_prices) >= 2:  # Need at least 2 points
                    validated_data[symbol] = clean_prices

            except Exception as e:
                error_msg = f"Error validating price data for {symbol}: {e}"
                if self.strict_mode:
                    raise ValidationError(error_msg)
                logger.warning(error_msg)

        return validated_data

    def handle_edge_cases(self, data: Any, context: str) -> Any:
        """
        Generic edge case handler for various data scenarios.

        Args:
            data: Data to check for edge cases
            context: Context description for logging

        Returns:
            Processed data with edge cases handled
        """
        if data is None:
            logger.warning(f"None data encountered in {context}")
            return None

        if isinstance(data, (list, pd.Series, pd.DataFrame)) and len(data) == 0:
            logger.warning(f"Empty data encountered in {context}")
            return data

        if isinstance(data, (int, float)):
            if np.isnan(data) or np.isinf(data):
                logger.warning(f"Invalid numeric value in {context}: {data}")
                return 0.0

        return data

    def _handle_validation_error(self, field: str, message: str) -> bool:
        """
        Handle validation error based on strict mode setting.

        Args:
            field: Field name that failed validation
            message: Error message

        Returns:
            False if not in strict mode, raises exception if in strict mode
        """
        self.validation_results[field] = {'valid': False, 'error': message}
        self.error_count += 1

        if self.strict_mode:
            raise ValidationError(message)

        logger.warning(f"Validation warning for {field}: {message}")
        return False

    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of validation results.

        Returns:
            Dictionary containing validation summary
        """
        return {
            'total_errors': self.error_count,
            'validation_results': self.validation_results,
            'strict_mode': self.strict_mode
        }


def error_recovery_decorator(default_return=None, log_errors=True):
    """
    Decorator for automatic error recovery and logging.

    Args:
        default_return: Default value to return on error
        log_errors: Whether to log errors

    Returns:
        Decorated function with error handling
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                return default_return
        return wrapper
    return decorator


def validate_input_data(required_fields: List[str], optional_fields: List[str] = None):
    """
    Decorator for input data validation.

    Args:
        required_fields: List of required field names
        optional_fields: List of optional field names

    Returns:
        Decorated function with input validation
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Assuming first argument is data dictionary
            if args and isinstance(args[0], dict):
                data = args[0]

                # Check required fields
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    raise ValidationError(f"Missing required fields: {missing_fields}")

                # Log optional missing fields
                if optional_fields:
                    missing_optional = [field for field in optional_fields if field not in data]
                    if missing_optional:
                        logger.info(f"Missing optional fields: {missing_optional}")

            return func(*args, **kwargs)
        return wrapper
    return decorator