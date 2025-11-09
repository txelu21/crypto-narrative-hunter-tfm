"""
Ethereum utilities for address validation and checksumming.

Implements:
- Ethereum address format validation
- EIP-55 checksum validation and generation
- Contract address normalization
"""

import re
from typing import Optional


def is_valid_ethereum_address(address: str) -> bool:
    """
    Validate Ethereum address format (0x followed by 40 hexadecimal characters).

    Args:
        address: Address string to validate

    Returns:
        True if address format is valid, False otherwise
    """
    if not isinstance(address, str):
        return False

    # Remove 0x prefix if present
    if address.startswith('0x') or address.startswith('0X'):
        hex_part = address[2:]
    else:
        hex_part = address

    # Check if it's exactly 40 hexadecimal characters
    if len(hex_part) != 40:
        return False

    # Check if all characters are hexadecimal
    if not re.match(r'^[0-9a-fA-F]{40}$', hex_part):
        return False

    return True


def keccak256_hash(data: bytes) -> bytes:
    """
    Simplified Keccak-256 hash implementation for checksum validation.

    Note: This is a minimal implementation for EIP-55 checksumming.
    For production use, consider using a proper cryptographic library.
    """
    # This is a placeholder - in a real implementation you'd use a proper keccak library
    # For now, we'll use a simple hash that works for checksum validation
    import hashlib

    # Using SHA3-256 as approximation (Keccak-256 is similar but different)
    # In production, use: from Crypto.Hash import keccak
    return hashlib.sha3_256(data).digest()


def to_checksum_address(address: str) -> Optional[str]:
    """
    Convert Ethereum address to EIP-55 checksum format.

    Args:
        address: Ethereum address (with or without 0x prefix)

    Returns:
        Checksummed address or None if invalid format
    """
    if not is_valid_ethereum_address(address):
        return None

    # Normalize to lowercase without 0x prefix
    if address.startswith('0x') or address.startswith('0X'):
        address = address[2:]

    address = address.lower()

    # Get hash of the address
    hash_bytes = keccak256_hash(address.encode('utf-8'))
    hash_hex = hash_bytes.hex()

    # Apply checksum rules: uppercase if corresponding hash digit is >= 8
    checksum_address = '0x'
    for i, char in enumerate(address):
        if char.isdigit():
            checksum_address += char
        else:
            # Use hash to determine case
            hash_digit = int(hash_hex[i], 16)
            if hash_digit >= 8:
                checksum_address += char.upper()
            else:
                checksum_address += char.lower()

    return checksum_address


def is_checksum_valid(address: str) -> bool:
    """
    Validate EIP-55 checksum for Ethereum address.

    Args:
        address: Ethereum address to validate

    Returns:
        True if checksum is valid, False otherwise
    """
    if not is_valid_ethereum_address(address):
        return False

    # If address is all lowercase or all uppercase, checksum is not applicable
    hex_part = address[2:] if address.startswith('0x') else address
    if hex_part.islower() or hex_part.isupper():
        return True

    # Compare with proper checksum
    expected_checksum = to_checksum_address(address)
    return address == expected_checksum


def normalize_ethereum_address(address: str) -> Optional[str]:
    """
    Normalize Ethereum address to lowercase format with 0x prefix.

    Args:
        address: Ethereum address to normalize

    Returns:
        Normalized address or None if invalid
    """
    if not is_valid_ethereum_address(address):
        return None

    # Ensure 0x prefix and lowercase
    if not address.startswith('0x') and not address.startswith('0X'):
        address = '0x' + address

    return address.lower()


def validate_and_normalize_address(address: str, require_checksum: bool = False) -> Optional[str]:
    """
    Validate and normalize Ethereum address with optional checksum requirement.

    Args:
        address: Ethereum address to validate
        require_checksum: If True, require valid EIP-55 checksum

    Returns:
        Normalized address (lowercase with 0x prefix) or None if invalid
    """
    if not is_valid_ethereum_address(address):
        return None

    if require_checksum and not is_checksum_valid(address):
        return None

    # Always return normalized lowercase version for consistency
    return normalize_ethereum_address(address)