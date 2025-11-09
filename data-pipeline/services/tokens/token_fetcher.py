"""
Token metadata fetcher with validation and deduplication.

Implements:
- Pagination for CoinGecko markets API filtered by "category=ethereum-ecosystem"
- Contract address lookup API integration for ERC-20 validation
- Ethereum address validation with checksum verification
- Deduplication logic prioritizing higher market cap when multiple symbols exist
- Market cap ranking preservation and validation
- Edge case handling: missing decimals, non-ETH chains, deprecated tokens
- Data validation using Pydantic models
"""

import time
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

from data_collection.common.logging_setup import get_logger
from .coingecko_client import CoinGeckoClient, CoinGeckoAPIError
from .ethereum_utils import validate_and_normalize_address


class TokenMetadata(BaseModel):
    """Pydantic model for validated token metadata"""

    token_address: str = Field(..., description="Normalized Ethereum contract address")
    symbol: str = Field(..., max_length=10, description="Token symbol")
    name: str = Field(..., max_length=100, description="Token name")
    decimals: Optional[int] = Field(None, ge=0, le=18, description="Token decimals")
    market_cap_rank: Optional[int] = Field(None, gt=0, description="CoinGecko market cap rank")
    market_cap_usd: Optional[Decimal] = Field(None, ge=0, description="Market cap in USD")
    volume_24h_usd: Optional[Decimal] = Field(None, ge=0, description="24h trading volume in USD")
    current_price_usd: Optional[Decimal] = Field(None, ge=0, description="Current price in USD")

    # Validation metadata
    coingecko_id: str = Field(..., description="CoinGecko coin ID")
    has_ethereum_contract: bool = Field(default=False, description="Has valid Ethereum contract")
    validation_flags: List[str] = Field(default_factory=list, description="Validation warnings/flags")

    @field_validator('token_address')
    @classmethod
    def validate_ethereum_address(cls, v):
        normalized = validate_and_normalize_address(v)
        if not normalized:
            raise ValueError(f"Invalid Ethereum address: {v}")
        return normalized

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()


@dataclass
class FetchStats:
    """Statistics for token fetching operation"""
    total_pages_fetched: int = 0
    total_tokens_found: int = 0
    ethereum_tokens_found: int = 0
    tokens_with_contracts: int = 0
    duplicates_found: int = 0
    invalid_addresses: int = 0
    missing_decimals: int = 0
    validation_errors: int = 0
    start_time: float = 0
    end_time: float = 0

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time if self.end_time > self.start_time else 0


class TokenFetcher:
    """
    Fetches and validates token metadata from CoinGecko with deduplication.
    """

    def __init__(self, client: CoinGeckoClient):
        self.client = client
        self.logger = get_logger("token_fetcher")

    def fetch_top_ethereum_tokens(self, target_count: int = 500) -> Tuple[List[TokenMetadata], FetchStats]:
        """
        Fetch top Ethereum tokens with complete metadata and validation.

        Args:
            target_count: Target number of tokens to fetch (default: 500)

        Returns:
            Tuple of (validated_tokens, fetch_statistics)
        """
        stats = FetchStats()
        stats.start_time = time.time()

        self.logger.log_operation(
            operation="fetch_ethereum_tokens",
            params={"target_count": target_count},
            status="started",
            message=f"Starting fetch of top {target_count} Ethereum tokens"
        )

        # Track tokens and handle deduplication
        tokens_by_address: Dict[str, TokenMetadata] = {}
        seen_symbols: Dict[str, List[TokenMetadata]] = {}
        all_raw_tokens = []

        try:
            # Fetch tokens with pagination
            page = 1
            per_page = 100  # CoinGecko max is 250, but 100 is more reliable
            total_fetched = 0

            # Calculate estimated pages needed (with buffer for filtering)
            estimated_pages = max(10, (target_count // per_page) + 5)

            with tqdm(total=estimated_pages, desc="Fetching token pages") as pbar:
                while total_fetched < target_count * 2:  # Fetch more to account for filtering
                    try:
                        self.logger.log_operation(
                            operation="fetch_page",
                            params={"page": page, "per_page": per_page},
                            status="started"
                        )

                        # Fetch page from CoinGecko
                        tokens_page = self.client.get_coins_markets(
                            per_page=per_page,
                            page=page,
                            category="ethereum-ecosystem"
                        )

                        if not tokens_page:
                            self.logger.log_operation(
                                operation="fetch_page",
                                params={"page": page},
                                status="completed",
                                message="No more tokens found, stopping pagination"
                            )
                            break

                        all_raw_tokens.extend(tokens_page)
                        total_fetched += len(tokens_page)
                        stats.total_pages_fetched = page
                        stats.total_tokens_found = total_fetched

                        self.logger.log_operation(
                            operation="fetch_page",
                            params={"page": page, "tokens_count": len(tokens_page)},
                            status="completed",
                            message=f"Fetched {len(tokens_page)} tokens from page {page}"
                        )

                        page += 1
                        pbar.update(1)

                        # Break if we got fewer tokens than requested (last page)
                        if len(tokens_page) < per_page:
                            self.logger.log_operation(
                                operation="fetch_pagination",
                                status="completed",
                                message="Reached last page of results"
                            )
                            break

                    except CoinGeckoAPIError as e:
                        self.logger.log_operation(
                            operation="fetch_page",
                            params={"page": page},
                            status="error",
                            error=str(e),
                            message=f"Failed to fetch page {page}"
                        )
                        # Stop pagination on error
                        break

            # Process and validate tokens
            self.logger.log_operation(
                operation="process_tokens",
                params={"raw_tokens_count": len(all_raw_tokens)},
                status="started",
                message="Processing and validating token metadata"
            )

            with tqdm(all_raw_tokens, desc="Processing tokens") as pbar:
                for raw_token in pbar:
                    try:
                        processed_token = self._process_raw_token(raw_token, stats)
                        if processed_token:
                            self._handle_token_deduplication(
                                processed_token, tokens_by_address, seen_symbols, stats
                            )

                        pbar.set_postfix({
                            'ethereum': stats.ethereum_tokens_found,
                            'contracts': stats.tokens_with_contracts,
                            'duplicates': stats.duplicates_found
                        })

                    except Exception as e:
                        stats.validation_errors += 1
                        self.logger.log_operation(
                            operation="process_token",
                            params={"token_id": raw_token.get("id", "unknown")},
                            status="error",
                            error=str(e),
                            message="Failed to process token"
                        )

            # Sort by market cap rank and take top N
            validated_tokens = sorted(
                tokens_by_address.values(),
                key=lambda t: t.market_cap_rank or 999999
            )[:target_count]

            stats.end_time = time.time()

            self.logger.log_operation(
                operation="fetch_ethereum_tokens",
                params={
                    "target_count": target_count,
                    "actual_count": len(validated_tokens),
                    "pages_fetched": stats.total_pages_fetched,
                    "duplicates_found": stats.duplicates_found
                },
                status="completed",
                duration_ms=int(stats.duration_seconds * 1000),
                message=f"Completed token fetch: {len(validated_tokens)} validated tokens"
            )

            return validated_tokens, stats

        except Exception as e:
            stats.end_time = time.time()
            self.logger.log_operation(
                operation="fetch_ethereum_tokens",
                status="error",
                error=str(e),
                duration_ms=int(stats.duration_seconds * 1000),
                message="Token fetching failed"
            )
            raise

    def _process_raw_token(self, raw_token: Dict, stats: FetchStats) -> Optional[TokenMetadata]:
        """Process raw token data from CoinGecko into validated TokenMetadata"""

        # Extract basic fields
        coingecko_id = raw_token.get("id")
        symbol = raw_token.get("symbol", "").upper()
        name = raw_token.get("name", "")
        market_cap_rank = raw_token.get("market_cap_rank")
        current_price = raw_token.get("current_price")
        market_cap = raw_token.get("market_cap")
        volume_24h = raw_token.get("total_volume")

        if not all([coingecko_id, symbol, name]):
            return None

        # Since platforms field is not available in markets API, we need to try
        # the contract lookup to see if this token has an Ethereum contract
        ethereum_address = None
        decimals = None
        validation_flags = []

        try:
            # Attempt to get coin details by ID to check for Ethereum contract
            # This uses a different endpoint: /coins/{id}
            coin_details = self.client.get_coin_details_by_id(coingecko_id)

            # Extract Ethereum contract address from platforms
            platforms = coin_details.get("platforms", {})
            ethereum_address = platforms.get("ethereum")

            if not ethereum_address:
                # Skip tokens without Ethereum contracts
                return None

            stats.ethereum_tokens_found += 1

            # Validate and normalize address
            normalized_address = validate_and_normalize_address(ethereum_address)
            if not normalized_address:
                stats.invalid_addresses += 1
                return None

            ethereum_address = normalized_address

            # Extract decimals from coin details
            detail_platforms = coin_details.get("detail_platforms", {})
            ethereum_details = detail_platforms.get("ethereum", {})
            decimals = ethereum_details.get("decimal_place")

            if decimals is None:
                stats.missing_decimals += 1
                validation_flags.append("missing_decimals")
                # Use common default for ERC-20 tokens
                decimals = 18

            stats.tokens_with_contracts += 1

        except CoinGeckoAPIError as e:
            self.logger.log_operation(
                operation="get_coin_details",
                params={"coin_id": coingecko_id},
                status="error",
                error=str(e),
                message="Failed to get coin details"
            )
            # Skip tokens we can't look up
            return None

        # Handle edge cases
        if market_cap_rank and market_cap_rank > 10000:
            validation_flags.append("low_market_cap_rank")

        if current_price and current_price < 0.000001:
            validation_flags.append("very_low_price")

        # Create validated token metadata
        try:
            token_metadata = TokenMetadata(
                token_address=ethereum_address,
                symbol=symbol,
                name=name,
                decimals=decimals,
                market_cap_rank=market_cap_rank,
                market_cap_usd=Decimal(str(market_cap)) if market_cap else None,
                volume_24h_usd=Decimal(str(volume_24h)) if volume_24h else None,
                current_price_usd=Decimal(str(current_price)) if current_price else None,
                coingecko_id=coingecko_id,
                has_ethereum_contract=True,
                validation_flags=validation_flags
            )

            return token_metadata

        except Exception as e:
            self.logger.log_operation(
                operation="validate_token_metadata",
                params={"symbol": symbol, "address": ethereum_address[:10] + "..."},
                status="error",
                error=str(e),
                message="Token metadata validation failed"
            )
            return None

    def _handle_token_deduplication(self,
                                  token: TokenMetadata,
                                  tokens_by_address: Dict[str, TokenMetadata],
                                  seen_symbols: Dict[str, List[TokenMetadata]],
                                  stats: FetchStats) -> None:
        """Handle token deduplication with priority for higher market cap"""

        # Check for address duplicates
        if token.token_address in tokens_by_address:
            existing_token = tokens_by_address[token.token_address]

            # Keep token with better market cap rank (lower number = better)
            existing_rank = existing_token.market_cap_rank or 999999
            new_rank = token.market_cap_rank or 999999

            if new_rank < existing_rank:
                self.logger.log_operation(
                    operation="resolve_address_duplicate",
                    params={
                        "address": token.token_address[:10] + "...",
                        "existing_symbol": existing_token.symbol,
                        "new_symbol": token.symbol,
                        "existing_rank": existing_rank,
                        "new_rank": new_rank
                    },
                    status="completed",
                    message=f"Replacing {existing_token.symbol} with {token.symbol} (better rank)"
                )
                tokens_by_address[token.token_address] = token
                stats.duplicates_found += 1
            else:
                stats.duplicates_found += 1
            return

        # Check for symbol duplicates
        if token.symbol in seen_symbols:
            seen_symbols[token.symbol].append(token)
            stats.duplicates_found += 1

            self.logger.log_operation(
                operation="symbol_duplicate_found",
                params={
                    "symbol": token.symbol,
                    "addresses": [t.token_address[:10] + "..." for t in seen_symbols[token.symbol]]
                },
                status="info",
                message=f"Multiple tokens found for symbol {token.symbol}"
            )
        else:
            seen_symbols[token.symbol] = [token]

        # Add to address map
        tokens_by_address[token.token_address] = token