import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    # Database
    database_url: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/crypto_narratives")

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: str = os.getenv("LOG_DIR", "./logs")

    # API Keys (principle of least privilege - only load when needed)
    dune_api_key: str = os.getenv("DUNE_API_KEY", "")
    alchemy_api_key: str = os.getenv("ALCHEMY_API_KEY", "")
    etherscan_api_key: str = os.getenv("ETHERSCAN_API_KEY", "")
    coingecko_api_key: str = os.getenv("COINGECKO_API_KEY", "")

    # HTTP Settings
    http_timeout: int = int(os.getenv("HTTP_TIMEOUT", "30"))  # seconds
    tls_verify: bool = os.getenv("TLS_VERIFY", "true").lower() == "true"
    max_retries: int = int(os.getenv("MAX_RETRIES", "5"))

    # Pinned Base URLs (prevent SSRF)
    dune_base_url: str = "https://api.dune.com/api/v1"
    alchemy_base_url: str = "https://eth-mainnet.g.alchemy.com/v2"
    etherscan_base_url: str = "https://api.etherscan.io/api"
    coingecko_base_url: str = "https://api.coingecko.com/api/v3"

    def validate(self):
        """Validate configuration on startup"""
        if not self.database_url:
            raise ValueError("DATABASE_URL must be set")

        # Only validate API keys if they're needed (will be checked when services are initialized)
        return True

settings = Settings()
settings.validate()

def get_config():
    """Get configuration settings"""
    return {
        "database_url": settings.database_url,
        "dune_cache_dir": "./cache/dune",
        "log_level": settings.log_level,
        "log_dir": settings.log_dir,
        "http_timeout": settings.http_timeout,
        "max_retries": settings.max_retries
    }
