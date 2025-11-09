"""
Dune Analytics Data Extraction Module
Handles data extraction from Dune Analytics API
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import requests
from dotenv import load_dotenv
from pathlib import Path
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DuneAnalyticsExtractor:
    """Extract data from Dune Analytics API"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Dune Analytics client"""
        self.api_key = api_key or os.getenv('DUNE_API_KEY')
        if not self.api_key:
            raise ValueError("Dune API key not found. Set DUNE_API_KEY environment variable.")

        self.base_url = "https://api.dune.com/api/v1"
        self.headers = {
            "X-Dune-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

        # Data directories
        self.project_root = Path(__file__).parent.parent.parent
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Query IDs (to be updated after creating queries in Dune)
        self.query_ids = {
            "active_wallets": None,  # Update with actual query ID
            "wallet_transactions": None,
            "token_transfers": None,
            "dex_swaps": None,
            "portfolio_snapshot": None,
            "gas_patterns": None,
            "protocol_interactions": None,
            "token_categories": None
        }

    def execute_query(self, query_id: int, params: Optional[Dict] = None) -> Dict:
        """Execute a Dune query"""
        logger.info(f"Executing query {query_id}")

        endpoint = f"{self.base_url}/query/{query_id}/execute"

        payload = {}
        if params:
            payload["parameters"] = params

        response = requests.post(endpoint, headers=self.headers, json=payload)

        if response.status_code != 200:
            logger.error(f"Query execution failed: {response.text}")
            response.raise_for_status()

        execution_id = response.json()["execution_id"]
        logger.info(f"Query submitted with execution ID: {execution_id}")

        return self.get_execution_results(execution_id)

    def get_execution_results(self, execution_id: str, max_retries: int = 60) -> Dict:
        """Get results from an execution"""
        endpoint = f"{self.base_url}/execution/{execution_id}/results"

        for attempt in range(max_retries):
            response = requests.get(endpoint, headers=self.headers)

            if response.status_code == 200:
                data = response.json()
                if data.get("state") == "QUERY_STATE_COMPLETED":
                    logger.info(f"Query completed successfully")
                    return data
                elif data.get("state") == "QUERY_STATE_FAILED":
                    logger.error(f"Query failed: {data}")
                    raise Exception(f"Query execution failed")

            logger.info(f"Query still running... (attempt {attempt + 1}/{max_retries})")
            time.sleep(5)  # Wait 5 seconds before retrying

        raise TimeoutError(f"Query execution timed out after {max_retries} attempts")

    def save_results_to_csv(self, results: Dict, filename: str) -> pd.DataFrame:
        """Save query results to CSV"""
        if "result" not in results or "rows" not in results["result"]:
            logger.error("No data found in results")
            return pd.DataFrame()

        df = pd.DataFrame(results["result"]["rows"])

        filepath = self.raw_data_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")

        return df

    def get_active_wallets(self, limit: int = 10000) -> pd.DataFrame:
        """Get top active wallets from Dune"""
        logger.info(f"Fetching top {limit} active wallets")

        # For now, we'll create a sample query
        # In production, this would use the actual query ID
        query_sql = """
        WITH wallet_activity AS (
            SELECT
                "from" as wallet_address,
                COUNT(*) as tx_count,
                SUM(CAST(value AS DOUBLE) / 1e18) as total_eth_moved,
                COUNT(DISTINCT DATE_TRUNC('day', block_time)) as active_days
            FROM ethereum.transactions
            WHERE block_time >= CURRENT_DATE - INTERVAL '6' MONTH
                AND "from" != '0x0000000000000000000000000000000000000000'
            GROUP BY "from"
            HAVING COUNT(*) >= 10
        )
        SELECT * FROM wallet_activity
        ORDER BY tx_count DESC
        LIMIT """ + str(limit)

        # TODO: Replace with actual query execution
        # results = self.execute_query(self.query_ids["active_wallets"])

        logger.info(f"Query prepared for {limit} wallets")
        return pd.DataFrame()  # Placeholder

    def extract_wallet_transactions(self, wallet_addresses: List[str]) -> pd.DataFrame:
        """Extract transactions for specified wallets"""
        logger.info(f"Extracting transactions for {len(wallet_addresses)} wallets")

        # Convert wallet list to parameter format
        params = {
            "wallet_addresses": wallet_addresses
        }

        # TODO: Execute actual query
        # results = self.execute_query(self.query_ids["wallet_transactions"], params)

        return pd.DataFrame()  # Placeholder

    def extract_token_transfers(self, wallet_addresses: List[str]) -> pd.DataFrame:
        """Extract ERC20 token transfers for wallets"""
        logger.info(f"Extracting token transfers for {len(wallet_addresses)} wallets")

        params = {
            "wallet_addresses": wallet_addresses
        }

        # TODO: Execute actual query
        # results = self.execute_query(self.query_ids["token_transfers"], params)

        return pd.DataFrame()  # Placeholder

    def run_full_extraction(self):
        """Run the complete data extraction pipeline"""
        logger.info("Starting full data extraction pipeline")

        try:
            # Step 1: Get active wallets
            logger.info("Step 1: Fetching active wallets")
            wallets_df = self.get_active_wallets(limit=10000)

            if not wallets_df.empty:
                wallet_list = wallets_df['wallet_address'].tolist()[:1000]  # Start with 1000

                # Step 2: Get transactions
                logger.info("Step 2: Extracting wallet transactions")
                transactions_df = self.extract_wallet_transactions(wallet_list)

                # Step 3: Get token transfers
                logger.info("Step 3: Extracting token transfers")
                transfers_df = self.extract_token_transfers(wallet_list)

                logger.info("Data extraction completed successfully")

                return {
                    "wallets": wallets_df,
                    "transactions": transactions_df,
                    "transfers": transfers_df
                }
            else:
                logger.warning("No active wallets found")
                return {}

        except Exception as e:
            logger.error(f"Data extraction failed: {str(e)}")
            raise

def main():
    """Main execution function"""
    try:
        # Initialize extractor
        extractor = DuneAnalyticsExtractor()

        # Run extraction
        results = extractor.run_full_extraction()

        logger.info("Data extraction complete!")

        # Print summary
        for key, df in results.items():
            if not df.empty:
                logger.info(f"{key}: {len(df)} records extracted")

    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()