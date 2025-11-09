"""
Data dictionary and schema documentation generator.

Generates comprehensive documentation for all database tables, fields,
relationships, and business rules for the Crypto Narrative Hunter project.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataDictionaryGenerator:
    """Generate comprehensive data dictionary and schema documentation"""

    def __init__(self, db_connection, output_path: str):
        """
        Initialize data dictionary generator.

        Args:
            db_connection: Database connection object
            output_path: Directory for documentation output
        """
        self.db = db_connection
        self.output_path = Path(output_path)
        self.logger = logging.getLogger(__name__)

        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

    def generate_comprehensive_data_dictionary(self) -> str:
        """Generate complete data dictionary for all datasets"""

        data_dictionary = {
            'metadata': {
                'generation_date': datetime.utcnow().isoformat(),
                'version': '1.0',
                'project': 'Crypto Narrative Hunter',
                'scope': 'Complete dataset documentation',
                'database': 'PostgreSQL',
                'maintainer': 'Data Engineering Team'
            },
            'tables': {}
        }

        # Define all tables
        table_definitions = [
            self._define_tokens_table(),
            self._define_wallets_table(),
            self._define_transactions_table(),
            self._define_wallet_balances_table(),
            self._define_eth_prices_table(),
            self._define_wallet_performance_table()
        ]

        for table_def in table_definitions:
            data_dictionary['tables'][table_def['table_name']] = table_def

        # Add relationships
        data_dictionary['relationships'] = self._define_relationships()

        # Add business rules
        data_dictionary['business_rules'] = self._define_business_rules()

        # Add usage guidelines
        data_dictionary['usage_guidelines'] = self._define_usage_guidelines()

        # Write data dictionary
        output_file = self.output_path / 'data_dictionary.json'
        with open(output_file, 'w') as f:
            json.dump(data_dictionary, f, indent=2, default=str)

        self.logger.info(f"Data dictionary generated: {output_file}")

        # Also create markdown version for readability
        self._generate_markdown_dictionary(data_dictionary)

        return str(output_file)

    def _define_tokens_table(self) -> Dict[str, Any]:
        """Define tokens table documentation"""
        return {
            'table_name': 'tokens',
            'description': 'ERC-20 token metadata with narrative categorization and liquidity analysis',
            'primary_key': 'token_address',
            'record_count_estimate': '500',
            'update_frequency': 'Static snapshot',
            'fields': {
                'token_address': {
                    'type': 'VARCHAR(42)',
                    'description': 'Ethereum contract address (checksummed)',
                    'constraints': ['PRIMARY KEY', 'NOT NULL', 'UNIQUE'],
                    'example': '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984',
                    'source': 'CoinGecko API',
                    'validation': 'Ethereum address format validation (0x + 40 hex characters)'
                },
                'symbol': {
                    'type': 'VARCHAR(10)',
                    'description': 'Token trading symbol',
                    'constraints': ['NOT NULL'],
                    'example': 'UNI',
                    'source': 'CoinGecko API',
                    'notes': 'May not be unique across tokens'
                },
                'name': {
                    'type': 'VARCHAR(100)',
                    'description': 'Full token name',
                    'constraints': ['NOT NULL'],
                    'example': 'Uniswap',
                    'source': 'CoinGecko API'
                },
                'decimals': {
                    'type': 'INTEGER',
                    'description': 'Token decimal places for amount calculations',
                    'constraints': ['NOT NULL', 'CHECK (decimals BETWEEN 0 AND 18)'],
                    'example': '18',
                    'source': 'On-chain contract call',
                    'calculation': 'ERC-20 decimals() function'
                },
                'narrative_category': {
                    'type': 'VARCHAR(50)',
                    'description': 'Token narrative classification',
                    'constraints': ['CHECK (narrative_category IN (\'DeFi\', \'Gaming\', \'AI\', \'Infrastructure\', \'Meme\', \'Stablecoin\', \'Other\'))'],
                    'example': 'DeFi',
                    'source': 'Rule-based classification + manual review',
                    'methodology': 'Keyword matching on token name/description, protocol analysis, manual validation'
                },
                'market_cap_rank': {
                    'type': 'INTEGER',
                    'description': 'Market capitalization ranking from CoinGecko',
                    'constraints': ['CHECK (market_cap_rank > 0)'],
                    'example': '15',
                    'source': 'CoinGecko API',
                    'update_frequency': 'Static snapshot at collection time'
                },
                'avg_daily_volume_usd': {
                    'type': 'DECIMAL(20,2)',
                    'description': '24-hour average trading volume in USD',
                    'example': '125000000.50',
                    'source': 'CoinGecko API',
                    'calculation': 'Rolling 24-hour average at collection time'
                },
                'liquidity_tier': {
                    'type': 'VARCHAR(10)',
                    'description': 'Liquidity tier based on DEX TVL analysis',
                    'constraints': ['CHECK (liquidity_tier IN (\'Tier 1\', \'Tier 2\', \'Tier 3\', \'Untiered\'))'],
                    'example': 'Tier 1',
                    'source': 'Dune Analytics + Uniswap/Curve API',
                    'methodology': 'Tier 1: >$10M TVL, Tier 2: $1M-$10M, Tier 3: <$1M'
                },
                'created_at': {
                    'type': 'TIMESTAMP',
                    'description': 'Record creation timestamp',
                    'constraints': ['NOT NULL', 'DEFAULT CURRENT_TIMESTAMP'],
                    'example': '2024-01-15T10:30:00Z'
                },
                'updated_at': {
                    'type': 'TIMESTAMP',
                    'description': 'Record last update timestamp',
                    'example': '2024-01-15T10:30:00Z'
                }
            }
        }

    def _define_wallets_table(self) -> Dict[str, Any]:
        """Define wallets table documentation"""
        return {
            'table_name': 'wallets',
            'description': 'Smart money wallet addresses with performance metrics and risk profiles',
            'primary_key': 'wallet_address',
            'record_count_estimate': '10000',
            'update_frequency': 'Static cohort after filtering',
            'fields': {
                'wallet_address': {
                    'type': 'VARCHAR(42)',
                    'description': 'Ethereum wallet address',
                    'constraints': ['PRIMARY KEY', 'NOT NULL', 'UNIQUE'],
                    'example': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
                    'source': 'Dune Analytics + on-chain discovery',
                    'validation': 'Ethereum address format + checksum validation'
                },
                'first_seen': {
                    'type': 'TIMESTAMP',
                    'description': 'First transaction timestamp in analysis period',
                    'constraints': ['NOT NULL'],
                    'source': 'On-chain transaction data'
                },
                'last_seen': {
                    'type': 'TIMESTAMP',
                    'description': 'Last transaction timestamp in analysis period',
                    'constraints': ['NOT NULL'],
                    'source': 'On-chain transaction data'
                },
                'total_transactions': {
                    'type': 'INTEGER',
                    'description': 'Total number of transactions in analysis period',
                    'constraints': ['NOT NULL', 'CHECK (total_transactions >= 0)'],
                    'source': 'Calculated from transactions table'
                },
                'total_volume_eth': {
                    'type': 'DECIMAL(30,18)',
                    'description': 'Total trading volume in ETH equivalent',
                    'constraints': ['CHECK (total_volume_eth >= 0)'],
                    'unit': 'ETH',
                    'calculation': 'Sum of all transaction amounts converted to ETH'
                },
                'unique_tokens': {
                    'type': 'INTEGER',
                    'description': 'Number of unique tokens traded',
                    'constraints': ['CHECK (unique_tokens >= 0)'],
                    'calculation': 'COUNT(DISTINCT token_address) from transactions'
                },
                'avg_profit_per_trade': {
                    'type': 'DECIMAL(20,8)',
                    'description': 'Average profit per trade in ETH',
                    'unit': 'ETH',
                    'calculation': 'Total profit / total trades'
                },
                'total_gas_spent': {
                    'type': 'DECIMAL(30,18)',
                    'description': 'Total gas fees paid in ETH',
                    'unit': 'ETH',
                    'calculation': 'Sum of (gas_used * gas_price) for all transactions'
                },
                'performance_score': {
                    'type': 'DECIMAL(5,4)',
                    'description': 'Composite performance score (0-1)',
                    'constraints': ['CHECK (performance_score BETWEEN 0 AND 1)'],
                    'calculation': 'Weighted combination of return, Sharpe ratio, win rate'
                },
                'risk_score': {
                    'type': 'DECIMAL(5,4)',
                    'description': 'Risk assessment score (0-1, higher = riskier)',
                    'constraints': ['CHECK (risk_score BETWEEN 0 AND 1)'],
                    'calculation': 'Based on volatility, max drawdown, portfolio concentration'
                },
                'sophistication_tier': {
                    'type': 'VARCHAR(20)',
                    'description': 'Trading sophistication classification',
                    'constraints': ['CHECK (sophistication_tier IN (\'Expert\', \'Advanced\', \'Intermediate\', \'Developing\'))'],
                    'methodology': 'Expert: >95th percentile, Advanced: 80-95th, Intermediate: 60-80th'
                },
                'created_at': {
                    'type': 'TIMESTAMP',
                    'description': 'Record creation timestamp',
                    'constraints': ['NOT NULL', 'DEFAULT CURRENT_TIMESTAMP']
                }
            }
        }

    def _define_transactions_table(self) -> Dict[str, Any]:
        """Define transactions table documentation"""
        return {
            'table_name': 'transactions',
            'description': 'DEX trading transactions for smart money wallets',
            'primary_key': 'tx_hash',
            'record_count_estimate': '1000000+',
            'partitioning': 'Monthly partitions by timestamp',
            'fields': {
                'tx_hash': {
                    'type': 'VARCHAR(66)',
                    'description': 'Ethereum transaction hash',
                    'constraints': ['PRIMARY KEY', 'NOT NULL', 'UNIQUE'],
                    'example': '0x123abc...',
                    'source': 'On-chain transaction logs'
                },
                'wallet_address': {
                    'type': 'VARCHAR(42)',
                    'description': 'Wallet that executed the transaction',
                    'constraints': ['NOT NULL', 'FOREIGN KEY REFERENCES wallets(wallet_address)'],
                    'indexed': True
                },
                'token_address': {
                    'type': 'VARCHAR(42)',
                    'description': 'Token being traded',
                    'constraints': ['NOT NULL', 'FOREIGN KEY REFERENCES tokens(token_address)'],
                    'indexed': True
                },
                'transaction_type': {
                    'type': 'VARCHAR(10)',
                    'description': 'Transaction direction',
                    'constraints': ['NOT NULL', 'CHECK (transaction_type IN (\'buy\', \'sell\', \'swap\'))'],
                    'source': 'Decoded from transaction logs'
                },
                'amount': {
                    'type': 'DECIMAL(30,18)',
                    'description': 'Token amount (in token\'s native decimals)',
                    'constraints': ['NOT NULL', 'CHECK (amount > 0)'],
                    'source': 'Transaction log data'
                },
                'amount_usd': {
                    'type': 'DECIMAL(20,2)',
                    'description': 'USD value at transaction time',
                    'unit': 'USD',
                    'calculation': 'amount * token_price_usd at timestamp'
                },
                'timestamp': {
                    'type': 'TIMESTAMP',
                    'description': 'Block timestamp',
                    'constraints': ['NOT NULL'],
                    'indexed': True,
                    'source': 'Block data'
                },
                'block_number': {
                    'type': 'BIGINT',
                    'description': 'Ethereum block number',
                    'constraints': ['NOT NULL'],
                    'source': 'Block data'
                },
                'gas_used': {
                    'type': 'INTEGER',
                    'description': 'Gas units consumed',
                    'unit': 'gas units',
                    'source': 'Transaction receipt'
                },
                'gas_price': {
                    'type': 'BIGINT',
                    'description': 'Gas price in wei',
                    'unit': 'wei',
                    'source': 'Transaction receipt'
                },
                'dex_protocol': {
                    'type': 'VARCHAR(50)',
                    'description': 'DEX protocol used',
                    'example': 'Uniswap V3, Curve, Balancer',
                    'source': 'Protocol identification from contract address'
                },
                'slippage': {
                    'type': 'DECIMAL(10,6)',
                    'description': 'Estimated slippage percentage',
                    'calculation': 'Difference between expected and actual price'
                },
                'created_at': {
                    'type': 'TIMESTAMP',
                    'description': 'Record creation timestamp',
                    'constraints': ['NOT NULL', 'DEFAULT CURRENT_TIMESTAMP']
                }
            }
        }

    def _define_wallet_balances_table(self) -> Dict[str, Any]:
        """Define wallet_balances table documentation"""
        return {
            'table_name': 'wallet_balances',
            'description': 'Daily balance snapshots for wallet portfolios',
            'primary_key': 'wallet_address, token_address, snapshot_date',
            'record_count_estimate': '500000+',
            'partitioning': 'Monthly partitions by snapshot_date',
            'fields': {
                'wallet_address': {
                    'type': 'VARCHAR(42)',
                    'description': 'Wallet address',
                    'constraints': ['NOT NULL', 'FOREIGN KEY REFERENCES wallets(wallet_address)'],
                    'indexed': True
                },
                'token_address': {
                    'type': 'VARCHAR(42)',
                    'description': 'Token address',
                    'constraints': ['NOT NULL', 'FOREIGN KEY REFERENCES tokens(token_address)'],
                    'indexed': True
                },
                'snapshot_date': {
                    'type': 'DATE',
                    'description': 'Balance snapshot date',
                    'constraints': ['NOT NULL'],
                    'indexed': True,
                    'frequency': 'Daily snapshots'
                },
                'balance': {
                    'type': 'DECIMAL(30,18)',
                    'description': 'Token balance (in token decimals)',
                    'constraints': ['NOT NULL', 'CHECK (balance >= 0)'],
                    'source': 'On-chain balance queries'
                },
                'balance_usd': {
                    'type': 'DECIMAL(20,2)',
                    'description': 'USD value of balance',
                    'unit': 'USD',
                    'calculation': 'balance * price_at_snapshot'
                },
                'price_at_snapshot': {
                    'type': 'DECIMAL(20,8)',
                    'description': 'Token price in USD at snapshot time',
                    'unit': 'USD',
                    'source': 'Price feeds (Chainlink, DEX oracles)'
                },
                'created_at': {
                    'type': 'TIMESTAMP',
                    'description': 'Record creation timestamp',
                    'constraints': ['NOT NULL', 'DEFAULT CURRENT_TIMESTAMP']
                }
            }
        }

    def _define_eth_prices_table(self) -> Dict[str, Any]:
        """Define eth_prices table documentation"""
        return {
            'table_name': 'eth_prices',
            'description': 'Daily ETH/USD price data for conversions',
            'primary_key': 'price_date',
            'record_count_estimate': '180-365',
            'update_frequency': 'Daily',
            'fields': {
                'price_date': {
                    'type': 'DATE',
                    'description': 'Price date',
                    'constraints': ['PRIMARY KEY', 'NOT NULL', 'UNIQUE']
                },
                'eth_price_usd': {
                    'type': 'DECIMAL(15,2)',
                    'description': 'ETH price in USD',
                    'constraints': ['NOT NULL', 'CHECK (eth_price_usd > 0)'],
                    'unit': 'USD',
                    'source': 'Chainlink ETH/USD oracle'
                },
                'source': {
                    'type': 'VARCHAR(50)',
                    'description': 'Price data source',
                    'example': 'Chainlink, CoinGecko',
                    'source': 'API or oracle identifier'
                },
                'created_at': {
                    'type': 'TIMESTAMP',
                    'description': 'Record creation timestamp',
                    'constraints': ['NOT NULL', 'DEFAULT CURRENT_TIMESTAMP']
                }
            }
        }

    def _define_wallet_performance_table(self) -> Dict[str, Any]:
        """Define wallet_performance table documentation"""
        return {
            'table_name': 'wallet_performance',
            'description': 'Calculated performance metrics for wallets',
            'primary_key': 'wallet_address, calculation_date',
            'record_count_estimate': '10000+',
            'update_frequency': 'Calculated at analysis time',
            'fields': {
                'wallet_address': {
                    'type': 'VARCHAR(42)',
                    'description': 'Wallet address',
                    'constraints': ['NOT NULL', 'FOREIGN KEY REFERENCES wallets(wallet_address)'],
                    'indexed': True
                },
                'calculation_date': {
                    'type': 'DATE',
                    'description': 'Performance calculation date',
                    'constraints': ['NOT NULL'],
                    'indexed': True
                },
                'total_return': {
                    'type': 'DECIMAL(10,6)',
                    'description': 'Total return percentage',
                    'calculation': '(End Value - Start Value) / Start Value',
                    'interpretation': '0.25 = 25% return'
                },
                'sharpe_ratio': {
                    'type': 'DECIMAL(10,4)',
                    'description': 'Risk-adjusted return',
                    'calculation': '(Return - Risk Free Rate) / Volatility',
                    'interpretation': '>1.0 excellent, >0.5 good'
                },
                'win_rate': {
                    'type': 'DECIMAL(5,4)',
                    'description': 'Percentage of profitable trades',
                    'constraints': ['CHECK (win_rate BETWEEN 0 AND 1)'],
                    'calculation': 'Profitable Trades / Total Trades'
                },
                'max_drawdown': {
                    'type': 'DECIMAL(10,6)',
                    'description': 'Maximum peak-to-trough loss',
                    'calculation': 'MIN((Valley Value - Peak Value) / Peak Value)',
                    'interpretation': 'Always negative or zero'
                },
                'volatility': {
                    'type': 'DECIMAL(10,6)',
                    'description': 'Standard deviation of returns',
                    'calculation': 'STDDEV(daily_returns)'
                },
                'total_trades': {
                    'type': 'INTEGER',
                    'description': 'Number of trades in period',
                    'constraints': ['CHECK (total_trades >= 0)']
                },
                'avg_position_size': {
                    'type': 'DECIMAL(10,6)',
                    'description': 'Average position size as fraction of portfolio',
                    'calculation': 'AVG(position_value / total_portfolio_value)'
                },
                'portfolio_concentration': {
                    'type': 'DECIMAL(10,6)',
                    'description': 'Herfindahl index of portfolio concentration',
                    'calculation': 'SUM(position_weight^2)',
                    'interpretation': 'Closer to 1 = more concentrated'
                },
                'created_at': {
                    'type': 'TIMESTAMP',
                    'description': 'Record creation timestamp',
                    'constraints': ['NOT NULL', 'DEFAULT CURRENT_TIMESTAMP']
                }
            }
        }

    def _define_relationships(self) -> Dict[str, Any]:
        """Define table relationships"""
        return {
            'tokens_to_transactions': {
                'type': 'one-to-many',
                'description': 'One token can appear in many transactions',
                'foreign_key': 'transactions.token_address -> tokens.token_address',
                'cardinality': '1:N'
            },
            'wallets_to_transactions': {
                'type': 'one-to-many',
                'description': 'One wallet can have many transactions',
                'foreign_key': 'transactions.wallet_address -> wallets.wallet_address',
                'cardinality': '1:N'
            },
            'wallets_to_balances': {
                'type': 'one-to-many',
                'description': 'One wallet has many balance snapshots',
                'foreign_key': 'wallet_balances.wallet_address -> wallets.wallet_address',
                'cardinality': '1:N'
            },
            'tokens_to_balances': {
                'type': 'one-to-many',
                'description': 'One token appears in many balance records',
                'foreign_key': 'wallet_balances.token_address -> tokens.token_address',
                'cardinality': '1:N'
            },
            'wallets_to_performance': {
                'type': 'one-to-many',
                'description': 'One wallet has performance metrics over time',
                'foreign_key': 'wallet_performance.wallet_address -> wallets.wallet_address',
                'cardinality': '1:N'
            }
        }

    def _define_business_rules(self) -> Dict[str, Any]:
        """Define business rules and constraints"""
        return {
            'token_filtering': {
                'rule': 'Only tokens with market cap rank < 500 and sufficient liquidity',
                'enforcement': 'Applied during token collection phase',
                'rationale': 'Focus on tradeable, liquid tokens'
            },
            'wallet_filtering': {
                'rule': 'Wallets must have >= 20 transactions and Sharpe ratio > 0.5',
                'enforcement': 'Applied during cohort filtering',
                'rationale': 'Identify skilled, active traders'
            },
            'transaction_validation': {
                'rule': 'All transactions must have valid amounts, timestamps, and gas data',
                'enforcement': 'Database constraints and validation logic',
                'rationale': 'Ensure data integrity'
            },
            'balance_reconciliation': {
                'rule': 'Daily balances must reconcile with transaction history',
                'enforcement': 'Cross-validation checks in quality assurance',
                'rationale': 'Detect data inconsistencies'
            },
            'price_validation': {
                'rule': 'Prices validated against multiple sources with < 5% deviation tolerance',
                'enforcement': 'Price validation service',
                'rationale': 'Ensure price accuracy'
            }
        }

    def _define_usage_guidelines(self) -> Dict[str, Any]:
        """Define usage guidelines for data consumers"""
        return {
            'recommended_applications': [
                'Smart money behavior analysis',
                'Narrative trend identification',
                'Trading strategy development',
                'Risk assessment and portfolio construction',
                'Performance benchmarking'
            ],
            'not_recommended_for': [
                'High-frequency trading signals',
                'Real-time portfolio tracking',
                'Regulatory compliance reporting',
                'Tax calculation (not audited)'
            ],
            'data_quality_notes': [
                'Data covers 6-month observation period',
                'Limited to Ethereum mainnet DEX activity',
                'Some exotic DEXs may have incomplete coverage',
                'Bot and Sybil filtered but not 100% guaranteed'
            ],
            'join_recommendations': {
                'wallet_analysis': 'JOIN wallets, transactions, wallet_balances ON wallet_address',
                'token_analysis': 'JOIN tokens, transactions ON token_address',
                'performance_analysis': 'JOIN wallets, wallet_performance ON wallet_address'
            },
            'query_optimization': [
                'Use date range filters on partitioned tables (transactions, wallet_balances)',
                'Index on wallet_address and token_address for joins',
                'Consider using Parquet exports for analytical queries'
            ]
        }

    def _generate_markdown_dictionary(self, data_dict: Dict[str, Any]) -> str:
        """Generate human-readable markdown version of data dictionary"""

        md_lines = []
        md_lines.append("# Crypto Narrative Hunter - Data Dictionary\n")
        md_lines.append(f"**Generated:** {data_dict['metadata']['generation_date']}\n")
        md_lines.append(f"**Version:** {data_dict['metadata']['version']}\n")
        md_lines.append(f"**Project:** {data_dict['metadata']['project']}\n\n")

        md_lines.append("## Tables\n")

        for table_name, table_def in data_dict['tables'].items():
            md_lines.append(f"### {table_name}\n")
            md_lines.append(f"**Description:** {table_def['description']}\n\n")
            md_lines.append(f"**Primary Key:** {table_def['primary_key']}\n\n")
            md_lines.append(f"**Estimated Records:** {table_def['record_count_estimate']}\n\n")

            if 'partitioning' in table_def:
                md_lines.append(f"**Partitioning:** {table_def['partitioning']}\n\n")

            md_lines.append("#### Fields\n\n")
            md_lines.append("| Field | Type | Description | Constraints |\n")
            md_lines.append("|-------|------|-------------|-------------|\n")

            for field_name, field_def in table_def['fields'].items():
                constraints = ', '.join(field_def.get('constraints', []))
                md_lines.append(f"| {field_name} | {field_def['type']} | {field_def['description']} | {constraints} |\n")

            md_lines.append("\n")

        # Add relationships
        md_lines.append("## Relationships\n\n")
        for rel_name, rel_def in data_dict['relationships'].items():
            md_lines.append(f"### {rel_name}\n")
            md_lines.append(f"- **Type:** {rel_def['type']}\n")
            md_lines.append(f"- **Description:** {rel_def['description']}\n")
            md_lines.append(f"- **Foreign Key:** {rel_def['foreign_key']}\n\n")

        # Add business rules
        md_lines.append("## Business Rules\n\n")
        for rule_name, rule_def in data_dict['business_rules'].items():
            md_lines.append(f"### {rule_name}\n")
            md_lines.append(f"- **Rule:** {rule_def['rule']}\n")
            md_lines.append(f"- **Enforcement:** {rule_def['enforcement']}\n")
            md_lines.append(f"- **Rationale:** {rule_def['rationale']}\n\n")

        # Add usage guidelines
        md_lines.append("## Usage Guidelines\n\n")
        md_lines.append("### Recommended Applications\n\n")
        for app in data_dict['usage_guidelines']['recommended_applications']:
            md_lines.append(f"- {app}\n")

        md_lines.append("\n### Not Recommended For\n\n")
        for item in data_dict['usage_guidelines']['not_recommended_for']:
            md_lines.append(f"- {item}\n")

        md_content = ''.join(md_lines)

        output_file = self.output_path / 'data_dictionary.md'
        with open(output_file, 'w') as f:
            f.write(md_content)

        self.logger.info(f"Markdown data dictionary generated: {output_file}")
        return str(output_file)