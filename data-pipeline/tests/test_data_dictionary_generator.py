"""
Tests for data dictionary generator.
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
from services.export.data_dictionary_generator import DataDictionaryGenerator


@pytest.fixture
def mock_db_connection():
    """Create mock database connection"""
    return Mock()


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for output"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def generator(mock_db_connection, temp_output_dir):
    """Create data dictionary generator instance"""
    return DataDictionaryGenerator(mock_db_connection, temp_output_dir)


def test_generator_initialization(generator, temp_output_dir):
    """Test that generator initializes correctly"""
    assert generator.output_path == Path(temp_output_dir)
    assert generator.output_path.exists()


def test_generate_comprehensive_data_dictionary(generator, temp_output_dir):
    """Test comprehensive data dictionary generation"""
    output_file = generator.generate_comprehensive_data_dictionary()

    # Verify JSON file was created
    assert os.path.exists(output_file)
    assert output_file.endswith('.json')

    # Verify markdown file was also created
    md_file = Path(temp_output_dir) / 'data_dictionary.md'
    assert md_file.exists()

    # Load and verify JSON structure
    with open(output_file, 'r') as f:
        data_dict = json.load(f)

    assert 'metadata' in data_dict
    assert 'tables' in data_dict
    assert 'relationships' in data_dict
    assert 'business_rules' in data_dict
    assert 'usage_guidelines' in data_dict


def test_metadata_structure(generator):
    """Test metadata structure in data dictionary"""
    output_file = generator.generate_comprehensive_data_dictionary()

    with open(output_file, 'r') as f:
        data_dict = json.load(f)

    metadata = data_dict['metadata']
    assert 'generation_date' in metadata
    assert 'version' in metadata
    assert metadata['version'] == '1.0'
    assert metadata['project'] == 'Crypto Narrative Hunter'
    assert 'scope' in metadata


def test_tokens_table_definition(generator):
    """Test tokens table is properly defined"""
    output_file = generator.generate_comprehensive_data_dictionary()

    with open(output_file, 'r') as f:
        data_dict = json.load(f)

    assert 'tokens' in data_dict['tables']
    tokens_table = data_dict['tables']['tokens']

    assert tokens_table['table_name'] == 'tokens'
    assert 'description' in tokens_table
    assert tokens_table['primary_key'] == 'token_address'
    assert 'fields' in tokens_table

    # Verify key fields exist
    fields = tokens_table['fields']
    assert 'token_address' in fields
    assert 'symbol' in fields
    assert 'name' in fields
    assert 'decimals' in fields
    assert 'narrative_category' in fields
    assert 'market_cap_rank' in fields
    assert 'liquidity_tier' in fields


def test_wallets_table_definition(generator):
    """Test wallets table is properly defined"""
    output_file = generator.generate_comprehensive_data_dictionary()

    with open(output_file, 'r') as f:
        data_dict = json.load(f)

    assert 'wallets' in data_dict['tables']
    wallets_table = data_dict['tables']['wallets']

    assert wallets_table['table_name'] == 'wallets'
    assert wallets_table['primary_key'] == 'wallet_address'

    fields = wallets_table['fields']
    assert 'wallet_address' in fields
    assert 'first_seen' in fields
    assert 'last_seen' in fields
    assert 'total_transactions' in fields
    assert 'performance_score' in fields
    assert 'sophistication_tier' in fields


def test_transactions_table_definition(generator):
    """Test transactions table is properly defined"""
    output_file = generator.generate_comprehensive_data_dictionary()

    with open(output_file, 'r') as f:
        data_dict = json.load(f)

    assert 'transactions' in data_dict['tables']
    transactions_table = data_dict['tables']['transactions']

    assert transactions_table['table_name'] == 'transactions'
    assert transactions_table['primary_key'] == 'tx_hash'
    assert 'partitioning' in transactions_table

    fields = transactions_table['fields']
    assert 'tx_hash' in fields
    assert 'wallet_address' in fields
    assert 'token_address' in fields
    assert 'transaction_type' in fields
    assert 'amount' in fields
    assert 'timestamp' in fields


def test_wallet_balances_table_definition(generator):
    """Test wallet_balances table is properly defined"""
    output_file = generator.generate_comprehensive_data_dictionary()

    with open(output_file, 'r') as f:
        data_dict = json.load(f)

    assert 'wallet_balances' in data_dict['tables']
    balances_table = data_dict['tables']['wallet_balances']

    assert balances_table['table_name'] == 'wallet_balances'
    assert 'partitioning' in balances_table

    fields = balances_table['fields']
    assert 'wallet_address' in fields
    assert 'token_address' in fields
    assert 'snapshot_date' in fields
    assert 'balance' in fields
    assert 'balance_usd' in fields


def test_eth_prices_table_definition(generator):
    """Test eth_prices table is properly defined"""
    output_file = generator.generate_comprehensive_data_dictionary()

    with open(output_file, 'r') as f:
        data_dict = json.load(f)

    assert 'eth_prices' in data_dict['tables']
    prices_table = data_dict['tables']['eth_prices']

    assert prices_table['table_name'] == 'eth_prices'
    assert prices_table['primary_key'] == 'price_date'

    fields = prices_table['fields']
    assert 'price_date' in fields
    assert 'eth_price_usd' in fields
    assert 'source' in fields


def test_wallet_performance_table_definition(generator):
    """Test wallet_performance table is properly defined"""
    output_file = generator.generate_comprehensive_data_dictionary()

    with open(output_file, 'r') as f:
        data_dict = json.load(f)

    assert 'wallet_performance' in data_dict['tables']
    performance_table = data_dict['tables']['wallet_performance']

    assert performance_table['table_name'] == 'wallet_performance'

    fields = performance_table['fields']
    assert 'wallet_address' in fields
    assert 'total_return' in fields
    assert 'sharpe_ratio' in fields
    assert 'win_rate' in fields
    assert 'max_drawdown' in fields


def test_field_definitions_complete(generator):
    """Test that field definitions have required attributes"""
    output_file = generator.generate_comprehensive_data_dictionary()

    with open(output_file, 'r') as f:
        data_dict = json.load(f)

    for table_name, table_def in data_dict['tables'].items():
        for field_name, field_def in table_def['fields'].items():
            # Each field must have type and description
            assert 'type' in field_def, f"{table_name}.{field_name} missing type"
            assert 'description' in field_def, f"{table_name}.{field_name} missing description"


def test_relationships_defined(generator):
    """Test that table relationships are documented"""
    output_file = generator.generate_comprehensive_data_dictionary()

    with open(output_file, 'r') as f:
        data_dict = json.load(f)

    relationships = data_dict['relationships']

    # Check key relationships exist
    assert 'tokens_to_transactions' in relationships
    assert 'wallets_to_transactions' in relationships
    assert 'wallets_to_balances' in relationships
    assert 'tokens_to_balances' in relationships

    # Verify relationship structure
    for rel_name, rel_def in relationships.items():
        assert 'type' in rel_def
        assert 'description' in rel_def
        assert 'foreign_key' in rel_def


def test_business_rules_defined(generator):
    """Test that business rules are documented"""
    output_file = generator.generate_comprehensive_data_dictionary()

    with open(output_file, 'r') as f:
        data_dict = json.load(f)

    business_rules = data_dict['business_rules']

    # Check key rules exist
    assert 'token_filtering' in business_rules
    assert 'wallet_filtering' in business_rules
    assert 'transaction_validation' in business_rules

    # Verify rule structure
    for rule_name, rule_def in business_rules.items():
        assert 'rule' in rule_def
        assert 'enforcement' in rule_def
        assert 'rationale' in rule_def


def test_usage_guidelines_defined(generator):
    """Test that usage guidelines are documented"""
    output_file = generator.generate_comprehensive_data_dictionary()

    with open(output_file, 'r') as f:
        data_dict = json.load(f)

    usage_guidelines = data_dict['usage_guidelines']

    assert 'recommended_applications' in usage_guidelines
    assert 'not_recommended_for' in usage_guidelines
    assert 'data_quality_notes' in usage_guidelines
    assert 'join_recommendations' in usage_guidelines
    assert 'query_optimization' in usage_guidelines

    # Verify lists are not empty
    assert len(usage_guidelines['recommended_applications']) > 0
    assert len(usage_guidelines['not_recommended_for']) > 0


def test_markdown_generation(generator, temp_output_dir):
    """Test that markdown version is generated correctly"""
    generator.generate_comprehensive_data_dictionary()

    md_file = Path(temp_output_dir) / 'data_dictionary.md'
    assert md_file.exists()

    # Read markdown content
    with open(md_file, 'r') as f:
        md_content = f.read()

    # Verify key sections are present
    assert '# Crypto Narrative Hunter - Data Dictionary' in md_content
    assert '## Tables' in md_content
    assert '### tokens' in md_content
    assert '### wallets' in md_content
    assert '## Relationships' in md_content
    assert '## Business Rules' in md_content
    assert '## Usage Guidelines' in md_content


def test_field_constraints_documented(generator):
    """Test that field constraints are properly documented"""
    output_file = generator.generate_comprehensive_data_dictionary()

    with open(output_file, 'r') as f:
        data_dict = json.load(f)

    # Check tokens.token_address has constraints
    token_address_field = data_dict['tables']['tokens']['fields']['token_address']
    assert 'constraints' in token_address_field
    assert 'PRIMARY KEY' in token_address_field['constraints']
    assert 'NOT NULL' in token_address_field['constraints']

    # Check narrative_category has CHECK constraint
    narrative_field = data_dict['tables']['tokens']['fields']['narrative_category']
    assert 'constraints' in narrative_field
    check_constraints = [c for c in narrative_field['constraints'] if c.startswith('CHECK')]
    assert len(check_constraints) > 0


def test_calculation_methodologies_documented(generator):
    """Test that calculation methodologies are documented for derived fields"""
    output_file = generator.generate_comprehensive_data_dictionary()

    with open(output_file, 'r') as f:
        data_dict = json.load(f)

    # Check wallet_performance metrics have calculations
    performance_table = data_dict['tables']['wallet_performance']

    assert 'calculation' in performance_table['fields']['total_return']
    assert 'calculation' in performance_table['fields']['sharpe_ratio']
    assert 'calculation' in performance_table['fields']['win_rate']

    # Check wallets table calculated fields
    wallets_table = data_dict['tables']['wallets']
    assert 'calculation' in wallets_table['fields']['total_volume_eth']


def test_data_sources_documented(generator):
    """Test that data sources are documented for fields"""
    output_file = generator.generate_comprehensive_data_dictionary()

    with open(output_file, 'r') as f:
        data_dict = json.load(f)

    # Check tokens table has source information
    tokens_table = data_dict['tables']['tokens']
    assert 'source' in tokens_table['fields']['symbol']
    assert 'CoinGecko' in tokens_table['fields']['symbol']['source']

    # Check transactions table has source info
    transactions_table = data_dict['tables']['transactions']
    assert 'source' in transactions_table['fields']['tx_hash']