"""
Tests for comprehensive documentation generator.
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from services.export.documentation_generator import ProjectDocumentationGenerator


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for output"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def doc_generator(temp_output_dir):
    """Create documentation generator instance"""
    return ProjectDocumentationGenerator(temp_output_dir)


def test_generator_initialization(doc_generator, temp_output_dir):
    """Test that generator initializes correctly"""
    assert doc_generator.output_path == Path(temp_output_dir)
    assert doc_generator.output_path.exists()
    assert doc_generator.generation_timestamp is not None


def test_generate_methodology_documentation(doc_generator):
    """Test methodology documentation generation"""
    json_file = doc_generator.generate_methodology_documentation()

    # Verify JSON file was created
    assert os.path.exists(json_file)
    assert json_file.endswith('.json')

    # Verify markdown file was also created
    md_file = doc_generator.output_path / 'methodology_documentation.md'
    assert md_file.exists()

    # Load and verify JSON structure
    with open(json_file, 'r') as f:
        doc = json.load(f)

    assert 'metadata' in doc
    assert 'executive_summary' in doc
    assert 'phase_1_token_universe' in doc
    assert 'phase_2_wallet_identification' in doc
    assert 'phase_3_transaction_analysis' in doc
    assert 'technical_architecture' in doc
    assert 'quality_assurance' in doc


def test_methodology_doc_structure(doc_generator):
    """Test methodology documentation has all required sections"""
    json_file = doc_generator.generate_methodology_documentation()

    with open(json_file, 'r') as f:
        doc = json.load(f)

    # Verify executive summary
    assert 'project_goal' in doc['executive_summary']
    assert 'methodology_overview' in doc['executive_summary']
    assert 'key_metrics' in doc['executive_summary']

    # Verify phase 1
    phase1 = doc['phase_1_token_universe']
    assert 'objective' in phase1
    assert 'data_sources' in phase1
    assert 'filtering_criteria' in phase1
    assert 'narrative_classification' in phase1

    # Verify technical architecture
    arch = doc['technical_architecture']
    assert 'database' in arch
    assert 'api_integration' in arch
    assert 'caching_strategy' in arch


def test_generate_data_lineage_documentation(doc_generator):
    """Test data lineage documentation generation"""
    json_file = doc_generator.generate_data_lineage_documentation()

    assert os.path.exists(json_file)

    # Verify markdown version
    md_file = doc_generator.output_path / 'data_lineage_documentation.md'
    assert md_file.exists()

    # Load and verify JSON structure
    with open(json_file, 'r') as f:
        doc = json.load(f)

    assert 'metadata' in doc
    assert 'source_systems' in doc
    assert 'data_flows' in doc
    assert 'transformation_logic' in doc
    assert 'checkpoint_and_recovery' in doc


def test_lineage_source_systems(doc_generator):
    """Test lineage documentation includes all source systems"""
    json_file = doc_generator.generate_data_lineage_documentation()

    with open(json_file, 'r') as f:
        doc = json.load(f)

    source_systems = doc['source_systems']
    assert 'coingecko_api' in source_systems
    assert 'dune_analytics' in source_systems
    assert 'alchemy_rpc' in source_systems
    assert 'chainlink_oracles' in source_systems

    # Verify each source has required fields
    for system_name, system_info in source_systems.items():
        assert 'description' in system_info
        assert 'data_extracted' in system_info
        assert 'reliability' in system_info


def test_lineage_data_flows(doc_generator):
    """Test lineage documentation includes data flows"""
    json_file = doc_generator.generate_data_lineage_documentation()

    with open(json_file, 'r') as f:
        doc = json.load(f)

    data_flows = doc['data_flows']
    assert 'flow_1_token_collection' in data_flows
    assert 'flow_2_wallet_identification' in data_flows
    assert 'flow_3_transaction_collection' in data_flows
    assert 'flow_4_balance_snapshots' in data_flows

    # Verify flow structure
    for flow_name, flow_info in data_flows.items():
        assert 'description' in flow_info
        assert 'steps' in flow_info
        assert 'transformations' in flow_info
        assert 'quality_gates' in flow_info


def test_generate_quality_certification_report(doc_generator):
    """Test quality certification report generation"""
    json_file = doc_generator.generate_quality_certification_report()

    assert os.path.exists(json_file)

    # Verify markdown version
    md_file = doc_generator.output_path / 'quality_certification_report.md'
    assert md_file.exists()

    # Load and verify JSON structure
    with open(json_file, 'r') as f:
        doc = json.load(f)

    assert 'certification_metadata' in doc
    assert 'executive_certification' in doc
    assert 'quality_metrics' in doc
    assert 'fitness_assessment' in doc
    assert 'limitations_and_caveats' in doc
    assert 'certification_decision' in doc


def test_certification_executive_summary(doc_generator):
    """Test certification report has executive summary"""
    json_file = doc_generator.generate_quality_certification_report()

    with open(json_file, 'r') as f:
        doc = json.load(f)

    exec_cert = doc['executive_certification']
    assert 'overall_quality_score' in exec_cert
    assert 'quality_grade' in exec_cert
    assert 'certification_status' in exec_cert
    assert exec_cert['certification_status'] == 'CERTIFIED FOR ANALYSIS'
    assert 'recommended_uses' in exec_cert
    assert 'not_recommended_for' in exec_cert


def test_certification_quality_metrics(doc_generator):
    """Test certification report includes quality metrics"""
    json_file = doc_generator.generate_quality_certification_report()

    with open(json_file, 'r') as f:
        doc = json.load(f)

    metrics = doc['quality_metrics']
    assert 'completeness' in metrics
    assert 'accuracy' in metrics
    assert 'consistency' in metrics
    assert 'timeliness' in metrics

    # Verify completeness metrics
    completeness = metrics['completeness']
    assert 'tokens' in completeness
    assert 'wallets' in completeness
    assert 'transactions' in completeness


def test_certification_with_custom_metrics(doc_generator):
    """Test certification report with custom metrics"""
    custom_metrics = {
        'overall_score': 0.95,
        'token_completeness': 1.0,
        'wallet_completeness': 0.98
    }

    json_file = doc_generator.generate_quality_certification_report(custom_metrics)

    with open(json_file, 'r') as f:
        doc = json.load(f)

    assert doc['executive_certification']['overall_quality_score'] == 0.95
    assert doc['quality_metrics']['completeness']['tokens'] == 1.0


def test_generate_user_guide(doc_generator):
    """Test user guide generation"""
    json_file = doc_generator.generate_user_guide()

    assert os.path.exists(json_file)

    # Verify markdown version
    md_file = doc_generator.output_path / 'user_guide.md'
    assert md_file.exists()

    # Load and verify JSON structure
    with open(json_file, 'r') as f:
        doc = json.load(f)

    assert 'metadata' in doc
    assert 'quick_start' in doc
    assert 'dataset_overview' in doc
    assert 'common_use_cases' in doc
    assert 'best_practices' in doc
    assert 'troubleshooting' in doc


def test_user_guide_quick_start(doc_generator):
    """Test user guide includes quick start"""
    json_file = doc_generator.generate_user_guide()

    with open(json_file, 'r') as f:
        doc = json.load(f)

    quick_start = doc['quick_start']
    assert 'overview' in quick_start
    assert 'prerequisites' in quick_start
    assert 'quick_example' in quick_start
    assert 'code' in quick_start['quick_example']


def test_user_guide_datasets(doc_generator):
    """Test user guide describes all datasets"""
    json_file = doc_generator.generate_user_guide()

    with open(json_file, 'r') as f:
        doc = json.load(f)

    datasets = doc['dataset_overview']['available_datasets']
    dataset_names = [d['name'] for d in datasets]

    assert 'tokens' in dataset_names
    assert 'wallets' in dataset_names
    assert 'transactions' in dataset_names
    assert 'wallet_balances' in dataset_names
    assert 'eth_prices' in dataset_names
    assert 'wallet_performance' in dataset_names


def test_user_guide_use_cases(doc_generator):
    """Test user guide includes common use cases"""
    json_file = doc_generator.generate_user_guide()

    with open(json_file, 'r') as f:
        doc = json.load(f)

    use_cases = doc['common_use_cases']
    assert 'narrative_analysis' in use_cases
    assert 'performance_analysis' in use_cases
    assert 'portfolio_tracking' in use_cases

    # Verify each use case has example code
    for use_case_name, use_case_info in use_cases.items():
        assert 'description' in use_case_info
        assert 'example_query' in use_case_info


def test_generate_handoff_manifest(doc_generator):
    """Test handoff package manifest generation"""
    json_file = doc_generator.generate_handoff_package_manifest()

    assert os.path.exists(json_file)

    # Verify markdown version
    md_file = doc_generator.output_path / 'handoff_package_manifest.md'
    assert md_file.exists()

    # Load and verify JSON structure
    with open(json_file, 'r') as f:
        doc = json.load(f)

    assert 'metadata' in doc
    assert 'deliverables' in doc
    assert 'success_metrics' in doc
    assert 'next_steps' in doc
    assert 'lessons_learned' in doc
    assert 'contact_and_support' in doc


def test_handoff_manifest_deliverables(doc_generator):
    """Test handoff manifest includes all deliverables"""
    json_file = doc_generator.generate_handoff_package_manifest()

    with open(json_file, 'r') as f:
        doc = json.load(f)

    deliverables = doc['deliverables']
    assert 'data_exports' in deliverables
    assert 'documentation' in deliverables
    assert 'code_and_config' in deliverables
    assert 'validation_results' in deliverables


def test_handoff_manifest_success_metrics(doc_generator):
    """Test handoff manifest includes success metrics"""
    json_file = doc_generator.generate_handoff_package_manifest()

    with open(json_file, 'r') as f:
        doc = json.load(f)

    success_metrics = doc['success_metrics']
    assert 'completion_criteria' in success_metrics
    assert 'technical_objectives' in success_metrics
    assert 'business_objectives' in success_metrics


def test_generate_all_documentation(doc_generator):
    """Test generating all documentation at once"""
    doc_files = doc_generator.generate_all_documentation()

    # Verify all documentation types were generated
    assert 'methodology' in doc_files
    assert 'data_lineage' in doc_files
    assert 'quality_certification' in doc_files
    assert 'user_guide' in doc_files
    assert 'handoff_manifest' in doc_files

    # Verify all files exist
    for doc_type, file_path in doc_files.items():
        assert os.path.exists(file_path), f"{doc_type} file not found"


def test_markdown_files_generated(doc_generator):
    """Test that markdown versions of all docs are generated"""
    doc_generator.generate_all_documentation()

    expected_md_files = [
        'methodology_documentation.md',
        'data_lineage_documentation.md',
        'quality_certification_report.md',
        'user_guide.md',
        'handoff_package_manifest.md'
    ]

    for md_file in expected_md_files:
        file_path = doc_generator.output_path / md_file
        assert file_path.exists(), f"{md_file} not found"

        # Verify file has content
        assert file_path.stat().st_size > 0, f"{md_file} is empty"


def test_json_validity(doc_generator):
    """Test that all generated JSON files are valid"""
    doc_files = doc_generator.generate_all_documentation()

    for doc_type, file_path in doc_files.items():
        with open(file_path, 'r') as f:
            doc = json.load(f)  # Will raise exception if invalid JSON

        assert isinstance(doc, dict), f"{doc_type} is not a valid JSON object"
        # Each doc has some form of metadata (metadata or certification_metadata, etc.)
        has_metadata = any(key.endswith('metadata') or key == 'metadata' for key in doc.keys())
        assert has_metadata, f"{doc_type} missing metadata section"


def test_markdown_headers_present(doc_generator):
    """Test that markdown files have proper headers"""
    doc_generator.generate_all_documentation()

    md_files = {
        'methodology_documentation.md': 'Methodology',
        'data_lineage_documentation.md': 'Lineage',
        'quality_certification_report.md': 'Certification',
        'user_guide.md': 'User Guide',
        'handoff_package_manifest.md': 'Handoff'
    }

    for md_file, expected_keyword in md_files.items():
        file_path = doc_generator.output_path / md_file
        with open(file_path, 'r') as f:
            content = f.read()

        # Verify markdown has headers
        assert content.startswith('#'), f"{md_file} doesn't start with markdown header"
        assert expected_keyword in content, f"{md_file} missing expected keyword"