#!/usr/bin/env python3
"""
CLI for DEX Liquidity Analysis
Provides command-line interface for running liquidity analysis workflows
"""

import click
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data_collection.common.db import DatabaseManager
from data_collection.common.config import get_config
from data_collection.common.checkpoints import CheckpointManager
from data_collection.common.logging_setup import setup_logging, get_logger
from services.tokens.dune_client import DuneClient, DuneClientError
from services.tokens.liquidity_analyzer import DEXLiquidityAnalyzer
from services.tokens.liquidity_validator import LiquidityValidator

logger = get_logger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', help='Path to config file')
@click.pass_context
def cli(ctx, verbose, config):
    """DEX Liquidity Analysis CLI"""
    ctx.ensure_object(dict)

    # Setup logging
    setup_logging()
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    if config:
        ctx.obj['config_path'] = config
    else:
        ctx.obj['config_path'] = None

    logger.info("DEX Liquidity Analysis CLI initialized")


@cli.command()
@click.option('--limit', '-l', type=int, help='Limit number of tokens to analyze')
@click.option('--date-range', '-d', type=int, default=30, help='Date range in days (default: 30)')
@click.option('--force', '-f', is_flag=True, help='Force analysis ignoring checkpoints')
@click.pass_context
def analyze(ctx, limit, date_range, force):
    """Run complete liquidity analysis workflow"""
    try:
        # Initialize components
        db_manager = DatabaseManager()
        checkpoint_manager = CheckpointManager("liquidity_analysis")
        dune_client = DuneClient()

        # Create analyzer
        analyzer = DEXLiquidityAnalyzer(dune_client, db_manager, checkpoint_manager)

        # Query IDs are already configured in the analyzer from config/dune_query_ids.yaml
        # No need to override them here - they're set in __init__

        if force:
            # Clear existing checkpoint
            checkpoint_manager.clear_checkpoint("liquidity_analysis")
            logger.info("Cleared existing checkpoint - forcing full analysis")

        click.echo(f"Starting liquidity analysis...")
        click.echo(f"Token limit: {limit or 'All tokens'}")
        click.echo(f"Date range: {date_range} days")

        # Run analysis
        result = analyzer.run_liquidity_analysis(
            token_limit=limit,
            date_range_days=date_range
        )

        if result["status"] == "completed":
            click.echo("\n✅ Liquidity analysis completed successfully!")
            click.echo(f"📊 Pools discovered: {result['pools_discovered']}")
            click.echo(f"🎯 Tokens analyzed: {result['tokens_analyzed']}")

            if "report" in result:
                report = result["report"]
                tier_dist = report["tier_distribution"]
                click.echo(f"\n📈 Tier Distribution:")
                for tier, count in tier_dist.items():
                    percentage = report["tier_percentages"][tier]
                    click.echo(f"  {tier}: {count} tokens ({percentage:.1f}%)")

        else:
            click.echo(f"❌ Analysis failed: {result.get('message', 'Unknown error')}")
            sys.exit(1)

    except DuneClientError as e:
        click.echo(f"❌ Dune API error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Analysis failed")
        click.echo(f"❌ Unexpected error: {e}")
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', help='Output file path for quality report')
@click.option('--export-pools', help='Export pool data to CSV file')
@click.option('--export-tiers', help='Export tier assignments to CSV file')
@click.pass_context
def validate(ctx, output, export_pools, export_tiers):
    """Validate liquidity analysis quality and generate reports"""
    try:
        # Initialize components
        db_manager = DatabaseManager()
        validator = LiquidityValidator(db_manager)

        click.echo("🔍 Running quality validation...")

        # Generate quality report
        output_path = output or f"./outputs/quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report = validator.generate_quality_report(output_path)

        # Display summary
        metadata = report["report_metadata"]
        click.echo(f"\n📋 Quality Report Summary:")
        click.echo(f"Overall Score: {metadata['overall_score']:.1f}/100")
        click.echo(f"Status: {metadata['status'].upper()}")
        click.echo(f"Critical Issues: {metadata['critical_issues']}")
        click.echo(f"High Severity Anomalies: {metadata['high_severity_anomalies']}")

        # Show validation results
        click.echo(f"\n🧪 Validation Results:")
        for validation in report["validation_results"]:
            status = "✅ PASS" if validation["passed"] else "❌ FAIL"
            click.echo(f"  {status} {validation['test_name']}: {validation['score']:.1f}/100")

        # Show recommendations
        if report["summary_recommendations"]:
            click.echo(f"\n💡 Recommendations:")
            for rec in report["summary_recommendations"]:
                click.echo(f"  • {rec}")

        click.echo(f"\n📄 Full report saved to: {output_path}")

        # Export data if requested
        if export_pools:
            validator.export_pool_data_for_review(export_pools)
            click.echo(f"📊 Pool data exported to: {export_pools}")

        if export_tiers:
            validator.export_tier_assignments(export_tiers)
            click.echo(f"🏷️ Tier assignments exported to: {export_tiers}")

    except Exception as e:
        logger.exception("Validation failed")
        click.echo(f"❌ Validation error: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Show current analysis status and statistics"""
    try:
        db_manager = DatabaseManager()

        with db_manager.get_connection() as conn:
            # Get checkpoint status
            checkpoint_result = conn.execute("""
                SELECT status, last_processed_date, records_collected, updated_at
                FROM collection_checkpoints
                WHERE collection_type = 'liquidity_analysis'
            """).fetchone()

            # Get tier statistics
            tier_stats = conn.execute("""
                SELECT liquidity_tier, COUNT(*) as count
                FROM tokens
                WHERE liquidity_tier IS NOT NULL
                GROUP BY liquidity_tier
                ORDER BY
                    CASE liquidity_tier
                        WHEN 'Tier 1' THEN 1
                        WHEN 'Tier 2' THEN 2
                        WHEN 'Tier 3' THEN 3
                        WHEN 'Untiered' THEN 4
                    END
            """).fetchall()

            # Get pool statistics
            pool_stats = conn.execute("""
                SELECT
                    COUNT(DISTINCT token_address) as tokens_with_pools,
                    COUNT(*) as total_pools,
                    COUNT(DISTINCT dex_name) as dex_count,
                    AVG(tvl_usd) as avg_tvl,
                    MAX(last_updated) as last_pool_update
                FROM token_pools
            """).fetchone()

        # Display checkpoint status
        click.echo("📊 Liquidity Analysis Status\n")

        if checkpoint_result:
            status, last_date, records, updated = checkpoint_result
            click.echo(f"Checkpoint Status: {status.upper()}")
            click.echo(f"Last Run: {updated}")
            click.echo(f"Records Collected: {records}")
            if last_date:
                click.echo(f"Last Processed Date: {last_date}")
        else:
            click.echo("❌ No liquidity analysis checkpoint found")

        # Display tier distribution
        if tier_stats:
            click.echo(f"\n🏷️ Tier Distribution:")
            total_tokens = sum(row[1] for row in tier_stats)
            for tier, count in tier_stats:
                percentage = (count / total_tokens * 100) if total_tokens > 0 else 0
                click.echo(f"  {tier}: {count} tokens ({percentage:.1f}%)")

        # Display pool statistics
        if pool_stats and pool_stats[0]:
            tokens_with_pools, total_pools, dex_count, avg_tvl, last_update = pool_stats
            click.echo(f"\n💧 Pool Statistics:")
            click.echo(f"  Tokens with pools: {tokens_with_pools}")
            click.echo(f"  Total pools: {total_pools}")
            click.echo(f"  DEX platforms: {dex_count}")
            click.echo(f"  Average TVL: ${avg_tvl:,.2f}" if avg_tvl else "  Average TVL: N/A")
            click.echo(f"  Last updated: {last_update}" if last_update else "  Last updated: N/A")

    except Exception as e:
        logger.exception("Status check failed")
        click.echo(f"❌ Error checking status: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def reset(ctx):
    """Reset liquidity analysis (clear checkpoints and data)"""
    if not click.confirm("⚠️ This will clear all liquidity analysis data. Continue?"):
        click.echo("Operation cancelled")
        return

    try:
        db_manager = DatabaseManager()
        checkpoint_manager = CheckpointManager(db_manager)

        with db_manager.get_connection() as conn:
            # Clear token tiers
            conn.execute("UPDATE tokens SET liquidity_tier = NULL")

            # Clear pool data
            conn.execute("DELETE FROM token_pools")

            # Clear checkpoint
            checkpoint_manager.clear_checkpoint("liquidity_analysis")

            conn.commit()

        click.echo("✅ Liquidity analysis data cleared successfully")

    except Exception as e:
        logger.exception("Reset failed")
        click.echo(f"❌ Reset error: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def setup(ctx):
    """Setup database schema for liquidity analysis"""
    try:
        db_manager = DatabaseManager()

        # Run schema updates
        schema_file = Path(__file__).parent / "sql" / "liquidity_schema_update.sql"

        if not schema_file.exists():
            click.echo(f"❌ Schema file not found: {schema_file}")
            sys.exit(1)

        with open(schema_file, 'r') as f:
            schema_sql = f.read()

        with db_manager.get_connection() as conn:
            # Execute schema updates
            for statement in schema_sql.split(';'):
                if statement.strip():
                    conn.execute(statement)
            conn.commit()

        click.echo("✅ Database schema setup completed")

    except Exception as e:
        logger.exception("Setup failed")
        click.echo(f"❌ Setup error: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def test_dune(ctx):
    """Test Dune Analytics API connectivity"""
    try:
        dune_client = DuneClient()

        # Test basic connectivity
        click.echo("🔗 Testing Dune Analytics API connectivity...")

        # Check cache stats
        cache_stats = dune_client.get_cache_stats()
        click.echo(f"📁 Cache directory: {cache_stats['cache_directory']}")
        click.echo(f"📊 Cached queries: {cache_stats['cached_queries']}")
        click.echo(f"💾 Cache size: {cache_stats['total_size_mb']:.2f} MB")

        click.echo("✅ Dune client initialized successfully")

        # Test simple query (if query ID provided)
        if click.confirm("Would you like to test a simple query? (requires valid query ID)"):
            query_id = click.prompt("Enter Dune query ID", type=int)
            execution_id = dune_client.execute_query(query_id, {})
            click.echo(f"📋 Query submitted with execution ID: {execution_id}")

    except DuneClientError as e:
        click.echo(f"❌ Dune API error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Dune test failed")
        click.echo(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli()