"""
Report generation and export functionality for wallet performance analysis.

This module provides comprehensive reporting capabilities including performance
summaries, statistical analysis, dashboard exports, and validation reports
across multiple output formats.
"""

import json
import csv
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive performance reports and exports.

    This class provides methods for creating various types of reports
    including performance summaries, detailed breakdowns, dashboard
    exports, and validation reports in multiple formats.
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the ReportGenerator.

        Args:
            output_dir: Directory for report output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_performance_summary_csv(
        self,
        wallet_data: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """
        Generate performance summary CSV for analysis and visualization.

        Args:
            wallet_data: List of wallet performance dictionaries
            filename: Optional custom filename

        Returns:
            Path to generated CSV file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_summary_{timestamp}.csv"

        filepath = self.output_dir / filename

        # Define columns for CSV export
        columns = [
            'wallet_address', 'calculation_date', 'time_period',
            'total_trades', 'win_rate', 'avg_return_per_trade',
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
            'calmar_ratio', 'var_95', 'total_gas_cost_usd',
            'volume_per_gas', 'net_return_after_costs',
            'unique_tokens_traded', 'hhi_concentration',
            'max_position_size', 'effective_tokens',
            'diversification_score', 'consistency_score',
            'positive_days_pct', 'time_weighted_return'
        ]

        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()

                for wallet in wallet_data:
                    # Filter and format data for CSV
                    csv_row = {}
                    for col in columns:
                        value = wallet.get(col, '')

                        # Format different data types appropriately
                        if isinstance(value, (date, datetime)):
                            csv_row[col] = value.strftime('%Y-%m-%d')
                        elif isinstance(value, (int, float, Decimal)):
                            csv_row[col] = value
                        else:
                            csv_row[col] = str(value) if value is not None else ''

                    writer.writerow(csv_row)

            logger.info(f"Generated performance summary CSV: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error generating CSV report: {e}")
            raise

    def generate_detailed_breakdown_json(
        self,
        wallet_address: str,
        performance_data: Dict[str, Any],
        time_series_data: Optional[List[Dict]] = None,
        diversification_data: Optional[Dict] = None,
        benchmark_data: Optional[Dict] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate detailed performance breakdown by time period.

        Args:
            wallet_address: Wallet address
            performance_data: Comprehensive performance metrics
            time_series_data: Optional time-series data
            diversification_data: Optional diversification analysis
            benchmark_data: Optional benchmark comparison data
            filename: Optional custom filename

        Returns:
            Path to generated JSON file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_address = wallet_address.replace('0x', '').lower()[:8]
            filename = f"detailed_breakdown_{safe_address}_{timestamp}.json"

        filepath = self.output_dir / filename

        # Build comprehensive report structure
        report = {
            'wallet_address': wallet_address,
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_return': performance_data.get('total_return', 0),
                'annualized_return': performance_data.get('annualized_return', 0),
                'sharpe_ratio': performance_data.get('sharpe_ratio', 0),
                'max_drawdown': performance_data.get('max_drawdown', 0),
                'win_rate': performance_data.get('win_rate', 0),
                'volatility': performance_data.get('volatility', 0)
            },
            'risk_metrics': {
                'value_at_risk_95': performance_data.get('var_95', 0),
                'sortino_ratio': performance_data.get('sortino_ratio', 0),
                'calmar_ratio': performance_data.get('calmar_ratio', 0),
                'max_drawdown': performance_data.get('max_drawdown', 0)
            },
            'efficiency_metrics': {
                'total_gas_cost_usd': performance_data.get('total_gas_cost_usd', 0),
                'volume_per_gas': performance_data.get('volume_per_gas', 0),
                'net_return_after_costs': performance_data.get('net_return_after_costs', 0)
            },
            'trading_metrics': {
                'total_trades': performance_data.get('total_trades', 0),
                'avg_return_per_trade': performance_data.get('avg_return_per_trade', 0),
                'consistency_score': performance_data.get('consistency_score', 0),
                'positive_days_pct': performance_data.get('positive_days_pct', 0)
            }
        }

        # Add diversification analysis if provided
        if diversification_data:
            report['diversification'] = {
                'hhi_concentration': diversification_data.get('hhi_concentration', 0),
                'effective_tokens': diversification_data.get('effective_tokens', 0),
                'max_position_size': diversification_data.get('max_position_size', 0),
                'diversification_score': diversification_data.get('diversification_score', 0),
                'unique_tokens_traded': diversification_data.get('unique_tokens_traded', 0)
            }

        # Add benchmark comparison if provided
        if benchmark_data:
            report['benchmark_comparison'] = benchmark_data

        # Add time series data if provided
        if time_series_data:
            report['time_series'] = time_series_data

        # Add metadata
        report['metadata'] = {
            'calculation_date': performance_data.get('calculation_date', date.today()).isoformat(),
            'time_period': performance_data.get('time_period', 'all_time'),
            'data_quality_score': self._calculate_data_quality_score(performance_data)
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as jsonfile:
                json.dump(report, jsonfile, indent=2, default=self._json_serializer)

            logger.info(f"Generated detailed breakdown JSON: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error generating JSON report: {e}")
            raise

    def generate_dashboard_export(
        self,
        top_performers: List[Dict[str, Any]],
        performance_distribution: Dict[str, Any],
        market_overview: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Generate dashboard data export for monitoring.

        Args:
            top_performers: List of top performing wallets
            performance_distribution: Performance distribution statistics
            market_overview: Market overview metrics
            filename: Optional custom filename

        Returns:
            Path to generated dashboard JSON file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_export_{timestamp}.json"

        filepath = self.output_dir / filename

        dashboard_data = {
            'generated_at': datetime.now().isoformat(),
            'top_performers': {
                'by_return': sorted(
                    top_performers,
                    key=lambda x: x.get('total_return', 0),
                    reverse=True
                )[:20],
                'by_sharpe': sorted(
                    top_performers,
                    key=lambda x: x.get('sharpe_ratio', 0),
                    reverse=True
                )[:20],
                'by_consistency': sorted(
                    top_performers,
                    key=lambda x: x.get('consistency_score', 0),
                    reverse=True
                )[:20]
            },
            'performance_distribution': performance_distribution,
            'market_overview': market_overview,
            'summary_statistics': self._calculate_summary_statistics(top_performers)
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as jsonfile:
                json.dump(dashboard_data, jsonfile, indent=2, default=self._json_serializer)

            logger.info(f"Generated dashboard export: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error generating dashboard export: {e}")
            raise

    def generate_statistical_summary(
        self,
        wallet_data: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """
        Generate statistical summary reports with distribution analysis.

        Args:
            wallet_data: List of wallet performance data
            filename: Optional custom filename

        Returns:
            Path to generated statistical summary file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"statistical_summary_{timestamp}.json"

        filepath = self.output_dir / filename

        # Convert to DataFrame for statistical analysis
        df = pd.DataFrame(wallet_data)

        # Key metrics for analysis
        metrics = [
            'total_return', 'sharpe_ratio', 'volatility', 'max_drawdown',
            'win_rate', 'diversification_score', 'consistency_score'
        ]

        statistical_summary = {
            'generated_at': datetime.now().isoformat(),
            'sample_size': len(wallet_data),
            'metrics_analysis': {}
        }

        for metric in metrics:
            if metric in df.columns:
                series = pd.to_numeric(df[metric], errors='coerce').dropna()

                if not series.empty:
                    statistical_summary['metrics_analysis'][metric] = {
                        'count': len(series),
                        'mean': float(series.mean()),
                        'median': float(series.median()),
                        'std': float(series.std()),
                        'min': float(series.min()),
                        'max': float(series.max()),
                        'percentiles': {
                            '10th': float(series.quantile(0.1)),
                            '25th': float(series.quantile(0.25)),
                            '75th': float(series.quantile(0.75)),
                            '90th': float(series.quantile(0.9)),
                            '95th': float(series.quantile(0.95)),
                            '99th': float(series.quantile(0.99))
                        },
                        'outlier_analysis': self._detect_outliers(series)
                    }

        # Correlation analysis
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            correlation_matrix = numeric_df.corr()
            statistical_summary['correlation_analysis'] = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'strongest_correlations': self._find_strongest_correlations(correlation_matrix)
            }

        try:
            with open(filepath, 'w', encoding='utf-8') as jsonfile:
                json.dump(statistical_summary, jsonfile, indent=2, default=self._json_serializer)

            logger.info(f"Generated statistical summary: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error generating statistical summary: {e}")
            raise

    def generate_top_performer_report(
        self,
        top_performers: List[Dict[str, Any]],
        ranking_criteria: List[str],
        filename: Optional[str] = None
    ) -> str:
        """
        Generate top performer identification and ranking reports.

        Args:
            top_performers: List of top performing wallets
            ranking_criteria: List of metrics used for ranking
            filename: Optional custom filename

        Returns:
            Path to generated top performer report
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"top_performers_{timestamp}.json"

        filepath = self.output_dir / filename

        # Create rankings for each criteria
        rankings_by_criteria = {}

        for criterion in ranking_criteria:
            if top_performers and criterion in top_performers[0]:
                sorted_performers = sorted(
                    top_performers,
                    key=lambda x: x.get(criterion, 0),
                    reverse=True
                )

                rankings_by_criteria[criterion] = {
                    'top_10': sorted_performers[:10],
                    'median_value': sorted_performers[len(sorted_performers)//2].get(criterion, 0),
                    'threshold_top_10_percent': sorted_performers[max(0, len(sorted_performers)//10)].get(criterion, 0)
                }

        # Identify consistent top performers (appear in multiple rankings)
        consistent_performers = self._find_consistent_performers(rankings_by_criteria)

        report = {
            'generated_at': datetime.now().isoformat(),
            'total_wallets_analyzed': len(top_performers),
            'ranking_criteria': ranking_criteria,
            'rankings_by_criteria': rankings_by_criteria,
            'consistent_top_performers': consistent_performers,
            'performance_tiers': self._create_performance_tiers(top_performers),
            'summary_insights': self._generate_performance_insights(top_performers, rankings_by_criteria)
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as jsonfile:
                json.dump(report, jsonfile, indent=2, default=self._json_serializer)

            logger.info(f"Generated top performer report: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error generating top performer report: {e}")
            raise

    def generate_validation_report(
        self,
        validation_results: Dict[str, Any],
        data_quality_metrics: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Generate performance validation reports for quality assurance.

        Args:
            validation_results: Results from validation checks
            data_quality_metrics: Data quality assessment metrics
            filename: Optional custom filename

        Returns:
            Path to generated validation report
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_report_{timestamp}.json"

        filepath = self.output_dir / filename

        validation_report = {
            'generated_at': datetime.now().isoformat(),
            'validation_summary': {
                'total_checks': len(validation_results),
                'passed_checks': sum(1 for result in validation_results.values() if result.get('passed', False)),
                'failed_checks': sum(1 for result in validation_results.values() if not result.get('passed', False)),
                'overall_status': 'PASS' if all(result.get('passed', False) for result in validation_results.values()) else 'FAIL'
            },
            'detailed_results': validation_results,
            'data_quality': data_quality_metrics,
            'recommendations': self._generate_validation_recommendations(validation_results, data_quality_metrics)
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as jsonfile:
                json.dump(validation_report, jsonfile, indent=2, default=self._json_serializer)

            logger.info(f"Generated validation report: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error generating validation report: {e}")
            raise

    def export_to_parquet(
        self,
        wallet_data: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """
        Export data to Parquet format for large-scale analytics.

        Args:
            wallet_data: List of wallet performance data
            filename: Optional custom filename

        Returns:
            Path to generated Parquet file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"wallet_performance_{timestamp}.parquet"

        filepath = self.output_dir / filename

        try:
            df = pd.DataFrame(wallet_data)

            # Optimize data types for Parquet
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to convert to numeric if possible
                    try:
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except:
                        pass

            df.to_parquet(filepath, compression='snappy', index=False)

            logger.info(f"Generated Parquet export: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error generating Parquet export: {e}")
            raise

    def _json_serializer(self, obj):
        """Custom JSON serializer for special data types."""
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif pd.isna(obj):
            return None
        elif isinstance(obj, (int, float, str, bool, list, dict)):
            return obj  # Return as-is for basic JSON types
        else:
            return str(obj)

    def _calculate_data_quality_score(self, performance_data: Dict[str, Any]) -> float:
        """Calculate data quality score based on completeness and validity."""
        required_fields = [
            'total_return', 'sharpe_ratio', 'win_rate', 'total_trades',
            'volatility', 'max_drawdown'
        ]

        present_fields = sum(1 for field in required_fields if field in performance_data and performance_data[field] is not None)
        completeness_score = present_fields / len(required_fields)

        # Additional validity checks
        validity_score = 1.0
        if performance_data.get('win_rate', 0) > 1:
            validity_score -= 0.1
        if performance_data.get('total_trades', 0) < 0:
            validity_score -= 0.1

        return min(1.0, completeness_score * validity_score)

    def _calculate_summary_statistics(self, performers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for dashboard."""
        if not performers:
            return {}

        df = pd.DataFrame(performers)

        return {
            'total_wallets': len(performers),
            'avg_return': float(df.get('total_return', pd.Series()).mean()),
            'avg_sharpe_ratio': float(df.get('sharpe_ratio', pd.Series()).mean()),
            'return_distribution': {
                'positive_returns': int((df.get('total_return', pd.Series()) > 0).sum()),
                'negative_returns': int((df.get('total_return', pd.Series()) < 0).sum())
            }
        }

    def _detect_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = series[(series < lower_bound) | (series > upper_bound)]

        return {
            'count': len(outliers),
            'percentage': float(len(outliers) / len(series) * 100),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }

    def _find_strongest_correlations(self, corr_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find strongest correlations in the matrix."""
        correlations = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if not pd.isna(corr_value):
                    correlations.append({
                        'metric1': corr_matrix.columns[i],
                        'metric2': corr_matrix.columns[j],
                        'correlation': float(corr_value)
                    })

        # Return top 10 strongest correlations (by absolute value)
        return sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)[:10]

    def _find_consistent_performers(self, rankings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find wallets that consistently appear in top rankings."""
        wallet_appearances = {}

        for criterion, ranking_data in rankings.items():
            for wallet in ranking_data.get('top_10', []):
                wallet_addr = wallet.get('wallet_address')
                if wallet_addr:
                    if wallet_addr not in wallet_appearances:
                        wallet_appearances[wallet_addr] = {'count': 0, 'criteria': [], 'wallet_data': wallet}
                    wallet_appearances[wallet_addr]['count'] += 1
                    wallet_appearances[wallet_addr]['criteria'].append(criterion)

        # Return wallets that appear in multiple rankings
        consistent = [
            {
                'wallet_address': addr,
                'appearances': data['count'],
                'criteria': data['criteria'],
                'performance_data': data['wallet_data']
            }
            for addr, data in wallet_appearances.items()
            if data['count'] > 1
        ]

        return sorted(consistent, key=lambda x: x['appearances'], reverse=True)

    def _create_performance_tiers(self, performers: List[Dict[str, Any]]) -> Dict[str, List]:
        """Create performance tiers based on overall performance."""
        if not performers:
            return {}

        # Sort by total return
        sorted_performers = sorted(performers, key=lambda x: x.get('total_return', 0), reverse=True)

        tier_size = len(sorted_performers) // 4

        return {
            'tier_1_elite': sorted_performers[:tier_size],
            'tier_2_high': sorted_performers[tier_size:tier_size*2],
            'tier_3_medium': sorted_performers[tier_size*2:tier_size*3],
            'tier_4_low': sorted_performers[tier_size*3:]
        }

    def _generate_performance_insights(
        self,
        performers: List[Dict[str, Any]],
        rankings: Dict[str, Any]
    ) -> List[str]:
        """Generate insights from performance analysis."""
        insights = []

        if not performers:
            return insights

        df = pd.DataFrame(performers)

        # Return distribution insight
        positive_returns = (df.get('total_return', pd.Series(dtype=float)) > 0).sum()
        total_wallets = len(performers)
        positive_pct = (positive_returns / total_wallets * 100) if total_wallets > 0 else 0

        insights.append(f"{positive_pct:.1f}% of analyzed wallets achieved positive returns")

        # Sharpe ratio insight
        high_sharpe = (df.get('sharpe_ratio', pd.Series(dtype=float)) > 1.0).sum()
        if high_sharpe > 0:
            insights.append(f"{high_sharpe} wallets achieved Sharpe ratio above 1.0")

        # Diversification insight
        high_diversification = (df.get('diversification_score', pd.Series(dtype=float)) > 70).sum()
        if high_diversification > 0:
            insights.append(f"{high_diversification} wallets showed high diversification (score > 70)")

        return insights

    def _generate_validation_recommendations(
        self,
        validation_results: Dict[str, Any],
        data_quality: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Check for failed validations
        failed_checks = [name for name, result in validation_results.items() if not result.get('passed', False)]

        if failed_checks:
            recommendations.append(f"Address failed validation checks: {', '.join(failed_checks)}")

        # Data quality recommendations
        completeness = data_quality.get('completeness_score', 1.0)
        if completeness < 0.9:
            recommendations.append("Improve data completeness - some required fields are missing")

        accuracy = data_quality.get('accuracy_score', 1.0)
        if accuracy < 0.95:
            recommendations.append("Review data accuracy - some values appear to be outside expected ranges")

        if not recommendations:
            recommendations.append("All validation checks passed - data quality is good")

        return recommendations