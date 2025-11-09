"""
Audit Trail and Documentation System

This module implements comprehensive audit trail tracking, filtering decision documentation,
and reproducibility frameworks for the wallet filtering process.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from decimal import Decimal
import logging
import json
import hashlib
import pickle
from pathlib import Path
from enum import Enum
import yaml

from .wallet_filter import FilterResult, FilterCriteria
from .cohort_optimizer import CohortResult
from .manual_review import ReviewCase

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events to audit."""
    FILTER_APPLIED = "filter_applied"
    THRESHOLD_CHANGED = "threshold_changed"
    MANUAL_REVIEW = "manual_review"
    OVERRIDE_APPLIED = "override_applied"
    COHORT_SELECTED = "cohort_selected"
    VALIDATION_PERFORMED = "validation_performed"
    PROCESS_STARTED = "process_started"
    PROCESS_COMPLETED = "process_completed"


class DecisionType(Enum):
    """Types of filtering decisions."""
    AUTOMATED_PASS = "automated_pass"
    AUTOMATED_FAIL = "automated_fail"
    MANUAL_OVERRIDE = "manual_override"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    COHORT_OPTIMIZATION = "cohort_optimization"


@dataclass
class AuditEvent:
    """Represents a single audit event."""
    event_id: str
    timestamp: datetime
    event_type: EventType
    decision_type: DecisionType
    wallet_address: Optional[str]

    # Event data
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    parameters: Dict[str, Any]

    # Process context
    process_id: str
    user_id: Optional[str]
    system_version: str

    # Decision rationale
    decision_rationale: str
    confidence_score: Optional[float]

    # Reproducibility data
    data_hash: str
    config_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['decision_type'] = self.decision_type.value
        return data


@dataclass
class FilteringRun:
    """Represents a complete filtering run."""
    run_id: str
    start_timestamp: datetime
    end_timestamp: Optional[datetime]
    status: str  # "running", "completed", "failed"

    # Configuration
    filter_criteria: FilterCriteria
    optimization_parameters: Dict[str, Any]

    # Results
    input_wallet_count: int
    output_wallet_count: int
    final_cohort_size: Optional[int]

    # Quality metrics
    avg_quality_score: Optional[float]
    quality_distribution: Dict[str, float]

    # Audit events
    events: List[AuditEvent] = field(default_factory=list)

    # Reproducibility
    data_sources: List[str] = field(default_factory=list)
    environment_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParameterJustification:
    """Justification for filtering parameters."""
    parameter_name: str
    parameter_value: Any
    justification: str
    data_source: str
    last_updated: datetime
    updated_by: str
    validation_results: Optional[Dict[str, Any]] = None


@dataclass
class DecisionTree:
    """Represents filtering decision logic."""
    node_id: str
    node_type: str  # "condition", "action", "leaf"
    condition: Optional[str]
    threshold: Optional[float]
    action: Optional[str]
    children: List['DecisionTree'] = field(default_factory=list)
    description: str = ""


class AuditTrailManager:
    """
    Comprehensive audit trail and documentation system for wallet filtering.

    This class tracks all filtering decisions, parameter changes, and process steps
    to ensure full reproducibility and compliance.
    """

    def __init__(self,
                 audit_directory: str = "./audit_logs",
                 max_log_age_days: int = 365,
                 compression_enabled: bool = True):
        """
        Initialize the audit trail manager.

        Args:
            audit_directory: Directory to store audit logs
            max_log_age_days: Maximum age of logs before archival
            compression_enabled: Whether to compress archived logs
        """
        self.audit_dir = Path(audit_directory)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.max_log_age = max_log_age_days
        self.compression_enabled = compression_enabled

        # Current run state
        self.current_run: Optional[FilteringRun] = None
        self.current_process_id: Optional[str] = None

        # Parameter documentation
        self.parameter_justifications: Dict[str, ParameterJustification] = {}
        self.decision_trees: Dict[str, DecisionTree] = {}

        # Configuration versioning
        self.config_versions: List[Dict[str, Any]] = []

        logger.info(f"AuditTrailManager initialized with directory: {audit_directory}")

    def start_filtering_run(self,
                          filter_criteria: FilterCriteria,
                          optimization_parameters: Dict[str, Any],
                          input_wallet_count: int,
                          user_id: str = "system") -> str:
        """
        Start a new filtering run and initialize audit trail.

        Args:
            filter_criteria: Filtering criteria configuration
            optimization_parameters: Cohort optimization parameters
            input_wallet_count: Number of input wallets
            user_id: User initiating the run

        Returns:
            Run ID for the filtering process
        """
        try:
            run_id = self._generate_run_id()
            process_id = self._generate_process_id()

            self.current_run = FilteringRun(
                run_id=run_id,
                start_timestamp=datetime.now(),
                end_timestamp=None,
                status="running",
                filter_criteria=filter_criteria,
                optimization_parameters=optimization_parameters,
                input_wallet_count=input_wallet_count,
                output_wallet_count=0,
                final_cohort_size=None,
                avg_quality_score=None,
                quality_distribution={},
                data_sources=[],
                environment_info=self._capture_environment_info()
            )

            self.current_process_id = process_id

            # Log process start event
            self.log_event(
                event_type=EventType.PROCESS_STARTED,
                decision_type=DecisionType.AUTOMATED_PASS,
                input_data={
                    "input_wallet_count": input_wallet_count,
                    "filter_criteria": asdict(filter_criteria),
                    "optimization_parameters": optimization_parameters
                },
                output_data={},
                parameters=asdict(filter_criteria),
                decision_rationale="Filtering process initiated",
                user_id=user_id
            )

            logger.info(f"Started filtering run {run_id} with process ID {process_id}")
            return run_id

        except Exception as e:
            logger.error(f"Error starting filtering run: {e}")
            raise

    def log_event(self,
                  event_type: EventType,
                  decision_type: DecisionType,
                  input_data: Dict[str, Any],
                  output_data: Dict[str, Any],
                  parameters: Dict[str, Any],
                  decision_rationale: str,
                  wallet_address: Optional[str] = None,
                  user_id: Optional[str] = None,
                  confidence_score: Optional[float] = None) -> str:
        """
        Log a filtering decision or event.

        Args:
            event_type: Type of event
            decision_type: Type of decision
            input_data: Input data for the event
            output_data: Output data from the event
            parameters: Parameters used in the decision
            decision_rationale: Human-readable rationale
            wallet_address: Wallet address if applicable
            user_id: User making the decision
            confidence_score: Confidence in the decision

        Returns:
            Event ID
        """
        try:
            event_id = self._generate_event_id()

            # Calculate data hashes for reproducibility
            data_hash = self._calculate_data_hash(input_data)
            config_hash = self._calculate_config_hash(parameters)

            event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                event_type=event_type,
                decision_type=decision_type,
                wallet_address=wallet_address,
                input_data=input_data,
                output_data=output_data,
                parameters=parameters,
                process_id=self.current_process_id or "unknown",
                user_id=user_id,
                system_version=self._get_system_version(),
                decision_rationale=decision_rationale,
                confidence_score=confidence_score,
                data_hash=data_hash,
                config_hash=config_hash
            )

            # Add to current run if active
            if self.current_run:
                self.current_run.events.append(event)

            # Persist event immediately
            self._persist_event(event)

            logger.debug(f"Logged event {event_id}: {event_type.value}")
            return event_id

        except Exception as e:
            logger.error(f"Error logging event: {e}")
            return ""

    def log_filtering_decision(self,
                             wallet_address: str,
                             filter_result: FilterResult,
                             decision_rationale: str) -> str:
        """
        Log a specific wallet filtering decision.

        Args:
            wallet_address: Address of the wallet
            filter_result: Result of the filtering
            decision_rationale: Rationale for the decision

        Returns:
            Event ID
        """
        return self.log_event(
            event_type=EventType.FILTER_APPLIED,
            decision_type=DecisionType.AUTOMATED_PASS if filter_result.passed else DecisionType.AUTOMATED_FAIL,
            input_data={
                "wallet_address": wallet_address,
                "filter_scores": filter_result.scores
            },
            output_data={
                "passed": filter_result.passed,
                "reason": filter_result.reason.value if filter_result.reason else None
            },
            parameters=filter_result.details,
            decision_rationale=decision_rationale,
            wallet_address=wallet_address,
            confidence_score=self._calculate_decision_confidence(filter_result)
        )

    def log_manual_override(self,
                          wallet_address: str,
                          original_decision: bool,
                          override_decision: bool,
                          override_reason: str,
                          authorized_by: str) -> str:
        """
        Log a manual override decision.

        Args:
            wallet_address: Address of the wallet
            original_decision: Original automated decision
            override_decision: Override decision
            override_reason: Reason for override
            authorized_by: Who authorized the override

        Returns:
            Event ID
        """
        return self.log_event(
            event_type=EventType.OVERRIDE_APPLIED,
            decision_type=DecisionType.MANUAL_OVERRIDE,
            input_data={
                "original_decision": original_decision,
                "wallet_address": wallet_address
            },
            output_data={
                "override_decision": override_decision,
                "authorized_by": authorized_by
            },
            parameters={
                "override_reason": override_reason
            },
            decision_rationale=f"Manual override: {override_reason}",
            wallet_address=wallet_address,
            user_id=authorized_by
        )

    def log_cohort_selection(self,
                           cohort_result: CohortResult,
                           selection_rationale: str) -> str:
        """
        Log final cohort selection.

        Args:
            cohort_result: Result of cohort optimization
            selection_rationale: Rationale for selection

        Returns:
            Event ID
        """
        return self.log_event(
            event_type=EventType.COHORT_SELECTED,
            decision_type=DecisionType.COHORT_OPTIMIZATION,
            input_data={
                "optimization_iterations": cohort_result.optimization_iterations,
                "final_thresholds": cohort_result.final_thresholds
            },
            output_data={
                "selected_count": len(cohort_result.selected_wallets),
                "rejected_count": len(cohort_result.rejected_wallets),
                "target_achieved": cohort_result.target_achieved
            },
            parameters=cohort_result.final_thresholds,
            decision_rationale=selection_rationale
        )

    def complete_filtering_run(self,
                             output_wallet_count: int,
                             final_cohort_size: int,
                             quality_metrics: Dict[str, float]) -> None:
        """
        Complete the current filtering run and finalize audit trail.

        Args:
            output_wallet_count: Number of wallets that passed filtering
            final_cohort_size: Final cohort size after optimization
            quality_metrics: Final quality metrics
        """
        try:
            if not self.current_run:
                logger.warning("No active filtering run to complete")
                return

            # Update run information
            self.current_run.end_timestamp = datetime.now()
            self.current_run.status = "completed"
            self.current_run.output_wallet_count = output_wallet_count
            self.current_run.final_cohort_size = final_cohort_size
            self.current_run.avg_quality_score = quality_metrics.get('mean', 0)
            self.current_run.quality_distribution = quality_metrics

            # Log completion event
            self.log_event(
                event_type=EventType.PROCESS_COMPLETED,
                decision_type=DecisionType.AUTOMATED_PASS,
                input_data={
                    "process_duration": (self.current_run.end_timestamp - self.current_run.start_timestamp).total_seconds()
                },
                output_data={
                    "output_wallet_count": output_wallet_count,
                    "final_cohort_size": final_cohort_size,
                    "quality_metrics": quality_metrics
                },
                parameters={},
                decision_rationale="Filtering process completed successfully"
            )

            # Persist complete run
            self._persist_filtering_run()

            logger.info(f"Completed filtering run {self.current_run.run_id}")

            # Clear current run
            self.current_run = None
            self.current_process_id = None

        except Exception as e:
            logger.error(f"Error completing filtering run: {e}")

    def document_parameter_justification(self,
                                       parameter_name: str,
                                       parameter_value: Any,
                                       justification: str,
                                       data_source: str,
                                       updated_by: str) -> None:
        """
        Document justification for filtering parameters.

        Args:
            parameter_name: Name of the parameter
            parameter_value: Value of the parameter
            justification: Business/technical justification
            data_source: Source of validation data
            updated_by: Who set/updated the parameter
        """
        try:
            self.parameter_justifications[parameter_name] = ParameterJustification(
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                justification=justification,
                data_source=data_source,
                last_updated=datetime.now(),
                updated_by=updated_by
            )

            # Persist justification
            self._persist_parameter_justifications()

            logger.info(f"Documented justification for parameter {parameter_name}")

        except Exception as e:
            logger.error(f"Error documenting parameter justification: {e}")

    def create_decision_tree(self,
                           tree_name: str,
                           decision_logic: Dict[str, Any]) -> None:
        """
        Create documentation for complex filtering decision logic.

        Args:
            tree_name: Name of the decision tree
            decision_logic: Hierarchical decision logic structure
        """
        try:
            tree = self._build_decision_tree(decision_logic)
            self.decision_trees[tree_name] = tree

            # Persist decision tree
            self._persist_decision_trees()

            logger.info(f"Created decision tree documentation: {tree_name}")

        except Exception as e:
            logger.error(f"Error creating decision tree: {e}")

    def generate_reproducibility_report(self, run_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive reproducibility report for a filtering run.

        Args:
            run_id: ID of the filtering run

        Returns:
            Reproducibility report
        """
        try:
            # Load run data
            run_data = self._load_filtering_run(run_id)
            if not run_data:
                return {"error": f"Run {run_id} not found"}

            # Generate report
            report = {
                "run_metadata": {
                    "run_id": run_id,
                    "start_time": run_data.start_timestamp.isoformat(),
                    "end_time": run_data.end_timestamp.isoformat() if run_data.end_timestamp else None,
                    "duration_seconds": (run_data.end_timestamp - run_data.start_timestamp).total_seconds() if run_data.end_timestamp else None
                },
                "configuration": {
                    "filter_criteria": asdict(run_data.filter_criteria),
                    "optimization_parameters": run_data.optimization_parameters,
                    "parameter_justifications": {
                        name: asdict(justification)
                        for name, justification in self.parameter_justifications.items()
                    }
                },
                "results": {
                    "input_wallet_count": run_data.input_wallet_count,
                    "output_wallet_count": run_data.output_wallet_count,
                    "final_cohort_size": run_data.final_cohort_size,
                    "avg_quality_score": run_data.avg_quality_score,
                    "quality_distribution": run_data.quality_distribution
                },
                "audit_trail": [event.to_dict() for event in run_data.events],
                "environment": run_data.environment_info,
                "data_sources": run_data.data_sources,
                "reproducibility_hashes": {
                    "config_hash": self._calculate_config_hash(asdict(run_data.filter_criteria)),
                    "parameter_hash": self._calculate_parameter_hash()
                },
                "decision_trees": {
                    name: self._serialize_decision_tree(tree)
                    for name, tree in self.decision_trees.items()
                }
            }

            # Save report
            report_path = self.audit_dir / f"reproducibility_report_{run_id}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Generated reproducibility report for run {run_id}")
            return report

        except Exception as e:
            logger.error(f"Error generating reproducibility report: {e}")
            return {"error": str(e)}

    def get_filter_effectiveness_metrics(self, time_period_days: int = 30) -> Dict[str, Any]:
        """
        Calculate filter effectiveness metrics over a time period.

        Args:
            time_period_days: Time period for analysis

        Returns:
            Filter effectiveness metrics
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=time_period_days)

            # Load recent events
            events = self._load_events_since(cutoff_date)

            # Calculate metrics
            total_decisions = len([e for e in events if e.event_type == EventType.FILTER_APPLIED])
            automated_passes = len([e for e in events
                                  if e.event_type == EventType.FILTER_APPLIED
                                  and e.decision_type == DecisionType.AUTOMATED_PASS])

            manual_overrides = len([e for e in events if e.event_type == EventType.OVERRIDE_APPLIED])

            # Calculate override rate
            override_rate = manual_overrides / total_decisions if total_decisions > 0 else 0

            # Calculate average confidence
            confidence_scores = [e.confidence_score for e in events
                               if e.confidence_score is not None]
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0

            # Filter distribution
            filter_reasons = defaultdict(int)
            for event in events:
                if event.event_type == EventType.FILTER_APPLIED:
                    reason = event.output_data.get('reason', 'unknown')
                    filter_reasons[reason] += 1

            return {
                "time_period_days": time_period_days,
                "total_decisions": total_decisions,
                "automated_pass_rate": automated_passes / total_decisions if total_decisions > 0 else 0,
                "manual_override_rate": override_rate,
                "average_confidence": avg_confidence,
                "filter_reason_distribution": dict(filter_reasons),
                "effectiveness_score": (1 - override_rate) * avg_confidence
            }

        except Exception as e:
            logger.error(f"Error calculating filter effectiveness: {e}")
            return {}

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(datetime.now().microsecond).encode()).hexdigest()[:8]
        return f"RUN_{timestamp}_{random_suffix}"

    def _generate_process_id(self) -> str:
        """Generate unique process ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"PROC_{timestamp}"

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"EVT_{timestamp}"

    def _calculate_data_hash(self, data: Dict[str, Any]) -> str:
        """Calculate hash of input data for reproducibility."""
        try:
            data_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(data_str.encode()).hexdigest()
        except Exception:
            return "hash_error"

    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration for reproducibility."""
        try:
            config_str = json.dumps(config, sort_keys=True, default=str)
            return hashlib.sha256(config_str.encode()).hexdigest()
        except Exception:
            return "hash_error"

    def _calculate_parameter_hash(self) -> str:
        """Calculate hash of all parameter justifications."""
        try:
            params = {name: asdict(justification)
                     for name, justification in self.parameter_justifications.items()}
            params_str = json.dumps(params, sort_keys=True, default=str)
            return hashlib.sha256(params_str.encode()).hexdigest()
        except Exception:
            return "hash_error"

    def _calculate_decision_confidence(self, filter_result: FilterResult) -> float:
        """Calculate confidence score for a filtering decision."""
        if not filter_result.scores:
            return 0.5

        # Simple confidence based on how close scores are to thresholds
        score_distances = []

        sharpe = filter_result.scores.get('sharpe_ratio', 0)
        score_distances.append(abs(sharpe - 0.5) / 2.0)  # Normalize by reasonable range

        win_rate = filter_result.scores.get('win_rate', 0)
        score_distances.append(abs(win_rate - 0.55) / 0.45)

        return min(1.0, np.mean(score_distances) + 0.5)

    def _capture_environment_info(self) -> Dict[str, Any]:
        """Capture environment information for reproducibility."""
        import sys
        import platform

        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "system": platform.system(),
            "timestamp": datetime.now().isoformat(),
            "working_directory": str(Path.cwd())
        }

    def _get_system_version(self) -> str:
        """Get system version for audit trail."""
        return "wallet_filter_v1.0"

    def _persist_event(self, event: AuditEvent) -> None:
        """Persist individual event to storage."""
        try:
            event_file = self.audit_dir / f"events_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(event_file, 'a') as f:
                f.write(json.dumps(event.to_dict(), default=str) + '\n')
        except Exception as e:
            logger.error(f"Error persisting event: {e}")

    def _persist_filtering_run(self) -> None:
        """Persist complete filtering run."""
        try:
            if not self.current_run:
                return

            run_file = self.audit_dir / f"run_{self.current_run.run_id}.json"
            run_data = {
                "run_metadata": asdict(self.current_run),
                "events": [event.to_dict() for event in self.current_run.events]
            }

            with open(run_file, 'w') as f:
                json.dump(run_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error persisting filtering run: {e}")

    def _persist_parameter_justifications(self) -> None:
        """Persist parameter justifications."""
        try:
            justifications_file = self.audit_dir / "parameter_justifications.yaml"
            justifications_data = {
                name: asdict(justification)
                for name, justification in self.parameter_justifications.items()
            }

            with open(justifications_file, 'w') as f:
                yaml.dump(justifications_data, f, default_flow_style=False)

        except Exception as e:
            logger.error(f"Error persisting parameter justifications: {e}")

    def _persist_decision_trees(self) -> None:
        """Persist decision tree documentation."""
        try:
            trees_file = self.audit_dir / "decision_trees.yaml"
            trees_data = {
                name: self._serialize_decision_tree(tree)
                for name, tree in self.decision_trees.items()
            }

            with open(trees_file, 'w') as f:
                yaml.dump(trees_data, f, default_flow_style=False)

        except Exception as e:
            logger.error(f"Error persisting decision trees: {e}")

    def _build_decision_tree(self, logic: Dict[str, Any]) -> DecisionTree:
        """Build decision tree from logic dictionary."""
        # Simplified implementation - would need more sophisticated parsing
        return DecisionTree(
            node_id=logic.get('id', 'root'),
            node_type=logic.get('type', 'condition'),
            condition=logic.get('condition'),
            threshold=logic.get('threshold'),
            action=logic.get('action'),
            description=logic.get('description', '')
        )

    def _serialize_decision_tree(self, tree: DecisionTree) -> Dict[str, Any]:
        """Serialize decision tree to dictionary."""
        return asdict(tree)

    def _load_filtering_run(self, run_id: str) -> Optional[FilteringRun]:
        """Load filtering run from storage."""
        try:
            run_file = self.audit_dir / f"run_{run_id}.json"
            if not run_file.exists():
                return None

            with open(run_file, 'r') as f:
                data = json.load(f)

            # Reconstruct FilteringRun object (simplified)
            return FilteringRun(**data['run_metadata'])

        except Exception as e:
            logger.error(f"Error loading filtering run {run_id}: {e}")
            return None

    def _load_events_since(self, since_date: datetime) -> List[AuditEvent]:
        """Load events since a specific date."""
        events = []
        try:
            # Load from daily event files
            current_date = since_date.date()
            while current_date <= datetime.now().date():
                event_file = self.audit_dir / f"events_{current_date.strftime('%Y%m%d')}.jsonl"

                if event_file.exists():
                    with open(event_file, 'r') as f:
                        for line in f:
                            try:
                                event_data = json.loads(line)
                                event_timestamp = datetime.fromisoformat(event_data['timestamp'])

                                if event_timestamp >= since_date:
                                    # Reconstruct AuditEvent (simplified)
                                    event = AuditEvent(
                                        event_id=event_data['event_id'],
                                        timestamp=event_timestamp,
                                        event_type=EventType(event_data['event_type']),
                                        decision_type=DecisionType(event_data['decision_type']),
                                        wallet_address=event_data.get('wallet_address'),
                                        input_data=event_data['input_data'],
                                        output_data=event_data['output_data'],
                                        parameters=event_data['parameters'],
                                        process_id=event_data['process_id'],
                                        user_id=event_data.get('user_id'),
                                        system_version=event_data['system_version'],
                                        decision_rationale=event_data['decision_rationale'],
                                        confidence_score=event_data.get('confidence_score'),
                                        data_hash=event_data['data_hash'],
                                        config_hash=event_data['config_hash']
                                    )
                                    events.append(event)
                            except Exception as e:
                                logger.warning(f"Error parsing event line: {e}")
                                continue

                current_date += timedelta(days=1)

        except Exception as e:
            logger.error(f"Error loading events: {e}")

        return events