"""
Manual Review and Edge Case Handling System

This module implements manual review processes for borderline performance cases,
unusual trading patterns, and complex edge cases that require human assessment.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from decimal import Decimal
import logging
from enum import Enum
import json
import hashlib
from collections import defaultdict

from .wallet_filter import FilterResult, FilterReason, WalletData
from .sybil_detector import SybilDetectionResult

logger = logging.getLogger(__name__)


class ReviewStatus(Enum):
    """Status of manual review cases."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_SECONDARY_REVIEW = "needs_secondary_review"
    ESCALATED = "escalated"


class ReviewPriority(Enum):
    """Priority levels for manual review."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ReviewReason(Enum):
    """Reasons for manual review."""
    BORDERLINE_PERFORMANCE = "borderline_performance"
    UNUSUAL_PATTERN = "unusual_pattern"
    SYBIL_SUSPECTED = "sybil_suspected"
    CONFLICTING_SIGNALS = "conflicting_signals"
    HIGH_POTENTIAL = "high_potential"
    EDGE_CASE = "edge_case"
    APPEAL_REQUEST = "appeal_request"


@dataclass
class ReviewCase:
    """Represents a manual review case."""
    case_id: str
    wallet_address: str
    status: ReviewStatus
    priority: ReviewPriority
    reason: ReviewReason

    # Review context
    filter_result: FilterResult
    sybil_result: Optional[SybilDetectionResult] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

    # Review process
    created_timestamp: datetime = field(default_factory=datetime.now)
    assigned_reviewer: Optional[str] = None
    assignment_timestamp: Optional[datetime] = None
    review_deadline: Optional[datetime] = None

    # Review decisions
    reviewer_notes: str = ""
    reviewer_decision: Optional[bool] = None  # True = approve, False = reject
    reviewer_confidence: Optional[float] = None  # 0.0 to 1.0
    override_applied: bool = False

    # Quality assurance
    qa_reviewer: Optional[str] = None
    qa_notes: str = ""
    qa_approved: Optional[bool] = None
    qa_timestamp: Optional[datetime] = None

    # Final outcome
    final_decision: Optional[bool] = None
    final_timestamp: Optional[datetime] = None
    decision_rationale: str = ""


@dataclass
class ReviewMetrics:
    """Metrics for manual review performance."""
    total_cases: int
    pending_cases: int
    completed_cases: int
    approval_rate: float
    avg_review_time_hours: float
    qa_override_rate: float
    reviewer_performance: Dict[str, Dict[str, float]]


@dataclass
class FlaggedPattern:
    """Represents an unusual trading pattern flagged for review."""
    pattern_id: str
    pattern_type: str
    description: str
    severity: float  # 0.0 to 1.0
    evidence: Dict[str, Any]
    auto_generated: bool = True


class ManualReviewSystem:
    """
    Comprehensive manual review system for wallet filtering edge cases.

    This system handles borderline performance cases, unusual trading patterns,
    and complex scenarios that require human judgment.
    """

    def __init__(self,
                 performance_threshold_buffer: float = 0.05,
                 review_queue_max_size: int = 1000,
                 auto_assignment: bool = True,
                 qa_sample_rate: float = 0.2):
        """
        Initialize the manual review system.

        Args:
            performance_threshold_buffer: Buffer around thresholds for borderline cases
            review_queue_max_size: Maximum size of review queue
            auto_assignment: Whether to automatically assign cases to reviewers
            qa_sample_rate: Percentage of cases to route for QA review
        """
        self.performance_buffer = performance_threshold_buffer
        self.max_queue_size = review_queue_max_size
        self.auto_assignment = auto_assignment
        self.qa_sample_rate = qa_sample_rate

        # Review state
        self.review_cases: Dict[str, ReviewCase] = {}
        self.review_queue: List[str] = []  # Case IDs ordered by priority
        self.completed_cases: Dict[str, ReviewCase] = {}

        # Reviewer management
        self.active_reviewers: Set[str] = set()
        self.reviewer_workloads: Dict[str, int] = defaultdict(int)
        self.reviewer_performance: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Pattern detection
        self.flagged_patterns: Dict[str, List[FlaggedPattern]] = defaultdict(list)

        # Audit trail
        self.decision_audit: List[Dict[str, Any]] = []

        logger.info(f"ManualReviewSystem initialized with buffer={performance_threshold_buffer}")

    def evaluate_for_manual_review(self, wallet_address: str,
                                 filter_result: FilterResult,
                                 sybil_result: Optional[SybilDetectionResult] = None,
                                 wallet_data: Optional[WalletData] = None) -> Optional[ReviewCase]:
        """
        Evaluate if a wallet requires manual review and create case if needed.

        Args:
            wallet_address: Wallet address to evaluate
            filter_result: Result from automated filtering
            sybil_result: Sybil detection result if available
            wallet_data: Complete wallet data for pattern analysis

        Returns:
            ReviewCase if manual review is needed, None otherwise
        """
        try:
            # Check if manual review is needed
            review_reasons = self._determine_review_reasons(
                filter_result, sybil_result, wallet_data
            )

            if not review_reasons:
                return None

            # Determine priority based on reasons and scores
            priority = self._calculate_review_priority(review_reasons, filter_result, sybil_result)

            # Create review case
            case_id = self._generate_case_id(wallet_address)
            primary_reason = review_reasons[0]  # Use the first (most important) reason

            review_case = ReviewCase(
                case_id=case_id,
                wallet_address=wallet_address,
                status=ReviewStatus.PENDING,
                priority=priority,
                reason=primary_reason,
                filter_result=filter_result,
                sybil_result=sybil_result,
                additional_data={
                    'all_review_reasons': [r.value for r in review_reasons],
                    'flagged_patterns': self.flagged_patterns.get(wallet_address, [])
                }
            )

            # Set review deadline based on priority
            review_case.review_deadline = self._calculate_review_deadline(priority)

            # Add to review system
            self._add_to_review_queue(review_case)

            # Flag unusual patterns if wallet data available
            if wallet_data:
                self._flag_unusual_patterns(wallet_address, wallet_data)

            logger.info(f"Created review case {case_id} for wallet {wallet_address} "
                       f"with priority {priority.value} and reason {primary_reason.value}")

            return review_case

        except Exception as e:
            logger.error(f"Error evaluating wallet {wallet_address} for manual review: {e}")
            return None

    def assign_case_to_reviewer(self, case_id: str, reviewer_id: str) -> bool:
        """
        Assign a review case to a specific reviewer.

        Args:
            case_id: ID of the case to assign
            reviewer_id: ID of the reviewer

        Returns:
            True if assignment successful, False otherwise
        """
        try:
            if case_id not in self.review_cases:
                logger.error(f"Case {case_id} not found")
                return False

            case = self.review_cases[case_id]

            if case.status != ReviewStatus.PENDING:
                logger.error(f"Case {case_id} is not in pending status")
                return False

            # Update case
            case.assigned_reviewer = reviewer_id
            case.assignment_timestamp = datetime.now()
            case.status = ReviewStatus.IN_REVIEW

            # Update reviewer workload
            self.reviewer_workloads[reviewer_id] += 1
            self.active_reviewers.add(reviewer_id)

            # Log assignment
            self._audit_decision(case_id, "case_assigned", {
                'reviewer': reviewer_id,
                'assignment_time': case.assignment_timestamp.isoformat()
            })

            logger.info(f"Assigned case {case_id} to reviewer {reviewer_id}")
            return True

        except Exception as e:
            logger.error(f"Error assigning case {case_id} to reviewer {reviewer_id}: {e}")
            return False

    def submit_review_decision(self, case_id: str, reviewer_id: str,
                             decision: bool, confidence: float,
                             notes: str = "") -> bool:
        """
        Submit a review decision for a case.

        Args:
            case_id: ID of the case
            reviewer_id: ID of the reviewer making the decision
            decision: True for approve, False for reject
            confidence: Confidence level (0.0 to 1.0)
            notes: Reviewer notes and rationale

        Returns:
            True if submission successful, False otherwise
        """
        try:
            if case_id not in self.review_cases:
                logger.error(f"Case {case_id} not found")
                return False

            case = self.review_cases[case_id]

            if case.assigned_reviewer != reviewer_id:
                logger.error(f"Case {case_id} not assigned to reviewer {reviewer_id}")
                return False

            if case.status != ReviewStatus.IN_REVIEW:
                logger.error(f"Case {case_id} is not in review status")
                return False

            # Update case with review decision
            case.reviewer_decision = decision
            case.reviewer_confidence = confidence
            case.reviewer_notes = notes

            # Determine if QA review is needed
            needs_qa = self._needs_qa_review(case, decision, confidence)

            if needs_qa:
                case.status = ReviewStatus.NEEDS_SECONDARY_REVIEW
                logger.info(f"Case {case_id} flagged for QA review")
            else:
                # Finalize decision
                case.final_decision = decision
                case.final_timestamp = datetime.now()
                case.status = ReviewStatus.APPROVED if decision else ReviewStatus.REJECTED
                case.decision_rationale = notes

                # Move to completed cases
                self._complete_case(case)

            # Update reviewer workload
            self.reviewer_workloads[reviewer_id] -= 1

            # Log decision
            self._audit_decision(case_id, "review_decision_submitted", {
                'reviewer': reviewer_id,
                'decision': decision,
                'confidence': confidence,
                'needs_qa': needs_qa
            })

            logger.info(f"Review decision submitted for case {case_id}: {decision} "
                       f"(confidence: {confidence:.2f})")

            return True

        except Exception as e:
            logger.error(f"Error submitting review decision for case {case_id}: {e}")
            return False

    def apply_manual_override(self, case_id: str, override_decision: bool,
                            override_reason: str, authorized_by: str) -> bool:
        """
        Apply a manual override to a review case decision.

        Args:
            case_id: ID of the case
            override_decision: Override decision (True/False)
            override_reason: Reason for the override
            authorized_by: Who authorized the override

        Returns:
            True if override applied successfully, False otherwise
        """
        try:
            if case_id not in self.review_cases and case_id not in self.completed_cases:
                logger.error(f"Case {case_id} not found")
                return False

            # Get case from appropriate storage
            case = self.review_cases.get(case_id) or self.completed_cases[case_id]

            # Apply override
            original_decision = case.final_decision
            case.final_decision = override_decision
            case.override_applied = True
            case.final_timestamp = datetime.now()
            case.decision_rationale += f"\n\nOVERRIDE: {override_reason} (by {authorized_by})"

            # Update status
            case.status = ReviewStatus.APPROVED if override_decision else ReviewStatus.REJECTED

            # Move to completed cases if not already there
            if case_id in self.review_cases:
                self._complete_case(case)

            # Log override
            self._audit_decision(case_id, "manual_override_applied", {
                'original_decision': original_decision,
                'override_decision': override_decision,
                'override_reason': override_reason,
                'authorized_by': authorized_by
            })

            logger.info(f"Manual override applied to case {case_id}: "
                       f"{original_decision} -> {override_decision}")

            return True

        except Exception as e:
            logger.error(f"Error applying override to case {case_id}: {e}")
            return False

    def perform_qa_review(self, case_id: str, qa_reviewer_id: str,
                         qa_decision: bool, qa_notes: str = "") -> bool:
        """
        Perform quality assurance review on a case.

        Args:
            case_id: ID of the case
            qa_reviewer_id: ID of the QA reviewer
            qa_decision: QA approval decision
            qa_notes: QA reviewer notes

        Returns:
            True if QA review completed successfully, False otherwise
        """
        try:
            if case_id not in self.review_cases:
                logger.error(f"Case {case_id} not found")
                return False

            case = self.review_cases[case_id]

            if case.status != ReviewStatus.NEEDS_SECONDARY_REVIEW:
                logger.error(f"Case {case_id} is not pending QA review")
                return False

            # Update case with QA decision
            case.qa_reviewer = qa_reviewer_id
            case.qa_approved = qa_decision
            case.qa_notes = qa_notes
            case.qa_timestamp = datetime.now()

            # Determine final decision
            if qa_decision:
                # QA approves original decision
                case.final_decision = case.reviewer_decision
                case.status = ReviewStatus.APPROVED if case.reviewer_decision else ReviewStatus.REJECTED
            else:
                # QA overrides original decision
                case.final_decision = not case.reviewer_decision
                case.status = ReviewStatus.APPROVED if not case.reviewer_decision else ReviewStatus.REJECTED

            case.final_timestamp = datetime.now()
            case.decision_rationale = f"Primary review: {case.reviewer_notes}\nQA review: {qa_notes}"

            # Move to completed cases
            self._complete_case(case)

            # Update reviewer performance metrics
            self._update_reviewer_performance(case)

            # Log QA decision
            self._audit_decision(case_id, "qa_review_completed", {
                'qa_reviewer': qa_reviewer_id,
                'qa_decision': qa_decision,
                'final_decision': case.final_decision
            })

            logger.info(f"QA review completed for case {case_id}: QA decision {qa_decision}")

            return True

        except Exception as e:
            logger.error(f"Error performing QA review for case {case_id}: {e}")
            return False

    def get_review_queue(self, reviewer_id: Optional[str] = None,
                        priority_filter: Optional[ReviewPriority] = None) -> List[ReviewCase]:
        """
        Get the current review queue, optionally filtered.

        Args:
            reviewer_id: Filter by assigned reviewer
            priority_filter: Filter by priority level

        Returns:
            List of review cases
        """
        try:
            cases = []

            for case_id in self.review_queue:
                if case_id in self.review_cases:
                    case = self.review_cases[case_id]

                    # Apply filters
                    if reviewer_id and case.assigned_reviewer != reviewer_id:
                        continue

                    if priority_filter and case.priority != priority_filter:
                        continue

                    cases.append(case)

            return cases

        except Exception as e:
            logger.error(f"Error getting review queue: {e}")
            return []

    def get_review_metrics(self) -> ReviewMetrics:
        """Get comprehensive review system metrics."""
        try:
            total_cases = len(self.review_cases) + len(self.completed_cases)
            pending_cases = len([c for c in self.review_cases.values()
                               if c.status == ReviewStatus.PENDING])
            completed_cases = len(self.completed_cases)

            # Calculate approval rate
            completed_with_decisions = [c for c in self.completed_cases.values()
                                     if c.final_decision is not None]
            if completed_with_decisions:
                approval_rate = sum(1 for c in completed_with_decisions
                                  if c.final_decision) / len(completed_with_decisions)
            else:
                approval_rate = 0.0

            # Calculate average review time
            review_times = []
            for case in self.completed_cases.values():
                if case.assignment_timestamp and case.final_timestamp:
                    review_time = (case.final_timestamp - case.assignment_timestamp).total_seconds() / 3600
                    review_times.append(review_time)

            avg_review_time = np.mean(review_times) if review_times else 0.0

            # Calculate QA override rate
            qa_cases = [c for c in self.completed_cases.values() if c.qa_approved is not None]
            if qa_cases:
                qa_overrides = sum(1 for c in qa_cases if c.qa_approved != (c.reviewer_decision == c.final_decision))
                qa_override_rate = qa_overrides / len(qa_cases)
            else:
                qa_override_rate = 0.0

            return ReviewMetrics(
                total_cases=total_cases,
                pending_cases=pending_cases,
                completed_cases=completed_cases,
                approval_rate=approval_rate,
                avg_review_time_hours=avg_review_time,
                qa_override_rate=qa_override_rate,
                reviewer_performance=dict(self.reviewer_performance)
            )

        except Exception as e:
            logger.error(f"Error calculating review metrics: {e}")
            return ReviewMetrics(0, 0, 0, 0.0, 0.0, 0.0, {})

    def _determine_review_reasons(self, filter_result: FilterResult,
                                sybil_result: Optional[SybilDetectionResult],
                                wallet_data: Optional[WalletData]) -> List[ReviewReason]:
        """Determine reasons why a wallet needs manual review."""
        reasons = []

        # Check for borderline performance
        if not filter_result.passed and filter_result.reason == FilterReason.PERFORMANCE_INSUFFICIENT:
            if self._is_borderline_performance(filter_result):
                reasons.append(ReviewReason.BORDERLINE_PERFORMANCE)

        # Check for Sybil suspicion
        if sybil_result and sybil_result.sybil_score > 0.5:
            reasons.append(ReviewReason.SYBIL_SUSPECTED)

        # Check for conflicting signals
        if self._has_conflicting_signals(filter_result, sybil_result):
            reasons.append(ReviewReason.CONFLICTING_SIGNALS)

        # Check for high potential cases
        if self._has_high_potential(filter_result):
            reasons.append(ReviewReason.HIGH_POTENTIAL)

        # Check for unusual patterns
        if wallet_data and self._has_unusual_patterns(wallet_data):
            reasons.append(ReviewReason.UNUSUAL_PATTERN)

        return reasons

    def _is_borderline_performance(self, filter_result: FilterResult) -> bool:
        """Check if performance is borderline (close to thresholds)."""
        scores = filter_result.scores

        # Check if any score is within the buffer of the threshold
        borderline_checks = [
            abs(scores.get('sharpe_ratio', 0) - 0.5) <= self.performance_buffer,
            abs(scores.get('win_rate', 0) - 0.55) <= self.performance_buffer,
            abs(scores.get('total_return', 0) - 0.10) <= self.performance_buffer
        ]

        return any(borderline_checks)

    def _has_conflicting_signals(self, filter_result: FilterResult,
                               sybil_result: Optional[SybilDetectionResult]) -> bool:
        """Check for conflicting signals between different assessments."""
        if not sybil_result:
            return False

        # Example: High performance but high Sybil score
        if (filter_result.scores.get('sharpe_ratio', 0) > 1.0 and
            sybil_result.sybil_score > 0.7):
            return True

        return False

    def _has_high_potential(self, filter_result: FilterResult) -> bool:
        """Check if wallet shows high potential despite failing some criteria."""
        scores = filter_result.scores

        # High performance in some areas
        high_performance_indicators = [
            scores.get('sharpe_ratio', 0) > 1.5,
            scores.get('win_rate', 0) > 0.7,
            scores.get('unique_tokens', 0) > 10
        ]

        return sum(high_performance_indicators) >= 2

    def _has_unusual_patterns(self, wallet_data: WalletData) -> bool:
        """Check for unusual trading patterns that merit review."""
        # This would implement more sophisticated pattern detection
        # For now, simple heuristics

        if len(wallet_data.trades) == 0:
            return False

        # Unusual volume spikes
        volumes = [float(trade.amount) for trade in wallet_data.trades]
        if volumes:
            volume_cv = np.std(volumes) / np.mean(volumes)
            if volume_cv > 3.0:  # Very high coefficient of variation
                return True

        return False

    def _calculate_review_priority(self, reasons: List[ReviewReason],
                                 filter_result: FilterResult,
                                 sybil_result: Optional[SybilDetectionResult]) -> ReviewPriority:
        """Calculate priority level for a review case."""
        priority_score = 0

        # Base score from reasons
        reason_weights = {
            ReviewReason.SYBIL_SUSPECTED: 3,
            ReviewReason.CONFLICTING_SIGNALS: 2,
            ReviewReason.HIGH_POTENTIAL: 2,
            ReviewReason.BORDERLINE_PERFORMANCE: 1,
            ReviewReason.UNUSUAL_PATTERN: 1,
            ReviewReason.EDGE_CASE: 1
        }

        for reason in reasons:
            priority_score += reason_weights.get(reason, 1)

        # Adjust based on scores
        if sybil_result and sybil_result.sybil_score > 0.8:
            priority_score += 2

        # Determine priority level
        if priority_score >= 5:
            return ReviewPriority.URGENT
        elif priority_score >= 3:
            return ReviewPriority.HIGH
        elif priority_score >= 2:
            return ReviewPriority.MEDIUM
        else:
            return ReviewPriority.LOW

    def _calculate_review_deadline(self, priority: ReviewPriority) -> datetime:
        """Calculate review deadline based on priority."""
        hours_mapping = {
            ReviewPriority.URGENT: 4,
            ReviewPriority.HIGH: 24,
            ReviewPriority.MEDIUM: 72,
            ReviewPriority.LOW: 168  # 1 week
        }

        hours = hours_mapping[priority]
        return datetime.now() + timedelta(hours=hours)

    def _generate_case_id(self, wallet_address: str) -> str:
        """Generate unique case ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        address_hash = hashlib.md5(wallet_address.encode()).hexdigest()[:8]
        return f"REVIEW_{timestamp}_{address_hash}"

    def _add_to_review_queue(self, case: ReviewCase) -> None:
        """Add case to review queue with priority ordering."""
        self.review_cases[case.case_id] = case

        # Insert in priority order
        priority_order = [ReviewPriority.URGENT, ReviewPriority.HIGH,
                         ReviewPriority.MEDIUM, ReviewPriority.LOW]

        insert_index = len(self.review_queue)
        for i, existing_case_id in enumerate(self.review_queue):
            existing_case = self.review_cases[existing_case_id]
            existing_priority_index = priority_order.index(existing_case.priority)
            new_priority_index = priority_order.index(case.priority)

            if new_priority_index < existing_priority_index:
                insert_index = i
                break

        self.review_queue.insert(insert_index, case.case_id)

        # Auto-assign if enabled and reviewers available
        if self.auto_assignment and self.active_reviewers:
            self._auto_assign_case(case.case_id)

    def _auto_assign_case(self, case_id: str) -> None:
        """Automatically assign case to available reviewer."""
        if not self.active_reviewers:
            return

        # Find reviewer with lowest workload
        available_reviewer = min(self.active_reviewers,
                               key=lambda r: self.reviewer_workloads[r])

        self.assign_case_to_reviewer(case_id, available_reviewer)

    def _needs_qa_review(self, case: ReviewCase, decision: bool, confidence: float) -> bool:
        """Determine if a case needs QA review."""
        # Low confidence decisions
        if confidence < 0.7:
            return True

        # Random sampling for QA
        if np.random.random() < self.qa_sample_rate:
            return True

        # High-priority cases
        if case.priority in [ReviewPriority.URGENT, ReviewPriority.HIGH]:
            return True

        return False

    def _complete_case(self, case: ReviewCase) -> None:
        """Move case from active to completed."""
        if case.case_id in self.review_cases:
            del self.review_cases[case.case_id]

        if case.case_id in self.review_queue:
            self.review_queue.remove(case.case_id)

        self.completed_cases[case.case_id] = case

    def _update_reviewer_performance(self, case: ReviewCase) -> None:
        """Update reviewer performance metrics."""
        if not case.assigned_reviewer:
            return

        reviewer_id = case.assigned_reviewer

        # Initialize metrics if needed
        if reviewer_id not in self.reviewer_performance:
            self.reviewer_performance[reviewer_id] = {
                'cases_reviewed': 0,
                'avg_confidence': 0.0,
                'qa_agreement_rate': 0.0,
                'avg_review_time_hours': 0.0
            }

        metrics = self.reviewer_performance[reviewer_id]
        metrics['cases_reviewed'] += 1

        # Update confidence
        if case.reviewer_confidence:
            metrics['avg_confidence'] = (
                (metrics['avg_confidence'] * (metrics['cases_reviewed'] - 1) +
                 case.reviewer_confidence) / metrics['cases_reviewed']
            )

        # Update QA agreement if applicable
        if case.qa_approved is not None:
            qa_agreed = case.qa_approved == (case.reviewer_decision == case.final_decision)
            metrics['qa_agreement_rate'] = (
                (metrics['qa_agreement_rate'] * (metrics['cases_reviewed'] - 1) +
                 (1 if qa_agreed else 0)) / metrics['cases_reviewed']
            )

    def _flag_unusual_patterns(self, wallet_address: str, wallet_data: WalletData) -> None:
        """Flag unusual trading patterns for a wallet."""
        patterns = []

        if not wallet_data.trades:
            return

        # Pattern 1: Extreme volume variations
        volumes = [float(trade.amount) for trade in wallet_data.trades]
        if volumes:
            volume_cv = np.std(volumes) / np.mean(volumes)
            if volume_cv > 2.0:
                patterns.append(FlaggedPattern(
                    pattern_id=f"volume_variation_{wallet_address}",
                    pattern_type="volume_anomaly",
                    description=f"Extreme volume variation (CV: {volume_cv:.2f})",
                    severity=min(volume_cv / 5.0, 1.0),
                    evidence={'coefficient_of_variation': volume_cv, 'trade_count': len(volumes)}
                ))

        # Pattern 2: Unusual timing patterns
        timestamps = [trade.timestamp for trade in wallet_data.trades]
        if len(timestamps) > 5:
            # Check for clustering in specific hours
            hours = [ts.hour for ts in timestamps]
            hour_counts = Counter(hours)
            max_hour_ratio = max(hour_counts.values()) / len(hours)

            if max_hour_ratio > 0.5:  # More than 50% of trades in one hour
                patterns.append(FlaggedPattern(
                    pattern_id=f"timing_cluster_{wallet_address}",
                    pattern_type="temporal_anomaly",
                    description=f"Trades clustered in specific hour ({max_hour_ratio:.1%})",
                    severity=max_hour_ratio,
                    evidence={'max_hour_ratio': max_hour_ratio, 'hour_distribution': dict(hour_counts)}
                ))

        self.flagged_patterns[wallet_address] = patterns

    def _audit_decision(self, case_id: str, action: str, details: Dict[str, Any]) -> None:
        """Log decision to audit trail."""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'case_id': case_id,
            'action': action,
            'details': details
        }

        self.decision_audit.append(audit_entry)

        # Keep audit trail manageable
        if len(self.decision_audit) > 10000:
            self.decision_audit = self.decision_audit[-5000:]  # Keep last 5000 entries