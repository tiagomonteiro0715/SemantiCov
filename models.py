"""
Models and Constants for Semantic Coverage Analyzer
====================================================
Contains data classes, block weights, and descriptions used across the analyzer.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# =============================================================================
# BLOCK WEIGHTS - Criticality scores for each semantic block type
# =============================================================================
BLOCK_WEIGHTS = {
    "core_functionality": 10,
    "error_handling": 9,
    "boundary_conditions": 8,
    "integration_points": 7,
    "security_features": 6,
    "performance_scalability": 5,
    "output_consistency": 4,
    "configuration_environment": 3,
    "ui_interactions": 2,
}

BLOCK_DESCRIPTIONS = {
    "core_functionality": "Core Functionality - Key business logic and expected outputs",
    "error_handling": "Error Handling - Exception management and unexpected conditions",
    "boundary_conditions": "Boundary Conditions - Edge cases and invalid input handling",
    "integration_points": "Integration Points - External system interactions (APIs, DBs)",
    "security_features": "Security Features - Data protection and vulnerability prevention",
    "performance_scalability": "Performance & Scalability - Load handling and optimization",
    "output_consistency": "Output Consistency - Logs, reports, and output accuracy",
    "configuration_environment": "Configuration & Environment - Cross-environment operation",
    "ui_interactions": "UI Interactions - User inputs and experience",
}


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class FunctionInfo:
    """Represents an extracted function with metadata."""
    name: str
    filepath: str
    start_line: int
    end_line: int
    source_code: str
    numbered_code: str
    blocks: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class BlockCoverage:
    """Coverage information for a semantic block."""
    block_type: str
    total_lines: int
    covered_lines: int
    line_numbers: List[int]
    covered_line_numbers: List[int]
    weight: int
    coverage_pct: float
    semantic_score: float


@dataclass
class FunctionCoverage:
    """Coverage information for a function."""
    function_name: str
    filepath: str
    blocks: List[BlockCoverage]
    total_lines: int
    covered_lines: int
    statement_coverage: float
    semantic_coverage: float
    semantic_score: float
    max_semantic_score: float