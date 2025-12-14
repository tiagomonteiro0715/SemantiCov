#!/usr/bin/env python3
"""
Semantic Coverage Analyzer
==========================
A tool that analyzes Python code coverage with semantic understanding.
Uses Ollama Mistral:7b for AI-powered code classification.

Features:
- Function extraction with line number tracking
- AI-powered classification into 9 semantic block types
- Block importance weighting
- Dependency analysis between blocks
- Traditional + semantic coverage calculation
- Meaningful quality metrics reporting

Usage:
  python main.py
  python main.py ./my_project
  python main.py ./src --output ./reports

Requirements:
  - Ollama running locally (ollama serve)
  - Mistral model installed (ollama pull mistral:7b)
  - Test files named test_*.py in your project
"""

import sys
import argparse
from pathlib import Path

from analyzers import (
    FunctionExtractor,
    CodeClassifier,
    DependencyAnalyzer,
    CoverageAnalyzer,
    ReportGenerator
)


# Default folder to test when no argument is provided
FOLDER_TO_TEST = "example_project_1"


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
class SemanticCoverageAnalyzer:
    """Main orchestrator for the semantic coverage analysis."""

    def __init__(self, project_dir: str, output_dir: str = None, model_name: str = "mistral:7b"):
        self.project_dir = Path(project_dir).resolve()
        self.output_dir = Path(output_dir) if output_dir else self.project_dir / "coverage_report"

        self.extractor = FunctionExtractor(str(self.project_dir))
        self.classifier = CodeClassifier(model_name)
        self.dependency_analyzer = DependencyAnalyzer()
        self.coverage_analyzer = CoverageAnalyzer(str(self.project_dir))

    def run(self):
        """Run the complete analysis pipeline."""
        print("=" * 80)
        print("SEMANTIC COVERAGE ANALYZER")
        print(f"Project: {self.project_dir}")
        print("=" * 80)

        # Step 1: Extract functions
        functions = self.extractor.extract_all()
        if not functions:
            print("\n[ERROR] No functions found to analyze.")
            return

        # Step 2: Classify code with AI
        functions = self.classifier.classify_all(functions)

        # Step 3: Analyze dependencies
        dependency_data = self.dependency_analyzer.analyze(functions)

        # Step 4: Run tests and collect coverage
        self.coverage_analyzer.run_tests_with_coverage()

        # Step 5: Load coverage data
        self.coverage_analyzer.load_coverage_data()

        # Step 6: Calculate semantic coverage
        coverage_results = self.coverage_analyzer.calculate_semantic_coverage(functions)

        # Generate reports
        reporter = ReportGenerator(functions, coverage_results, dependency_data)
        reporter.print_summary_report()

        # Save JSON report
        self.output_dir.mkdir(parents=True, exist_ok=True)
        reporter.save_json_report(str(self.output_dir / "semantic_coverage.json"))

        return coverage_results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Semantic Coverage Analyzer - AI-powered code coverage analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py ./my_project
  python main.py ./src --output ./reports

Requirements:
  - Ollama running locally (ollama serve)
  - Mistral model installed (ollama pull mistral:7b)
  - Test files named test_*.py in your project
        """
    )
    parser.add_argument("project_dir", nargs="?", default=FOLDER_TO_TEST, 
                        help="Path to the Python project directory (default: example_project)")
    parser.add_argument("--output", "-o", help="Output directory for reports", default=None)
    parser.add_argument("--model", "-m", help="Ollama model to use (default: mistral:7b)", default="mistral:7b")

    args = parser.parse_args()

    if not Path(args.project_dir).exists():
        print(f"[ERROR] Directory not found: {args.project_dir}")
        sys.exit(1)

    analyzer = SemanticCoverageAnalyzer(args.project_dir, args.output, args.model)
    analyzer.run()


if __name__ == "__main__":
    main()