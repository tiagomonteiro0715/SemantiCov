"""
Analyzers for Semantic Coverage Analyzer
========================================
Contains all analyzer classes:
- FunctionExtractor: Extracts functions from Python source files
- CodeClassifier: AI-powered classification using Ollama Mistral:7b
- DependencyAnalyzer: Analyzes variable dependencies between blocks
- CoverageAnalyzer: Collects and analyzes code coverage
- ReportGenerator: Generates coverage reports
"""

import os
import sys
import ast
import re
import json
import builtins
import keyword
import subprocess
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

from models import (
    FunctionInfo, BlockCoverage, FunctionCoverage,
    BLOCK_WEIGHTS, BLOCK_DESCRIPTIONS
)
from cache_manager import CacheManager
import concurrent.futures

# Import dependencies (install if needed)
try:
    import coverage
except ImportError:
    print("Installing coverage library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "coverage", "-q"])
    import coverage

try:
    import networkx as nx
except ImportError:
    print("Installing networkx library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx", "-q"])
    import networkx as nx

try:
    import requests
except ImportError:
    print("Installing requests library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests


# =============================================================================
# FUNCTION EXTRACTION
# =============================================================================
class FunctionExtractor:
    """Extracts functions from Python source files with line numbers."""

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.functions: List[FunctionInfo] = []

    def get_python_files(self) -> List[Path]:
        """Get all Python files in the project directory."""
        py_files = []
        for root, _, files in os.walk(self.project_dir):
            for file in files:
                if file.endswith(".py") and not file.startswith("test_") and not file.startswith("."):
                    py_files.append(Path(root) / file)
        return py_files

    def extract_functions_from_file(self, filepath: Path) -> List[FunctionInfo]:
        """Extract all functions from a single Python file."""
        functions = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)
            lines = source.splitlines()

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    try:
                        start_line = node.lineno
                        end_line = node.body[-1].end_lineno if node.body else node.lineno
                        func_code = "\n".join(lines[start_line - 1:end_line])
                        numbered_code = self._add_line_numbers(func_code)

                        func_info = FunctionInfo(
                            name=node.name,
                            filepath=str(filepath),
                            start_line=start_line,
                            end_line=end_line,
                            source_code=func_code,
                            numbered_code=numbered_code
                        )
                        functions.append(func_info)
                    except Exception as e:
                        print(f"  [WARN] Skipping function {node.name}: {e}")

        except Exception as e:
            print(f"  [ERROR] Could not parse {filepath}: {e}")

        return functions

    def _add_line_numbers(self, func_code: str) -> str:
        """Add line numbers to function code for tracking."""
        lines = func_code.split("\n")
        code_line_number = 0
        numbered_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip function definition line
            if stripped.startswith("def "):
                numbered_lines.append(line)
                continue

            # Skip empty lines, comments, and docstrings
            if stripped and not stripped.startswith("#") and not stripped.startswith('"""') and not stripped.startswith("'''"):
                code_line_number += 1
                numbered_lines.append(f"#{code_line_number} {line}")
            else:
                numbered_lines.append(line)

        return "\n".join(numbered_lines)

    def extract_all(self) -> List[FunctionInfo]:
        """Extract all functions from the project."""
        print(f"\n[1/6] EXTRACTING FUNCTIONS from {self.project_dir}")
        print("-" * 60)

        py_files = self.get_python_files()
        print(f"  Found {len(py_files)} Python files")

        for filepath in py_files:
            funcs = self.extract_functions_from_file(filepath)
            self.functions.extend(funcs)
            if funcs:
                print(f"  {filepath.name}: {len(funcs)} functions")

        print(f"  Total functions extracted: {len(self.functions)}")
        return self.functions


# =============================================================================
# AI-POWERED CODE CLASSIFICATION (Ollama Mistral:7b)
# =============================================================================
class CodeClassifier:
    """Classifies code lines into semantic blocks using Ollama Mistral:7b."""

    OLLAMA_URL = "http://localhost:11434/api/generate"

    def __init__(self, model_name: str = "mistral:7b"):
        self.model = model_name
        self.cache = CacheManager()

    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                # Check for exact match or if the requested model is a prefix of an available model
                # e.g. "llama3" matches "llama3:latest"
                if any(self.model == name or name.startswith(self.model + ":") for name in model_names):
                    print(f"  [INFO] Using Ollama model: {self.model}")
                    return True
                    
                # Also try fuzzy match if exact/prefix failed, for backward compatibility with "mistral"
                if any(self.model in name for name in model_names):
                    print(f"  [INFO] Found model matching '{self.model}'")
                    return True

                print(f"  [WARN] Model '{self.model}' not found. Available models: {model_names}")
                print(f"  Run: ollama pull {self.model}")
                return False
        except requests.exceptions.ConnectionError:
            print("  [ERROR] Ollama is not running. Start it with: ollama serve")
            return False
        return False

    def _build_prompt(self, func_info: FunctionInfo) -> str:
        """Build the classification prompt for a function."""
        return f"""You are a code analyzer. Categorize each numbered line of this Python function into exactly ONE of these 9 categories:

1. core_functionality - Key business logic, main computations, return statements with results
2. error_handling - Try/except blocks, raise statements, error checking
3. boundary_conditions - Edge case handling, input validation, limit checks
4. integration_points - API calls, database operations, file I/O, external services
5. security_features - Authentication, authorization, encryption, sanitization
6. performance_scalability - Caching, optimization, async operations, batching
7. output_consistency - Logging, printing, formatting output, reports
8. configuration_environment - Config loading, environment variables, setup
9. ui_interactions - User input handling, display formatting, prompts

Function to analyze:
```python
{func_info.numbered_code}
```

RULES:
- Every numbered line (lines starting with #1, #2, etc.) MUST be categorized
- Each line belongs to exactly ONE category
- Group consecutive lines of the same category together
- Output ONLY valid JSON, no other text

Output format (JSON only):
{{
    "core_functionality": ["#1 line content", "#2 line content"],
    "error_handling": ["#3 line content"],
    ...
}}

Respond with ONLY the JSON object, nothing else."""

    def _parse_response(self, response_text: str) -> Dict[str, List[str]]:
        """Parse the JSON response from Ollama."""
        # Try to extract JSON from the response
        try:
            # First, try direct JSON parse
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        try:
            cleaned_text = re.sub(r',\s*([\]}])', r'\1', response_text)
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass
            
        # Return None if parsing fails
        print(f"    [WARN] Failed to parse JSON. Raw response:\n{response_text[:500]}...")
        return None

    def classify_function(self, func_info: FunctionInfo) -> Tuple[Dict[str, List[str]], bool]:
        """Classify a single function's lines into semantic blocks. Returns (blocks, is_cached)."""
        
        # 1. Check Cache
        cached_result = self.cache.get_classification(func_info.numbered_code, self.model)
        if cached_result:
            return cached_result, True

        # 2. Build Prompt
        prompt = self._build_prompt(func_info)

        try:
            response = requests.post(
                self.OLLAMA_URL,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "format": "json", # Enforce JSON mode
                    "stream": False,
                    "options": {
                        "temperature": 0.0, # Enforce determinism
                        "num_predict": 2000
                    }
                },
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                blocks = self._parse_response(response_text)
                
                # 3. Save to Cache if valid (even if empty)
                if blocks is not None:
                    self.cache.save_classification(func_info.numbered_code, self.model, blocks)
                
                # Return empty dict for None to avoid breaking callers
                return blocks or {}, False
            else:
                print(f"    [ERROR] Ollama returned status {response.status_code}")
                return {}, False

        except Exception as e:
            print(f"    [ERROR] Classification failed for {func_info.name}: {e}")
            return {}, False

    def classify_all(self, functions: List[FunctionInfo]) -> List[FunctionInfo]:
        """Classify all functions using parallel processing."""
        print(f"\n[2/6] CLASSIFYING CODE with Ollama {self.model}")
        print("-" * 60)

        if not self._check_ollama_available():
            print("  [ERROR] Cannot proceed without Ollama. Using fallback classification.")
            return self._fallback_classification(functions)

        # Use ThreadPoolExecutor for parallel processing
        # Adjust max_workers based on typical LLM server capacity or CPU cores
        # For local Ollama, 1-4 is usually good. Too many might overload it.
        max_workers = 4
        print(f"  [INFO] Starting parallel classification with {max_workers} workers...")

        completed_count = 0
        total_functions = len(functions)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a dictionary to map futures to functions
            future_to_func = {
                executor.submit(self.classify_function, func): func 
                for func in functions
            }

            for future in concurrent.futures.as_completed(future_to_func):
                func = future_to_func[future]
                completed_count += 1
                try:
                    blocks, is_cached = future.result()
                    func.blocks = blocks
                    block_count = sum(len(lines) for lines in func.blocks.values())
                    status = "Cached" if is_cached else "Generated"
                    print(f"  [{completed_count}/{total_functions}] {func.name}: {block_count} lines ({status})")
                except Exception as e:
                    print(f"  [{completed_count}/{total_functions}] {func.name}: [ERROR] {e}")
                    func.blocks = {} # fallback empty

        return functions

    def _fallback_classification(self, functions: List[FunctionInfo]) -> List[FunctionInfo]:
        """Simple rule-based fallback classification."""
        for func in functions:
            blocks = defaultdict(list)
            lines = func.numbered_code.split("\n")

            for line in lines:
                if not line.strip() or line.strip().startswith("def "):
                    continue

                match = re.match(r"(#\d+\s+)(.*)", line)
                if not match:
                    continue

                prefix, content = match.groups()
                stripped = content.strip()

                # Simple rule-based classification
                if any(kw in stripped for kw in ["try:", "except", "raise", "finally:"]):
                    blocks["error_handling"].append(line)
                elif any(kw in stripped for kw in ["if", "elif", "else:"]) and any(c in stripped for c in ["not", "is None", "< ", "> ", "==", "!="]):
                    blocks["boundary_conditions"].append(line)
                elif any(kw in stripped for kw in ["requests.", "open(", "connect", "cursor", "execute"]):
                    blocks["integration_points"].append(line)
                elif any(kw in stripped for kw in ["print(", "logging.", "logger.", "log("]):
                    blocks["output_consistency"].append(line)
                elif any(kw in stripped for kw in ["config", "environ", "settings", "CONFIG"]):
                    blocks["configuration_environment"].append(line)
                elif any(kw in stripped for kw in ["input(", "prompt", "click.", "argparse"]):
                    blocks["ui_interactions"].append(line)
                elif any(kw in stripped for kw in ["encrypt", "decrypt", "hash", "token", "auth", "password"]):
                    blocks["security_features"].append(line)
                elif any(kw in stripped for kw in ["cache", "async", "await", "thread", "pool"]):
                    blocks["performance_scalability"].append(line)
                else:
                    blocks["core_functionality"].append(line)

            func.blocks = dict(blocks)

        return functions


# =============================================================================
# DEPENDENCY ANALYSIS
# =============================================================================
class DependencyAnalyzer:
    """Analyzes variable dependencies between semantic blocks."""

    def __init__(self):
        self.python_keywords = set(keyword.kwlist)
        self.builtins = set(dir(builtins))
        self.common_methods = {"strip", "title", "lower", "upper", "replace", "find", "split",
                               "append", "extend", "pop", "get", "items", "keys", "values",
                               "join", "format", "encode", "decode"}

    def extract_variables(self, code_lines: List[str]) -> Set[str]:
        """Extract variable names from code lines using AST."""
        variables = set()
        
        # Combine lines into a single string, stripping line numbers
        clean_lines = []
        for line in code_lines:
            # Remove line number prefix (#123 )
            clean_line = re.sub(r'^#\d+\s*', '', line)
            clean_lines.append(clean_line)
            
        source = "\n".join(clean_lines)
        
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    name = node.id
                    if name in self.python_keywords or name in self.builtins:
                        continue
                    variables.add(name)
        except SyntaxError:
            # Fallback to simple regex if AST parsing fails (e.g. partial code)
            for line in clean_lines:
                 matches = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', line)
                 for match in matches:
                     if match not in self.python_keywords and match not in self.builtins:
                         variables.add(match)
                         
        return variables

    def build_dependency_graph(self, functions: List[FunctionInfo]) -> nx.DiGraph:
        """Build a dependency graph across all functions and blocks using inverted index."""
        G = nx.DiGraph()
        
        # 1. Build Inverted Index: Variable -> List of (NodeName, FunctionName, BlockType)
        var_to_blocks = defaultdict(list)
        
        # Add all nodes first
        for func in functions:
            for block_type, lines in func.blocks.items():
                node_name = f"{func.name}:{block_type}"
                variables = self.extract_variables(lines)
                
                # Add node to graph
                G.add_node(node_name, function=func.name, block_type=block_type, variables=variables)
                
                # Populate inverted index
                for var in variables:
                    var_to_blocks[var].append(node_name)

        # 2. systematic edge creation
        # For each variable, connect all blocks that use it
        # Note: This creates a clique for each variable. 
        # To avoid explosion, we might want to directionality or just undirected connectivity.
        # But original logic was: if they share vars, they are connected.
        
        # Original logic was roughly: if A and B share vars, connect A-B and B-A (implied by nested loop?)
        # Actually original was DiGraph but logic `nodes[i+1:]` implies undirected checking? No, DiGraph adds directed edges by default.
        # But `G.add_edge(node1, node2)` only adds one way. 
        # The original code `for node2 in nodes[i+1:]` checks pair (A, B). If shared, adds A->B. 
        # It missed B->A if the relationship is symmetric. 
        # Let's assume symmetric dependency for "sharing variables".
        
        # Optimized approach:
        edge_weights = defaultdict(int)
        
        for var, nodes in var_to_blocks.items():
            if len(nodes) < 2:
                continue
            
            # Connect all pairs in this list
            # If list is too long, this is still O(N^2) for that variable. 
            # But usually variables are local or shared by few.
            # Large globals might be an issue.
            
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    n1, n2 = nodes[i], nodes[j]
                    # Sort to ensure consistent key for undirected weight accumulation
                    u, v = sorted((n1, n2))
                    edge_weights[(u, v)] += 1

        # Add edges to graph
        for (u, v), weight in edge_weights.items():
            # Add bidirectional edges since they "share" variables
            G.add_edge(u, v, weight=weight)
            G.add_edge(v, u, weight=weight)

        return G

    def analyze(self, functions: List[FunctionInfo]) -> Dict:
        """Perform dependency analysis."""
        print(f"\n[3/6] ANALYZING DEPENDENCIES")
        print("-" * 60)

        G = self.build_dependency_graph(functions)

        # Calculate metrics
        block_dependencies = {}
        for node in G.nodes():
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
            block_dependencies[node] = {
                "in_degree": in_deg,
                "out_degree": out_deg,
                "total_connections": in_deg + out_deg
            }

        # Sort by connectivity
        sorted_blocks = sorted(block_dependencies.items(), key=lambda x: x[1]["total_connections"], reverse=True)

        print(f"  Total blocks analyzed: {len(G.nodes())}")
        print(f"  Total dependencies found: {len(G.edges())}")

        if sorted_blocks:
            print(f"  Most connected blocks:")
            for block, deps in sorted_blocks[:5]:
                print(f"    - {block}: {deps['total_connections']} connections")

        return {
            "graph": G,
            "block_dependencies": block_dependencies,
            "sorted_blocks": sorted_blocks
        }


# =============================================================================
# COVERAGE COLLECTION & SEMANTIC CALCULATION
# =============================================================================
class CoverageAnalyzer:
    """Collects and analyzes code coverage with semantic weighting."""

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.cov = coverage.Coverage(data_file=str(self.project_dir / ".coverage"))
        self.covered_lines: Dict[str, Set[int]] = {}

    def find_test_files(self) -> List[Path]:
        """Find all test files in the project."""
        test_files = []
        for root, _, files in os.walk(self.project_dir):
            for file in files:
                # Match test_*.py, *_test.py, and *tests.py patterns
                is_test = (
                    file.startswith("test_") or 
                    file.endswith("_test.py") or 
                    file.endswith("_tests.py") or
                    file == "tests.py"
                )
                if is_test and file.endswith(".py"):
                    test_files.append(Path(root) / file)
        return test_files

    def _check_test_imports(self, test_file: Path, source_modules: List[str]) -> List[str]:
        """Check if test file imports match available source modules."""
        warnings = []
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Find import statements
            import_pattern = r'(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            imports = re.findall(import_pattern, content)
            
            for imp in imports:
                # Skip standard library and common packages
                if imp in ['unittest', 'pytest', 'os', 'sys', 're', 'json', 'random', 
                          'typing', 'collections', 'datetime', 'math', 'itertools']:
                    continue
                # Check if import matches a source module
                if imp not in source_modules and imp != test_file.stem:
                    warnings.append(f"Import '{imp}' not found in project (available: {source_modules})")
        except Exception:
            pass
        return warnings

    def run_tests_with_coverage(self) -> bool:
        """Run tests and collect coverage data using subprocess for isolation."""
        print(f"\n[4/6] RUNNING TESTS WITH COVERAGE")
        print("-" * 60)

        test_files = self.find_test_files()
        if not test_files:
            print("  [WARN] No test files found")
            return False

        print(f"  Found {len(test_files)} test file(s). Running with coverage...")

        # 1. Analyze for needed aliases (legacy support for example projects)
        source_modules = []
        for root, _, files in os.walk(self.project_dir):
            for file in files:
                if file.endswith(".py") and not any(p in file for p in ["test_", "_test", "tests"]):
                    source_modules.append(file.replace(".py", ""))
        
        unknown_imports = set()
        for tf in test_files:
            warnings = self._check_test_imports(tf, source_modules)
            for w in warnings:
                match = re.search(r"Import '(\w+)'", w)
                if match:
                    imp_name = match.group(1)
                    if imp_name not in ['TestCase', 'main', 'unittest', 'pytest']:
                        unknown_imports.add(imp_name)

        alias_map = {}
        if unknown_imports and source_modules:
            primary = source_modules[0]
            for unknown in unknown_imports:
                alias_map[unknown] = primary
            print(f"  [INFO] Will alias {alias_map} for legacy support")

        # 2. Create a temporary custom test runner script
        runner_content = f"""
import sys
import unittest
import os
import importlib.util

def setup_aliases():
    aliases = {json.dumps(alias_map)}
    if not aliases:
        return
        
    # Load primary module
    for alias, target in aliases.items():
        if target in sys.modules:
            sys.modules[alias] = sys.modules[target]
            continue
            
        # Try to load target if not loaded
        try:
             # Basic import attempt
             module = __import__(target)
             sys.modules[alias] = module
             print(f"[Runner] Aliased '{{alias}}' to '{{target}}'")
        except ImportError:
             print(f"[Runner] Failed to alias '{{alias}}' to '{{target}}'")

if __name__ == "__main__":
    # Add current dir to path
    sys.path.insert(0, os.getcwd())
    
    setup_aliases()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Discover and load
    discovered = loader.discover(start_dir=".", pattern="test_*.py")
    suite.addTests(discovered)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    sys.exit(0 if result.wasSuccessful() else 1)
"""
        runner_path = self.project_dir / "_semantic_coverage_runner.py"
        try:
            with open(runner_path, "w") as f:
                f.write(runner_content)
                
            # 3. Run coverage on the runner
            cmd = [
                sys.executable, "-m", "coverage", "run", 
                "--source", ".",
                "_semantic_coverage_runner.py"
            ]
            
            result = subprocess.run(
                cmd, 
                cwd=self.project_dir,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)

            if result.returncode != 0:
                print(f"  [WARN] Tests failed or had errors")
            
            if (self.project_dir / ".coverage").exists():
                 print("  ✓ Coverage data generated")
                 return True
            else:
                 print("  [ERROR] No .coverage file generated")
                 return False

        except Exception as e:
            print(f"  [ERROR] Failed to run tests: {e}")
            return False
        finally:
            # Cleanup
            if runner_path.exists():
                runner_path.unlink()

    def load_coverage_data(self) -> Dict[str, Set[int]]:
        """Load coverage data for all source files using canonical paths."""
        print(f"\n[5/6] LOADING COVERAGE DATA")
        print("-" * 60)

        try:
            self.cov.load()
            data = self.cov.get_data()

            measured = data.measured_files()
            print(f"  Files with coverage data: {len(measured)}")

            for filepath in measured:
                lines = data.lines(filepath)
                if lines:
                    # Resolve to canonical absolute path
                    canonical_path = str(Path(filepath).resolve())
                    self.covered_lines[canonical_path] = set(lines)
                    
                    try:
                        rel_path = Path(filepath).relative_to(Path.cwd())
                    except ValueError:
                        rel_path = filepath
                    
                    print(f"    {rel_path}: {len(lines)} lines covered")

            if not self.covered_lines:
                print("  [WARN] No coverage data found!")
                print("  [INFO] This could mean:")
                print("         - Tests didn't import/call the source code")
                print("         - Source files were not in coverage scope")

        except Exception as e:
            print(f"  [WARN] Could not load coverage data: {e}")
            import traceback
            traceback.print_exc()

        return self.covered_lines

    def calculate_semantic_coverage(self, functions: List[FunctionInfo]) -> List[FunctionCoverage]:
        """Calculate semantic coverage for all functions."""
        print(f"\n[6/6] CALCULATING SEMANTIC COVERAGE")
        print("-" * 60)

        results = []

        for func in functions:
            # Use canonical path matching
            filepath = str(Path(func.filepath).resolve())
            
            file_covered = self.covered_lines.get(filepath, set())
            
            if file_covered:
                print(f"  {func.name}: Found coverage data ({len(file_covered)} lines)")
            else:
                # Fallback: try filename match if path resolution fails (e.g. if coverage uses rel paths)
                stem = Path(filepath).stem
                found = False
                for cov_path, lines in self.covered_lines.items():
                    if Path(cov_path).stem == stem:
                        file_covered = lines
                        found = True
                        print(f"  {func.name}: Matched by filename ({len(file_covered)} lines)")
                        break
                if not found:
                     print(f"  {func.name}: No coverage data found")

            block_coverages = []
            total_func_lines = 0
            total_func_covered = 0
            total_semantic_score = 0
            max_semantic_score = 0

            for block_type, lines in func.blocks.items():
                # Extract actual line numbers from numbered lines
                block_line_nums = []
                for line in lines:
                    match = re.match(r'#(\d+)', line)
                    if match:
                        relative_num = int(match.group(1))
                        # Convert relative to absolute line number
                        actual_line = func.start_line + relative_num
                        block_line_nums.append(actual_line)

                # Check intersection with covered lines
                covered_in_block = [ln for ln in block_line_nums if ln in file_covered]
                weight = BLOCK_WEIGHTS.get(block_type, 1)

                coverage_pct = (len(covered_in_block) / len(block_line_nums) * 100) if block_line_nums else 0
                semantic_score = (coverage_pct / 100) * weight

                block_cov = BlockCoverage(
                    block_type=block_type,
                    total_lines=len(block_line_nums),
                    covered_lines=len(covered_in_block),
                    line_numbers=block_line_nums,
                    covered_line_numbers=covered_in_block,
                    weight=weight,
                    coverage_pct=coverage_pct,
                    semantic_score=semantic_score
                )
                block_coverages.append(block_cov)

                total_func_lines += len(block_line_nums)
                total_func_covered += len(covered_in_block)
                total_semantic_score += semantic_score
                max_semantic_score += weight

            stmt_coverage = (total_func_covered / total_func_lines * 100) if total_func_lines else 0
            sem_coverage = (total_semantic_score / max_semantic_score * 100) if max_semantic_score else 0

            func_cov = FunctionCoverage(
                function_name=func.name,
                filepath=func.filepath,
                blocks=block_coverages,
                total_lines=total_func_lines,
                covered_lines=total_func_covered,
                statement_coverage=stmt_coverage,
                semantic_coverage=sem_coverage,
                semantic_score=total_semantic_score,
                max_semantic_score=max_semantic_score
            )
            results.append(func_cov)

        return results


# =============================================================================
# REPORTING
# =============================================================================
class ReportGenerator:
    """Generates coverage reports with semantic insights."""

    def __init__(self, functions: List[FunctionInfo], coverage_results: List[FunctionCoverage],
                 dependency_data: Dict):
        self.functions = functions
        self.coverage_results = coverage_results
        self.dependency_data = dependency_data

    def print_summary_report(self):
        """Print a summary report to console."""
        print("\n" + "=" * 80)
        print("SEMANTIC COVERAGE REPORT")
        print("=" * 80)

        # Function-level summary
        print(f"\n{'Function':<30} | {'Stmt %':<10} | {'Sem %':<10} | {'Score':<10} | {'Priority'}")
        print("-" * 80)

        sorted_results = sorted(self.coverage_results, key=lambda x: x.semantic_coverage)

        for func_cov in sorted_results:
            priority = "HIGH" if func_cov.semantic_coverage < 50 else "MEDIUM" if func_cov.semantic_coverage < 80 else "LOW"
            print(f"{func_cov.function_name:<30} | {func_cov.statement_coverage:<10.1f} | "
                  f"{func_cov.semantic_coverage:<10.1f} | {func_cov.semantic_score:<10.1f} | {priority}")

        # Block-level breakdown
        print("\n" + "-" * 80)
        print("BLOCK-LEVEL BREAKDOWN")
        print("-" * 80)

        for func_cov in sorted_results:
            if not func_cov.blocks:
                continue

            print(f"\n  {func_cov.function_name}:")
            sorted_blocks = sorted(func_cov.blocks, key=lambda b: b.weight, reverse=True)

            for block in sorted_blocks:
                status = "✓" if block.coverage_pct == 100 else "○" if block.coverage_pct > 0 else "✗"
                print(f"    {status} {block.block_type:<30} | {block.covered_lines}/{block.total_lines} lines | "
                      f"{block.coverage_pct:.0f}% | weight={block.weight}")

        # Overall metrics
        total_score = sum(r.semantic_score for r in self.coverage_results)
        max_score = sum(r.max_semantic_score for r in self.coverage_results)
        overall_semantic = (total_score / max_score * 100) if max_score else 0

        total_lines = sum(r.total_lines for r in self.coverage_results)
        covered_lines = sum(r.covered_lines for r in self.coverage_results)
        overall_statement = (covered_lines / total_lines * 100) if total_lines else 0

        print("\n" + "=" * 80)
        print("OVERALL PROJECT METRICS")
        print("=" * 80)
        print(f"  Total Functions Analyzed: {len(self.functions)}")
        print(f"  Total Lines Analyzed:     {total_lines}")
        print(f"  Lines Covered:            {covered_lines}")
        print(f"\n  Statement Coverage:       {overall_statement:.1f}%")
        print(f"  Semantic Coverage:        {overall_semantic:.1f}%")
        print(f"  Semantic Score:           {total_score:.1f} / {max_score:.1f}")

        # Quality assessment
        print("\n" + "-" * 80)
        print("QUALITY ASSESSMENT")
        print("-" * 80)

        if overall_semantic >= 80:
            print("  ✓ EXCELLENT - High-value code paths are well tested")
        elif overall_semantic >= 60:
            print("  ○ GOOD - Most critical code is tested, some gaps remain")
        elif overall_semantic >= 40:
            print("  △ NEEDS IMPROVEMENT - Critical business logic may be undertested")
        else:
            print("  ✗ POOR - Significant gaps in testing critical functionality")

        # Recommendations
        print("\n  Recommendations:")
        low_coverage_blocks = []
        for func_cov in self.coverage_results:
            for block in func_cov.blocks:
                if block.coverage_pct < 50 and block.weight >= 7:
                    low_coverage_blocks.append((func_cov.function_name, block))

        if low_coverage_blocks:
            print("  Priority areas to improve test coverage:")
            for func_name, block in sorted(low_coverage_blocks, key=lambda x: x[1].weight, reverse=True)[:5]:
                print(f"    - {func_name}: {block.block_type} (weight={block.weight}, coverage={block.coverage_pct:.0f}%)")
        else:
            print("    - All high-priority blocks have reasonable coverage")

    def save_json_report(self, output_path: str):
        """Save detailed report as JSON."""
        report = {
            "summary": {
                "total_functions": len(self.functions),
                "total_lines": sum(r.total_lines for r in self.coverage_results),
                "covered_lines": sum(r.covered_lines for r in self.coverage_results),
                "statement_coverage": sum(r.covered_lines for r in self.coverage_results) /
                                      max(sum(r.total_lines for r in self.coverage_results), 1) * 100,
                "semantic_coverage": sum(r.semantic_score for r in self.coverage_results) /
                                     max(sum(r.max_semantic_score for r in self.coverage_results), 1) * 100
            },
            "functions": []
        }

        for func_cov in self.coverage_results:
            func_report = {
                "name": func_cov.function_name,
                "filepath": func_cov.filepath,
                "statement_coverage": func_cov.statement_coverage,
                "semantic_coverage": func_cov.semantic_coverage,
                "blocks": [
                    {
                        "type": b.block_type,
                        "weight": b.weight,
                        "total_lines": b.total_lines,
                        "covered_lines": b.covered_lines,
                        "coverage_pct": b.coverage_pct,
                        "semantic_score": b.semantic_score
                    }
                    for b in func_cov.blocks
                ]
            }
            report["functions"].append(func_report)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n  JSON report saved to: {output_path}")