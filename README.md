# SemantiCov


Not all lines of code are created equal. 


SemantiCovuses LLMs to analyze your Python code semantically, categorizing it into importance-weighted blocks and giving you coverage metrics that actually reflect risk.

## Why SemantiCov?

Traditional code coverage tools treat every line equally.


A print statement counts the same as critical error handling. 


This creates a false sense of security: you might have 90% line coverage but still miss testing your most important code paths.


**SemantiCov changes that.** 


It classifies your code into semantic blocks and weights them by business criticality:

- **Core Functionality** (weight: 10) - Your main business logic
- **Error Handling** (weight: 9) - Exception management
- **Boundary Conditions** (weight: 8) - Edge cases and validation
- **Integration Points** (weight: 7) - APIs, databases, external services
- **Security Features** (weight: 6) - Auth, encryption, sanitization
- **Performance & Scalability** (weight: 5) - Caching, optimization
- **Output Consistency** (weight: 4) - Logging and reporting
- **Configuration** (weight: 3) - Environment setup
- **UI Interactions** (weight: 2) - User input handling

## Features

**AI-Powered Classification** - Uses local LLMs (via Ollama) to understand code semantically  
**Weighted Coverage Metrics** - See both traditional and semantic coverage percentages  
**Priority Recommendations** - Know exactly which untested code poses the highest risk  
**Smart Caching** - Classifications are cached to speed up subsequent runs  
**Detailed Reports** - Console summaries and JSON output for CI/CD integration

## Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Ollama** - For local LLM inference
   ```bash
   # Install Ollama (macOS/Linux)
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Start Ollama service
   ollama serve
   
   # Pull the Mistral model (in a new terminal)
   ollama pull mistral:7b
   ```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SemantiCov.git
cd SemantiCov

# Install dependencies
pip install coverage networkx requests
```

## How It Works

1. **Function Extraction** - Parses your Python files and extracts all functions with line numbers
2. **AI Classification** - Sends each function to a local LLM to categorize lines into semantic blocks
3. **Semantic Calculation** - Computes weighted coverage scores based on block importance
4. **Reporting** - Generates actionable insights about your test suite's quality

## Project Structure

```
SemantiCov/
├── main.py              # CLI entry point and orchestrator
├── analyzers.py         # Core analysis classes
├── models.py            # Data models and block weights
├── cache_manager.py     # SQLite-based classification cache
├── example_project_1/   # Sample project (BlackJack game)
├── example_project_2/   # Sample project (Tic-Tac-Toe)
└── README.md
```


### Model Selection

SemantiCov works with any Ollama model. Good choices:

- `mistral:7b` (default) - Good balance of speed and accuracy
- `llama3:8b` - Faster, slightly less accurate
- `codellama:13b` - Better code understanding, slower
- `mixtral:8x7b` - Most accurate, requires more resources

## Limitations

- **Python Only** - Currently supports Python projects only
- **Local LLM Required** - Requires Ollama running locally (no cloud API support yet)
- **Classification Accuracy** - LLM classification depends on model quality and prompt engineering

## License

This project is licensed under the MIT License

## Contributors

For questions about this project:
- Vandit Vasa: https://www.linkedin.com/in/vasa-vandit/
- Muhammed Bilal: https://www.linkedin.com/in/mbilal-ai/
- Tiago Monteiro: https://www.linkedin.com/in/tiago-monteiro-/


