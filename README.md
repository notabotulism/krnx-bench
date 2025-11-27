# KRNX Benchmark Harness

A rigorous, reproducible test harness for evaluating KRNX temporal memory infrastructure against baseline memory systems.

## Overview

This harness proves four guarantees that KRNX provides to AI agents:

| Guarantee | What It Means | How We Test It |
|-----------|---------------|----------------|
| **Durability** | Events survive crashes | Kill process mid-write, verify recovery |
| **Consistency** | Current state is always correct | Update facts, verify latest is returned |
| **Auditability** | Trace any output to its causes | Walk hash-chain provenance |
| **Replay** | Reconstruct state at any point | Replay to timestamp, verify accuracy |

## Quick Start

```bash
# Install
pip install -e .

# List available scenarios
krnx-bench scenarios

# Run a single scenario
krnx-bench run fact_correction --adapter=all

# Run the full suite
krnx-bench suite

# Generate report from run
krnx-bench report outputs/runs/<timestamp>/
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (Typer)                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Suite Runner                              │
│            Orchestrates scenarios × adapters × trials           │
└─────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│  KRNX Adapter     │ │  RAG Adapter      │ │  Baseline Adapter │
│  (Docker)         │ │  (Docker)         │ │  (In-process)     │
└───────────────────┘ └───────────────────┘ └───────────────────┘
```

## Systems Under Test

| System | Description | Docker |
|--------|-------------|--------|
| `krnx` | KRNX temporal memory kernel | Yes |
| `naive_rag` | Qdrant vector store (top-k retrieval) | Yes |
| `baseline` | No memory (raw LLM) | No |

## Scenarios

### Durability
- `crash_recovery`: Write events, kill process, verify recovery

### Consistency  
- `fact_correction`: Update facts over time, verify latest is returned
- `temporal_versioning`: Query facts at specific timestamps

### Auditability
- `provenance_chain`: Verify hash-chain integrity for multi-step workflows

### Replay
- `point_in_time`: Reconstruct state at arbitrary timestamps
- `determinism`: Verify replay produces identical results

### Baseline
- `niah`: Needle-in-haystack sanity check

## Configuration

Edit `config/default.yaml` for LLM settings:

```yaml
llm:
  provider: "openai"  # or "anthropic"
  model: "gpt-4-turbo-preview"
  temperature: 0.0
```

Edit `config/scenarios.yaml` for test parameters:

```yaml
consistency:
  fact_correction:
    versions: 5
    distractors_per_version: 100
    trials: 50
```

## Output

### Raw Results
```
outputs/runs/<timestamp>/
├── manifest.json
├── raw/
│   ├── fact_correction_krnx.jsonl
│   ├── fact_correction_naive_rag.jsonl
│   └── ...
└── logs/
```

### Reports
```
outputs/reports/<timestamp>/
├── tables/
│   ├── consistency.md
│   ├── durability.md
│   └── replay.md
├── figures/
│   ├── consistency_comparison.png
│   ├── replay_scaling.png
│   └── durability_comparison.png
└── summary.md
```

## Requirements

- Python 3.11+
- Docker with Docker Compose
- OpenAI API key (or Anthropic)

## Environment Variables

```bash
# Copy the example and add your keys
cp .env.example .env

# Edit .env with your API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Or export directly:
```bash
export OPENAI_API_KEY="sk-..."
```

## License

MIT
