# APEDP-Distributed

Autonomous Distributed & Out-of-Core Chunked Sweep Execution Framework for APEDP-X.

## Installation

```bash
pip install -e packages/apedp-distributed
```

## Features

- **Bounded-Memory Streaming**: Streamingly yields results with $\mathcal{O}(\text{chunk\_size})$ auxiliary RAM.
- **Fault-Tolerant Checkpointing**: Automatic partition persistence and resumption.
- **Multi-Backend Scheduling**: Serial, thread pool, process pool, and Ray cluster hooks.
