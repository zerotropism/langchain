# LangChain LLM Integration Toolkit

An introduction to LangChain through a clean, object-oriented Python codebase.
It provides a comprehensive, modular implementation for working with local LLMs
(served via [Ollama](https://ollama.com/)) across the core LangChain concepts:
prompting, parsing, memory, chaining, retrieval-augmented generation (RAG),
agents, and evaluation.

## Requirements

* Python 3.12
* [Ollama](https://ollama.com/) running locally with the required models pulled:
  * Chat model: `gemma3:12b` (configurable in `src/config.yml`)
  * Embedding model: `nomic-embed-text` (used by the RAG system)
* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## Definition

A comprehensive, object-oriented implementation for working with LLMs through
LangChain, with a clean separation of concerns across specialized components:

* **LLM client communication** — `LLMClient` (+ `CustomTokenCountLLM`) in `src/llm.py`
* **Prompt template management** — `PromptManager` in `src/prompting.py`
* **Structured output parsing** — `OutputParser` in `src/parsing.py`
* **Text processing operations** — `TextProcessor` in `src/processing.py`
* **Memory management** — `MemoryFactory` and manager classes (`buffer`, `window`,
  `token`, `summary`) in `src/memory.py`, plus `MessageHistoryMemoryManager` in `src/history.py`
* **Retrieval Augmented Generation** — `RAGSystem` and `EmbeddingService` in `src/rag.py`
* **Agents** — `AgentFactory`, `Tools`, and `AgentRunner` in `src/agent.py`
* **Configuration** — `ConfigManager` (YAML-based) in `src/config.py`
* **Cross-cutting utilities** — `handle_exception` and `timing_decorator` in `src/decorators.py`

## Usage

Run the interactive entry point and pick a mode from the menu:

```bash
cd src
python main.py
```

### Available modes

| # | Mode | Description |
|---|------|-------------|
| 1 | `prompt` | Generate text from templates (simple completion, translation, structured extraction). |
| 2 | `chat_memory` | Chat with an LLM using legacy conversation memory. |
| 3 | `chat_history` | Chat with an LLM using a runnable message-history memory. |
| 4 | `simple_rag` | Run a standalone document-based retrieval-augmented query. |
| 5 | `chat_rag` | Chat with RAG capabilities. |
| 6 | `chat_rag_memory` | Chat with combined RAG and memory capabilities. |
| 7 | `agent` | Set up and converse with a specialized agent (math solver, Wikipedia search, Python coder, current date, or a combined "custom" agent). |
| 8 | `evaluate` | Set up an automated LLM-based evaluator for a simple RAG output. |

## Configuration

All runtime settings live in `src/config.yml` and are loaded through
`ConfigManager`. You can adjust:

* **Model** — name, temperature, `top_k`, `top_p`, context length
* **Memory** — type (`buffer`, `window`, `token`, `summary`), window size,
  token limit, verbosity
* **Prompts** — system prompt and templates for translation, extraction, RAG,
  and agent examples

## Project structure

```
src/                Factored, production-style source code
  main.py           Interactive entry point with all modes
  config.py|.yml    Configuration manager and default settings
  llm.py            LLM client (ChatOllama) and token-counting variant
  prompting.py      Prompt template management
  parsing.py        Structured output parsing
  processing.py     High-level text processing tasks
  memory.py         Memory managers and factory
  history.py        Message-history-based memory manager
  rag.py            RAG system and embedding service
  agent.py          Agent factory, tools, and runner
  decorators.py     Exception handling and timing decorators
notebooks/          Step-by-step progression by chapter/task
  1_parsing.ipynb   2_memory.ipynb   3_chaining.ipynb
  4_qadocs.ipynb    5_evaluate.ipynb 6_agents.ipynb
  notebook.ipynb    Full end-to-end draft progression
data/               Example datasets (amazon.csv, clothing.csv)
generated/          Generated variants of the code
archives/           Archived configuration/code
logs/               Runtime logs (e.g. errors.log)
```

## Learning path

The `notebooks/` folder mirrors the source modules and is meant to be followed
in order: parsing → memory → chaining → question answering over documents (RAG)
→ evaluation → agents.