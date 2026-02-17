# Dependency Analysis

Analysis of each dependency in `pyproject.toml` — whether it's actually used in `src/`.

| Dependency | Used in src/? | Verdict |
|---|---|---|
| `langchain` | Yes (`langchain_core` prompts, runnables) | Keep |
| `langchain-openai` | No (only `evals/` + README) | **Removed** |
| `pandas` | Yes (throughout) | Keep |
| `python-dotenv` | No (only `.env.example`) | Not needed |
| `langfuse` | No | Not needed |
| `boto3` | Yes (Bedrock client in CLI) | Keep |
| `scikit-learn` | Yes (evaluate metrics in CLI) | Keep |
| `openpyxl` | No | Not needed |
| `pyarrow` | No | Not needed |
| `toml` | No | Not needed |

## New dependencies added

| Dependency | Reason |
|---|---|
| `langchain-aws` | `ChatBedrockConverse` LLM for the CLI |
| `typer[all]` | CLI framework (includes `rich` for output) |
| `tiktoken` | Token counting in `llm_batch_processor.py` (was transitive via `langchain-openai`) |

## Note on openai removal

`langchain-openai` was removed — it was only used in `evals/` and README.
The `openai` import in `llm_batch_processor.py` (for `BadRequestError` in an
except clause) was made conditional with a placeholder class so the module
loads without `openai` installed. `tiktoken` was promoted to a direct
dependency since it's actively used for token counting.
