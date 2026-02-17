# ThemeFinder CLI

CLI fork of [i-dot-ai/themefinder](https://github.com/i-dot-ai/themefinder) for running the full pipeline from the command line. Uses AWS Bedrock.

## Install

```bash
uv sync
```

## Commands

### `themefinder discover` — find themes in responses

Runs sentiment analysis, theme generation, condensation, and refinement. Outputs a themes CSV.

```bash
themefinder discover responses.csv \
  -q "What do you think about the proposed policy changes?" \
  -c response_text --id-col id \
  -o themes.csv \
  --profile my-aws-profile --region eu-west-2
```

Use `--cluster` to add hierarchical clustering before refinement (reduces to `--target-themes` themes, default 10):

```bash
themefinder discover responses.csv \
  -q "What do you think about the proposed policy changes?" \
  --cluster --target-themes 8 \
  -o themes.csv
```

### `themefinder classify` — code responses against existing themes

Maps responses to a theme set. Outputs a wide CSV with `code_1`, `code_2`, ... columns.

```bash
themefinder classify responses.csv \
  -t themes.csv \
  -q "What do you think about the proposed policy changes?" \
  -c response_text --id-col id \
  -o coded.csv
```

Add `--detail` to also run detail/evidence detection on each response.

### `themefinder evaluate` — compare coded output to a reference

Computes F1, precision, recall, exact match, and overlap rate. No LLM or AWS credentials needed.

```bash
themefinder evaluate coded.csv reference.csv
themefinder evaluate coded.csv reference.csv -o metrics.json
```

Both CSVs need `response_id` and `code_*` columns (e.g. `code_1`, `code_2`).

## Common options

| Flag | Description | Default |
|---|---|---|
| `-c` / `--column` | Response text column name | `response` |
| `--id-col` | Response ID column name | `response_id` |
| `--model` | Bedrock model ID | `anthropic.claude-3-7-sonnet-20250219-v1:0` |
| `--region` | AWS region | `eu-west-2` |
| `--profile` | AWS named profile | (default credentials) |
| `--concurrency` | Max parallel LLM calls | `10` |

## Input format

Responses CSV must have an ID column and a text column. Use `--id-col` and `-c` / `--text-col` if they aren't named `response_id` and `response`.

Themes CSV must have `topic_id` and either a `topic` column (`"label: description"` format) or separate `topic_label` + `topic_description` columns.
