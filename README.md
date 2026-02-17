# ThemeFinder

ThemeFinder is a topic modelling Python package designed for analysing one-to-many question-answer data (i.e. survey responses, public consultations, etc.). See the [docs](https://i-dot-ai.github.io/themefinder/) for more info.

> [!IMPORTANT]
> Incubation project: This project is an incubation project; as such, we don't recommend using this for critical use cases yet. We are currently in a research stage, trialling the tool for case studies across the Civil Service. Find out more about our projects at https://ai.gov.uk/. 


## Quickstart

### Install using your package manager of choice

For example `pip install themefinder` or `poetry add themefinder`.

### Usage

ThemeFinder takes as input a [pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) with two columns:
- `response_id`: A unique identifier for each response
- `response`: The free text survey response

ThemeFinder now supports a range of language models through structured outputs.

The function `find_themes` identifies common themes in responses and labels them, it also outputs results from intermediate steps in the theme finding pipeline.

For this example, import the following Python packages into your virtual environment: `asyncio`, `pandas`, `lanchain`. And import `themefinder` as described above.

If you are using environment variables (eg for API keys), you can use `python-dotenv` to read variables from a `.env` file. 

If you are using an Azure OpenAI endpoint, you will need the following variables:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `OPENAI_API_VERSION`
- `DEPLOYMENT_NAME`
- `AZURE_OPENAI_BASE_URL`

Otherwise you will need whichever variables [LangChain](https://www.langchain.com/) requires for your LLM of choice.

```python
import asyncio
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import AzureChatOpenAI
from themefinder import find_themes

# If needed, load LLM API settings from .env file
load_dotenv()

# Initialise your LLM of choice using langchain
llm = AzureChatOpenAI(
    model="gpt-4o",
    temperature=0,
)

# Set up your data
responses_df = pd.DataFrame({
   "response_id": ["1", "2", "3", "4", "5"],
   "response": ["I think it's awesome, I can use it for consultation analysis.", 
   "It's great.", "It's a good approach to topic modelling.", "I'm not sure, I need to trial it more.", "I don't like it so much."]
})

# Add your question
question = "What do you think of ThemeFinder?"

# Make the system prompt specific to your use case 
system_prompt = "You are an AI evaluation tool analyzing survey responses about a Python package."

# Run the function to find themes, we use asyncio to query LLM endpoints asynchronously, so we need to await our function
async def main():
    result = await find_themes(responses_df, llm, question, system_prompt=system_prompt)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## CLI

ThemeFinder includes a command-line interface for running the full pipeline without writing Python. It uses AWS Bedrock as the LLM provider.

### Install

```bash
uv sync
```

### Commands

#### `themefinder discover` — find themes in responses

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

#### `themefinder classify` — code responses against existing themes

Maps responses to a theme set. Outputs a wide CSV with `code_1`, `code_2`, ... columns.

```bash
themefinder classify responses.csv \
  -t themes.csv \
  -q "What do you think about the proposed policy changes?" \
  -c response_text --id-col id \
  -o coded.csv
```

Add `--detail` to also run detail/evidence detection on each response.

#### `themefinder evaluate` — compare coded output to a reference

Computes F1, precision, recall, exact match, and overlap rate. No LLM or AWS credentials needed.

```bash
themefinder evaluate coded.csv reference.csv
themefinder evaluate coded.csv reference.csv -o metrics.json
```

Both CSVs need `response_id` and `code_*` columns (e.g. `code_1`, `code_2`).

### Common options

| Flag | Description | Default |
|---|---|---|
| `-c` / `--column` | Response text column name | `response` |
| `--id-col` | Response ID column name | `response_id` |
| `--model` | Bedrock model ID | `anthropic.claude-3-5-sonnet-20241022-v2:0` |
| `--region` | AWS region | `eu-west-2` |
| `--profile` | AWS named profile | (default credentials) |
| `--concurrency` | Max parallel LLM calls | `10` |

### Input format

Responses CSV must have an ID column and a text column. Use `--id-col` and `-c` / `--text-col` if they aren't named `response_id` and `response`.

Themes CSV must have `topic_id` and either a `topic` column (`"label: description"` format) or separate `topic_label` + `topic_description` columns.


## ThemeFinder pipeline

ThemeFinder's pipeline consists of five distinct stages, each utilizing a specialized LLM prompt:

### Sentiment analysis
- Analyses the emotional tone and position of each response using sentiment-focused prompts
- Provides structured sentiment categorisation based on LLM analysis

### Theme generation
- Uses exploratory prompts to identify initial themes from response batches
- Groups related responses for better context through guided theme extraction

### Theme condensation
- Employs comparative prompts to combine similar or overlapping themes
- Reduces redundancy in identified topics through systematic theme evaluation

### Theme refinement
- Leverages standardisation prompts to normalise theme descriptions
- Creates clear, consistent theme definitions through structured refinement

### Theme target alignment
- Optional step to consolidate themes down to a target number

### Theme mapping
- Utilizes classification prompts to map individual responses to refined themes
- Supports multiple theme assignments per response through detailed analysis


The prompts used at each stage can be found in `src/themefinder/prompts/`.

The file `src/themefinder.core.py` contains the function `find_themes` which runs the pipline. It also contains functions fo each individual stage.


**For more detail - see the docs: [https://i-dot-ai.github.io/themefinder/](https://i-dot-ai.github.io/themefinder/).**


## Model Compatibility

ThemeFinder's structured output approach makes it compatible with a wide range of language models from various providers. This list is non-exhaustive, and other models may also work effectively:

### OpenAI Models
- GPT-4, GPT-4o, GPT-4.1
- All Azure OpenAI deployments

### Google Models
- Gemini series (1.5 Pro, 2.0 Pro, etc.)

### Anthropic Models
- Claude series (Claude 3 Opus, Sonnet, Haiku, etc.)

### Open Source Models
- Llama 2, Llama 3
- Mistral models (e.g., Mistral 7B, Mixtral)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The documentation is [© Crown copyright](https://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/) and available under the terms of the [Open Government 3.0 licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).


## Feedback

Contact us with questions or feedback at packages@cabinetoffice.gov.uk.
