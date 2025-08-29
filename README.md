# AQuAECHR Benchmark

This repository contains all the code needed to evaluate models on xAQuA!


## Setup Instructions

#### Create an environment and install dependencies

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


#### Environment variables

Our main Evaluation is done using LLMs accessed via APIs.

```
GOOGLE_API_KEY=<your gemini api key>
OPENROUTER_API_KEY=<used for all models not in gemini api>
```

if you want to use the batch processing, also set the following:

```
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_CLOUD_PROJECT=<your google cloud project>
GOOGLE_APPLICATION_CREDENTIALS=env/application_default_credentials.json
GCS_BATCH_BUCKET=<your gcs bucket name for batch results>
```

# ECHR-QA Dataset Generation

This repository contains all code used to curate the ECHR-QA dataset.

The dataset creation can be reproduced using the following steps:

1. Download the ECHR case law guides from the ECHR website:
    you can use the script [download_guides_pdf.py](utils/download_guides_pdf.py)

2. Use marker-pdf to extract the markdowns from the PDFs:
    ! This needs enough VRAM to run, I used Google Colab with a T4 GPU
    [colab_marker_pdf.ipynb](parsing/colab_marker_pdf.ipynb)

3. Parse the markdowns to extract their paragraphs. Unfotunately due to many inconsistencies in the markdowns this includes manual work. The parsing logic is in [guide_parser.py](parsing/guide_parser.py).
The manual work which rules out the inconsistencies is in [guide_parser_config.py](parsing/guide_parser_config.py).
And the script can be run from [guide_parser.ipynb](parsing/guide_parser.ipynb).

4. I embedded every paragraph using Googles embedding models.
[paragraph_embedding.py](paragraph_embedding.py) is called from [paragraph_embedding.ipynb](paragraph_embedding.ipynb)
Note you need to create a [.env file](.env) with a GOOGLE_API_KEY=<your key from aistudio.google.com>.

5. For splitting the paragraphs into sentences and extracting the citations I use an LLM ('gemini-2.0-flash') in [markdown_citations_llm_extraction.py](citations/markdown_citations_llm_extraction.py).

6. To use paragraphs that have a coherent topic I use clustering methods to group paragraphs in [group_paragraphs.py](grouping/group_paragraphs.py).

7. Finally [generation_pipeline.py](generation/generation_pipeline.py) is used to generate the question-answer pairs.

    i) In question_gen() it uses the previously generated clusters (groups) of paragraphs to generate a question using different prompts
    ii) filter valid citations, it filters...





## Preembedding for Retrieval


First the questions need to be embedded in qwen3 embeddings (You might need to lower the batch size)
```
bash setup/run_create_qwen3_embeddings.sh
bash setup/run_preembed_qwen3_queries.sh
```

## Running Experiments

The names of all experiments can be found in `src/experiments/load_experiment.py`

```
python execution/experiment.py --experiment <experiment name>
```

## Running evaluations

To use llm-as-a-judge evaluations run:

```
python src/models/llm_judge/llm_judge.py <filename of experiment output>
```
