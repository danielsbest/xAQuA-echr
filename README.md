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

## Running experiments

The names of all experiments can be found in `src/experiments/load_experiment.py`

```
python experiment.py --experiment <experiment name>
```

## Running evaluations


