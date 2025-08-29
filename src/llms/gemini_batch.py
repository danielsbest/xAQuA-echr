import time
import json
import uuid
import os
from typing import List, Dict, Any

from google.cloud import storage
from google import genai
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions, ThinkingConfig

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable first!
# Also this code is unneeded complex / deprecated after Google released a proper batch api recently!

class GeminiBatchProcessor:
    """
    Handles the entire process of running a batch prediction job with Gemini.
    This includes:
    1. Formatting requests into a JSONL file.
    2. Uploading the request file to Google Cloud Storage (GCS).
    3. Creating and monitoring the Gemini batch job.
    4. Downloading and parsing the results from GCS.
    5. Cleaning up the created files in GCS.
    """

    def __init__(self, gcs_bucket_name: str, gcs_prefix: str = "gemini-batch-jobs"):
        """
        Initializes the batch processor.

        Args:
            gcs_bucket_name: The name of the GCS bucket to use for staging requests and results.
            gcs_prefix: A prefix within the bucket to organize batch job files.
        """
        if not gcs_bucket_name:
            raise ValueError("A GCS bucket name is required for batch processing.")
            
        self.gcs_bucket_name = gcs_bucket_name
        self.gcs_prefix = gcs_prefix
        
        self.genai_client = genai.Client(http_options=HttpOptions(api_version="v1"))
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.gcs_bucket_name)

    def _prepare_and_upload_requests(self, model_name: str, prompts: List[Dict[str, Any]]) -> str:
        """Formats prompts into JSONL, uploads to GCS, and returns the GCS URI."""
        job_id = str(uuid.uuid4())
        filename = f"{self.gcs_prefix}/{job_id}/requests.jsonl"
        
        batch_requests = []
        for i, p_info in enumerate(prompts):
            prompt_text = p_info['prompt']
            config_kwargs = {}
            if 'temperature' in p_info:
                config_kwargs['temperature'] = p_info['temperature']
            if 'thinking_budget' in p_info:
                config_kwargs['thinking_config'] = {
                                                    "includeThoughts": False,
                                                    "thinkingBudget": int(p_info['thinking_budget'])
                                                    }
            
            req_body: Dict[str, Any] = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt_text}
                        ]
                    }
                ]
            }
            if 'system_prompt' in p_info:
                req_body["system_instruction"] = {"parts": [{"text": p_info['system_prompt']}]}  
            if config_kwargs:
                req_body["generationConfig"] = config_kwargs
            
            # Add a request_id to reorder responses later
            batch_requests.append(json.dumps({"request_id": i, "request": req_body}, ensure_ascii=False))

        jsonl_content = "\n".join(batch_requests)
        
        blob = self.bucket.blob(filename)
        print(f"Uploading request file to gs://{self.gcs_bucket_name}/{filename}")
        blob.upload_from_string(jsonl_content, content_type="application/json")
        
        return f"gs://{self.gcs_bucket_name}/{filename}"

    def _download_and_parse_results(self, gcs_output_prefix: str, num_prompts: int) -> List[str]:
        """Downloads result files from GCS, parses them, and returns responses in order."""
        blobs = self.storage_client.list_blobs(self.gcs_bucket_name, prefix=gcs_output_prefix)
        
        # Initialize list with placeholders to ensure correct order
        responses = [None] * num_prompts
        
        for blob in blobs:
            if blob.name.endswith("predictions.jsonl"):
                print(f"Downloading results from {blob.name}...")
                result_content = blob.download_as_text()
                for line in result_content.strip().split("\n"):
                    try:
                        data = json.loads(line)
                        request_id = int(data['request_id'])
                        response_text = data['response']['candidates'][0]['content']['parts'][0]['text']
                        
                        if 0 <= request_id < num_prompts:
                            responses[request_id] = response_text
                        else:
                            print(f"Warning: Found out-of-bounds request_id {request_id}.")
                            
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        print(f"Warning: Could not parse line in result file: {line}. Error: {e}")
        
        return responses

    def _cleanup_gcs_files(self, gcs_prefix: str):
        """Deletes all files under a given GCS prefix."""
        print(f"Cleaning up GCS files under prefix: {gcs_prefix}")
        blobs_to_delete = list(self.storage_client.list_blobs(self.gcs_bucket_name, prefix=gcs_prefix))
        self.bucket.delete_blobs(blobs=blobs_to_delete)
        print("Cleanup complete.")

    def run_batch_job(self, model_name: str, prompts: List[Dict[str, Any]]) -> List[str]:
        """Orchestrates the entire batch job workflow."""
        gcs_input_uri = self._prepare_and_upload_requests(model_name, prompts)
        gcs_input_prefix = os.path.dirname(gcs_input_uri.replace(f"gs://{self.gcs_bucket_name}/", ""))
        gcs_output_uri = f"gs://{self.gcs_bucket_name}/{gcs_input_prefix}/output/"

        # try:
        job = self.genai_client.batches.create(
            model=f"{model_name}",
            src=gcs_input_uri,
            config=CreateBatchJobConfig(dest=gcs_output_uri),
        )
        print(f"Started batch job: {job.name}")
        print(f"Initial job state: {job.state.name}")

        completed_states = {
            JobState.JOB_STATE_SUCCEEDED, JobState.JOB_STATE_FAILED, JobState.JOB_STATE_CANCELLED, JobState.JOB_STATE_EXPIRED,
        }

        while job.state not in completed_states:
            time.sleep(30)
            job = self.genai_client.batches.get(name=job.name)
            print(f"Job state: {job.state.name}")

        if job.state != JobState.JOB_STATE_SUCCEEDED:
            raise RuntimeError(f"Batch job failed with state: {job.state.name}. Error: {job.error}")

        print("Batch job succeeded. Downloading results...")
        results = self._download_and_parse_results(
            os.path.dirname(gcs_output_uri.replace(f"gs://{self.gcs_bucket_name}/", "")),
            num_prompts=len(prompts)
        )
        
        if len(results) != len(prompts):
                print(f"Warning: Number of results ({len(results)}) does not match number of prompts ({len(prompts)}).")
                raise RuntimeError("Mismatch in number of results and prompts.")

        return results
        # finally:
            # self._cleanup_gcs_files(gcs_input_prefix)

    # def run_batch_job(self, model_name: str, prompts: List[Dict[str, Any]]) -> List[str]:
    #     """
    #     Use this version when the batch job has already been run and you just want to
    #     download the results from a known GCS location.
    #     """
    #     # Manually specify the GCS output path where the results are stored.
    #     # gcs_output_uri = "gs://<your-bucket-name>/<path-to-your-output-folder>/"
    #     gcs_output_uri = "gs://batch_processing3/gemini-batch-jobs/4f7620dd-1aa1-4e09-b16f-a0820a47f90e/output/prediction-model-2025-08-21T11:34:53.785890Z/predictions.jsonl"
    #     if not gcs_output_uri.endswith("/"):
    #         gcs_output_uri = gcs_output_uri.rsplit("/", 1)[0] + "/"
    #     print(f"Downloading results from manually specified path: {gcs_output_uri}")

    #     results = self._download_and_parse_results(
    #         os.path.dirname(gcs_output_uri.replace(f"gs://{self.gcs_bucket_name}/", "")),
    #         num_prompts=len(prompts)
    #     )

    #     if len(results) != len(prompts):
    #         print(f"Warning: Number of results ({len(results)}) does not match number of prompts ({len(prompts)}).")
    #         raise RuntimeError("Mismatch in number of results and prompts.")

    #     return results
