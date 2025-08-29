import os
from time import sleep
import re
from typing import Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.llms.llm import LLM
from src.llms.gemini_batch import GeminiBatchProcessor

load_dotenv(override=True)

client = genai.Client()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_response(prompt: str, model: str, temperature: float, thinking: bool, system_prompt: str, thinking_budget: int = None, prompt_size: int = 5000) -> str:
    try:
        config_kwargs = {}
        if temperature is not None:
            config_kwargs['temperature'] = temperature
        if thinking_budget is not None:
            config_kwargs['thinking_config'] = types.ThinkingConfig(thinking_budget=thinking_budget)
        elif thinking:
            config_kwargs['thinking_config'] = types.ThinkingConfig(thinking_budget=-1)
        else:
            config_kwargs['thinking_config'] = types.ThinkingConfig(thinking_budget=0)
        if system_prompt:
            config_kwargs['system_instruction'] = system_prompt

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=config
        )
        return response.text
    except Exception as e:
        print(e)
        if len(prompt.split()) >= prompt_size:
            prompt = " ".join(prompt.split()[:prompt_size])
        sleep(60)  # avoid rate limiting
        return get_response(prompt, model, temperature, thinking, system_prompt, thinking_budget, prompt_size - 500)


class Gemini(LLM):
    def __init__(self, model_name: str, gcs_bucket_name: str = None):
        super().__init__(model_name)
        self.gcs_bucket_name = gcs_bucket_name or os.getenv("GCS_BATCH_BUCKET")
        if self.gcs_bucket_name:
            print(f"Gemini model configured for batch processing with GCS bucket: {self.gcs_bucket_name}")


    def _infer(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get("temperature")
        thinking = kwargs.get("thinking", True)
        thinking_budget = kwargs.get("thinking_budget")
        system_prompt = kwargs.get("system_prompt")
        return get_response(
            prompt,
            self.model_name,
            temperature=temperature,
            thinking=thinking,
            system_prompt=system_prompt,
            thinking_budget=thinking_budget,
        )

    def _infer_batch(self, prompts: list[dict[str, Any]]) -> list[str]:
        if not self.gcs_bucket_name:
            raise ValueError(
                "A GCS bucket name must be provided for batch inference. "
                "Set it during Gemini model initialization or via the GCS_BATCH_BUCKET environment variable."
            )
        
        processor = GeminiBatchProcessor(gcs_bucket_name=self.gcs_bucket_name)
        return processor.run_batch_job(self.model_name, prompts)

class Gemma3(Gemini):
    def __init__(self):
        super().__init__("gemma-3-27b-it")

class GeminiFlash2_0(Gemini):
    def __init__(self):
        super().__init__("gemini-2.0-flash")

    def _infer(self, prompt: str, **kwargs) -> str:
        """Force thinking off for gemini-2.0-flash.

        This model doesn't support thinking tokens / budget, so we always
        disable it regardless of user-supplied kwargs.
        """
        kwargs.pop("thinking", None)
        kwargs.pop("thinking_budget", None)
        return super()._infer(prompt, thinking=False, **kwargs)

class GeminiFlash2_5(Gemini):
    def __init__(self):
        super().__init__("gemini-2.5-flash")

class GeminiPro2_5(Gemini):
    def __init__(self):
        super().__init__("gemini-2.5-pro")
        
class GPToss120(Gemini):
    def __init__(self):
        if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "") != "true":
            raise EnvironmentError("GOOGLE_GENAI_USE_VERTEXAI environment variable must be set to 'true' to use Gemini.")
        super().__init__("openai/gpt-oss-120b-maas")


class R1_0528(Gemini):
    def __init__(self):
        if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "") != "true":
            raise EnvironmentError("GOOGLE_GENAI_USE_VERTEXAI environment variable must be set to 'true' to use Gemini.")
        super().__init__("deepseek-ai/deepseek-r1-0528-maas")
        
    def _infer(self, prompt: str, **kwargs) -> tuple[str, str]:
        kwargs.pop("thinking", None)
        kwargs.pop("thinking_budget", None)
        response_text: str = super()._infer(prompt, thinking=False, **kwargs)

        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        match = think_pattern.search(response_text)

        if match:
            thoughts = match.group(1).strip()
            response_without_think = think_pattern.sub("", response_text, 1)
            response_without_think = response_without_think.strip()
        else:
            thoughts = ""
            response_without_think = response_text.strip()
            
        return response_without_think, thoughts


if __name__ == "__main__":
    model = GeminiFlash2_5()
    prompt = "What is the capital of France?"
    response = model.infer_completion(prompt, temperature=0.7, system_prompt="Antworte auf Deutsch.")
    print(f"Response: {response}")