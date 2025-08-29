import os
import re
from time import sleep
from openai import OpenAI
from dotenv import load_dotenv


import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


from src.llms.llm import LLM

load_dotenv(override=True)

def get_response(client: OpenAI, model: str, prompt: str, temperature: float, system_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    completion_kwargs = {}
    if temperature is not None:
        completion_kwargs['temperature'] = temperature
        
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            **completion_kwargs
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"OpenRouter API call failed: {e}")
        sleep(60)
        # a simple retry
        return get_response(client, model, prompt, temperature, system_prompt)


class OpenRouter(LLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def _infer(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get("temperature")
        system_prompt = kwargs.get("system_prompt")
        
        return get_response(
            client=self.client,
            model=self.model_name,
            prompt=prompt,
            temperature=temperature,
            system_prompt=system_prompt
        )

class R1_0528(OpenRouter):
    def __init__(self):
        super().__init__("deepseek/deepseek-r1-0528:free")
        
    def _infer(self, prompt: str, **kwargs) -> str | tuple[str, str]:
        response_text: str = super()._infer(prompt, **kwargs)
        
        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        match = think_pattern.search(response_text)

        if match:
            thoughts = match.group(1).strip()
            response_without_think = think_pattern.sub("", response_text, 1)
            response_without_think = response_without_think.strip()
        else:
            thoughts = ""
            response_without_think = response_text.strip()
            return response_without_think
            
        return response_without_think, thoughts

class K2(OpenRouter):
    def __init__(self):
        super().__init__("moonshotai/kimi-k2:free")



        
    
if __name__ == "__main__":




    model = R1_0528()
    prompt = "What is the capital of France?"
    response, thoughts = model.infer_completion(prompt, temperature=0.7, system_prompt="Antworte auf Deutsch.")
    print(f"Response: {response}\nThoughts: {thoughts}")