from abc import ABC, abstractmethod
from .caching import Cache
from typing import List, Dict, Any, Optional
from tqdm import tqdm


class LLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.cache = Cache()

    @abstractmethod
    def _infer(self, prompt: str, **kwargs) -> str|tuple[str, str]:
        ...

    def _infer_batch(self, prompts: List[Dict[str, Any]]) -> List[Optional[str|tuple[str, str]]]:
        """Default sequential implementation for batch inference. Subclasses should override for native batching."""

        results: List[Optional[str|tuple[str, str]]] = []
        print(f"No native batch inference available, using sequential inference for {len(prompts)} prompts.")
        for p_info in tqdm(prompts, desc="Processing uncached prompts sequentially", unit="prompt"):
            kwargs = p_info.copy()
            prompt_text = kwargs.pop('prompt')

            result = self._infer(prompt_text, **kwargs)

            if result is None:
                results.append(None)
                continue

            cache_kwargs = kwargs.copy()
            if isinstance(result, tuple):
                response, thoughts = result
                cache_kwargs['thoughts'] = thoughts
            else:
                response = result

            if isinstance(response, str) and response.startswith("```json"):
                response_to_cache = response[7:-3].strip()
            else:
                response_to_cache = response

            temp_value = float(cache_kwargs.pop('temperature', 1.0))

            try:
                self.cache.set(
                    prompt_text,
                    self.model_name,
                    response_to_cache,
                    temp_value,
                    **cache_kwargs,
                )
            except Exception as e:
                print(f"Warning: failed to cache batch item: {e}")

            results.append(result)
        return results

    def infer_completion(self, prompt: str, temperature: float = 1.0, **kwargs) -> str:
        cached_response = self.cache.get(prompt, self.model_name, temperature=temperature, **kwargs)
        if cached_response:
            return cached_response

        result = self._infer(prompt, **kwargs)
        if isinstance(result, tuple):
            response, thoughts = result
            kwargs['thoughts'] = thoughts
        else:
            response = result
        if response.startswith("```json"):
            response = response[7:-3].strip()
        self.cache.set(prompt, self.model_name, response, temperature= temperature, **kwargs)
        return response
    
    def infer_batch_completion(self, prompts: List[Dict[str, Any]]) -> List[Optional[str]]:
        """
        Infers completions for a batch of prompts, using caching.

        Args:
            prompts: A list of dictionaries, where each dict contains a 'prompt' key
                     and other optional parameters like 'temperature'.

        Returns:
            A list of response strings (or None if a result failed) in the same order as the input prompts.
        """
        cached_responses, uncached_prompts = self.cache.get_many(prompts, self.model_name)

        if not uncached_prompts:
            print("Batch cache hit for all items.")
            all_responses_map = cached_responses
        else:
            print(f"Batch cache miss for {len(uncached_prompts)}/{len(prompts)} items. Running batch inference.")
            new_results = self._infer_batch(uncached_prompts)

            new_responses_map = {}
            processed_responses = []
            for i, result in enumerate(new_results):
                p_info = uncached_prompts[i]
                if result is None:
                    processed_responses.append(None)
                    continue
                    
                if isinstance(result, tuple):
                    response, thoughts = result
                    p_info['thoughts'] = thoughts
                else:
                    response = result

                if response.startswith("```json"):
                    response = response[7:-3].strip()
                
                processed_responses.append(response)

                # Construct the unique composite key for this request
                prompt_hash = self.cache._get_prompt_hash(**p_info)
                temperature = p_info.get('temperature', 1.0)
                request_key = (prompt_hash, self.model_name, float(temperature))
                new_responses_map[request_key] = response

            # Only cache non-None responses
            valid_prompts = []
            valid_responses = []
            for i, response in enumerate(processed_responses):
                if response is not None:
                    valid_prompts.append(uncached_prompts[i])
                    valid_responses.append(response)
            
            if valid_prompts:
                self.cache.set_many(valid_prompts, valid_responses, self.model_name)
            all_responses_map = {**cached_responses, **new_responses_map}

        final_ordered_responses = []
        for p_info in prompts:
            # Reconstruct the unique composite key to look up the correct response
            prompt = p_info['prompt']
            temperature = p_info.get('temperature', 1.0)
            prompt_hash = self.cache._get_prompt_hash(**p_info)
            request_key = (prompt_hash, self.model_name, float(temperature))
            response = all_responses_map.get(request_key, None)
            final_ordered_responses.append(response)
            
        return final_ordered_responses
