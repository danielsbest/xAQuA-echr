import os
import sys
import re
import json
import logging
import logging.config
from typing import List, Union, overload
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.llms.llm import LLM
from src.llms.gemini import GeminiFlash2_5

from src.dataset_generation.generation_prompts import ANSWER_QUALITY_FILTERING_PROMPT_2, ANSWER_GROUNDEDNESS_FILTERING_PROMPT_1
from src.dataset_generation.generation_utility import numbered_string

logging_config_path = project_root / 'src' / 'logging' / 'logging.json'
with open(logging_config_path, 'rt') as f:
    config = json.load(f)

for handler in config.get('handlers', {}).values():
    if 'filename' in handler:
        handler['filename'] = os.path.join(project_root, handler['filename'])

logging.config.dictConfig(config)
logger = logging.getLogger("generation_logs")


class QualityFilter:
    def __init__(self, llm: LLM = None):
        if llm is None:
            self.llm = GeminiFlash2_5()
        self.thinking_budget = 5000


    def get_score(self, response: str, score: str):
        match = re.search(rf"{score} Score\**:.*?\s*.*?(\d+)", response, re.DOTALL)
        if not match:
            raise ValueError(f"Score not found in the response: {response}")
        return int(match.group(1))

    @overload
    def are_quality_pairs(self, question: str, answer: str) -> bool: ...

    @overload
    def are_quality_pairs(self, question: List[str], answer: List[str]) -> List[bool]: ...

    def are_quality_pairs(self, question: Union[str, List[str]], answer: Union[str, List[str]]) -> Union[bool, List[bool]]:
        def _apply_prompt(q: str, a: str) -> dict:
            prompt = ANSWER_QUALITY_FILTERING_PROMPT_2.format(
                question=q, answer=a
            )
            return {
                "prompt": prompt,
                # "system_prompt": "",
                "temperature": 0.0,
                "thinking_budget": self.thinking_budget,
            }
        
        def _parse_result(result: str) -> bool:
            try:
                llm_conciseness = self.get_score(result, "Conciseness")
                llm_comprehensiveness = self.get_score(result, "Comprehensiveness")
                llm_proper_answer = self.get_score(result, "Question Assessment")
                llm_annotation = min(llm_conciseness, llm_comprehensiveness, llm_proper_answer)
                if llm_annotation >= 4:
                    return True
                else:
                    logger.info(f"Answer declined in quality check. Scores: Conciseness: {llm_conciseness}, Comprehensiveness: {llm_comprehensiveness}, Question Assessment: {llm_proper_answer}")
                    return False
            except ValueError as e:
                logger.error(f"Error getting score from LLM response for QA quality filtering: {e}")
                return False

        if isinstance(question, list) and isinstance(answer, list):
            is_batch = True
        else:
            is_batch = False
            question = [question]
            answer = [answer]

        if len(question) != len(answer):
            raise ValueError("Question and answer lists must have the same length.")
            
        prompts_to_run = []
        for q, a in zip(question, answer):
            prompts_to_run.append(_apply_prompt(q, a))
        
        if is_batch:  
            results_texts = self.llm.infer_batch_completion(prompts_to_run)
        else:
            results_texts = [self.llm.infer_completion(prompts_to_run[0])]
        
        parsed_results = []
        for result_text in results_texts:
            parsed_results.append(_parse_result(result_text))

        return parsed_results if is_batch else parsed_results[0]


    @overload
    def are_grounded(self, answer: str, reference_text: str, citation_strings: list[str]) -> bool: ...

    @overload
    def are_grounded(self, answer: List[str], reference_text: List[str], citation_strings: List[list[str]]) -> List[bool]: ...

    def are_grounded(self, answer: Union[str, List[str]], reference_text: Union[str, List[str]], citation_strings: Union[list[str], List[list[str]]]) -> Union[bool, List[bool]]:
        def _apply_prompt(a: str, ref: str, cit: list[str]) -> dict:
            prompt = ANSWER_GROUNDEDNESS_FILTERING_PROMPT_1.format(
                answer=a,
                reference=ref,
                citation_numbers=numbered_string(cit)
            )
            return {
                "prompt": prompt,
                # "system_prompt": "",
                "temperature": 0.0,
                "thinking_budget": self.thinking_budget,
            }
        
        def _parse_result(result: str) -> bool:
            try:
                llm_groundedness = self.get_score(result, "Groundedness")
                llm_citation_correctness = self.get_score(result, "Citation Correctness")
                llm_annotation = min(llm_groundedness, llm_citation_correctness)
                if llm_annotation >= 4:
                    return True
                else:
                    logger.info(f"Answer declined in groundedness check. Scores: Groundedness: {llm_groundedness}, Citation Correctness: {llm_citation_correctness}")
                    return False
            except ValueError as e:
                logger.error(f"Error getting score from LLM response for Groundedness quality filtering: {e}")
                return False

        if isinstance(answer, list) and isinstance(reference_text, list) and isinstance(citation_strings, list):
            is_batch = True
        else:
            is_batch = False
            answer = [answer]
            reference_text = [reference_text]
            citation_strings = [citation_strings]

        if len(answer) != len(reference_text) or len(answer) != len(citation_strings):
            raise ValueError("Answer, reference_text, and citation_strings lists must have the same length.")
            
        prompts_to_run = []
        for a, ref, cit in zip(answer, reference_text, citation_strings):
            prompts_to_run.append(_apply_prompt(a, ref, cit))
        
        if is_batch:  
            results_texts = self.llm.infer_batch_completion(prompts_to_run)
        else:
            results_texts = [self.llm.infer_completion(prompts_to_run[0])]
        
        parsed_results = []
        for result_text in results_texts:
            parsed_results.append(_parse_result(result_text))

        return parsed_results if is_batch else parsed_results[0]