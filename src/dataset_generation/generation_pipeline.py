import os
import sys
import re
import json
import time
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import List, Union, Tuple, Optional, Any
import sqlite3

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from dotenv import load_dotenv
from tqdm import tqdm



project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from dataset_generation import generation_prompts
from dataset_generation.generation_utility import select_random_sample, numbered_string, filter_duplicates, write_to_db
from dataset_generation.quality_filtering import QualityFilter
from paragraph_embedding import get_cluster_embedder
from dataset_generation.citations.citation import Citation
from llms.gemini import GeminiFlash2_5
import logging

def get_logger(name):
    return logging.getLogger(name)

logger = get_logger("generation_logs")


class EmbeddingError(Exception):
    """Custom exception for embedding failures after all retries."""
    pass


class Fact(BaseModel):
    """Data model for a single extracted fact with citations."""
    text: str
    citations: list[Citation]


@dataclass
class GenerationConfig:
    lang_code: str
    grouping_method: str
    question_prompt_template: str
    answer_prompt_template: str
    model_name: str
    num_context_paras: int
    batch_mode: bool = False
    sequential_embedding: bool = False

    def __init__(self, lang_code, grouping_method, question_prompt_template, answer_prompt_template, num_context_paras, model_name, batch_mode=False, sequential_embedding=False):
        self.lang_code = lang_code
        self.grouping_method = grouping_method
        self.question_prompt_template = question_prompt_template.replace("LANG", lang_code.upper())
        self.answer_prompt_template = answer_prompt_template.replace("LANG", lang_code.upper())
        self.num_context_paras = num_context_paras
        self.model_name = model_name
        self.batch_mode = batch_mode
        self.sequential_embedding = sequential_embedding


def get_citations(guide_id: str, paragraph_ids: list[int], citations_db_path: str = os.path.join(project_root, "data/extracted_citations.db")) -> dict[int, dict[str, list[Any]]]:
    """Retrieve citations and errors for the given paragraphs from the citations db"""

    if not paragraph_ids:
        return {}

    # Ensure all IDs are ints
    paragraph_ids = [int(pid) for pid in paragraph_ids]

    conn = sqlite3.connect(f"file:{citations_db_path}?mode=ro", timeout=30.0, uri=True)
    cursor = conn.cursor()
    
    placeholders = ','.join('?' for _ in paragraph_ids)
    
    query = f"""
        SELECT paragraph_id, citations, errors
        FROM citation_extractions
        WHERE guide_id = ? AND paragraph_id IN ({placeholders})
    """
    cursor.execute(query, [guide_id] + paragraph_ids)
    results = cursor.fetchall()
    conn.close()

    citations_dict: dict[int, dict[str, list]] = {}
    for paragraph_id_str, citations_str, errors_str in results:
        try:
            citations_json = json.loads(citations_str)
        except Exception:
            citations_json = []
        try:
            errors = json.loads(errors_str)
        except Exception:
            errors = []

        citations: list[Citation] = []

        for sentence_cits in citations_json:
            for cit_dict in sentence_cits:
                cit = Citation.from_json(cit_dict)
                if any(existing.considered_same(cit) for existing in citations):
                    continue
                citations.append(cit)

        citations_dict[int(paragraph_id_str)] = {
            'citations': citations,
            'errors': errors
        }

    return citations_dict


def has_valid_citations(paragraph_ids: list[int], citation_data: dict[int, dict[str, list]]) -> bool:
    """Return True if
    - all paragraphs have a row in the citation database
    - at least one citation exists
    - there are no fatal errors.
    """

    if not citation_data:
        return False

    IGNORED_ERRORS = set([
        "This is an example of an error that can be ignored",
    ])

    citation_count = 0

    for paragraph_id in paragraph_ids:
        pid = int(paragraph_id)
        if pid not in citation_data:
            return False

        entry = citation_data[pid]

        relevant_errors = [e for e in entry.get("errors", []) if e not in IGNORED_ERRORS]
        if relevant_errors:
            return False

        citation_count += len(entry.get("citations", []))

    return citation_count > 0


def get_context_batch(questions: Union[str, List[str]], groups: Union[pd.DataFrame, List[pd.DataFrame]], guide_df: pd.DataFrame, guide_ids: Union[str, List[str]], group_ids: Union[int, List[int]], num_context_paras: int, cluster_embedder, sequential_embedding: bool = False) -> Union[Tuple[str, List[Citation], List[int]], List[Tuple[str, List[Citation], List[int]]]]:
    """
    Batch-aware context retrieval. Handles both single and batch processing.
    """
    is_batch = isinstance(questions, list)
    
    if not is_batch:
        questions = [questions]
        groups = [groups]
        guide_ids = [guide_ids]
        group_ids = [group_ids]
    
    question_embeddings = []
    MAX_RETRIES = 5
    
    for attempt in range(MAX_RETRIES):
        try:
            if is_batch and not sequential_embedding:
                question_embeddings = cluster_embedder.embed_documents(questions)
            else:
                question_embeddings = [cluster_embedder.embed_query(q) for q in tqdm(questions, desc="embed batch:", ncols=100, bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")]
            break
        except Exception as e:
            log_msg = f"Embedding failed on attempt {attempt + 1}/{MAX_RETRIES}."
            if not is_batch:
                log_msg += f" for guide {guide_ids[0]}, group {group_ids[0]}."
            log_msg += f" Error: {e}"
            logger.warning(log_msg)

            if attempt < MAX_RETRIES - 1:
                seconds_to_wait = 61 - datetime.now().second
                logger.info(f"Waiting for {seconds_to_wait} seconds before retrying.")
                time.sleep(seconds_to_wait)
            else:
                logger.error("Embedding failed after all retries. Raising error to halt processing.")
                raise EmbeddingError(
                    "Failed to get embedding after multiple retries. NOTE retry per batch, (increase MAX_RETRIES if needed)."
                ) from e

    results = []
    embeddings_matrix = np.array([eval(embedding) for embedding in guide_df['cluster_embedding']])

    for i, question_embedding in enumerate(question_embeddings):
        guide_id = guide_ids[i]
        group = groups[i]
        
        question_embedding_array = np.array(question_embedding).reshape(1, -1)
       
        similarities = cosine_similarity(question_embedding_array, embeddings_matrix)[0]
        guide_df_copy = guide_df.copy()
        guide_df_copy['similarity'] = similarities
        top_n_similar = guide_df_copy.nlargest(num_context_paras, 'similarity')
        candidate_ids = [int(pid) for pid in top_n_similar['paragraph_id'].tolist()
                         if int(pid) not in map(int, group['paragraph_id'].tolist())]
        
        if not candidate_ids:
            results.append(("", [], []))
            continue

        citation_info = get_citations(guide_id, candidate_ids)

        context_texts = []
        context_citations: list[Citation] = []
        context_paragraph_ids = []

        for pid in candidate_ids:
            entry = citation_info.get(pid)
            if not entry or not has_valid_citations([pid], {pid: entry}):
                continue
            text = guide_df.loc[guide_df['paragraph_id'] == pid, 'text'].iloc[0]
            context_texts.append(text)
            context_citations.extend(entry['citations'])
            context_paragraph_ids.append(pid)

        results.append(("\n".join(context_texts), context_citations, context_paragraph_ids))
    
    return results if is_batch else results[0]


def extract_question(response: str) -> str:
    question_words = ['Question', 'Întrebare']
    patterns = []
    for question_word in question_words:
        patterns.extend([f"\\*\\*{question_word}:\\*\\*", f"{question_word}:"])

    pattern = f"(?:{'|'.join(patterns)})\\s*(.*)"
    match = re.search(
        pattern,
        response,
        re.DOTALL
    )
    if match:
        question_text = match.group(1).strip()
    else:
        question_text = response.strip()
        logger.warning("No 'Question:' found in the response. Using the entire response as the question.")

    question_text = question_text.split("?")[0] + "?"
    return question_text


def question_gen_batch(groups: Union[pd.DataFrame, List[pd.DataFrame]], guide_ids: Union[str, List[str]], group_ids: Union[int, List[int]], config: GenerationConfig, llm) -> Union[str, List[str]]:
    """
    Batch-aware question generation. Handles both single and batch processing.
    """
    is_batch = isinstance(groups, list)
    
    if not is_batch:
        groups = [groups]
        guide_ids = [guide_ids]
        group_ids = [group_ids]
    
    prompts_to_run = []
    prompt_template = getattr(generation_prompts, config.question_prompt_template)
    
    for i, group in enumerate(groups):
        paragraphs = group["text"].str.cat(sep=' ')
        prompt = prompt_template.format(paragraphs=paragraphs)
        prompts_to_run.append({
            "prompt": prompt,
            "system_prompt": "Inside the answer template only place plain text, no formatting.",
            "guide_id": guide_ids[i],
            "group_id": group_ids[i],
            "grouping_method": config.grouping_method,
            "question_prompt_template": config.question_prompt_template
        })
    
    # Execute batch or single processing
    if is_batch and config.batch_mode:
        responses = llm.infer_batch_completion(prompts_to_run)
    else:
        responses = [llm.infer_completion(prompt_data["prompt"], **{k: v for k, v in prompt_data.items() if k != "prompt"}) for prompt_data in prompts_to_run]
    
    # Extract questions from responses
    questions = [extract_question(response) for response in responses]

    return questions if is_batch else questions[0]


def extract_bracket_nums(text: str):
    """
    Extracts citation numbers from the text and returns a tuple:
    (cleaned_text, set_of_citation_numbers)
    - Removes all citation patterns from the text.
    - Logs a warning if a citation is found more than once.
    """
    pattern = r" ?\[\d+(?:-\d+)?(?:, ?\d+(?:-\d+)?)*\]"
    all_ints = set()
    matches = list(re.finditer(pattern, text))
    for match in matches:
        items = re.findall(r"\d+(?:-\d+)?", match.group())
        for item in items:
            if "-" in item:
                start, end = map(int, item.split("-"))
                nums = range(start, end + 1)
            else:
                nums = [int(item)]
            for num in nums:
                if num in all_ints:
                    logger.warning(f"Citation number {num} found more than once in this split: {text}")
                all_ints.add(num)

    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text, all_ints


def parse_answer(answer: str, all_citations: list[Citation]) -> list[Fact]:
    """
    Parse the answer text into individual facts and associated citations.
    Splits sentences
    removes citation markers, and returns a list of Fact objects.
    """
    split_pattern = r'(?<=\]\.)'
    raw_sentences = re.split(split_pattern, answer)
    facts: list[Fact] = []
    for raw in raw_sentences:
        sentence = raw.strip()
        if not sentence:
            continue

        # Extract citation numbers
        clean_text, nums= extract_bracket_nums(sentence)
        citations_objs: list[Citation] = []
        for num in nums:
            if num <= 0 or num > len(all_citations):
                logger.warning(f"Citation index {num} out of range for all_citations (1-{len(all_citations)})")
                continue
            citations_objs.append(all_citations[num-1])

        facts.append(Fact(text=clean_text, citations=citations_objs))
    return facts


def answer_gen_batch(questions: Union[str, List[str]], paragraphs: Union[str, List[str]], citations: Union[List[str], List[List[str]]], guide_ids: Union[str, List[str]], group_ids: Union[int, List[int]], config: GenerationConfig, llm) -> Union[str, List[str]]:
    """
    Batch-aware answer generation. Handles both single and batch processing.
    """
    is_batch = isinstance(questions, list)
    
    if not is_batch:
        questions = [questions]
        paragraphs = [paragraphs]
        citations = [citations]
        guide_ids = [guide_ids]
        group_ids = [group_ids]
    
    # Prepare prompts for all questions
    prompts_to_run = []
    prompt_template = getattr(generation_prompts, config.answer_prompt_template)
    
    for i, question in enumerate(questions):
        try:
            prompt = prompt_template.format(
                question=question,
                paragraphs=paragraphs[i],
                judgement_citations=numbered_string(citations[i])
            )
        except KeyError as e:
            msg = f"Your prompt template is incorrect. It is missing the key: {e}"
            raise ValueError(msg)
        
        prompts_to_run.append({
            "prompt": prompt,
            "system_prompt": "Respond with plain text, no formatting.",
            "guide_id": guide_ids[i],
            "group_id": group_ids[i],
            "grouping_method": config.grouping_method,
            "answer_prompt_template": config.answer_prompt_template,
            "question": question
        })
    
    # Execute batch or single processing
    if is_batch and config.batch_mode:
        responses = llm.infer_batch_completion(prompts_to_run)
    else:
        responses = [llm.infer_completion(prompt_data["prompt"], **{k: v for k, v in prompt_data.items() if k != "prompt"}) for prompt_data in prompts_to_run]
    
    # Clean responses
    answers = [response.strip() for response in responses]
    
    return answers if is_batch else answers[0]


def create_result_row(guide_id, group_id, question, answer, group_data, config, context_paragraph_ids, guides_df):
    """Helper function to create a result row with consistent structure"""
    answer_json = json.dumps([fact.model_dump() for fact in answer], ensure_ascii=False)
    main_paragraph_ids = group_data['paragraph_id'].tolist()
    all_paragraph_ids = main_paragraph_ids.copy()
    if context_paragraph_ids:
        all_paragraph_ids += [pid for pid in context_paragraph_ids if pid not in all_paragraph_ids]

    all_paragraphs_text = guides_df[
        (guides_df['guide_id'] == guide_id) & (guides_df['paragraph_id'].isin(all_paragraph_ids))
    ]["text"].str.cat(sep=' ')

    return {
        'guide_id': guide_id,
        'group_id': int(group_id),
        'question': question,
        'answer': answer_json,
        'paragraphs': all_paragraphs_text,
        'paragraph_nums': json.dumps(all_paragraph_ids),
        'num_paragraphs': len(group_data),
        'grouping_method': config.grouping_method,
        'model_name': config.model_name,
        'lang_code': config.lang_code
    }


class GenerationPipeline:
    """
    Main pipeline class for generating questions and answers with batch processing support.
    """
    
    def __init__(self, config: GenerationConfig, llm, cluster_embedder, guides_df, nlp, output_db_path: str):
        self.config = config
        self.llm = llm
        self.cluster_embedder = cluster_embedder
        self.guides_df = guides_df
        self.nlp = nlp
        self.output_db_path = output_db_path
    
    def process_groups_unified(self, groups_data: Union[Tuple[pd.DataFrame, str, int], List[Tuple[pd.DataFrame, str, int]]]) -> Union[Tuple[Optional[str], Optional[List[Fact]], List[int]], List[Tuple[Optional[str], Optional[List[Fact]], List[int]]]]:
        """
        Unified processing method that handles both single and batch mode.
        Similar to llm_judge claim_recall_llm pattern.
        """
        is_batch = isinstance(groups_data, list)
        
        if not is_batch:
            groups_data = [groups_data]
        
        # Filter groups with valid citations first
        valid_groups_data = []
        valid_indices = []
        for i, (group, guide_id, group_id) in enumerate(groups_data):
            paragraph_ids = group['paragraph_id'].tolist()
            citation_data = get_citations(guide_id, paragraph_ids=paragraph_ids)
            if has_valid_citations(paragraph_ids, citation_data):
                valid_groups_data.append((group, guide_id, group_id, citation_data))
                valid_indices.append(i)

        print(f"Valid groups found: {len(valid_groups_data)}")

        # Extract data for batch processing
        groups = [item[0] for item in valid_groups_data]
        guide_ids = [item[1] for item in valid_groups_data]
        group_ids = [item[2] for item in valid_groups_data]
        citation_data_list = [item[3] for item in valid_groups_data]
        
        # Generate questions (batch-aware)
        questions = question_gen_batch(groups, guide_ids, group_ids, self.config, self.llm)
        if not isinstance(questions, list):
            questions = [questions]
        logger.info(f"Generated {len(questions)} questions for {len(groups)} groups.")
        
        # Get context for each question, batched by guide_id
        questions_by_guide = {}
        for i, q in enumerate(questions):
            guide_id = guide_ids[i]
            if guide_id not in questions_by_guide:
                questions_by_guide[guide_id] = []
            questions_by_guide[guide_id].append({
                "question": q,
                "group": groups[i],
                "group_id": group_ids[i],
                "original_index": i
            })

        all_context_results_list = [None] * len(questions)
        desc = "Embedding questions and retrieving context"
        for guide_id, guide_questions_data in tqdm(questions_by_guide.items(), desc=desc, ncols=100, bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
            guide_df = self.guides_df[self.guides_df['guide_id'] == guide_id].copy()

            guide_questions = [d["question"] for d in guide_questions_data]
            guide_groups = [d["group"] for d in guide_questions_data]
            guide_group_ids = [d["group_id"] for d in guide_questions_data]
            
            guide_guide_ids = [guide_id] * len(guide_questions)

            context_results_batch = get_context_batch(
                guide_questions,
                guide_groups,
                guide_df,
                guide_guide_ids,
                guide_group_ids,
                self.config.num_context_paras,
                self.cluster_embedder,
                sequential_embedding=self.config.sequential_embedding
            )

            for i, context_result in enumerate(context_results_batch):
                original_index = guide_questions_data[i]["original_index"]
                all_context_results_list[original_index] = context_result
        
        all_context_results = all_context_results_list
        
        # Prepare data for answer generation
        all_paragraphs_texts = []
        citation_strings_list = []
        all_citations_list = []
        
        for i, (context_paragraphs, context_citations, context_paragraph_ids) in enumerate(all_context_results):
            all_citations = filter_duplicates(
                [cit for entry in citation_data_list[i].values() for cit in entry.get('citations', [])],
                context_citations
            )
            citation_strings = [citation.citation_string() for citation in all_citations]
            
            paragraphs_text = groups[i]["text"].str.cat(sep='\n')
            all_paragraphs_text = paragraphs_text + "\n" + context_paragraphs
            
            all_paragraphs_texts.append(all_paragraphs_text)
            citation_strings_list.append(citation_strings)
            all_citations_list.append(all_citations)
        
        # Generate answers
        print(f"Generating answers for {len(questions)} questions.")
        answers = answer_gen_batch(questions, all_paragraphs_texts, citation_strings_list, guide_ids, group_ids, self.config, self.llm)
        if not isinstance(answers, list):
            answers = [answers]
        
        # Quality checks
        q = QualityFilter()
        quality_results = q.are_quality_pairs(question=questions, answer=answers)   

        # Only run grounding check on quality pairs to save compute
        quality_passed_indices = [i for i, passed in enumerate(quality_results) if passed]
        quality_passed_answers = [answers[i] for i in quality_passed_indices]
        quality_passed_paragraphs = [all_paragraphs_texts[i] for i in quality_passed_indices]
        quality_passed_citations = [citation_strings_list[i] for i in quality_passed_indices]
        
        print(f"Running grounding check on quality passed pairs {len(quality_passed_indices)}/{len(answers)}.")
        grounding_results_subset = q.are_grounded(quality_passed_answers, quality_passed_paragraphs, quality_passed_citations)
        
        # Transform back to original indexes with None where calculation was not done
        grounding_results = [None] * len(questions)
        for i, original_idx in enumerate(quality_passed_indices):
            grounding_results[original_idx] = grounding_results_subset[i]
        
        # Statistics tracking
        total_pairs = len(groups_data)
        no_valid_citations = total_pairs - len(valid_groups_data)
        quality_passed = sum(quality_results)
        quality_failed = len(quality_results) - quality_passed
        grounding_passed = sum(1 for r in grounding_results if r is True)
        grounding_failed = sum(1 for r in grounding_results if r is False)
        
        logger.info(f"Batch statistics: Total pairs: {total_pairs}, No valid citations: {no_valid_citations}, Quality passed: {quality_passed}, Quality failed: {quality_failed}, Grounding passed: {grounding_passed}, Grounding failed: {grounding_failed}")
        
        # Process results
        final_results = [(None, None, []) for _ in groups_data]
        
        for i, (question, answer, quality_ok, grounding_ok) in enumerate(zip(questions, answers, quality_results, grounding_results)):
            original_index = valid_indices[i]
            
            if not quality_ok:
                logger.info(f"QA pair for guide {guide_ids[i]}, group {group_ids[i]} did not meet quality standards.")
                continue
            
            if not grounding_ok:
                logger.info(f"QA pair for guide {guide_ids[i]}, group {group_ids[i]} did not meet groundedness standards.")
                continue
            
            answer_facts: list[Fact] = parse_answer(answer, all_citations_list[i])
            context_paragraph_ids = all_context_results[i][2]
            
            final_results[original_index] = (question, answer_facts, context_paragraph_ids)
        
        return final_results if is_batch else final_results[0]
    
    def process_single_group(self, group: pd.DataFrame, guide_id: str, group_id: int) -> Tuple[Optional[str], Optional[List[Fact]], List[int]]:
        """
        Process a single group - delegates to unified method.
        """
        return self.process_groups_unified((group, guide_id, group_id))
    
    def process_batch_groups(self, groups_data: List[Tuple[pd.DataFrame, str, int]]) -> List[Tuple[Optional[str], Optional[List[Fact]], List[int]]]:
        """
        Process multiple groups in batch mode - delegates to unified method.
        """
        return self.process_groups_unified(groups_data)


def process_groups(pipeline: GenerationPipeline, grouped_df) -> pd.DataFrame:
    """
    Process groups either sequentially or in batch mode based on configuration.
    """
    results = []
    
    if pipeline.config.batch_mode:
        # Collect all groups for batch processing
        all_groups_data = []
        for (guide_id, group_id), group_data in grouped_df:
            all_groups_data.append((group_data.copy(), guide_id, int(group_id)))
        
        # Process all groups in one batch
        try:
            batch_results = pipeline.process_batch_groups(all_groups_data)
            
            for i, (question, answer, context_paragraph_ids) in enumerate(batch_results):
                if question is None or answer is None:
                    continue
                
                group_data, guide_id, group_id = all_groups_data[i]
                result_row = create_result_row(
                    guide_id, group_id, question, answer, group_data, pipeline.config, context_paragraph_ids, pipeline.guides_df
                )
                results.append(result_row)
                
                # Save to database immediately
                write_to_db(result_row, output_db_path=pipeline.output_db_path)
                
        except EmbeddingError as e:
            logger.critical(
                f"Critical embedding error: {e}. Stopping processing and saving progress."
            )
        except Exception as e:
            logger.error(
                f"Unhandled error in batch processing: {type(e).__name__}: {e}."
            )
    else:
        # Sequential processing
        for (guide_id, group_id), group_data in grouped_df:
            try:
                question, answer, context_paragraph_ids = pipeline.process_single_group(
                    group_data.copy(), guide_id, int(group_id)
                )

                if question is None or answer is None:
                    continue

                result_row = create_result_row(
                    guide_id, group_id, question, answer, group_data, pipeline.config, context_paragraph_ids, pipeline.guides_df
                )
                results.append(result_row)
                
                # Save to database immediately
                write_to_db(result_row, output_db_path=pipeline.output_db_path)
                
            except EmbeddingError as e:
                logger.critical(
                    f"Critical embedding error: {e}. Stopping processing and saving progress."
                )
                break
            except Exception as e:
                logger.error(
                    f"Unhandled error processing guide_id={guide_id}, group_id={group_id}: {type(e).__name__}: {e}. Skipping item."
                )
                continue

    return pd.DataFrame(results)


def parse_arguments():
    """
    Parse command line arguments for the generation pipeline.
    """
    parser = argparse.ArgumentParser(description="Generate questions and answers from legal documents")
    
    parser.add_argument("--lang-code", default="ron", help="Language code (default: ron)")
    parser.add_argument("--grouping-method", default="group_full_threshold_0.94_max_length_5000", 
                        help="Grouping method (default: group_full_threshold_0.94_max_length_5000)")
    parser.add_argument("--question-template", default="SYSTEMATIC_LEGAL_QUESTION_GENERATION_ENG_2_LANG",
                        help="Question prompt template (default: SYSTEMATIC_LEGAL_QUESTION_GENERATION_ENG_2_LANG)")
    parser.add_argument("--answer-template", default="SIMPLE_ANSWER_PROSE_GENERATION_ENG_2_RON",
                        help="Answer prompt template (default: SIMPLE_ANSWER_PROSE_GENERATION_ENG_2_RON)")
    parser.add_argument("--num-context-paras", type=int, default=5,
                        help="Number of context paragraphs (default: 5)")
    parser.add_argument("--model-name", default="gemini-2.5-flash-preview-05-20",
                        help="Model name (default: gemini-2.5-flash-preview-05-20)")
    parser.add_argument("--batch", action="store_true", help="Enable batch processing mode")
    parser.add_argument("--sequential-embedding", action="store_true", help="Enable sequential embedding mode")
    parser.add_argument("--n", type=int, default=-1, help="Number of samples to process (-1 for all)")
    
    return parser.parse_args()


def main():
    load_dotenv()
    
    args = parse_arguments()

    args.batch = True
    
    config = GenerationConfig(
        lang_code=args.lang_code,
        grouping_method=args.grouping_method,
        question_prompt_template=args.question_template.replace("LANG", args.lang_code.upper()),
        answer_prompt_template=args.answer_template.replace("LANG", args.lang_code.upper()),
        num_context_paras=args.num_context_paras,
        model_name=args.model_name,
        batch_mode=args.batch,
        sequential_embedding=args.sequential_embedding
    )
    
    logger.info(f"Starting generation pipeline with batch_mode={config.batch_mode}")
    
    # Initialize components
    llm = GeminiFlash2_5()
    cluster_embedder = get_cluster_embedder()
    
    output_filename = os.path.join(project_root, f"data/generated_questions_{config.lang_code}_{config.grouping_method}_{config.model_name.replace('/', '_')}.csv")
    output_db_path = os.path.join(project_root, "data/output_generation_pipeline.db")
    
    # download first using: python -m spacy download <model_name>
    spacy_models = {
        "eng": "en_core_web_trf",
        "ron": "ro_core_news_lg",
        "fra": "fr_dep_news_trf",
    }
    # nlp = spacy.load(spacy_models[config.lang_code])
    nlp = None

    # Load guides data
    guides_df_path = os.path.join(project_root, f"data/echr_case_law_guides_grouped_{config.lang_code}.csv")
    guides_df = pd.read_csv(guides_df_path)
    
    # Apply sampling if specified
    if args.n > 0:
        guides_df = select_random_sample(
            guides_df=guides_df, 
            grouping_method=config.grouping_method, 
            num_guides=1, 
            num_groups=15, 
            random_seed=113
        )
    
    print(f"Selected number of groups: {guides_df[[config.grouping_method, 'guide_id']].drop_duplicates().shape[0]}")
    
    # Create pipeline and process groups
    pipeline = GenerationPipeline(config, llm, cluster_embedder, guides_df, nlp, output_db_path)
    grouped_df = guides_df.groupby(["guide_id", config.grouping_method])
    
    results_df = process_groups(pipeline, grouped_df)
    
    # Save results to CSV
    results_df.to_csv(output_filename, index=False)
    logger.info(f"Results saved to {output_filename}")
    logger.info(f"Generated {len(results_df)} questions")
    
    return results_df


if __name__ == "__main__":
    main()