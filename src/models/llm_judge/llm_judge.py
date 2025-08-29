import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import re
import json
import pandas as pd
from typing import List, Optional, Tuple, cast
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.llms.llm import LLM
from src.models.llm_judge import judge_prompts
from src.column import Column


class LLMasJudge:
    def __init__(self, model: LLM):
        self.llm = model
        self.judgement_paragraphs_df = pd.read_csv('data/echr_case_paragraphs.csv')


    def claim_recall_llm(self, gold_answer: List[str], gen_answer: List[str]) -> List[int]:
        """
        Evaluates claim recall for batches of gold and generated answers.
        
        Args:
            gold_answer: List of gold standard answers
            gen_answer: List of generated answers to evaluate
            
        Returns:
            List of claim recall scores (0-5 scale, -1 for parsing errors, -2 for missing data)
        """
        def _apply_prompt(g_ans: str, gen_ans: str) -> dict:
            system = judge_prompts.CLAIM_RECALL_9_SYSTEM
            output_structure = judge_prompts.CLAIM_RECALL_9_OUTPUT_STRUCTURE
            prompt = judge_prompts.CLAIM_RECALL_9.format(
                gold_answer=g_ans,
                gen_answer=gen_ans,
                ouput_structure=output_structure,
            )
            return {
                "prompt": prompt,
                "system_prompt": system,
                "temperature": 0.5,
            }
        
        def _parse_json_result(result: str) -> tuple[int, str]:
            if not result:
                raise ValueError("Received an empty result string.")
            if result.startswith("```json"):
                result = result[7:-3].strip()
            try:  
                json_result = json.loads(result)
                rating_reasoning = json_result['evaluation']['evaluation_reasoning']['reasoning']
                rating_score = int(json_result['evaluation']['rating'])
            except json.JSONDecodeError as e:
                rating_pattern = r'"rating":\s*"(\d)"'
                reasoning_pattern = r'"reasoning":\s"([^}]*)"'
                rating_score = re.search(rating_pattern, result)
                rating_reasoning = re.findall(reasoning_pattern, result)
                if not rating_score or not rating_reasoning:
                    raise ValueError(f"Could not parse JSON result: {result}. Error: {e}")
                rating_score = int(rating_score.group(1))
                rating_reasoning = rating_reasoning[-1]
            return rating_score, rating_reasoning


        if len(gold_answer) != len(gen_answer):
            raise ValueError("Gold_answer and gen_answer lists must have the same length.")
            
        prompts_to_run: List[Optional[dict]] = []
        for idx, (g_ans, gen_ans) in enumerate(zip(gold_answer, gen_answer)):
            if pd.isna(gen_ans):
                print(f"Skipping claim recall for item {idx}: generated answer is NaN")
                prompts_to_run.append(None)
            else:
                prompts_to_run.append(_apply_prompt(g_ans, gen_ans))
        
        results_texts: List[Optional[str]] = []
        valid_prompts = [p for p in prompts_to_run if p is not None]
        if valid_prompts:
            try:
                batch_results = self.llm.infer_batch_completion(valid_prompts)
                if not batch_results:
                    raise ValueError("LLM returned empty results")
            except Exception as e:
                print(f"Critical error: LLM batch inference failed: {e}")
                raise


            batch_idx = 0
            for p in prompts_to_run:
                if p is None:
                    results_texts.append(None)
                else:
                    results_texts.append(batch_results[batch_idx])
                    batch_idx += 1
        else:
            results_texts = [None] * len(prompts_to_run)
        
        parsed_scores: List[int] = []
        for i, result_text in enumerate(results_texts):
            try:
                if result_text is None:
                    # Skipped due to missing generated answer (NaN)
                    parsed_scores.append(-2)
                    continue
                if not result_text:
                    print(f"Error: Empty LLM response for batch item {i}. Appending -1 as score.")
                    parsed_scores.append(-1)
                    continue
                    
                score, _ = _parse_json_result(result_text)
                if score < 0 or score > 6:
                    print(f"Error: Invalid rating value {score} for batch item {i}. Appending -1 as score.")
                    parsed_scores.append(-1)
                else:
                    parsed_scores.append(score)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error: JSON parsing or regex pattern matching failed for batch item {i}: {e}. Appending -1 as score.")
                parsed_scores.append(-1)
            except Exception as e:
                print(f"Unexpected error parsing batch item {i}: {e}. Appending -1 as score.")
                parsed_scores.append(-1)

        return parsed_scores
    

    def answer_faithfullness_llm(self, gen_answers_with_citations_batch: List[str]) -> List[tuple[int, int]]:
        """
        Evaluates faithfulness and citation correctness for batches of generated answers.
        
        Args:
            gen_answers_with_citations_batch: List of JSON strings, each containing
                                             one answer split into sentences with citations
                                              
        Returns:
            List of tuples containing (faithfulness_score, citation_correctness_score)
            Scores are 0-5 scale, (-1, -1) for parsing errors, (-2, -2) for missing/invalid data
        """
        def _transform_citations(gen_answers_with_citations_batch: List[str]) -> List[str]:
            transformed_answers_batch = []
            for json_answer in gen_answers_with_citations_batch:
                try:
                    if pd.isna(json_answer):
                        raise TypeError("Input is NaN")
                    answer_data = json.loads(json_answer)
                    transformed_sentences = []
                    
                    for sentence_item in answer_data:
                        new_item = {"sentence": sentence_item["sentence"]}
                        new_citations = {}
                        for i, citation_obj in enumerate(sentence_item["citations"], 1):
                            new_citations[str(i)] = citation_obj["paragraph_text"]
                        new_item["citations"] = new_citations
                        transformed_sentences.append(new_item)
                    
                    transformed_answers_batch.append(json.dumps(transformed_sentences, indent=4, ensure_ascii=False))
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"Error: json.loads or _transform_citations failed due to missing sentence or citations: {e}")
                    transformed_answers_batch.append(None)
            return transformed_answers_batch

        def _apply_faithfullness_prompt(sentences_with_citations: str) -> dict:
            system = judge_prompts.GROUNDEDNESS_SYSTEM
            output_structure = judge_prompts.GROUNDEDNESS_OUTPUT_STRUCTURE
            prompt = judge_prompts.GROUNDEDNESS_PROMPT.format(
                sentences_with_citations=sentences_with_citations,
                output_structure=output_structure,
            )
            return {
                "prompt": prompt,
                "system_prompt": system,
                "temperature": 0.5,
            }
        
        def _parse_json_result(result: str):
            if not result:
                raise ValueError("Received an empty result string.")
            if result.startswith("```json"):
                result = result[7:-3].strip()
            try:  
                json_result = json.loads(result)
                faithfullness_reasoning = json_result['faithfullness_evaluation']['faithfullness_evaluation_reasoning']['reasoning']
                faithfullness_score = int(json_result['faithfullness_evaluation']['faithfullness_rating'])
                citation_correctness_reasoning = json_result['citation_correctness_evaluation']['citation_correctness_evaluation_reasoning']['reasoning']
                citation_correctness_score = int(json_result['citation_correctness_evaluation']['citation_correctness_rating'])
            except json.JSONDecodeError as e:
                faithfullness_pattern = r'"faithfullness_rating":\s*"(\d)"'
                citation_pattern = r'"citation_correctness_rating":\s*"(\d)"'
                faithfullness_reasoning_pattern = r'"faithfullness_evaluation_reasoning":\s"([^}]*)"'
                citation_reasoning_pattern = r'"citation_correctness_evaluation_reasoning":\s"([^}]*)"'
                
                faithfullness_score = re.search(faithfullness_pattern, result)
                citation_correctness_score = re.search(citation_pattern, result)
                faithfullness_reasoning = re.search(faithfullness_reasoning_pattern, result)
                citation_correctness_reasoning = re.search(citation_reasoning_pattern, result)
                
                if not faithfullness_score or not citation_correctness_score:
                    raise ValueError(f"Could not parse JSON result: {result}. Error: {e}")
                
                faithfullness_score = int(faithfullness_score.group(1))
                citation_correctness_score = int(citation_correctness_score.group(1))
                faithfullness_reasoning = faithfullness_reasoning.group(1) if faithfullness_reasoning else ""
                citation_correctness_reasoning = citation_correctness_reasoning.group(1) if citation_correctness_reasoning else ""
            
            return faithfullness_score, faithfullness_reasoning, citation_correctness_score, citation_correctness_reasoning

        transformed_answers = _transform_citations(gen_answers_with_citations_batch)
        prompts_to_run: List[Optional[dict]] = []
        for transformed_answer in transformed_answers:
            if transformed_answer is None:
                prompts_to_run.append(None)
            else:
                prompts_to_run.append(_apply_faithfullness_prompt(transformed_answer))

        results_texts: List[Optional[str]] = []
        valid_prompts = [p for p in prompts_to_run if p is not None]
        if valid_prompts:
            try:
                batch_results = self.llm.infer_batch_completion(valid_prompts)
                if not batch_results:
                    raise ValueError("LLM returned empty results")
            except Exception as e:
                print(f"Critical error: LLM batch inference failed: {e}")
                raise

            batch_idx = 0
            for p in prompts_to_run:
                if p is None:
                    results_texts.append(None)
                else:
                    results_texts.append(batch_results[batch_idx])
                    batch_idx += 1
        else:
            results_texts = [None] * len(prompts_to_run)
        
        parsed_scores = []
        for i, result_text in enumerate(results_texts):
            try:
                if result_text is None:
                    # Skipped due to invalid input JSON for generated citations
                    parsed_scores.append((-2, -2))
                    continue
                if not result_text:
                    print(f"Error: Empty LLM response for batch item {i}. Appending (-1, -1) as score.")
                    parsed_scores.append((-1, -1))
                    continue
                    
                faithfullness_score, _, citation_score, _ = _parse_json_result(result_text)
                
                if (faithfullness_score < 0 or faithfullness_score > 5 or 
                    citation_score < 0 or citation_score > 5):
                    print(f"Error: Invalid rating values ({faithfullness_score}, {citation_score}) for batch item {i}. Appending (-1, -1) as score.")
                    parsed_scores.append((-1, -1))
                else:
                    parsed_scores.append((faithfullness_score, citation_score))
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error: JSON parsing or regex pattern matching failed for batch item {i}: {e}. Appending (-1, -1) as score.")
                parsed_scores.append((-1, -1))
            except Exception as e:
                print(f"Unexpected error parsing batch item {i}: {e}. Appending (-1, -1) as score.")
                parsed_scores.append((-1, -1))
        
        return parsed_scores


    def citation_semantic_similarity_llm(self, questions_batch: List[str], gold_answers_with_citations_batch: List[str], gen_answers_with_citations_batch: List[str]) -> List[int]:
        """
        Evaluates citation semantic similarity for batches of questions and answers.
        
        Args:
            questions_batch: List of questions
            gold_answers_with_citations_batch: List of JSON strings with gold answer citations
            gen_answers_with_citations_batch: List of JSON strings with generated answer citations
            
        Returns:
            List of citation semantic similarity scores (0-5 scale, -1 for errors, -2 for missing/invalid data)
        """
        def _get_unique_gen_citations(
            gen_sentence_with_citations: str,
        ) -> Optional[List[Tuple[str, int]]]:
            """
            Extracts unique citations from a JSON string of generated sentences.
            Preserves the order based on first appearance.
            Returns None if parsing fails.
            """
            seen: set[Tuple[str, int]] = set()
            ordered: List[Tuple[str, int]] = []
            try:
                if pd.isna(gen_sentence_with_citations):
                    raise TypeError("Input is NaN")
                data = json.loads(gen_sentence_with_citations)
                if not data:
                    return []
                for item in data:
                    for citation in item.get("citations", []):
                        case_id = citation.get("case_id")
                        para_num = citation.get("paragraph_number")
                        if case_id and para_num is not None:
                            key = (str(case_id), int(para_num))
                            if key not in seen:
                                seen.add(key)
                                ordered.append(key)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Error parsing gen_sentence_with_citations: {e}")
                return None
            return ordered


        def _get_unique_gold_citations(
            gold_answer_with_citations: str, language: str = "eng"
        ) -> Optional[List[Tuple[str, int]]]:
            """
            Extracts unique citations from a JSON string of gold sentences.
            Preserves the order based on first appearance.
            Returns None if a citation is missing a language-specific ID.
            """
            seen: set[Tuple[str, int]] = set()
            ordered: List[Tuple[str, int]] = []
            try:
                data = json.loads(gold_answer_with_citations)
                if not data:
                    return []
                for item in data:
                    for citation in item.get("citations", []):
                        lang_citation = citation.get("multilingual", {}).get(language)
                        if not lang_citation or "id" not in lang_citation:
                            original_url = citation.get("original_url", "Unknown URL")
                            print(
                                f"Error: No '{language}' citation id for citation: "
                                f"{original_url}"
                            )
                            return None

                        case_id = lang_citation.get("id")
                        paragraphs = citation.get("citation_paragraphs", [])
                        for para_num in paragraphs:
                            if case_id and para_num is not None:
                                key = (str(case_id), int(para_num))
                                if key not in seen:
                                    seen.add(key)
                                    ordered.append(key)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Error parsing gold_answer_with_citations: {e}")
                return None
            return ordered

        def _prompt_citations(gold_citations: List[List[tuple]], gen_citations: List[List[tuple]]) -> tuple[List[List[tuple]], List[List[tuple]], List[List[tuple]]]:
            """
            Compares gold and generated citations and returns unique and overlapping citations for batches.
            Returns three lists: unique_gen_batch, overlap_gen_gold_batch, unique_gold_batch
            """
            unique_gen_batch = []
            overlap_gen_gold_batch = []
            unique_gold_batch = []
            
            for gold, gen in zip(gold_citations, gen_citations):
                gold_set = set(gold) if gold is not None else set()
                gen_set = set(gen)

                unique_gen = [c for c in gen if c not in gold_set]
                overlap_gen_gold = [c for c in gen if c in gold_set]
                unique_gold = [c for c in gold if c not in gen_set]
                
                unique_gen_batch.append(unique_gen)
                overlap_gen_gold_batch.append(overlap_gen_gold)
                unique_gold_batch.append(unique_gold)
            
            return unique_gen_batch, overlap_gen_gold_batch, unique_gold_batch
        
        def _fetch_paragraphs(citations: List[tuple]) -> Optional[List[str]]:
            if not citations:
                return []
                
            try:
                judgement_paragraphs_df = self.judgement_paragraphs_df
                paragraphs = []
                missing_citations = []
                
                for case_id, paragraph_num in citations:
                    matching_row = judgement_paragraphs_df[
                        (judgement_paragraphs_df[Column.ECHR_CASE_ID] == case_id) & 
                        (judgement_paragraphs_df[Column.ECHR_CASE_PARAGRAPH_NUM] == paragraph_num)
                    ]
                    if not matching_row.empty:
                        case_name = matching_row.iloc[0][Column.ECHR_CASE_NAME]
                        paragraph_text = matching_row.iloc[0][Column.ECHR_CASE_PARAGRAPH_TEXT]
                        paragraphs.append(f"{case_name}, paragraph {paragraph_num}: {paragraph_text}")
                    else:
                        missing_citations.append((case_id, paragraph_num))
                
                if missing_citations:
                    print(f"Error: Could not fetch paragraphs for citations: {missing_citations}")
                    return None
                    
                return paragraphs
            except Exception as e:
                print(f"Error fetching paragraphs for citations {citations}: {e}")
                return None

        def _apply_ctoc_prompt(question: str, exclusive_gen: List[str], overlap_gen_gold: List[str], exclusive_gold: List[str]) -> dict:
            system = judge_prompts.CTOC_SYSTEM
            output_structure = judge_prompts.CTOC_OUTPUT_STRUCTURE
            prompt = judge_prompts.CTOC_PROMPT.format(
                question=question,
                exclusive_gen='\n'.join(exclusive_gen) if exclusive_gen else "None",
                overlap_gen_gold='\n'.join(overlap_gen_gold) if overlap_gen_gold else "None", 
                exclusive_gold='\n'.join(exclusive_gold) if exclusive_gold else "None",
                output_structure=output_structure
            )
            return {
                "prompt": prompt,
                "system_prompt": system,
                "temperature": 0.5,
            }

        def _parse_json_result(result: str) -> tuple[int, str]:
            if not result:
                raise ValueError("Received an empty result string.")
            if result.startswith("```json"):
                result = result[7:-3].strip()
            try:  
                json_result = json.loads(result)
                rating_reasoning = json_result['evaluation']['evaluation_reasoning']['reasoning']
                rating_score = int(json_result['evaluation']['rating'])
            except json.JSONDecodeError as e:
                rating_pattern = r'"rating":\s"(\d)"'
                reasoning_pattern = r'"reasoning":\s"([^}]*)"'
                rating_score = re.search(rating_pattern, result)
                rating_reasoning = re.findall(reasoning_pattern, result)
                if not rating_score:
                    raise ValueError(f"Could not parse JSON result: {result}. Error: {e}")
                rating_score = int(rating_score.group(1))
                rating_reasoning = rating_reasoning[-1]
            return rating_score, rating_reasoning

        if len(questions_batch) != len(gold_answers_with_citations_batch) or len(questions_batch) != len(gen_answers_with_citations_batch):
            raise ValueError("All input lists must have the same length.")

        unique_gen_citations = [_get_unique_gen_citations(gen_answer_with_citations) for gen_answer_with_citations in gen_answers_with_citations_batch]
        unique_gold_citations = [_get_unique_gold_citations(gold_answer_with_citations) for gold_answer_with_citations in gold_answers_with_citations_batch]
        
        prompts_to_run = []
        for i, question in enumerate(questions_batch):
            # Check if any citation extraction failed
            if unique_gold_citations[i] is None:
                print(f"Error: Gold citations judgement paragraph not available in selected language for batch item {i}. Skipping citation semantic similarity (will return -2).")
                prompts_to_run.append(None)
                continue
                
            if unique_gen_citations[i] is None:
                print(f"Error: Generated citations parsing failed for batch item {i}. Skipping citation semantic similarity (will return -2).")
                prompts_to_run.append(None)
                continue
            
            # Get citation sets for this specific item
            try:
                gold_cits = unique_gold_citations[i]
                gen_cits = unique_gen_citations[i]
                # cast for type checkers
                gold_cits_nn = cast(List[Tuple[str, int]], gold_cits)
                gen_cits_nn = cast(List[Tuple[str, int]], gen_cits)
                exclusive_gen, overlap_gen_gold, exclusive_gold = _prompt_citations([gold_cits_nn], [gen_cits_nn])
                paragraphs_gen = _fetch_paragraphs(exclusive_gen[0])
                paragraphs_overlap = _fetch_paragraphs(overlap_gen_gold[0])
                paragraphs_gold = _fetch_paragraphs(exclusive_gold[0])
                
                # Check if paragraph fetching failed
                if paragraphs_gen is None or paragraphs_overlap is None or paragraphs_gold is None:
                    print(f"Error: Paragraph fetching failed for batch item {i}. Skipping citation semantic similarity (will return -2).")
                    prompts_to_run.append(None)
                    continue
                    
                prompts_to_run.append(_apply_ctoc_prompt(question, paragraphs_gen, paragraphs_overlap, paragraphs_gold))
            except Exception as e:
                print(f"Error: Processing citations failed for batch item {i}: {e}. Skipping citation semantic similarity (will return -2).")
                prompts_to_run.append(None)

        results_texts = []
        valid_prompts = [p for p in prompts_to_run if p is not None]
        if valid_prompts:
            batch_results = self.llm.infer_batch_completion(valid_prompts)
            batch_idx = 0
            for prompt in prompts_to_run:
                if prompt is None:
                    results_texts.append(None)
                else:
                    results_texts.append(batch_results[batch_idx])
                    batch_idx += 1
        else:
            results_texts = [None] * len(prompts_to_run)

        parsed_scores = []
        for i, result_text in enumerate(results_texts):
            if result_text is None:
                # Skipped due to missing/invalid data (e.g., NaN input, failed citation extraction)
                parsed_scores.append(-2)
            else:
                try:
                    if not result_text:
                        print(f"Error: Empty LLM response for batch item {i}. Appending -1 as score.")
                        parsed_scores.append(-1)
                        continue
                        
                    score, _ = _parse_json_result(result_text)
                    
                    if score < 0 or score > 5:
                        print(f"Error: Invalid rating value {score} for batch item {i}. Appending -1 as score.")
                        parsed_scores.append(-1)
                    else:
                        parsed_scores.append(score)
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error: JSON parsing or regex pattern matching failed for batch item {i}: {e}. Appending -1 as score.")
                    parsed_scores.append(-1)
                except Exception as e:
                    print(f"Unexpected error parsing batch item {i}: {e}. Appending -1 as score.")
                    parsed_scores.append(-1)

        return parsed_scores

    def evaluate_csv_and_add_scores(
        self,
        csv_path: str,
        output_csv_path: Optional[str] = None,
        eval_n_rows_only: int = 2
    ):
        """
        Evaluates the CSV data using LLM judge sequentially and adds scores to the CSV.
        
        Args:
            csv_path: Path to the CSV file to evaluate
            output_csv_path: Path to save the updated CSV. If None, overwrites original.
            eval_n_rows_only: Number of rows to evaluate (default: 2, set to 0 or negative for all rows)
        """
        df = pd.read_csv(csv_path)

        if eval_n_rows_only > 0:
            df = df.head(eval_n_rows_only).copy()
            print(f"Evaluating first {eval_n_rows_only} rows only")

        # Initialize all score columns with -1
        df[Column.LLM_CLAIM_RECALL] = -1
        df[Column.LLM_FAITHFULNESS] = -1
        df[Column.LLM_CITATION_CORRECTNESS] = -1
        df[Column.LLM_CITATION_SEMANTIC_SIMILARITY] = -1

        # Extract all data for processing
        gold_answers_batch = df[Column.TARGET_ANSWER].tolist()
        generated_answers_batch = df[Column.GENERATED_ANSWER].tolist()
        questions_batch = df[Column.QUESTION].tolist()
        generated_citations_batch = df[Column.GENERATED_CITATIONS].tolist()

        print(f"Processing {len(df)} rows sequentially...")

        # Run claim recall
        print("Starting claim recall evaluation...")
        claim_recall_scores = self.claim_recall_llm(
            gold_answers_batch, generated_answers_batch
        )
        skipped_claim_recall = [
            i for i, score in enumerate(claim_recall_scores) if score == -2
        ]
        if skipped_claim_recall:
            print(
                f"Skipped {len(skipped_claim_recall)} rows in claim recall due to missing data (NaN): {skipped_claim_recall}"
            )
        print(f"Claim recall completed: {claim_recall_scores}")

        # Run faithfulness
        print("Starting faithfulness evaluation...")
        faithfulness_scores = self.answer_faithfullness_llm(generated_citations_batch)
        skipped_faithfulness = [
            i for i, (faith_score, cite_score) in enumerate(faithfulness_scores) 
            if faith_score == -2 or cite_score == -2
        ]
        if skipped_faithfulness:
            print(
                f"Skipped {len(skipped_faithfulness)} rows in faithfulness due to invalid citation JSON: {skipped_faithfulness}"
            )
        if faithfulness_scores:
            print(
                f"Faithfulness completed: "
                f"{ [f[0] for f in faithfulness_scores]} / {[f[1] for f in faithfulness_scores]}"
            )
        else:
            print("Faithfulness completed: [] / []")

        # Run semantic similarity
        print("Starting citation semantic similarity evaluation...")
        semantic_similarity_scores = self.citation_semantic_similarity_llm(
            questions_batch, gold_answers_batch, generated_citations_batch
        )
        skipped_rows = [
            i for i, score in enumerate(semantic_similarity_scores) if score == -2
        ]
        if skipped_rows:
            print(
                f"Skipped {len(skipped_rows)} rows due to citation/paragraph issues: {skipped_rows}"
            )
        print(f"Citation semantic similarity completed: {semantic_similarity_scores}")

        # Update dataframe with results
        df[Column.LLM_CLAIM_RECALL] = claim_recall_scores
        df[Column.LLM_FAITHFULNESS] = [s[0] for s in faithfulness_scores]
        df[Column.LLM_CITATION_CORRECTNESS] = [s[1] for s in faithfulness_scores]
        df[Column.LLM_CITATION_SEMANTIC_SIMILARITY] = semantic_similarity_scores

        # Save updated CSV
        output_path = output_csv_path if output_csv_path else csv_path
        df.to_csv(output_path, index=False)
        print(f"Updated CSV saved to: {output_path}")

        return df



if __name__ == "__main__":
    import argparse
    from src.llms.gemini import GeminiFlash2_5

    parser = argparse.ArgumentParser(description="Evaluate a CSV file with LLM as a Judge.")
    parser.add_argument(
        "filename",
        help="Path to the CSV file (relative to project root or absolute)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=-1,
        help="Number of last rows to evaluate. If not specified or < 1, all rows are evaluated.",
    )
    
    args = parser.parse_args()

    csv_path = args.filename
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        sys.exit(1)

    output_dir = os.path.dirname(csv_path)
    output_path = os.path.join(output_dir, f"eval_{os.path.basename(csv_path)}")

    llm = GeminiFlash2_5() 
    judge = LLMasJudge(llm) 
    
    judge.evaluate_csv_and_add_scores(csv_path, output_path, eval_n_rows_only=args.n)



OTHER_IDEAS = """
        1. **Relevance**: How well does the generated answer address the key points of the claim?
        2. **Accuracy**: Does the generated answer accurately reflect the legal principles and facts of the case?
        3. **Clarity**: Is the generated answer clear, concise, and well-structured?
        4. **Completeness**: Does the generated answer cover all necessary aspects of the claim?        5. **Legal Reasoning**: Does the generated answer demonstrate sound legal reasoning and analysis
        6. **Language Quality**: Is the generated answer free from grammatical errors and typos?
        7. **Consistency**: Is the generated answer consistent with the gold standard answer in terms of legal interpretation and conclusions?        Your evaluation should be a score between 0 and 1, where 0 means the generated answer is completely inadequate and 1 means it is perfect. 
        Please provide a detailed explanation of your evaluation, highlighting the strengths and weaknesses of the generated answer compared to the gold standard answer.
        Gold Answer: {gold_answer}
        Generated Answer: {gen_answer}
        Evaluation Score (0-1):
        """
