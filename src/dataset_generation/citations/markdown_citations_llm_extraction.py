"""Refactored citation extraction with improved thread management."""
import argparse
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FuturesTimeoutError
from queue import Queue, Empty
import itertools

import pandas as pd
from langchain.prompts.prompt import PromptTemplate

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.llms.llm import LLM
from src.llms.gemini import GeminiFlash2_0, GeminiFlash2_5
from src.dataset_generation.citations.citation import Citation
from src.dataset_generation.citations.judgement_fetching import JudgementFetcher
from src.dataset_generation.citations.config import config
from src.dataset_generation.citations.database_manager import db_manager

# logging
import logging
import logging.config
logging_config_path = os.path.join(project_root, 'src/logging', 'logging.json')
with open(logging_config_path, 'rt') as f:
    logging_config = json.load(f)
logging_dir = os.path.dirname(logging_config_path)
for handler in logging_config.get('handlers', {}).values():
    if 'filename' in handler:
        handler['filename'] = os.path.join(logging_dir, handler['filename'])
        log_file_dir = os.path.dirname(handler['filename'])
        if not os.path.exists(log_file_dir):
            os.makedirs(log_file_dir)
logging.config.dictConfig(logging_config)
logger = logging.getLogger("md_citations_extraction_logs")



SYSTEM_PROMPT = """
You need to perform a task in sentence extraction.
You are given a paragraph and need to extract sentences with their according citations from it.

The stripped sentences do NOT contain citations and are stripped by any markdown formatting.
E.g. there is no bold, italic, links, escape characters, etc.
If the citation says "ibidem", this means the citation is the same as the previous one! Put the previous citations text in "citation_extraction", and add ibidem to tags.
Ignore other citations that don't cite judgements. E.g. ignore (Culegere, vol. VI, p. 132).

Please only output the structured JSON object, without any additional text! Make sure to escape any double quotes in the JSON output.
"""

USER_PROMPT_TEMPLATE = """
Here is an example for your reference:
EXAMPLE********
Example input:
{INPUT_EXAMPLE}

Example output:
{FORMATED_OUTPUT_EXAMPLE}
EXAMPLE********


Now it's your turn:

Real input:
paragraph_text: '{INPUT}'"""


INPUT_EXAMPLE = """
paragraph_text: \"În temeiul art. 1 din Convenție, angajamentul asumat de statele contractante se limitează la \"recunoașterea\" (în limba engleză, \"*to secure*\") drepturilor și libertăților enumerate ale persoanelor aflate sub \"jurisdicția\" lor. \"Jurisdicția\" în sensul art. 1 este o condiție *sine qua non*. Acesta trebuie să fie exercitată pentru ca un stat contractant să poată fi considerat răspunzător pentru acte sau omisiuni imputabile, care se află la originea unei presupuse încălcări a drepturilor și libertăților enunțate în Convenție [*[Catan și alții împotriva Republicii Moldova și Rusiei](http://hudoc.echr.coe.int/fre?i=001-114222)* (MC), pct. 103, 2012, și jurisprudența citată]. Cererile îndreptate împotriva statelor membre cu privire la punerea în aplicare de către acestea a dreptului UE nu sunt, în principiu, incompatibile *ratione personae* cu Convenția [*[Bosphorus Hava](https://hudoc.echr.coe.int/eng?i=001-69564) [Yolları Turizm ve Ticaret Anonim Şirketi împotriva Irlandei](https://hudoc.echr.coe.int/eng?i=001-69564)* (MC), 2005, pct. 137; *[Michaud împotriva](https://hudoc.echr.coe.int/eng?i=001-115377) [Franței](https://hudoc.echr.coe.int/eng?i=001-115377)*, 2012, pct. 100, 102; *[Avotiņš împotriva Letoniei](https://hudoc.echr.coe.int/eng?i=001-163114)* (MC), 2016, pct. 101-105]. Având în vedere rolul partidelor politice, măsurile luate împotriva acestora afectează libertatea de asociere și, prin urmare, starea emocrației din țara respectivă (ibidem, pct. 31).\"
"""

FORMATED_OUTPUT_EXAMPLE_3 = r"""
{
    "sentences": [
        {
            "stripped_sentence": "În temeiul art. 1 din Convenție, angajamentul asumat de statele contractante se limitează la \"recunoașterea\" (în limba engleză, \"to secure\") drepturilor și libertăților enumerate ale persoanelor aflate sub \"jurisdicția\" lor. \"Jurisdicția\" în sensul art. 1 este o condiție sine qua non.",
            "citations": null,
        }
        {
            "stripped_sentence": "Acesta trebuie să fie exercitată pentru ca un stat contractant să poată fi considerat răspunzător pentru acte sau omisiuni imputabile, care se află la originea unei presupuse încălcări a drepturilor și libertăților enunțate în Convenție.",
            "citations": [
                {
                    "citation_url": "http://hudoc.echr.coe.int/fre?i=001-114222",
                    "citation_year": 2012,
                    "citation_tags": ["MC", "și jurisprudența citată"],
                    "citation_paragraphs": [103],
                }
        {
            "stripped_sentence": "Cererile îndreptate împotriva statelor membre cu privire la punerea în aplicare de către acestea a dreptului UE nu sunt, în principiu, incompatibile ratione personae cu Convenția.",
            "citations": [
                {
                    "citation_url": "https://hudoc.echr.coe.int/eng?i=001-69564",
                    "citation_year": 2005,
                    "citation_tags": ["MC"],
                    "citation_paragraphs": [137],
                },
                {
                    "citation_url": "https://hudoc.echr.coe.int/eng?i=001-115377",
                    "citation_year": 2012,
                    "citation_tags": [],
                    "citation_paragraphs": [100, 102],
                },
                {
                    "citation_url": "https://hudoc.echr.coe.int/eng?i=001-163114",
                    "citation_year": 2016,
                    "citation_tags": ["MC"],
                    "citation_paragraphs": [101, 102, 103, 104, 105],
                }
            ]
        }
        {
            "stripped_sentence": "Având în vedere rolul partidelor politice, măsurile luate împotriva acestora afectează libertatea de asociere și, prin urmare, starea emocrației din țara respectivă.",
            "citations": [
                {
                    "citation_url": "https://hudoc.echr.coe.int/eng?i=001-163114",
                    "citation_year": 2016,
                    "citation_tags": ["MC", "ibidem"],
                    "citation_paragraphs": [31],
                }
            ]
        }
    ]
}
"""


@dataclass
class ExtractionTask:
    """Represents a single extraction task."""
    guide_id: str
    paragraph_id: int
    paragraph_text: str
    lang_code: str
    attempt: int = 0
    start_time: float = field(default_factory=time.time)
    future: Optional[Future] = None


@dataclass
class ExtractionResult:
    """Result of an extraction task."""
    task: ExtractionTask
    sentences: List[str]
    citations: List[List[Citation]]
    errors: List[str]
    processing_time: float


class TaskQueue:
    """Thread-safe task queue with priority support."""
    
    def __init__(self, max_size: int = 10000):
        self._queue = Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._completed: Set[Tuple[str, int]] = set()
        self._in_progress: Dict[Tuple[str, int], ExtractionTask] = {}
    
    def add(self, task: ExtractionTask) -> bool:
        """Add task to queue if not already processed."""
        key = (task.guide_id, task.paragraph_id)
        
        with self._lock:
            if key in self._completed or key in self._in_progress:
                return False
            
            try:
                self._queue.put_nowait(task)
                return True
            except:
                return False
    
    def get(self, timeout: float = 1.0) -> Optional[ExtractionTask]:
        """Get next task from queue."""
        try:
            task = self._queue.get(timeout=timeout)
            with self._lock:
                key = (task.guide_id, task.paragraph_id)
                self._in_progress[key] = task
            return task
        except Empty:
            return None
    
    def mark_completed(self, task: ExtractionTask):
        """Mark task as completed."""
        key = (task.guide_id, task.paragraph_id)
        with self._lock:
            self._in_progress.pop(key, None)
            self._completed.add(key)
    
    def mark_failed(self, task: ExtractionTask):
        """Mark task as failed (can be retried)."""
        key = (task.guide_id, task.paragraph_id)
        with self._lock:
            self._in_progress.pop(key, None)
    
    def get_stuck_tasks(self, timeout: float) -> List[ExtractionTask]:
        """Get tasks that have been running too long."""
        current_time = time.time()
        stuck = []
        
        with self._lock:
            for task in self._in_progress.values():
                if current_time - task.start_time > timeout:
                    stuck.append(task)
        
        return stuck
    
    @property
    def size(self) -> int:
        """Get queue size."""
        return self._queue.qsize()
    
    @property
    def in_progress_count(self) -> int:
        """Get number of tasks in progress."""
        with self._lock:
            return len(self._in_progress)
    
    @property
    def completed_count(self) -> int:
        """Get number of completed tasks."""
        with self._lock:
            return len(self._completed)


class CitationExtractor:
    """Improved citation extractor with better error handling."""
    
    def __init__(self, llm: LLM, judgement_fetcher: JudgementFetcher):
        self.llm = llm
        self.judgement_fetcher = judgement_fetcher
        self._extraction_stats = {
            'successful': 0,
            'failed': 0,
            'retried': 0,
            'timeouts': 0
        }
        self._stats_lock = threading.Lock()
    
    def _update_stats(self, stat_name: str):
        """Update extraction statistics."""
        with self._stats_lock:
            self._extraction_stats[stat_name] += 1
    
    def _is_valid_hudoc_url(self, url: str) -> bool:
        """Check if URL is a valid HUDOC URL."""
        return url.startswith('https://hudoc.echr.coe.int/')
    
    def _validate_citation_availability(self, citation_obj: Citation, sentence_index: int) -> List[str]:
        """Validate if citation can be fetched and check year consistency."""
        errors = []
        
        try:
            if not citation_obj.multilingual:
                errors.append(f"No metadata for {citation_obj.original_url}")
                return errors
            
            eng_version = citation_obj.multilingual.get("eng")
            if not eng_version:
                errors.append(f"No English translation for {citation_obj.original_url}")
                return errors
            
            eng_case_id = eng_version.get("id")
            if not eng_case_id:
                errors.append(f"Missing English case ID for {citation_obj.original_url}")
                return errors
            
            # Check year mismatch between citation_year and HUDOC metadata
            if citation_obj.citation_year and citation_obj.citation_date:
                metadata_year = int(citation_obj.citation_date[:4])
                if citation_obj.citation_year != metadata_year:
                    errors.append(
                        f"Year mismatch for citation in sentence {sentence_index + 1} ({citation_obj.original_url}). "
                        f"Citation year {citation_obj.citation_year} does not match "
                        f"HUDOC date {citation_obj.citation_date}."
                    )
            
            # Check if judgment can be fetched
            fetched = self.judgement_fetcher.fetch_judgement(case_id=eng_case_id)
            if not fetched:
                errors.append(f"Failed to fetch English judgment {eng_case_id}")
            elif fetched == 'Paragraph missing':
                errors.append(f"Paragraph missing for {eng_case_id}")
            
        except Exception as e:
            errors.append(f"Validation error for {citation_obj.original_url}: {e}")
        
        return errors
    
    def _extract_json(self, response: str) -> Tuple[Optional[Dict], List[str]]:
        """Extract and fix JSON from LLM response."""
        errors = []
        
        # Try to extract JSON from markdown code block
        pattern = r'```json\n([\s\S]*?)\n```'
        match = re.search(pattern, response)
        json_str = match.group(1) if match else response.strip()
        
        try:
            return json.loads(json_str), errors
        except json.JSONDecodeError as e:
            # Try to recover from common JSON errors
            try:
                # Fix unescaped quotes
                fixed_json = self._fix_json_quotes(json_str)
                return json.loads(fixed_json), errors
            except:
                errors.append(f"Invalid JSON: {e}")
                return None, errors
    
    def _fix_json_quotes(self, json_str: str) -> str:
        """Attempt to fix unescaped quotes in JSON."""
        # This is a simplified fix - in production, use a proper JSON repair library
        result = []
        in_string = False
        escape_next = False
        
        for i, char in enumerate(json_str):
            if escape_next:
                result.append(char)
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                result.append(char)
            elif char == '"':
                # Check context to determine if this should be escaped
                if in_string and i + 1 < len(json_str):
                    next_char = json_str[i + 1]
                    if next_char not in [',', ':', '}', ']', '\n', ' ']:
                        result.append('\\"')
                    else:
                        result.append(char)
                        in_string = False
                else:
                    result.append(char)
                    in_string = not in_string
            else:
                result.append(char)
        
        return ''.join(result)
    
    def _validate_links(
        self,
        citations: List[List[Citation]],
        sentences: List[str],
        paragraph_text: str
    ) -> List[str]:
        """Validate extracted citations against paragraph links."""
        errors = []
        
        # Extract links from paragraph
        links = re.findall(r'\((http.*?)\)', paragraph_text)
        links = [re.sub(r'http(?!s)', 'https', link) for link in links]
        # Normalize URLs to handle both formats
        links = [Citation.normalize_hudoc_url(link) for link in links]
        links = [key for key, _ in itertools.groupby(links)]  # Remove consecutive duplicates
        links = [link for link in links if self._is_valid_hudoc_url(link)]
        
        # Validate citation links match paragraph links
        j = 0
        for i, sentence_citations in enumerate(citations):
            if not sentence_citations:
                continue
            
            for citation in sentence_citations:
                if j < len(links) and citation.original_url == links[j]:
                    j += 1
                elif j > 0 and citation.original_url == links[j - 1]:
                    # Same link as previous (e.g., ibidem)
                    pass
                else:
                    errors.append(
                        f"Link mismatch for citation in sentence {i+1} ({citation.original_url}). "
                        f"Expected link around '{links[j] if j < len(links) else 'end of paragraph links (or no more links expected)'}'. "
                        f"Actual URL: {citation.original_url}"
                    )
        
        if j < len(links):
            errors.append(
                f"There are {len(links) - j} unconsumed links in the paragraph text, "
                f"starting with '{links[j]}'."
            )
        
        # Check for links in stripped sentences
        if re.findall(r'\((http.*?)\)', ' '.join(sentences)):
            errors.append(
                "There are still links in the stripped sentences, which should not be the case."
            )
        
        return errors
    
    def extract_citations(
        self,
        paragraph_text: str,
        guide_id: str,
        paragraph_id: int,
        lang_code: str
    ) -> Tuple[List[str], List[List[Citation]], List[str]]:
        """Extract citations from paragraph text."""
        try:
            # Prepare prompt
            prompt = PromptTemplate.from_template(USER_PROMPT_TEMPLATE).format(
                INPUT_EXAMPLE=INPUT_EXAMPLE,
                FORMATED_OUTPUT_EXAMPLE=FORMATED_OUTPUT_EXAMPLE_3,
                INPUT=paragraph_text
            )
            
            # Call LLM with timeout
            response = None
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.llm.infer_completion,
                    prompt=prompt,
                    temperature=0.0,
                    system_prompt=SYSTEM_PROMPT
                )
                
                try:
                    response = future.result(timeout=config.threading.llm_inference_timeout)
                except FuturesTimeoutError:
                    self._update_stats('timeouts')
                    return [], [], ["LLM timeout"]
                except Exception as e:
                    return [], [], [f"LLM error: {e}"]
            
            if not response:
                return [], [], ["No LLM response"]
            
            # Parse JSON response
            json_data, json_errors = self._extract_json(response)
            if json_errors:
                return [], [], json_errors
            
            if not json_data or 'sentences' not in json_data:
                return [], [], ["Invalid response structure"]
            
            # Extract sentences and citations
            sentences = []
            citations = []
            errors = []
            
            for sentence_idx, sentence_data in enumerate(json_data['sentences']):
                # Extract sentence
                sentence = sentence_data.get('stripped_sentence', '')
                sentence = re.sub(r'http(?!s)', 'https', sentence)
                sentences.append(sentence)
                
                # Extract citations
                sentence_citations = []
                citations_data = sentence_data.get('citations')
                
                if citations_data:
                    for citation_data in citations_data:
                        try:
                            url = re.sub(r'http(?!s)', 'https', citation_data['citation_url'])
                            
                            # Normalize the URL format before validation
                            url = Citation.normalize_hudoc_url(url)
                            
                            if not self._is_valid_hudoc_url(url):
                                continue
                            
                            # Check for empty paragraphs
                            if not citation_data.get('citation_paragraphs'):
                                errors.append(
                                    f"No paragraphs provided for citation in sentence {sentence_idx + 1} ({url})"
                                )
                            
                            citation = Citation.from_llm_extraction(
                                citation_url=url,
                                citation_year=citation_data.get('citation_year'),
                                citation_tags=citation_data.get('citation_tags', []),
                                citation_paragraphs=citation_data.get('citation_paragraphs', []),
                                lang_code=lang_code,
                                timeout=config.threading.api_call_timeout
                            )
                            
                            # Validate citation with sentence index for better error messages
                            validation_errors = self._validate_citation_availability(citation, sentence_idx)
                            errors.extend(validation_errors)
                            
                            sentence_citations.append(citation)
                            
                        except Exception as e:
                            errors.append(f"Citation extraction error: {e}")
                
                citations.append(sentence_citations)
            
            # Validate links only if there were no critical JSON errors
            if not json_errors:
                validation_errors = self._validate_links(citations, sentences, paragraph_text)
                errors.extend(validation_errors)
            else:
                errors.append("Skipping link validation due to earlier critical JSON structure errors.")
            
            if not errors:
                self._update_stats('successful')
            else:
                self._update_stats('failed')
            
            return sentences, citations, errors
            
        except Exception as e:
            logger.error(f"Extraction failed for {guide_id}/{paragraph_id}: {e}")
            self._update_stats('failed')
            return [], [], [f"Extraction error: {e}"]
    
    def get_stats(self) -> Dict[str, int]:
        """Get extraction statistics."""
        with self._stats_lock:
            return dict(self._extraction_stats)


class ExtractionWorker(threading.Thread):
    """Worker thread for processing extraction tasks."""
    
    def __init__(
        self,
        worker_id: int,
        task_queue: TaskQueue,
        result_queue: Queue,
        primary_extractor: CitationExtractor,
        fallback_extractor: CitationExtractor,
        shutdown_event: threading.Event
    ):
        super().__init__(name=f"Worker-{worker_id}")
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.primary_extractor = primary_extractor
        self.fallback_extractor = fallback_extractor
        self.shutdown_event = shutdown_event
        self.tasks_processed = 0
    
    def run(self):
        """Main worker loop."""
        logger.info(f"Worker {self.worker_id} started")
        
        while not self.shutdown_event.is_set():
            # Get next task
            task = self.task_queue.get(timeout=1.0)
            if not task:
                continue
            
            try:
                # Process task
                result = self._process_task(task)
                
                # Submit result
                self.result_queue.put(result)
                self.task_queue.mark_completed(task)
                self.tasks_processed += 1
                
                if self.tasks_processed % 10 == 0:
                    logger.info(f"Worker {self.worker_id} processed {self.tasks_processed} tasks")
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                self.task_queue.mark_failed(task)
                
                # Create error result
                error_result = ExtractionResult(
                    task=task,
                    sentences=[],
                    citations=[],
                    errors=[f"Worker error: {e}"],
                    processing_time=time.time() - task.start_time
                )
                self.result_queue.put(error_result)
        
        logger.info(f"Worker {self.worker_id} shutting down after {self.tasks_processed} tasks")
    
    def _process_task(self, task: ExtractionTask) -> ExtractionResult:
        """Process a single extraction task."""
        start_time = time.time()
        
        # Try primary extractor
        sentences, citations, errors = self.primary_extractor.extract_citations(
            task.paragraph_text,
            task.guide_id,
            task.paragraph_id,
            task.lang_code
        )
        
        # Retry with fallback if needed
        if self._should_retry(errors) and task.attempt == 0:
            logger.info(f"Retrying {task.guide_id}/{task.paragraph_id} with fallback")
            sentences, citations, errors = self.fallback_extractor.extract_citations(
                task.paragraph_text,
                task.guide_id,
                task.paragraph_id,
                task.lang_code
            )
        
        processing_time = time.time() - start_time
        
        return ExtractionResult(
            task=task,
            sentences=sentences,
            citations=citations,
            errors=errors,
            processing_time=processing_time
        )
    
    def _should_retry(self, errors: List[str]) -> bool:
        """Check if errors warrant a retry."""
        if not errors:
            return False
        
        retry_triggers = [
            'unconsumed links',
            'links in the stripped sentences',
            'Missing field',
            'Invalid JSON',
            'LLM timeout',
            'Invalid response structure'
        ]
        
        return all(
            any(trigger in error for trigger in retry_triggers)
            for error in errors
        )


class ExtractionOrchestrator:
    """Orchestrates the extraction process."""
    
    def __init__(
        self,
        num_workers: int = 16,
        lang_code: str = 'ron'
    ):
        self.num_workers = num_workers
        self.lang_code = lang_code
        self.task_queue = TaskQueue()
        self.result_queue = Queue()
        self.shutdown_event = threading.Event()
        self.workers: List[ExtractionWorker] = []
        
        # Initialize components
        self._initialize_extractors()
        self._initialize_workers()
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed': 0,
            'failed': 0,
            'timed_out': 0
        }
        self.stats_lock = threading.Lock()
    
    def _initialize_extractors(self):
        """Initialize extractors."""
        # Create LLMs
        primary_llm = GeminiFlash2_0()
        fallback_llm = GeminiFlash2_5()
        
        # Create shared judgement fetcher
        self.judgement_fetcher = JudgementFetcher()
        
        # Create extractors
        self.primary_extractor = CitationExtractor(primary_llm, self.judgement_fetcher)
        self.fallback_extractor = CitationExtractor(fallback_llm, self.judgement_fetcher)
    
    def _initialize_workers(self):
        """Initialize worker threads."""
        for i in range(self.num_workers):
            worker = ExtractionWorker(
                worker_id=i,
                task_queue=self.task_queue,
                result_queue=self.result_queue,
                primary_extractor=self.primary_extractor,
                fallback_extractor=self.fallback_extractor,
                shutdown_event=self.shutdown_event
            )
            worker.start()
            self.workers.append(worker)
    
    def process_dataframe(self, df: pd.DataFrame) -> List[ExtractionResult]:
        """Process all rows in dataframe."""
        results = []
        
        # Add all tasks to queue
        for _, row in df.iterrows():
            task = ExtractionTask(
                guide_id=row['guide_id'],
                paragraph_id=row['paragraph_id'],
                paragraph_text=row['paragraph'],
                lang_code=self.lang_code
            )
            
            if self.task_queue.add(task):
                with self.stats_lock:
                    self.stats['total_tasks'] += 1
        
        logger.info(f"Added {self.stats['total_tasks']} tasks to queue")
        
        # Process results
        result_thread = threading.Thread(
            target=self._process_results,
            args=(results,)
        )
        result_thread.start()
        
        # Monitor progress
        self._monitor_progress()
        
        # Wait for completion
        result_thread.join()
        
        return results
    
    def _process_results(self, results: List[ExtractionResult]):
        """Process results from workers."""
        while True:
            # Check if we're done
            with self.stats_lock:
                if self.stats['completed'] >= self.stats['total_tasks']:
                    break
            
            try:
                # Get result with timeout
                result = self.result_queue.get(timeout=1.0)
                
                # Save to database
                self._save_result(result)
                
                # Add to results list
                results.append(result)
                
                # Update statistics
                with self.stats_lock:
                    self.stats['completed'] += 1
                    if result.errors:
                        self.stats['failed'] += 1
                
                # Log progress
                if self.stats['completed'] % 50 == 0:
                    self._log_progress()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing result: {e}")
    
    def _save_result(self, result: ExtractionResult):
        """Save result to database."""
        try:
            # Serialize citations
            serialized_citations = [
                [c.to_json() for c in sentence_citations]
                for sentence_citations in result.citations
            ]
            
            # Save to database
            with db_manager.get_citations_connection(timeout=10.0) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO citation_extractions 
                    (guide_id, paragraph_id, paragraph_text, sentences, citations, errors, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    result.task.guide_id,
                    result.task.paragraph_id,
                    result.task.paragraph_text,
                    json.dumps(result.sentences, ensure_ascii=False),
                    json.dumps(serialized_citations, ensure_ascii=False),
                    json.dumps(result.errors, ensure_ascii=False)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save result for {result.task.guide_id}/{result.task.paragraph_id}: {e}")
    
    def _monitor_progress(self):
        """Monitor extraction progress."""
        last_check = time.time()
        stuck_check_interval = 60  # Check for stuck tasks every minute
        
        while True:
            # Check if done
            with self.stats_lock:
                if self.stats['completed'] >= self.stats['total_tasks']:
                    break
            
            # Check for stuck tasks periodically
            if time.time() - last_check > stuck_check_interval:
                stuck_tasks = self.task_queue.get_stuck_tasks(
                    timeout=config.threading.stuck_threshold
                )
                
                if stuck_tasks:
                    logger.warning(f"Found {len(stuck_tasks)} stuck tasks")
                    for task in stuck_tasks:
                        # Mark as timed out
                        with self.stats_lock:
                            self.stats['timed_out'] += 1
                        
                        # Create timeout result
                        timeout_result = ExtractionResult(
                            task=task,
                            sentences=[],
                            citations=[],
                            errors=[f"Task timed out after {time.time() - task.start_time:.0f}s"],
                            processing_time=time.time() - task.start_time
                        )
                        self.result_queue.put(timeout_result)
                
                last_check = time.time()
                self._log_progress()
            
            time.sleep(5)
    
    def _log_progress(self):
        """Log current progress."""
        with self.stats_lock:
            total = self.stats['total_tasks']
            completed = self.stats['completed']
            failed = self.stats['failed']
            timed_out = self.stats['timed_out']
            
            if total > 0:
                progress = 100 * completed / total
                logger.info(
                    f"Progress: {completed}/{total} ({progress:.1f}%) | "
                    f"Failed: {failed} | Timed out: {timed_out} | "
                    f"Queue: {self.task_queue.size} | "
                    f"In progress: {self.task_queue.in_progress_count}"
                )
    
    def shutdown(self):
        """Shutdown orchestrator."""
        logger.info("Shutting down orchestrator...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=10)
        
        # Clean up resources
        self.judgement_fetcher.close()
        
        # Log final statistics
        logger.info(f"Final statistics: {self.stats}")
        logger.info(f"Primary extractor stats: {self.primary_extractor.get_stats()}")
        logger.info(f"Fallback extractor stats: {self.fallback_extractor.get_stats()}")
        logger.info(f"Judgement fetcher stats: {self.judgement_fetcher.get_stats()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Extract citations from ECHR case law guides')
    parser.add_argument('--threads', type=int, default=16, help='Number of worker threads')
    parser.add_argument('--lang', type=str, default='ron', help='Language code')
    args = parser.parse_args()
    
    # Load data
    csv_path = project_root / f"data/echr_case_law_guides_{args.lang}.csv"
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} paragraphs from {csv_path}")
    
    # Create orchestrator
    orchestrator = ExtractionOrchestrator(
        num_workers=args.threads,
        lang_code=args.lang
    )
    
    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        orchestrator.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Process data
        results = orchestrator.process_dataframe(df)
        
        logger.info(f"Processing complete. Extracted {len(results)} results")
        
        # Post-processing
        logger.info("Running post-processing tasks...")
        orchestrator.judgement_fetcher.fix_missing_case_names()
        orchestrator.judgement_fetcher.export_to_csv()
        
        # Export results to CSV
        if results:
            output_data = []
            for result in results:
                output_data.append({
                    'guide_id': result.task.guide_id,
                    'paragraph_id': result.task.paragraph_id,
                    'paragraph_text': result.task.paragraph_text,
                    'sentences': json.dumps(result.sentences, ensure_ascii=False),
                    'citations': json.dumps(
                        [[c.to_json() for c in sc] for sc in result.citations],
                        ensure_ascii=False
                    ),
                    'errors': json.dumps(result.errors, ensure_ascii=False),
                    'processing_time': result.processing_time
                })
            
            output_df = pd.DataFrame(output_data)
            output_path = project_root / "data/extracted_sentences_and_citations.csv"
            output_df.to_csv(output_path, index=False)
            logger.info(f"Saved results to {output_path}")
        
    finally:
        # Clean up
        orchestrator.shutdown()
        db_manager.close()


if __name__ == "__main__":
    main()