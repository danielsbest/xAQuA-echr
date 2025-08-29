import json
import pandas as pd
from typing import Optional
from langchain_core.documents import Document

from src.column import Column
from src.experiments.load_experiment import Experiment, load_experiment_df
from src.llms.llm import LLM
from src.experiments import prompts
from src.models.citation import Citation, SentenceWithCitations
from src.retrievers.retriever import Retriever
from src.utils.text_utils import get_sentences, get_citations, remove_citations, remove_par_text_prefix
from src.translations.translation import Translator


def generate_response(question: str, llm: LLM, retriever: Retriever, k: int = 10, prompt_template: str = 'RAG_EXTRACTIVE', translator: Optional[Translator] = None, experiment_name: str = ""):
    docs = retriever.retrieve(question, k=k)

    docs = [doc[0] for doc in docs]

    if 'fullnative' in experiment_name:
        doc_contents = [doc.page_content for doc in docs]
        translated_contents = translator.translate_batch(doc_contents, target_lang='ron')
        for i, doc in enumerate(docs):
            doc.page_content = translated_contents[i]

    paragraphs = "\n".join(
        [
            f"[Doc {i+1}]: "
            + remove_par_text_prefix(
                doc.page_content,
                doc.metadata["case_name"],
                doc.metadata["paragraph_number"],
            )
            for i, doc in enumerate(docs)
        ]
    )

    prompt_template = getattr(prompts, prompt_template)
    prompt = prompt_template.format(
        paragraphs=paragraphs,
        question=question,
    )

    SYSTEM_PROMPT = "Work on the following with a structured and systematic approach. \nYour output must be plain text, no markdown or other formatting. \nFor every single sentence in your output, you must start a new line!"
    
    response = llm.infer_completion(prompt, system_prompt=SYSTEM_PROMPT)
    return response, docs


def parse_response(response: str, docs: list[Document]):
    response_sentences = get_sentences(response, simple=True)
    citations = []
    for sentence in response_sentences:
        doc_numbers = get_citations(sentence)
        doc_numbers = [d - 1 for d in doc_numbers if d - 1 < len(docs) and d - 1 >= 0]
        docs_in_sentence = [docs[doc_number] for doc_number in doc_numbers]

        citations.append(
            SentenceWithCitations(
                sentence=remove_citations(sentence),
                citations=[
                    Citation(
                        case_id=doc.metadata["case_id"],
                        case_name=doc.metadata["case_name"],
                        paragraph_number=doc.metadata["paragraph_number"],
                        paragraph_text=doc.page_content,
                    )
                    for doc in docs_in_sentence
                ],
            ).model_dump()
        )

    parsed_response = "\n".join([remove_citations(s) for s in response_sentences])

    return parsed_response, citations


def rag_loop(
        llm: LLM, 
        retriever: Retriever, 
        experiment: Experiment,
        custom_prompt: str,
        translator: Translator,
        k: int = 10,
        ):
    
    df, path = load_experiment_df(experiment)
    experiment_name = experiment.value

    for i, row in df.iterrows():
        if Column.GENERATED_ANSWER in row and pd.notnull(row[Column.GENERATED_ANSWER]):
            print("Skipping row", i, "as it already has a response")
            continue
        print("\n\nProcessing row", i)
        question = row[Column.QUESTION]

        if 'fulltranslation' in experiment_name or 'halftranslation' in experiment_name:
            in_question = translator.translate(question, target_lang='eng')
        else:
            in_question = question

        out_response, docs = generate_response(
            in_question,
            llm,
            retriever,
            k,
            prompt_template=custom_prompt,
            translator=translator,
            experiment_name=experiment_name
        )
        
        if 'fulltranslation' in experiment_name:
            response = translator.translate(out_response, target_lang='ron')
        else:
            response = out_response

        print(f"\nQuestion:\n{question}")
        print(f"\nResponse:\n{response}")

        print(f"\n\n\nTranslated Question:\n{translator.translate(question, target_lang='eng')}\n")

        if 'fulltranslation' in experiment_name:
            print(f"\nTranslated Response:\n{out_response}")
        else: #deactivate for full run to save compute
            print(f"\nTranslated Response:\n{translator.translate(response, target_lang='eng')}\n")

        parsed_response, citations = parse_response(response, docs)
        df.at[i, Column.GENERATED_ANSWER] = parsed_response
        citations = json.dumps(citations, indent=4, ensure_ascii=False)
        df.at[i, Column.GENERATED_CITATIONS] = citations

        print(f"\n\n\nParsed Response:\n\n{parsed_response}")
        print(f"Citations:\n\n{citations}")
        print("\n\n ------------------------------------------------------- \n\n\n\n")
        df.to_csv(path, index=False)
