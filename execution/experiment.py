import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# from src.experiments.base import base_completion_loop
# from src.experiments.llatrieval import llatrieval_loop
from src.experiments.load_experiment import Experiment
# from src.experiments.post_hoc import post_hoc_loop
from src.experiments.rag import rag_loop
# from src.experiments.rarr import rarr_loop
# from src.llms.gemini import Gemma3, GeminiFlash2_5
from src.llms.openrouter import R1_0528, K2
from src.llms.gemini import GPToss120
# from src.retrievers.bm25 import BM25
# from src.retrievers.gtr_t5 import GTR
from src.retrievers.qwen3 import QwenRetriever
from src.translations.llm_translation import LLM_Translator


def main():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment to run",
        choices=[e.value for e in Experiment],
        required=True,
    )

    parser.add_argument(
        "--source_language",
        type=str,
        help="Source language for multilingual experiments",
        default="English",
    )

    parser.add_argument(
        "--device",
        type=str,
        help="Device to run the experiment on",
        default="cpu",
    )

    args = parser.parse_args()

    print(f"Running experiment {args.experiment}")
    match args.experiment:
        # Base Experiments


        ## Ron RAG Qwen3 Experiments

        # R1_0528 Baseline
        case Experiment.RON_RAG_QWEN3_DOC_NO_INSTRUCT_QUERY_BASE_R10528_NOTRANSLATION:
            rag_loop(
                llm=R1_0528(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db', query_db_path='data/query_embeddings/qwen3_query_embeddings_base_ron.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_NO_INSTRUCT_QUERY_BASE_R10528_NOTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        # R1_0528 Best performer
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_R10528_NOTRANSLATION:
            rag_loop(
                llm=R1_0528(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_echr_retrieve.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_R10528_NOTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        # R1_0528 Query instruct in english revealing romanian
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_RON_R10528_NOTRANSLATION:
            rag_loop(
                llm=R1_0528(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_echr_retrieve_ron.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_RON_R10528_NOTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        # R1_0528 Query instruct in romanian
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_RON_ECHR_RETRIEVE_RON_R10528_NOTRANSLATION:
            rag_loop(
                llm=R1_0528(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_ron_echr_retrieve_ron.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_RON_ECHR_RETRIEVE_RON_R10528_NOTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        # R1_0528 Translation variants
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_R10528_HALFTRANSLATION:
            rag_loop(
                llm=R1_0528(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_translation_echr_retrieve.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_R10528_HALFTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_R10528_FULLTRANSLATION:
            rag_loop(
                llm=R1_0528(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_translation_echr_retrieve.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_R10528_FULLTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE",
                translator=LLM_Translator(),
            )
        # R1_0528 Fullnative
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_R10528_FULLNATIVE:
            rag_loop(
                llm=R1_0528(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_echr_retrieve.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_R10528_FULLNATIVE,
                k=10,
                custom_prompt="RAG_RON_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        # R1_0528 Legacy variants
        case Experiment.RON_RAG_QWEN3_DOC_RON_ECHR_TOPIC_QUERY_RON_BASE_RON_R10528_NOTRANSLATION:
            rag_loop(
                llm=R1_0528(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_ron_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_ron_base_ron.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_RON_ECHR_TOPIC_QUERY_RON_BASE_RON_R10528_NOTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_BASE_R10528_HALFTRANSLATION:
            rag_loop(
                llm=R1_0528(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_translation_base.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_BASE_R10528_HALFTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_BASE_R10528_FULLTRANSLATION:
            rag_loop(
                llm=R1_0528(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_translation_base.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_BASE_R10528_FULLTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE",
                translator=LLM_Translator(),
            )


        # GPT OSS 120B Notranslation as baseline
        case Experiment.RON_RAG_QWEN3_DOC_NO_INSTRUCT_QUERY_BASE_GPTOSS_NOTRANSLATION:
            rag_loop(
                llm=GPToss120(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db', query_db_path='data/query_embeddings/qwen3_query_embeddings_base_ron.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_NO_INSTRUCT_QUERY_BASE_GPTOSS_NOTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        # GPT OSS 120B Notranslation as best performer 
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_GPTOSS_NOTRANSLATION:
            rag_loop(
                llm=GPToss120(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_echr_retrieve.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_GPTOSS_NOTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        # GPT OSS 120B Notranslation as query query instruct in english revealing romanian
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_RON_GPTOSS_NOTRANSLATION:
            rag_loop(
                llm=GPToss120(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_echr_retrieve_ron.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_RON_GPTOSS_NOTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        # GPT OSS 120B Notranslation as query query instruct in romanian
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_RON_ECHR_RETRIEVE_RON_GPTOSS_NOTRANSLATION:
            rag_loop(
                llm=GPToss120(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_ron_echr_retrieve_ron.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_RON_ECHR_RETRIEVE_RON_GPTOSS_NOTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        # GPT OSS 120B Halftranslation best performance
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_GPTOSS_HALFTRANSLATION:
            rag_loop(
                llm=GPToss120(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_translation_echr_retrieve.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_GPTOSS_HALFTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        # GPT OSS 120B Fulltranslation best performance
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_GPTOSS_FULLTRANSLATION:
            rag_loop(
                llm=GPToss120(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_translation_echr_retrieve.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_GPTOSS_FULLTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE",
                translator=LLM_Translator(),
            )
        # GPT OSS 120B Fullnative
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_GPTOSS_FULLNATIVE:
            rag_loop(
                llm=GPToss120(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_echr_retrieve.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_GPTOSS_FULLNATIVE,
                k=10,
                custom_prompt="RAG_RON_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )

        
        # K2 Baseline
        case Experiment.RON_RAG_QWEN3_DOC_NO_INSTRUCT_QUERY_BASE_K2_NOTRANSLATION:
            rag_loop(
                llm=K2(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db', query_db_path='data/query_embeddings/qwen3_query_embeddings_base_ron.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_NO_INSTRUCT_QUERY_BASE_K2_NOTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        # K2 Best performer
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_K2_NOTRANSLATION:
            rag_loop(
                llm=K2(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_echr_retrieve.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_K2_NOTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        # K2 Query instruct in english revealing romanian
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_RON_K2_NOTRANSLATION:
            rag_loop(
                llm=K2(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_echr_retrieve_ron.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_RON_K2_NOTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        # K2 Query instruct in romanian
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_RON_ECHR_RETRIEVE_RON_K2_NOTRANSLATION:
            rag_loop(
                llm=K2(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_ron_echr_retrieve_ron.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_RON_ECHR_RETRIEVE_RON_K2_NOTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        # K2 Translation variants
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_K2_HALFTRANSLATION:
            rag_loop(
                llm=K2(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_translation_echr_retrieve.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_K2_HALFTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_K2_FULLTRANSLATION:
            rag_loop(
                llm=K2(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_translation_echr_retrieve.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_ECHR_RETRIEVE_K2_FULLTRANSLATION,
                k=10,
                custom_prompt="RAG_ABSTRACTIVE",
                translator=LLM_Translator(),
            )
        # K2 Fullnative
        case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_K2_FULLNATIVE:
            rag_loop(
                llm=K2(),
                retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_echr_retrieve.db'),
                experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_ECHR_RETRIEVE_K2_FULLNATIVE,
                k=10,
                custom_prompt="RAG_RON_ABSTRACTIVE_RON",
                translator=LLM_Translator(),
            )
            
        ## Legacy versions
        # case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_RON_BASE_RON_GPTOSS_NOTRANSLATION:
        #     rag_loop(
        #         llm=GPToss120(),
        #         retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_ron_base_ron.db'),
        #         experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_RON_BASE_RON_GPTOSS_NOTRANSLATION,
        #         k=10,
        #         custom_prompt="RAG_ABSTRACTIVE_RON",
        #         translator=LLM_Translator(),
        #     )
        # case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_BASE_GPTOSS_HALFTRANSLATION:
        #     rag_loop(
        #         llm=GPToss120(),
        #         retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_translation_base.db'),
        #         experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_BASE_GPTOSS_HALFTRANSLATION,
        #         k=10,
        #         custom_prompt="RAG_ABSTRACTIVE_RON",
        #         translator=LLM_Translator(),
        #     )
        # case Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_BASE_GPTOSS_FULLTRANSLATION:
        #     rag_loop(
        #         llm=GPToss120(),
        #         retriever=QwenRetriever(chroma_db_path='data/chromadbs/chroma_qwen3_db_instruct_echr_topic', query_db_path='data/query_embeddings/qwen3_query_embeddings_translation_base.db'),
        #         experiment=Experiment.RON_RAG_QWEN3_DOC_ECHR_TOPIC_QUERY_TRANSLATION_BASE_GPTOSS_FULLTRANSLATION,
        #         k=10,
        #         custom_prompt="RAG_ABSTRACTIVE",
        #         translator=LLM_Translator(),
        #     )

        case _:
            raise ValueError(f"Experiment {args.experiment} not implemented")

if __name__ == "__main__":
    main()
