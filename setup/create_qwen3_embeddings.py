import sys
import os
import time
import pandas as pd
from langchain_core.documents import Document
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrievers.qwen3 import get_db, QwenEmbeddings
from src.constants import ECHR_CASE_PARAGRAPHS_CSV_PATH

df = pd.read_csv(f"{ECHR_CASE_PARAGRAPHS_CSV_PATH}")

EMBED_INSTRUCTS = {
    'empty': None,
    'simple_doc': "Embed this legal judgment paragraph for retrieval by general legal questions.",
    'simple_doc_ron': "Embed this legal judgment paragraph for retrieval by general legal questions in Romanian.",
    'ron_simple_doc_ron': "Embedează acest paragraf dintr-o hotărâre judecătorească pentru regăsire prin întrebări juridice generale în limba română.",

    'echr_topic': "Embed this paragraph from an ECHR judgment, clearly capturing the primary topics and issues discussed, to facilitate accurate and topic-based retrieval by core questions about applicability of the European Convention on Human Rights.",
    'echr_topic_ron': "Embed this paragraph from an ECHR judgment, clearly capturing the primary topics and issues discussed, to facilitate accurate and topic-based retrieval by core questions in Romanian about applicability of the European Convention on Human Rights.",
    'ron_echr_topic': "Embedează acest paragraf dintr-o hotărâre a CEDO, surprinzând în mod clar temele și chestiunile principale discutate, pentru a facilita regăsirea precisă și tematică prin întrebări cheie în limba română privind aplicabilitatea Convenției Europene a Drepturilor Omului.",

    # not used
    'echr_doc': "Embed this paragraph from a judgement of the European Court of Human Rights (ECHR) for retrieval by European Human Rights Article legal questions.",
    'echr_doc_ron': "Embed this paragraph from a judgement of the European Court of Human Rights (ECHR) for retrieval by general legal questions in Romanian.",
    'question_doc': "Embed this legal judgment paragraph for retrieval by European Human Rights Law specific questions.",
    'question_doc_ron': "Embed this legal judgment paragraph for retrieval by European Human Rights Law specific questions in Romanian.",
}

parser = argparse.ArgumentParser(description="Create Qwen3 embeddings for ECHR cases.")
parser.add_argument("--instruct", type=str, required=True, help="The key for the embed instruction from EMBED_INSTRUCTS.")
args = parser.parse_args()
embed_instruct_name = args.instruct

if embed_instruct_name not in EMBED_INSTRUCTS:
    print(f"Error: Instruction key '{embed_instruct_name}' not found in EMBED_INSTRUCTS.")
    print("Available keys are:", list(EMBED_INSTRUCTS.keys()))
    sys.exit(1)

embed_instruct = EMBED_INSTRUCTS[embed_instruct_name]

all_paragraphs = [
    Document(
        page_content=f'{case_name}; § {paragraph_number}: {str(paragraph_text).replace(f"{paragraph_number}.", "").strip()}',
        metadata={
            "case_name": case_name,
            "paragraph_number": paragraph_number,
            "case_id": case_id,
        },
    )
    for case_name, paragraph_number, paragraph_text, case_id in zip(
        df["case_name"], df["paragraph_number"], df["paragraph_text"], df["case_id"]
    )
]

print(f"Embedding: {embed_instruct_name}")
    
db_path = "./data/chromadbs/chroma_qwen3_db"
if embed_instruct is not None:
    db_path += f"_instruct_{embed_instruct_name}"

embedding_function = QwenEmbeddings(embed_instruction=embed_instruct)
db = get_db(db_path, embedding_function=embedding_function)
entries = db._collection.count()
print("Entries:", entries)

# Process paragraphs in batches to avoid memory issues
n = 2000
loading_time = time.time()
i = entries

while i < len(all_paragraphs):
    paragraphs = all_paragraphs[i : i + n]

    start_time = time.time()

    # embeds and adds to chroma db
    db.add_documents(documents=paragraphs)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    print(f"\nLoading time between inference: {(time.time() - elapsed_time - loading_time):.2f} seconds")
    loading_time = time.time()
    print(f"Processed {len(paragraphs)} paragraphs in {int(minutes):02d}:{int(seconds):02d} min:sec)")
    paragraphs_left = len(all_paragraphs) - (i + len(paragraphs))
    time_left = int(paragraphs_left / n) * elapsed_time
    hours_left, remainder = divmod(time_left, 3600)
    minutes_left, _ = divmod(remainder, 60)
    print(f'Remaining {paragraphs_left} paragraphs in "{embed_instruct_name}" --- time left ~ {int(hours_left)}h {int(minutes_left)}min')
    i += n

print(f"Completed {embed_instruct_name}")
