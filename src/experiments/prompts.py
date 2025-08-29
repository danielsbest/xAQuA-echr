BASE_COMPLETION = "You are an ECHR legal expert tasked to answer the following question:\nQuestion: {question}\nAnswer:"

# RAG_EXTRACTIVE = """
#     You are an European Court of Human Rights (ECHR) legal expert tasked to answer a question.
#     The following retrieved documents are paragraphs extracted from official ECHR judgments and can help you answer the question:
#     {paragraphs}

#     Instructions:
#     Use the retrieved judgment paragraphs to answer the question. 
#     Reuse the language from the documents!
#     Cite relevant documents at the the end of a sentence! 
#     Accepted formats: sentence [citation(s)].
#     Valid citation formats: [Doc 1] or [Doc 1, Doc 2, Doc 3]
#     You must follow the [Doc i] format! Do NOT use the case names or paragraph numbers to cite documents!
#     Do not provide a list of all used citations at the end of your response!

#     Question: {question}
#     Answer:
#     """

RAG_ABSTRACTIVE = """
    You are an European Court of Human Rights (ECHR) legal expert tasked to answer a question.
    The following retrieved documents are paragraphs extracted from official ECHR judgments and should help you answer the question:
    {paragraphs}

    Instructions:
    Use the retrieved judgment paragraphs to answer the question. 
    Cite relevant documents at the the end of a sentence! 
    Accepted formats: sentence [citation(s)].
    Valid citation formats: [Doc 1] or [Doc 1, Doc 2, Doc 3]
    You must follow the [Doc i] format! Do NOT use the case names or paragraph numbers to cite documents!
    Do not provide a list of all used citations at the end of your response!

    Question: {question}
    Answer:
    """

RAG_ABSTRACTIVE_RON = """
    You are an European Court of Human Rights (ECHR) legal expert tasked to answer a question.
    The following retrieved documents are paragraphs extracted from official ECHR judgments and should help you answer the question:
    {paragraphs}

    Instructions:
    Use the retrieved judgment paragraphs to answer the question. 
    The answer must be in Romanian language.
    Cite relevant documents at the the end of a sentence! 
    Accepted formats: sentence [citation(s)].
    Valid citation formats: [Doc 1] or [Doc 1, Doc 2, Doc 3]
    You must follow the [Doc i] format! Do NOT use the case names or paragraph numbers to cite documents!
    Do not provide a list of all used citations at the end of your response!

    Question: {question}
    Answer:
    """

RAG_RON_ABSTRACTIVE_RON = """
    Sunteți un expert juridic specializat în jurisprudența Curții Europene a Drepturilor Omului (CEDO), iar sarcina dumneavoastră este să răspundeți la o întrebare.
    Următoarele documente sunt paragrafe extrase din hotărâri oficiale ale CEDO și ar trebui să vă ajute să răspundeți la întrebare:
    {paragraphs}

    Instrucțiuni:
    Folosiți paragrafele din hotărârile extrase pentru a răspunde la întrebare. 
    Răspunsul trebuie să fie în limba română.
    Faceți referire la documentele relevante la sfârșitul propoziției! 
    Formate acceptate: propoziție [referință/referințe].
    Formate de referință valide: [Doc 1] sau [Doc 1, Doc 2, Doc 3]
    Trebuie să respectați formatul [Doc i]! NU folosiți denumirile cauzelor sau numerele paragrafelor pentru a face referire la documente!
    Nu includeți o listă cu toate referințele folosite la finalul răspunsului!

    Întrebare: {question}
    Răspuns:
"""


LLATRIEVAL_SCORE = """
You are ScoreGPT as introduced below.
You are ScoreGPT, capable of scoring candidate documents based on their level of support for the corresponding question, with a rating range from 0 to 10.

Input:
- Question: The specific question.
- Candidate Documents: Documents whose combination may maximally support the corresponding question.

Skill:
1. Analyzing the given question(s) and understanding the required information.
2. Searching through documents to score them based on their level of support for the corresponding question(s),
with a rating range from 0 to 10.

Output:
- A score ranging from 0 to 10, where a higher score indicates greater support of the candidate documents for the corresponding question, and a lower score indicates lesser support.

Candidate Documents:
{documents}

Question: 
{question}

Output Format: (You MUST follow this output format!)
Thoughts: [Your thoughts about how well the candidate documents support the question]
Score: [SCORE]
"""

LLATRIEVAL_PROGRESSIVE_SELECTION = """
You are DocSelectorGPT, capable of selecting a specified number of documents for answering the user's specific question.

Input:
- Question: The specific question
- Candidate Documents: Documents contain supporting documents which can support answering the given questions. Candidate documents will have their own identifiers for FactRetrieverGPT to cite.

Skill:
1. Analyzing the given question and understanding the required information.
2. Searching through candidate documents to select k supporting documents whose combination can maximally support giving a direct, accurate, clear and engaging answer to the question and make the answer and is closely related to the core of the question.

Workflow:
1. Read and understand the questions posed by the user.
2. Browse through candidate documents to select k documents whose combination can maximally support giving a direct, accurate, clear and engaging answer to the question(s) and make the answer and is closely related to the core of the question.
3. List all selected documents.

Output:
- Selected Documents: The identifiers of selected supporting documents whose combination can maximally support giving an accurate and engaging answer to the question and make the answer and is closely related to the core of the question.

Output Example:
Selected Documents: Doc 2, Doc 6, Doc 8 (You MUST follow this format!)

Max number of selectable documents: {k}
- You can only select a maximum of {k} documents!

Candidate Documents:
{documents}

Question: 
{question}

Output Format (You MUST follow this output format!)
Thoughts: [Your thoughts about which candidate documents support the question well and why]
Selected Documents: [document identifiers] 
"""

LLATRIEVAL_PASSAGE_RETRIEVAL = """
You are a PassageRetriever, 
capable of identifying missing content that answers the given question but does not exist in the given possible answering
passages and then using your own knowledge to generate correct answering passages using missing content you identify.

Input:
- Question: The specific question.
- Answering Passages: Possible answering passages.

Output:
- Correct answering passages generated using missing content you identify based on your own knowledge.

Rules:
1. You have to use your own knowledge to generate correct answering passages using missing content you identify.
2. Only generate the required correct answering passages. Do not output anything else.
3. Directly use your own knowledge to generate correct answering passages if you think the given possible answering passages do not answer to the given question.

Workflow:
1. Read and understand the question and possible answering passages.
2. Identify missing content that answers the given question but does not exist in the given possible answering passages.
3. Directly use your own knowledge to generate correct answering passages if you think the given possible answering passages do not answer to the given question(s). 
4. Use your own knowledge to generate correct answering passages using missing content you identify.

Answering Passages:
{documents}

Question: 
{question}

Output Format: (You MUST follow this output format!)
Correct Answering Passages: [correct answering passages]
Missing Passages: [missing passages]
"""

RARR_RETRIEVE_EVIDENCE = """
You will be determining if a sentence of the answer should be supported by a case law citation.

## Context
Question: {question}
Answer: {answer}

## Now we will analyze if the following sentence should have a citation

The sentence for which to decide if it should have a citation: 
<sentence-start>{sentence}</sentence-end>

First, analyze the sentence and decide if it should be supported by a case law citation.
Note:
- General knowledge, headers, and other non-sentences do not require citations.
- Legal arguments, facts, examples, ... should be supported by citations.

The format of your response MUST look like this:
Thoughts: [Reason why the sentence <sentence-start>{sentence}</sentence-end> should have a citation]
Should have a supporting citation: [Yes/No]
"""

RARR_AGREEMENT = """
You will be determining if a piece of evidence agrees with, disagrees with, or is irrelevant to an sentence for a given question.

## Context
Question: {question}
Answer: {answer}

## Now we will analyze the evidence for the following sentence in the answer

Evidence: 
<evidence-start>{evidence}</evidence-end>

The sentence for which to decide if the evidence agrees, disagrees, or is irrelevant: 
<sentence-start>{sentence}</sentence-end>

Carefully analyze the evidence and explain in a reasoning step whether it agrees, contradicts, or is irrelevant for the sentence in <sentence-start>...</sentence-end> tags.
Then, based on your reasoning, provide your final classification, which MUST be one of ahe following:
- Agrees
- Disagrees
- Irrelevant

The format of your response MUST look like this:
Thoughts: [Reason weather the evidence agrees, disagrees or is irrelevant for the given sentence]
Final classification: [Your final classification here: Agrees/Disagrees/Irrelevant]
"""

RARR_EDIT_DISAGREEMENT = """
You will be editing a sentence based on the disagreement with the evidence.

## Context
Question: {question}
Answer: {answer}

## Now we will edit the following sentence in the answer based on the disagreement with the evidence

Evidence: 
<evidence-start>{evidence}</evidence-end>

The sentence in the answer the evidence disagrees with:
<sentence-start>{sentence}</sentence-end>

First, carefully analyze the sentence and identify the part that contains the disagreement with the evidence.
Then, rewrite the sentence with MINIMAL modification to resolve the disagreement.
We will not accept drastic changes to the sentence!

The format of your response MUST look like this:
Thoughts: [Reason why the sentence in <sentence-start>...</sentence-end> should be edited]
Fix with minimal edit: [The corrected entire sentence with MINIMAL modification to resolve the disagreement enclosed by <fixed-sentence-start>...<fixed-sentence-end>]
"""



