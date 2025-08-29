# Question generation prompts:

SIMPLE_QUESTION_GENERATION_ENG_1_RON = \
"""Case law paragraphs: 
{paragraphs}

Task:
Define a single challenging legal question that can be answered with the given case law paragraphs.
Reuse the language from the case law in the question. 
Make sure the question is general and does NOT mention specific cases. 
The question needs to be in Romanian as the paragraphs. 

Question: 
"""

SYSTEMATIC_LEGAL_QUESTION_GENERATION_ENG_1_RON = \
"""Your objective is to develop educational and challenging questions for lawyers working with ECHR case law and for judges who want to draft judgments based on ECHR case law.
Each question should be based on the provided paragraphs from the ECHR case law guides.
When formulating a question reuse the language from the ECHR case law and match legal doctrines to specific facts. 
Make sure the question is general and does NOT mention specific cases. 
Emphasize the patterns that link facts to specific legal doctrines.
The question needs to be in Romanian as the paragraphs. 

Doctrines and facts: 
{paragraphs}

Steps:
1. Identify how the margin of appreciation and positive obligations apply in relation to the State's discretion
2. Identify the reasons that justify necessity and pressing social needs
3. Identify the reasons that command that rights be effective in their application
4. Identify how reasonable measures apply in relation to the State's discretion and to restrictions imposed by States or private individuals
5. Identify the reasons set forth by the Court to defer to domestic reasons provided by domestic authorities
6. Define a question that can be answered exactly by the given legal doctrines and applicable facts to those doctrines

Answer Template:
Margin of appreciation: {{how do the margin of appreciation and positive obligations apply in relation to the State's discretion}}
Necessities: {{reasons that justify necessity and pressing social needs}}
Effectivity: {{reasons that command that rights be effective in their application}}
Reasonable Measures: {{how do reasonable measures apply in relation to the State's discretion and to restrictions imposed by States or private individuals?}}
Domestic Reasons: {{the reasons set forth by the Court to defer to domestic reasons provided by domestic authorities}}
Question: {{define a single question that can be answered exactly by the given legal doctrines and applicable facts reusing the language from the ECHR case law}}
"""

SYSTEMATIC_LEGAL_QUESTION_GENERATION_ENG_2_RON = \
"""Your objective is to develop educational and challenging questions for lawyers working with ECHR case law and for judges who want to draft judgments based on ECHR case law.
Each question should be based on the provided paragraphs from the ECHR case law guides.
When formulating a question reuse the language from the ECHR case law and match legal doctrines to specific facts.
Make sure the question is general and does NOT mention specific cases. 
Emphasize the patterns that link facts to specific legal doctrines.
The question needs to be in Romanian as the paragraphs. 

Doctrines and facts: 
{paragraphs}

Steps:
1. Identify what are the criteria under the Convention for applying the rights enshrined therein.
2. Identify the conditions that the Court sets forth with view to analyse the legality of domestic measures and restrictions
3. Identify the reasons provided by the Court to protect applicants and victims and to differentiate between them.
4. Identify the reasons set forth by the Court to distinguish between legal doctrines and contextual application of those doctrines.
5. Assign a set of facts to its corresponding Article and identify a sequence of reasons that justify the application of the Article to those facts.
6. Explain why analogies and comparisons between Article-fact pair are pertinent
7. Identify separately reasons that are linked to margin of appreciation of the State from those linked to the Court's appreciation of facts
8. Define a question that can be answered exactly by the given Article-facts correspondence

Answer Template:
Criteria for rights: {{how does the Court define the criteria for applying rights}}
Legality of domestic measures and restrictions: {{conditions that determine that domestic measures are compliant with the Convention}}
Protection and differentiation of applicants and victims: {{circumstances and conditions that limit or allow applicants and victims to present their case}}
Distinction between legal doctrines and contextual application: {{how and why and in what circumstances legal doctrines apply to specific facts}}
Article-facts correspondence: {{the reasons set forth by the Court to justify the application of the Article/Articles to those facts}}
Analogies and comparisons between Article-fact pair {{the reasons set forth by the Court to justify why articles and facts differ from one another}}
margin of appreciation of States and Court's appreciation {{the reasons, circumstances and conditions set forth by the Court to explain margin of appreciation of States and its own appreciation}}
Question: {{define a single question that can be answered exactly by the given legal doctrines and applicable facts and by the Article-fact pair, reusing the language from the ECHR case law}}
"""

SYSTEMATIC_LEGAL_QUESTION_GENERATION_ENG_3_RON = \
"""Your objective is to develop educational and challenging questions for lawyers working with ECHR case law and for judges who want to draft judgments based on ECHR case law.
Each question should be based on the provided paragraphs from the ECHR case law guides.
When formulating a question reuse the language from the ECHR case law and match legal doctrines to specific facts.
Make sure the question is general and does NOT mention specific cases. 
Emphasize the patterns that link facts to specific legal doctrines.
The question needs to be in Romanian as the paragraphs. 

Doctrines and facts: 
{paragraphs}

Steps:
1. Identify what are the criteria under the Convention for applying the rights enshrined therein.
2. Identify the conditions that the Court sets forth with view to analyse the legality of domestic measures and restrictions
3. Identify the reasons provided by the Court to protect applicants and victims and to differentiate between them.
4. Identify the reasons set forth by the Court to distinguish between legal doctrines and contextual application of those doctrines.
5. Assign a set of facts to its corresponding Article and identify a sequence of reasons that justify the application of the Article to those facts.
6. Explain why analogies and comparisons between Article-fact pair are pertinent
7. Identify separately reasons that are linked to margin of appreciation of the State from those linked to the Court's appreciation of facts
8. Define a question that can be answered exactly by the given Article-facts correspondence

Answer Template:
Criteria for rights: {{how does the Court define the criteria for applying rights}}
Legality of domestic measures and restrictions: {{conditions that determine that domestic measures are compliant with the Convention}}
Protection and differentiation of applicants and victims: {{circumstances and conditions that limit or allow applicants and victims to present their case}}
Distinction between legal doctrines and contextual application: {{how and why and in what circumstances legal doctrines apply to specific facts}}
Article-facts correspondence: {{the reasons set forth by the Court to justify the application of the Article/Articles to those facts}}
Analogies and comparisons between Article-fact pair {{the reasons set forth by the Court to justify why articles and facts differ from one another}}
margin of appreciation of States and Court's appreciation {{the reasons, circumstances and conditions set forth by the Court to explain margin of appreciation of States and its own appreciation}}
Question: {{define only a single question that can be answered exactly by the given legal doctrines and applicable facts and by the Article-fact pair, reusing the language from the ECHR case law}}
"""


# Answer generation prompts:

SIMPLE_ANSWER_ATTRIBUTION_GENERATION_ENG_2_RON = \
"""Your objective is to develop question-answer pairs that could help lawyers working with ECHR case law 
and judges who want to draft judgments based on ECHR find the best arguments to justify violations or non-violations of the ECHR.

Doctrines, articles, and facts:
{sentences}

Answer the following question following the sequence: characterization of facts according to ECHR doctrines and justification of those facts in relation to specific articles of the ECHR.
Use the provided doctrines and facts to answer the question.
Use citations! At the end of each sentence in your answer add the numbers of the used facts and doctrines in square brackets.
Answer the question in Romanian.

Question: 
{question}
Answer:
"""

SIMPLE_ANSWER_PROSE_GENERATION_ENG_1_RON = """\
Your objective is to develop question-answer pairs that could help lawyers working with ECHR case law 
and judges who want to draft judgments based on ECHR find the best arguments to justify violations or non-violations of the ECHR.
Keep the answer concise, only use the provided facts and doctrines to answer the question.
Your answer will be checked if all you say is supported by the provided doctrines and facts.

Doctrines, articles, and facts:
{paragraphs}


Question: 
{question}

Judgement citations:
{judgement_citations}

Your answer should directly characterize facts according to ECHR doctrines and justify those facts in relation to specific articles of the ECHR.
Start directly with the substance of the characterization and justification.
To provide evidence for your answer, undermine each of your points with citations from echr judgement decision which you extract from doctrines and facts.
All valid citations are in the list of judgement citations.

Cite them by adding the numbers of the cited judgements in square brackets at the end of each sentence in your answer. 
Answer the question in Romanian as the answer.


Provide your answer here:
Answer:
"""


SIMPLE_ANSWER_PROSE_GENERATION_ENG_2_RON = """\
Your objective is to synthesize the provided legal doctrines, articles, and facts into a concise answer to the question using the applicable legal principles.
The summary should state the law in a neutral, objective, and generalized manner.

The answer must be concise and based exclusively on the provided texts.
Every statement must be supported by the provided doctrines and facts.

Doctrines, articles, and facts:
{paragraphs}


Question:
{question}

Judgement citations:
{judgement_citations}

Instructions for the answer:
1.  Answer the question by first characterizing the facts according to ECHR doctrines, and then justifying those facts in relation to specific articles of the ECHR.
2.  Substantiate each point with citations from the ECHR judgments.
3.  All valid citations are in the list of 'judgement citations'. Cite them by adding the numbers in square brackets at the end of each sentence.
4.  The answer must be in Romanian.
5.  The answer must be a general statement of law, not advice directed at a specific person (e.g., do NOT start with "Un judecător ar trebui..." or similar phrasing).

Provide your answer here:
Answer:
"""

SIMPLE_ANSWER_PROSE_GENERATION_ENG_2_1_RON = """\
Your objective is to synthesize the provided legal doctrines, articles, and facts into a concise answer to the question using the applicable legal principles.
The summary should state the law in a neutral, objective, and generalized manner.


The answer must be concise and based exclusively on the provided texts.
Every statement must be supported by the provided doctrines and facts.


Doctrines, articles, and facts:
{paragraphs}




Question:
{question}


Judgement citations:
{judgement_citations}


Instructions for the answer:
1.  Answer the question by first characterizing the facts according to ECHR doctrines, and then justifying those facts in relation to specific articles of the ECHR.
2.  Substantiate each point with citations from the ECHR judgments.
3.  All valid citations are in the list of 'judgement citations'. Cite them by adding the numbers in square brackets at the end of each sentence. You don’t need to use all of them. When several citations apply to one sentence only cite the most relevant ones.
4.  The answer must be in Romanian.
5.  The answer must be a general statement of law, not advice directed at a specific person (e.g., do NOT start with "Un judecător ar trebui..." or similar phrasing).


Provide your answer here:
Answer:
"""


# QA pair quality filtering prompt:
ANSWER_QUALITY_FILTERING_PROMPT_1 = """\
You are a strict legal expert judging ECHR legal question-answer pairs. The answer might be bad, so be strict!

Question: 
```{question}```

Potential Answer: 
```{answer}```

You MUST answer each question in full sentences!

The response MUST follow this template:
Comprehensiveness Analysis: {{Go through the answer and analyze how well it answers the question. Does is cover all angles of the question? If the question is not a proper question or not a generic question (mentions a specific case), give a score of 1.}}
Comprehensiveness Score: {{A score from 1 (not comprehensive at all) to 5 (extremely comprehensive)}}
Conciseness Analysis: {{Is there any part in the answer irrelevant / unrelated to the question? If so, what is unneeded?}}
Conciseness Score: {{A score from 1 (not concise at all) to 5 (extremely concise)}}
"""

ANSWER_QUALITY_FILTERING_PROMPT_2 = """\
You are a strict legal expert judging European Court of Human Rights legal question-answer pairs. The answer might be bad, so be strict!


Question:
```{question}```


Potential Answer:
```{answer}```


You MUST answer each question in full sentences!

The response MUST follow this template:
Comprehensiveness Analysis: {{Go through the answer and analyze how well it answers the question. Does is cover all aspects of the question?}}
Comprehensiveness Score: {{A score from 1 (not comprehensive at all) to 5 (extremely comprehensive)}}
Conciseness Analysis: {{Is there any part in the answer irrelevant / unrelated to the question? If so, what is unneeded?}}
Conciseness Score: {{A score from 1 (not concise at all) to 5 (extremely concise)}}
Question Assessment Analysis: {{Analyze the question itself. Is it a proper, answerable question? Is it generic or does it focus on a too specific, non-generalizable case?}} 
Question Assessment Score: {{A score from 1 (not a proper question / unanswerable) to 5 (proper question)}}
"""


ANSWER_GROUNDEDNESS_FILTERING_PROMPT_1 = """\
You are a strict legal expert judging European Court of Human Rights legal answers faithfullness. The answer might be bad, so be strict!

Answer: 
```{answer}```

Reference text used to generate the answer:
```{reference}```

Citation numbers used in answer:
```{citation_numbers}```

You MUST answer each question in full sentences!

The response MUST follow this template:
Groundedness Analysis: {{Check every factual and legal statement in the answer. Is each claim directly supported by identifiable statements in the reference text? Any statements that are unsubstantiated, misattributed, factually incorrect reduce the score. A perfectly grounded answer would allow a legal professional to trace every claim back to the reference.}}
Groundedness Score: {{A score from 1 (not at all based on the reference text) to 5 (extremely grounded on the reference text)}}
Citation Correctness Analysis: {{Is each claim in the answer citing the same Citations as the statement in the reference text? Any misalignments or mixups reduce the score. Use the provided citation numbers.}}
Citation Correctness Score: {{A score from 1 (citations are incorrect or misaligned or hallucinated) to 5 (accurate citations use)}}
"""

# not in use
# ANSWER_GROUNDEDNESS_FILTERING_PROMPT_2 = """\
# You are an exceptionally meticulous and strict fact-checker for the European Human Rights Law Review. Your task is to analyze an AI-generated answer about ECHR law for faithfulness to a provided reference text. You must be rigorous and assume the answer may contain subtle errors.

# **Answer to Evaluate:**
# ```{answer}```

# **Reference Text:**
# ```{reference}```

# **Citation numbers used in answer**:
# ```{citation_numbers}```

# ---

# **Instructions:**

# First, perform a step-by-step verification process in your head (do not write this part in the final output).
# 1.  Go through the `Answer` sentence by sentence.
# 2.  For each factual or legal claim, locate the specific sentence(s) in the `Reference Text` that should support it.
# 3.  **For Groundedness:** Determine if the claim in the `Answer` is a direct and faithful representation of the information in the `Reference Text`. Note any hallucinations, contradictions, or unsubstantiated statements.
# 4.  **For Citation Correctness:** Check if the citation marker (e.g., `[1]`, `[2]`) in the `Answer` correctly points to the part of the `Reference Text` that supports that specific claim. Note any misaligned or incorrect citations.

# After your internal verification, you MUST provide your final analysis in the template below. You MUST answer each section in full sentences and provide specific examples to justify your analysis and scores.

# ---

# **Required Output Template:**

# **Groundedness Analysis:** {{Analyze whether every factual and legal claim in the Answer is directly supported by the Reference Text. A perfect answer is fully traceable to the reference. Identify and quote specific examples of any unsubstantiated claims, contradictions, or statements that go beyond the scope of the reference. State clearly what is correct and what is not.}}
# **Groundedness Score:** {{Provide a score from 1 to 5 based on the following rubric:
# 1: Not at all grounded. The answer is completely fabricated or contradicts the reference.
# 2: Poorly grounded. The answer contains significant factual errors or unsubstantiated claims.
# 3: Partially grounded. The answer has a mix of supported and unsupported claims.
# 4: Mostly grounded. The answer is largely faithful but contains minor inaccuracies or a few unsubstantiated details.
# 5: Perfectly grounded. Every single claim in the answer is directly and accurately supported by the reference text.}}

# **Citation Correctness Analysis:** {{Analyze whether the citation markers in the Answer are correctly placed. For each citation, does it point to the correct supporting statement in the Reference Text? Identify and quote specific examples of correctly placed citations and any misaligned, mixed-up, or missing citations.}}
# **Citation Correctness Score:** {{Provide a score from 1 to 5 based on the following rubric:
# 1: All citations are incorrect, misaligned, or hallucinated.
# 2: Poorly cited. The majority of citations are incorrect or misaligned.
# 3: Partially cited correctly. A significant number of citations are correct, but an equal number are misaligned.
# 4. Mostly cited correctly. The vast majority of citations are correct, with only minor misalignments.
# 5: Perfectly cited. Every citation marker accurately links the claim to the correct supporting evidence in the reference.}}
# """