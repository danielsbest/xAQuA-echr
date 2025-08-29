
CLAIM_RECALL_9_SYSTEM = """\
You are a legal expert in the field of European Court of Human Rights (ECHR) cases. 
Your task is to evaluate the quality of a generated answer to a legal query based on a provided golden standard answer.
Your output must strictly adhere to the given JSON output structure! It must be valid JSON and all inputs shall be plain text without any formatting!
"""

CLAIM_RECALL_9 = """\
Claims are statements that are verifiable.
Focus on whether the generated content mentions the most important criteria. If a main criteria is present, but it's key points are omitted consider it as covered!

The evaluation should follow the following strategy:


First list the claims made in the gold standard answer as a numbered list.
Then, list the claims made in the generated answer as a numbered list.
Finally check for every claim in the golden answer if it is sufficiently covered in the generated answer.


Output structure:
{ouput_structure}

Now it's your turn:
Golden Answer: {gold_answer}

Generated Answer: {gen_answer}
"""

CLAIM_RECALL_9_OUTPUT_STRUCTURE = r"""{
    "claims_in_gold_standard": {
    "gold_claim_1": "<gold_claim_1>",
    "gold_claim_2": "<gold_claim_2>",
    …                      
    },
    "claims_in_generated_answer": {
    "gen_claim_1": "<gen_claim_1>",
    "gen_claim_2": "<gen_claim_2>",
    …                     
    },
    "analysis": {
    "covered_by_generated_claims": {
        "gold_claim_1": {
        "gen_claims": ["gen_claim_2", "gen_claim_5"],
        "reasoning": "<nuanced reasoning why these claims cover gold_claim_1>"
        },
        "gold_claim_2": {
        "gen_claims": ["gen_claim_3"],
        "reasoning": "<…>"
        },
        …
        },
    },
    "evaluation": {
    "description": "Rate the output with one of the provided options.",
    "rating_options": {
      "0": "No key aspects or only minor aspects were addressed.",
      "1": "Any key aspects are addressed, but significant key aspects are missing or incorrect.",
      "2": "Several key aspects are addressed, including some of the most important ones, but with significant omissions.",
      "3": "Most of the crucial aspects are addressed, though some important points may still be missing or incomplete. This generally meets the minimum standard for a passing result.",
      "4": "Nearly all of the important aspects are thoroughly addressed, with only minor omissions or missing elaboration on aspects.",
      "5": "All essential and important aspects are addressed fully and correctly.",
      "6": "Goes beyond the gold standard, providing additional relevant accurate elaboration on requirements for aspects(exceptional)."
    }
    "rating": "<your rating choice>",
    "evaluation_reasoning": {
        "description": "Step-by-step, explicit reasoning for why the above boolean evaluations were chosen, referencing the specific coverage and gaps from the analysis above along with their severity. This section must explain the rationale for the overall judgment, including edge cases, ambiguity, and any uncertainty.",
        "reasoning": "<your reasoning>"
        }
    }
}
"""

GROUNDEDNESS_SYSTEM = """\
You are a legal expert in the field of European Court of Human Rights (ECHR) cases. 
Your task is to evaluate the faithfulness of an answer based on its citations.
Your output must strictly adhere to the given JSON output structure! It must be valid JSON and all inputs shall be plain text without any formatting!
"""

GROUNDEDNESS_PROMPT = """\

Here are the sentences along with their citations:
```{sentences_with_citations}```


You MUST answer each question in full sentences!


Your output MUST follow this json template:
{output_structure}
"""


GROUNDEDNESS_OUTPUT_STRUCTURE = r"""{
    "claims_in_sentence": {
    "sentence_1": {
        "s1_claim_1": "<first claim in sentence 1>",
        "s1_claim_2": "<second claim in sentence 1>",
        …
        },
    "sentence_2": {
        "s2_claim_1": "<first claim in sentence 2>",
        "s2_claim_2": "<second claim in sentence 2>",
        …
        },
    …                      
    },
    "claims_in_citations": {
    "citations_1": {
        "c1_claim_1": "<first claim in citations 1>",
        "c1_claim_2": "<second claim in citations 1>",
        …
        },
    "citations_2": {
        "c2_claim_1": "<first claim in citations 2>",
        "c2_claim_2": "<second claim in citations 2>",
        …
        },
    …                     
    },
    "analysis": {
        "covered_by_citations": {
            "sentence_1": {
                "s1_claim_1": {
                    "covered_by_citation_claims": ["c1_claim_1", "c1_claim_3"],
                    "reasoning": "<nuanced reasoning why these claims from citations cover claim 1 in sentence 1>",
                    },
                "s1_claim_2": {
                    "covered_by_citation_claims": ["c1_claim_2"],
                    "reasoning": "<…>"
                    },
                …
            },
            "sentence_2": {
                "s2_claim_1": {
                    "covered_by_citation_claims": ["c2_claim_1", "c2_claim_4"],
                    "reasoning": "<nuanced reasoning why these claims from citations cover claim 1 in sentence 2>",
                    },
                "s2_claim_2": {
                    "covered_by_citation_claims": ["c2_claim_2"],
                    "reasoning": "<nuanced reasoning why these claims from citations cover claim 2 in sentence 2>"
                    },
                …
                },
            …
            }
        },
    "faithfullness_evaluation": {
        "description": "Rate the total faithfullness with one of the provided options, it should answer wheter the information is supported by the sources at all. This does not include citation correctness.",
        "rating_options": {
            "0": "No citations or any completely hallucinated claims that are not even close supported by any citations or claims that are contradicted by citations.",
            "1": "Any key aspects are backed by citations, but significant key aspects are not backed by citations or misinterpret them.",
            "2": "Several key aspects are backed by citations, including some of the most important ones, but with significant omissions.",
            "3": "Most of the crucial aspects are backed by citations, though some important points may still be missing or incomplete. This generally meets the minimum standard for a passing result.",
            "4": "Nearly all of the important aspects are thoroughly backed by citations, with only minor claims not proved by citations.",
            "5": "All essential and important aspects are fully backed by citations.",
            },
        "faithfullness_rating": "<your 0-5 rating choice>",
        "faithfullness_evaluation_reasoning": {
            "description": "Step-by-step, explicit reasoning for why the above option was chosen, referencing the specific coverage and gaps from the analysis above along with their severity. This section must explain the rationale for the overall judgment, including edge cases, ambiguity, and any uncertainty.",
            "reasoning": "<your reasoning>"
            }
        },
    "citation_correctness_evaluation": {
        "description": "Rate the total citation correctness, it should answer whether the right citations are attached to the right sentences.",
        "rating_options": {
            "0": "Correctness doesn't apply due to missing citations or major hallucinations.",
            "1": "Any aspects were not backed at all by its own citations, but supported by other sentences citations.",
            "2": "All aspects are backed by its own citations, but other sentences citations would have been way better to prove the claim.",
            "3": "All aspects that are backed by citations are backed by its own citations and these are the most suitable citations for the claim.",
            },
        "citation_correctness_rating": "<your 0-3 rating choice>",
        "citation_correctness_evaluation_reasoning": {
            "description": "Step-by-step, explicit reasoning for why the above evaluation option was chosen, referencing the specific coverage and gaps from the analysis above along with their severity. This section must explain the rationale for the overall judgment, including edge cases, ambiguity, and any uncertainty.",
            "reasoning": "<your reasoning>"
            }
        }
}
"""


CTOC_SYSTEM = """\
You are a legal expert in the field of European Court of Human Rights (ECHR) cases. 
Your task is to evaluate semantic relevance of 'Retrieved Documents' in relation to a 'Question' and a set of 'Golden Reference Documents'.
All Documents are paragraphs from ECHR judgements.


Claims are statements that are verifiable


Your evaluation must strictly adhere to the given JSON output structure! It must be valid JSON and all inputs shall be plain text without any formatting!
"""


CTOC_PROMPT = """\
Your primary goal is to determine if the 'Retrieved Documents' contain information that is semantically equivalent or sufficiently similar to the 'Golden Reference Documents' to address the 'Question'. The phrasing might differ, but the core facts, answers, or concepts should be covered.

The evaluation should follow the following strategy:
First, list the key claims from the 'Golden Reference Documents' (from exclusive and overlapping documents) that are necessary to answer the question.
Second, list the key claims from the 'Retrieved Documents' (from exclusive and overlapping documents).
Third, for each claim in the golden documents, identify which claims from the retrieved documents cover it.
Fourth, for any claims in the retrieved documents that were not used for coverage, assess their relevance to the question.
Finally, provide an overall evaluation rating and reasoning based on your analysis.

Question:
```{question}```


Documents that are exclusively in the Retrieved Documents:
```{exclusive_gen}```


Documents that are both in the Retrieved Documents and in the Golden Documents:
```{overlap_gen_gold}```




Documents that are exclusively in the Golden Documents:
```{exclusive_gold}```



Output structure:
{output_structure}
"""




CTOC_OUTPUT_STRUCTURE = r"""{
    "claims_in_golden_documents": {
        "gold_claim_1": "<claim 1 from golden docs>",
        "gold_claim_2": "<claim 2 from golden docs>",
        "…": "…"
    },
    "claims_in_retrieved_documents": {
        "retrieved_claim_1": "<claim 1 from retrieved docs>",
        "retrieved_claim_2": "<claim 2 from retrieved docs>",
        "…": "…"
    },
    "analysis": {
        "coverage_of_golden_claims": {
            "gold_claim_1": {
                "retrieved_claims": ["retrieved_claim_2", "retrieved_claim_5"],
                "reasoning": "<nuanced reasoning why these retrieved claims cover gold_claim_1>"
            },
            "gold_claim_2": {
                "retrieved_claims": [],
                "reasoning": "<reasoning why this golden claim is not covered by any retrieved claim>"
            },
            "…": {
                "retrieved_claims": [],
                "reasoning": "<…>"
            }
        },
        "relevance_of_excess_retrieved_claims": {
            "retrieved_claim_3": {
                "is_relevant": "<true/false>",
                "reasoning": "<reasoning for relevance to the question>"
            },
            "…": {
                "is_relevant": "<true/false>",
                "reasoning": "<…>"
            }
        }
    },
    "evaluation": {
        "description": "Rate the alignment between the Retrieved Documents and the Golden Reference Documents with one of the provided options.",
        "rating_options": {
            "0": "No relevant alignment: Retrieved Documents are entirely irrelevant, incorrect, or fail to address the Golden Documents or the Question in any meaningful way.",
            "1": "Minimal alignment: Retrieved Documents contain only marginally relevant information or very limited overlap, missing nearly all essential aspects.",
            "2": "Partial but poor alignment: Retrieved Documents capture a few aspects correctly, but most key aspects are missing, incorrect, or require major inference.",
            "3": "Moderate alignment: Retrieved Documents address several key aspects, including some important ones, but with substantial omissions or incompleteness. Minimum threshold for acceptable match.",
            "4": "Strong alignment: Retrieved Documents cover nearly all essential aspects of the Golden Documents, missing only minor details or elaboration.",
            "5": "Full alignment: Retrieved Documents comprehensively and correctly cover all essential aspects of the Golden Documents, fully addressing the Question.",
        },
        "rating": "<your rating choice>",
        "evaluation_reasoning": {
            "description": "Step-by-step, explicit reasoning for why the above rating was chosen, referencing the specific coverage and relevance analysis above. This section must explain the rationale for the overall judgment, including edge cases, ambiguity, and any uncertainty.",
            "reasoning": "<your reasoning>"
        }
    }
}
"""