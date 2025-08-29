import re
import spacy


nlp = spacy.load("en_core_web_trf")

def remove_par_text_prefix(text: str, case_name: str, paragraph_number: str | int):
    text = text.replace(f"{case_name}; § {paragraph_number}: ", "")
    return text

def remove_unwanted_citations_at_the_end(sentences: list[str]):
    """
    LLMs often add a list of the used citations at the end of the response. We try to remove these citations.
    We remove all sentences at the end that do not end in a "." or start with a citation.
    """

    while sentences:
        if (
            not sentences[-1].strip().endswith(".")
            or sentences[-1].strip().startswith("[")
            or sentences[-1].strip().startswith("【")
            or "citations:" in sentences[-1].lower()
        ):
            sentences.pop()
        else:
            break

    return sentences


def get_sentences(response: str, simple: bool = False) -> list[str]:
    """
    We split the response into sentences using spaCy.
    """
    initial_sentences = response.split("\n")  # an llm does not randomly add newlines
    sentences = []
    
    if not simple:
        for sentence in initial_sentences:
            if sentence.strip():
                doc = nlp(sentence)
                sentences.extend([s.text for s in doc.sents if s.text.strip()])
    else:
        sentences = [sentence.strip() for sentence in initial_sentences if sentence.strip()]

    return sentences


def remove_citations(sentence: str):
    """
    Utility functions for extracting / removing citations from the response
    """
    pattern = r" \[.*?\]"
    return re.sub(pattern, "", sentence)


def get_citations(sentence: str) -> list[int]:
    """
    Extract citations from a sentence in the following forms:
    - Doc 1
    - Doc 1, 2
    - Doc 2-5
    - Doc 2, 3, and 4
    - Doc 1 and 2
    - [1]
    - [2, 3]
    - [2 and 3]
    - [3-5]

    Args:
    sentence (str): The input sentence containing citations.

    Returns:
    list of int: The list of extracted document numbers.
    """
    # Normalize the sentence by replacing variations of "and" with commas
    sentence = sentence.replace(", and", ",")
    sentence = sentence.replace(" and", ",")
    sentence = sentence.replace("【", "[")
    sentence = sentence.replace("】", "]")

    # Patterns to match the different citation styles
    patterns = [
        r"Doc\s(\d+(?:-\d+)?(?:,\s\d+)*)",  # Doc 1, Doc 1, 2, Doc 2-5, Doc 2, 3, and 4, Doc 1 and 2
        r"\[(\d+(?:-\d+)?(?:,\s\d+)*)\]",  # [1], [2, 3], [2 and 3], [3-5]
    ]

    citations = set()

    for pattern in patterns:
        matches = re.findall(pattern, sentence)
        for match in matches:
            parts = match.split(", ")
            for part in parts:
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    citations.update(range(start, end + 1))
                else:
                    citations.add(int(part))

    return sorted(citations)


def test_get_citations():
    test_cases = [
        ("This is a sentence with a citation [1].", [1]),
        ("This sentence has multiple citations [2, 3].", [2, 3]),
        ("This one has a range [3-5].", [3, 4, 5]),
        ("Citations with 'and' [2 and 3].", [2, 3]),
        ("Citations with ', and' [2, and 3].", [2, 3]),
        ("Doc style citation Doc 1.", [1]),
        ("Doc style with multiple Doc 1, 2.", [1, 2]),
        ("Doc style with range Doc 2-5.", [2, 3, 4, 5]),
        ("Doc style with 'and' Doc 1 and 2.", [1, 2]),
        ("Doc style with ', and' Doc 2, 3, and 4.", [2, 3, 4]),
        ("No citations here.", []),
        ("Mixed styles [1] and Doc 2.", [1, 2]),
        ("This is a sentence with a citation [1]. And another one [2].", [1, 2]),
        ("This is a sentence with a citation Doc 1. And another one Doc 2.", [1, 2]),
        ("This is a sentence with a citation [1-3]. And another one Doc 4, 5.", [1, 2, 3, 4, 5]),
        ("Prin urmare, dacă expulzarea este considerată legală și conformă cu Articolul 4, restricţia asupra libertăţii de circulație este justificată și nu constituie o încălcare a Articolului 2【Doc 2, Doc 6】.", [2, 6]),
        ("Acest principiu se bazează pe interpretarea Curţii că „expulsion” are sensul generic de îndepărtare forţată a unui străin din teritoriul unui stat, indiferent de legalitatea șederii, şi că această măsură prevalează asupra dreptului la libertatea de circulație atunci când există un ordin definitiv de expulzare [Doc 5].", [5]),
        ("În lipsa unei astfel de motivări specifice, refuzul este considerat arbitrar și încalcă dreptul la un proces echitabil, deoarece nu se respectă obligația de a furniza argumente clare și suficiente pentru decizia luată【Doc 60, Doc 61, Doc 62】.", [60, 61, 62]),
        ("Se verifică dacă motivarea răspunde criteriilor enumerate de CJUE; o motivare insuficientă sau absentă este considerată arbitrară și constituie încălcare a Articolului 6 alineatul 1 [Doc 5, Doc 8].", [5, 8]),
        ("justifică refuzul și de ce aceasta nu lasă loc de îndoială rezonabilă [Doc 1][Doc 6][Doc 10].", [1, 6, 10]),
        ("Această motivare trebuie să arate clar dacă motivul refuzului se încadrează în una dintre excepțiile CILFIT: întrebarea este irelevantă, prevederea UE a fost deja interpretată de Curtea de Justiție sau aplicarea corectă a dreptului UE este atât de evidentă încât nu există dubiu rezonabil [Doc 1, Doc 4].", [1, 4])
    ]

    for i, (sentence, expected) in enumerate(test_cases):
        result = get_citations(sentence)
        assert result == expected, f"Test case {i+1} failed: For '{sentence}', expected {expected}, but got {result}"
        print(f"Test case {i+1} passed!")

if __name__ == '__main__':
    test_get_citations()