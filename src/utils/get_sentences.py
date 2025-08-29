import spacy


nlp = spacy.load("en_core_web_trf")


def get_sentences(response: str):
    doc = nlp(response)
    sentences = [s.text for s in doc.sents if s.text.strip()]
    return sentences
