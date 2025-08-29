import sys
from typing import Optional
import pandas as pd
import pathlib

project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.llms.gemini import Gemini, GeminiFlash2_5
from src.translations.translation import Translator, LANGUAGE_CODES
from src.column import Column
import src.constants as constants


class LLM_Translator(Translator):
    def __init__(self, llm: Optional[Gemini] = None):
        if llm is None:
            self.llm = GeminiFlash2_5()
        else:
            self.llm = llm
        self.system_prompt = "You are a professional translator. Translate the following text to {target_lang}. Don't comment, don't change structure, just output the pure translation."

    def translate(self, text: str, target_lang: str = 'eng', source_lang: Optional[str] = None) -> Optional[str]:
        if text is None:
            return None
        if text == "":
            return ""

        language_name = LANGUAGE_CODES[target_lang.lower()]
        system_prompt = self.system_prompt.format(target_lang=language_name)

        translation = self.llm.infer_completion(
            prompt=text,
            system_prompt=system_prompt,
            temperature=0.0,
            thinking_budget=2000
        )
        return translation.strip()
    
    def translate_batch(self, texts: list[str], target_lang: str = 'eng', source_lang: Optional[str] = None) -> list[str]:
        language_name = LANGUAGE_CODES[target_lang.lower()]
        system_prompt = self.system_prompt.format(target_lang=language_name)

        prompts_to_run = [{"prompt": text, "system_prompt": system_prompt, "temperature": 0.0, "thinking_budget": 2000} for text in texts]
        translations = self.llm.infer_batch_completion(prompts_to_run)
        return [translation.strip() for translation in translations]
        
    

def pretranslate_questions()-> None:
    df = pd.read_csv(constants.ECHR_QA_CSV_PATH)
    questions = df[Column.QUESTION].tolist()
    translator = LLM_Translator()
    translations = translator.translate_batch(questions, target_lang='eng')
    df[Column.QUESTION_TRANSLATION] = translations
    df.to_csv(constants.ECHR_QA_CSV_PATH, index=False)

if __name__ == "__main__":
    pretranslate_questions()
    print("Pretranslation of questions completed.")