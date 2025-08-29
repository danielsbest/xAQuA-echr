import sys
import os
import re
import pandas as pd
from pdfminer.high_level import extract_text
from pydantic import BaseModel
import glob

from typing import List, Tuple

LANG_STRINGS = {
    "eng": {
        "start_string": "HUDOC keywords",
        "end_string": "List of cited cases",
        "clutter_patterns": [
            r"\sEuropean Court of Human Rights\s+\d+\/\d+\s+Last update: \d{1,2}\.\d{2}\.\d{4}\s*(.*?)\n"
        ],
    },
    "fre": {
        "start_string": "Mots-clés HUDOC",
        "end_string": "Liste des affaires citées",
        "clutter_patterns": [
            r"\sCour européenne des droits de l’homme\s+\d+\/\d+\s+Mise à jour : \d{1,2}\.\d{2}\.\d{4}\s*(.*?)\n"
        ],
    },
    "ron": {
        "start_string": "Cuvinte-cheie HUDOC",
        "end_string": "Lista cauzelor citate",
        "clutter_patterns": [
            r"Curtea Europeană a Drepturilor Omului\s+\d+\/\d+\s+.{1,8}?ctualizare: \d{1,2}\.\d{1,2}\.\s*\d{4}\s*(.*?)\n",
        ],
    },
}

# logging
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Project root set to: {project_root}")
import logging
import logging.config
import json

logging_config_path = os.path.join(project_root, "logging", "logging_config.json")
with open(logging_config_path, 'rt') as f:
    config = json.load(f)

for handler_name, handler_config in config.get("handlers", {}).items():
    if "filename" in handler_config:
        # Convert relative path to absolute path
        handler_config["filename"] = os.path.join(project_root, handler_config["filename"])
        # Ensure directory exists
        log_dir = os.path.dirname(handler_config["filename"])
        os.makedirs(log_dir, exist_ok=True)

logging.config.dictConfig(config)
logger = logging.getLogger("guide_parser_logs")

def number_of_paragraphs(text: str):
    i = 1
    while f"{i}." in text:
        i += 1
    return i - 1


def clean_paragraph(paragraph: str, superscripts: dict):

    paragraph = re.sub(r"\s+", " ", paragraph)  # remove double spaces
    paragraph = paragraph.strip()

    # add superscripts
    match = re.findall(r"<sup>(.*?)</sup>", paragraph)
    for sup in match:
        if superscripts.get(sup):
            paragraph += f"\n<sup>{sup}</sup>{superscripts[sup]}"
        elif superscripts.get(sup + "."):
            paragraph += f"\n<sup>{sup}</sup>{superscripts[sup + '.']}"
        else:
            logger.warning(f"Superscript {sup} not found in superscripts dictionary.")

    return paragraph


def extract_paragraph(i: int, text: str, superscripts: dict):
    split_on_par = text.split(f"\n{i+1}. ", maxsplit=1)

    if len(split_on_par) != 2:
        logger.debug(f'Failed to split at "\\n{i+1}. "') 
        split_on_par = text.split(f"\n- {i+1}.", maxsplit=1)
        
        
        if len(split_on_par) == 2:
            logger.debug(f'Successfully split at "\\n- {i+1}."')
        else:
            logger.debug(f'Failed to split at "\\n- {i+1}."')
            raise Exception("Failed to split on", i + 1)

    paragraph = clean_paragraph(split_on_par[0], superscripts)
    continued_text = split_on_par[1]

    return paragraph, continued_text

    
def get_markdown(guide_id: str):
    lang = guide_id[-3:]

    data_path = f"../data/parse_control/marker-md/{lang}/{guide_id}/{guide_id}.md"
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
            return text
    except FileNotFoundError:
        logger.error(f"Markdown file not found for guide {guide_id} in language {lang}.")
        raise FileNotFoundError(f"Markdown file not found for guide {guide_id} in language {lang}.")


class GuideParsingMeta(BaseModel):
    guide_id: str
    starting_string: str | None = None
    base_url: str = "https://ks.echr.coe.int/documents/d/echr-ks/"



class GuideParser:
    def __init__(self, guide_id: str, starting_string: str | None = None, base_url: str = "https://ks.echr.coe.int/documents/d/echr-ks/", remove_patterns: List[str] = [], replace_patterns: List[Tuple[str, str]] = []):
        self.guide_id = guide_id
        self.lang = guide_id[-3:]
        self.base_url = base_url
        self.starting_string = starting_string
        self.remove_patterns = remove_patterns
        self.replace_patterns = replace_patterns
        self.superscripts = {}

    def __trim_document(self, text: str):
        if self.starting_string:
            text = self.starting_string.replace("1. ", "") + text.split(self.starting_string, maxsplit=1)[1]
        else:
            text = text.split(LANG_STRINGS[self.lang]["start_string"], maxsplit=1)[1]
            text = text.split("\n1. ", maxsplit=1)[1]
        
        text = text.split(LANG_STRINGS[self.lang]["end_string"])[0]
        return text

    def __remove_clutter(self, text: str):
        all_patterns = []

        all_patterns.extend(self.replace_patterns)

        for pattern in self.remove_patterns:
            all_patterns.append((pattern, ""))

        # remove all markdown headings and subs
        headings_pattern = r"^\s*#{1,6}\s+.*$\n?"
        page_span_pattern = r'<span id=".*?"></span>'
        sup_pattern = r"^\s*<sup>(.*?)</sup>(.*)"

        all_patterns.append((headings_pattern, ""))
        all_patterns.append((page_span_pattern, ""))
        all_patterns.append((sup_pattern, ""))

        # Store the superscript texts
        sup_matches = re.findall(sup_pattern, text, re.MULTILINE)
        for match in sup_matches:
            self.superscripts[match[0]] = match[1]

        for pattern, replacement in all_patterns:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

        return text

    def __get_text(self):
        return get_markdown(self.guide_id)

    def __extract_content(self):
        text = self.__get_text()
        text = self.__trim_document(text)
        text = self.__remove_clutter(text)
        return text

    def _write_paragraphs_to_file(self, paragraphs: List[str]):
        """Write paragraphs to file and detect changes if file already exists."""
        # Create language-specific directory
        lang_dir = f"../data/parse_control/marker_txt_versions/{self.lang}"
        os.makedirs(lang_dir, exist_ok=True)
        
        # Generate new content
        new_content = ""
        for i, paragraph in enumerate(paragraphs):
            new_content += f"{i+1}. {paragraph}\n"
        
        # Find all existing versions in the new location
        pattern = f"{lang_dir}/*_paragraphs_{self.guide_id}.txt"
        existing_files = glob.glob(pattern)
         
        if existing_files:
            # Find highest existing version number and latest file
            max_version = -1  # Start at -1 so first version is 0
            latest_file = None
            for file_path in existing_files:
                filename = os.path.basename(file_path)
                if filename[0].isdigit():
                    version_num = int(filename.split('_')[0])
                    if version_num > max_version:
                        max_version = version_num
                        latest_file = file_path
            
            if latest_file is None:
                logger.error(f"No valid versioned files found for guide {self.guide_id}")
                return
            
            try:
                with open(latest_file, "r", encoding="utf-8") as existing_f:
                    existing_content = existing_f.read()
                
                if existing_content != new_content:
                    # Next version is one higher than the highest found
                    next_version = max_version + 1
                    
                    versioned_path = f"{lang_dir}/{next_version}_paragraphs_{self.guide_id}.txt"
                    logger.info(f"Writing new version to: {versioned_path}")
                    
                    # Write to versioned file
                    with open(versioned_path, "w", encoding="utf-8") as out_f:
                        out_f.write(new_content)
                    
                    return
                else:
                    return  # No need to write if content is the same
                    
            except Exception as e:
                logger.error(f"Error comparing files for guide {self.guide_id}: {e}")
        else:
            # No existing files, create the first version starting with 0
            logger.info(f"New file created for guide {self.guide_id}")
            versioned_path = f"{lang_dir}/0_paragraphs_{self.guide_id}.txt"
            with open(versioned_path, "w", encoding="utf-8") as out_f:
                out_f.write(new_content)

    def parse(self):
        logger.info(f"Starting to parse guide: {self.guide_id}")
        text = self.__extract_content()
        n = number_of_paragraphs(text)
        logger.debug(f"Number of paragraphs found: {n}")
        paragraphs = []
        for i in range(1, n):
            paragraph, text = extract_paragraph(i, text, self.superscripts)
            paragraphs.append(paragraph)
        paragraphs.append(clean_paragraph(text, self.superscripts))  # Last paragraph

        # write to file for using change detection
        self._write_paragraphs_to_file(paragraphs)

        return paragraphs

    def to_csv(self):
        paragraphs = self.parse()
        df = pd.DataFrame(paragraphs, columns=['paragraph'], index=range(1, len(paragraphs)+1))
        guide_ids = [self.guide_id] * len(paragraphs)
        df['guide_id'] = guide_ids
        df['paragraph_id'] = df.index
        return df