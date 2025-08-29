import sys
import os
import json
import requests
from string import Template
import urllib.parse
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# logging
import logging
import logging.config
logging_config_path = os.path.join(project_root, '..', 'logging', 'logging.json')
with open(logging_config_path, 'rt') as f:
    config = json.load(f)
logging_dir = os.path.dirname(logging_config_path)
for handler in config.get('handlers', {}).values():
    if 'filename' in handler:
        handler['filename'] = os.path.join(logging_dir, handler['filename'])
        log_file_dir = os.path.dirname(handler['filename'])
        if not os.path.exists(log_file_dir):
            os.makedirs(log_file_dir)
logging.config.dictConfig(config)
logger = logging.getLogger("md_citations_extraction_logs")

def get_metadata_for_case_id(case_id: str):
    
    url_template = Template('https://hudoc.echr.coe.int/app/query/results?query=((itemid%3A"$case_id"))&select=sharepointid,rank,echrranking,languagenumber,itemid,docname,doctype,application,appno,conclusion,importance,originatingbody,typedescription,kpdate,kpdateastext,documentcollectionid,documentcollectionid2,languageisocode,extractedappno,isplaceholder,doctypebranch,respondent,advopidentifier,advopstatus,ecli,appnoparts,sclappnos,ECHRConcepts&sort=&start=0&length=20&rankingModelId=11111111-0000-0000-0000-000000000000')
    url = url_template.substitute(case_id=case_id)
    res = requests.get(url)
    data = res.json()
    return data["results"][0]["columns"]


# Case-law guide uses different country codes than HUDOC
COUNTRY_CODE_HUDOC_TO_GUIDE = {
    "ENG": "eng",
    "FRE": "fre",
    "RUM": "ron",
}

def search_hudoc(appno=None, advno=None):    

    url = (
        f"https://hudoc.echr.coe.int/app/query/results?query=contentsitename:ECHR "
        f"AND (NOT (doctype=PR OR doctype=HFCOMOLD OR doctype=HECOMOLD)) "
    )
    if appno:
        fst = appno.split("/")[0]
        snd = appno.split("/")[1]
        url += f"AND ((appno={fst}/{snd})) "

    elif advno:
        advopidentifier = advno
        url += f'AND ((advopidentifier="{advopidentifier}")) '


    url += (
        # f'AND ((documentcollectionid="GRANDCHAMBER") OR (documentcollectionid="CHAMBER"))'
        f"&select=sharepointid,rank,echrranking,languagenumber,itemid,docname,doctype,application,appno,conclusion,importance,originatingbody,typedescription,kpdate,kpdateastext,documentcollectionid,documentcollectionid2,languageisocode,extractedappno,isplaceholder,doctypebranch,respondent,advopidentifier,advopstatus,ecli,appnoparts,sclappnos,ECHRConcepts&sort=&start=0&length=20&rankingModelId=22222222-eeee-0000-0000-000000000000"
    )
    urllib.parse.quote(url, safe=":/?=&,")
    res = requests.get(url)
    return res.json()



def find_translations(metadata):
    advno = None
    appno = None

    if not metadata.get("appno"):
        advno = metadata["advopidentifier"]
        data = search_hudoc(advno=advno)
    
    else:
        appno = metadata["appno"].split(";")[0]
        data = search_hudoc(appno=appno)
             
    # # Debug output of full response from API
    # print(f"For Appno: {appno}")
    # print("Data:\n", json.dumps(data, indent=2))


    # result dictionary
    result = {
        "orig_id": metadata["itemid"],
        "kpdate": metadata["kpdate"],
    }
    if appno:
        result["appno"] = appno
    if advno:
        result["advno"] = advno

    # Process all language versions
    for item in data["results"]:
        if all(col in item["columns"] for col in ["languageisocode", "itemid", "docname", "kpdate"]):

            if  metadata["kpdate"] == item["columns"]["kpdate"] \
                and "summary" not in item["columns"]["docname"] \
                and "Résumé juridique" not in item["columns"]["docname"] \
                and not item["columns"]["itemid"].startswith("002-") \
                and item['columns']['isplaceholder'] == 'False':
                    
                    lang_code = item["columns"]["languageisocode"].upper()
                    lang_code = COUNTRY_CODE_HUDOC_TO_GUIDE.get(lang_code, lang_code) # Case-law guide uses different country codes than HUDOC
                    item_id = item["columns"]["itemid"]
                    docname = item["columns"]["docname"]
                    date = item["columns"]["kpdate"]

                    # Doen't have to be an error as some cases are translated by several institutions
                    if lang_code in result:
                        logger.debug(f'Duplicate language version found for {lang_code}: {{\n\t"id": {item_id},\n\t"docname": {docname},\n\t"date": {date}\n}}')
                    else:
                        result[lang_code] = {
                            "id": item_id,
                            "docname": docname,
                        }
    
    return result


def multilingual_citations(url, lang_code):

    case_id = url.split("?i=")[-1].split("&", 1)[0]
    metadata = get_metadata_for_case_id(case_id)
    translations = find_translations(metadata)
    desired_keys = ["orig_id", "appno", "advno", "kpdate"]
    desired_langs = ["eng", "fre", lang_code.lower()]
    output = {}
    for key, value in translations.items():
        if key in desired_keys:
            output[key] = value
        elif key in desired_langs:
            if "lang_versions" not in output:
                output["lang_versions"] = {}
            output["lang_versions"][key.lower()] = value
    return output
