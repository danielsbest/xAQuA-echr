from pymongo import MongoClient
from collections import Counter
import os
from dotenv import load_dotenv
import json

def get_database_collection(collection_name="echr_documents"):
    """
    Establishes a connection to the MongoDB database and returns a collection object.
    """
    load_dotenv()
    MONGO_DB_USERNAME = os.getenv('MONGO_DB_USERNAME')
    MONGO_DB_PWD = os.getenv('MONGO_DB_PWD')
    URI = "mongodb://%s:%s@f27se1.in.tum.de:27017/echr" % (MONGO_DB_USERNAME, MONGO_DB_PWD)
    client = MongoClient(URI)
    database = client['new_echr']
    return database[collection_name]

def get_document_by_id(collection, itemid):
    return collection.find({'_id': itemid})

def get_documents_by_year(collection, year):
    """
    Retrieves all documents for a given year.
    """
    query = {
        'kpdate': {
            '$gte': f'{year}-01-01T00:00:00',
            '$lt': f'{year+1}-01-01T00:00:00'
        }
    }
    return collection.find(query)

def get_document_type_counts(documents):
    """
    Counts the document types from a list of documents.
    """
    doc_types = [doc['doctype'] for doc in documents]
    return Counter(doc_types)

if __name__ == "__main__":
    echr_collection = get_database_collection()
    
    ids = ['001-103259',] # '001-104367', '001-107033', '001-112101', '001-122881', '001-127410', '001-145277', '001-148683', '001-148927', '001-155895', '001-163926', '001-165078', '001-165265', '001-177319', '001-180706', '001-180802', '001-183961', '001-203533', '001-205509', '001-213717', '001-216748', '001-218132', '001-225324', '001-231739', '001-59159', '001-71314', '001-72364', '001-80616', '001-81182', '001-93373']

    for id in ids:
        document = get_document_by_id(echr_collection, id)
        def json_default(o):
            if isinstance(o, bytes):
                return "<binary data>"
            raise TypeError(repr(o) + " is not JSON serializable")

        for doc in document:
            print(json.dumps(doc, indent=4, default=json_default, ensure_ascii=False))


# Example document structure:
json_data = {
  "_id": "003-4930442-6035902",
  "originatingbody": "",
  "appnoparts": "",
  "representedby": "",
  "typedescription": "1002",
  "resolutionnumber": "",
  "nonviolation": "",
  "scl": "",
  "scl_array": [],
  "organisations": "ECHR;Committee of Ministers",
  "documentcollectionid": "PRESS;CHAMBERJUDGMENTS;ENG",
  "judges": "Aleš Pejchal;André Potocki;Angelika Nußberger;Ganna Yudkivska;Helena Jäderblom;Mark Villiger;Vincent A. De Gaetano",
  "courts": "Court of Cassation",
  "conclusion": "",
  "documentcollectionid2": "PRESS;CHAMBERJUDGMENTS;ENG",
  "meetingnumber": "",
  "externalsources": "",
  "doctypebranch": "",
  "appno": "",
  "respondent": "",
  "application": "ACROBAT",
  "importance": "",
  "extractedappno": "40014/10;61198/08;44446/10;60995/09;53406/10;30010/10;66069/09;130/10;3896/10",
  "rulesofcourt": "",
  "ecli": "",
  "isplaceholder": "False",
  "rank": "7.1376038",
  "violation": "",
  "publishedby": "",
  "judgementdate": "",
  "dmdocnumber": "",
  "sclappnos": "",
  "separateopinion": "",
  "doctype": "PR",
  "languageisocode": "ENG",
  "introductiondate": "",
  "reportdate": "",
  "kpthesaurus": "",
  "issue": "",
  "applicability": "",
  "languagenumber": 10,
  "docname": "Judgment Bodein v. France - possibility of review of life sentence",
  "article": "",
  "counter": 0,
  "kpdate": "0001-01-01T00:00:00",
  "referencedate": "",
  "decisiondate": "",
  "resolutiondate": "",
  "html": "",
  "pdf": "b'\\x80\\x04B\\xe7\\x9a\\x01\\x00%PDF-1.5\\r\\n4 0 obj\\r\\n<</Type /Page/Parent 3 0 R/Contents 5 0 R/MediaBox [0 0 595...Root 2 0 R>>\\r\\nstartxref\\r\\n103448\\r\\n%%EOF\\r\\n\\x94.'",
}
