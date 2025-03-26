import spacy

nlp = spacy.load("en_core_web_sm")

def extract_people_names(query):
    doc = nlp(query)
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

def extract_account_names(query, chunks):
    account_names = {chunk["project"] for chunk in chunks}
    lowered_query = query.lower()
    found_account_names = []
    
    for name in account_names:
        if name.lower() in lowered_query:
            found_account_names.append(name)

    return found_account_names
