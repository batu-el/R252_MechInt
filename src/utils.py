def preprocess_entry(entry):
    source = entry["source"]
    target = entry["target"]
    return source, target

def tokenize_data(tokenizer, data):
    return tokenizer(data['source'], padding="max_length", truncation=True, max_length=512)
