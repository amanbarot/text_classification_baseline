import nltk

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

LEMMATIZER = nltk.WordNetLemmatizer()

def tokenize(batch,
             is_batched=True):
    """
    Tokenizes the text in a batch of data.
    Args:
        batch (dict): A dictionary containing a 'text' key with a list of strings.
        is_batched (bool): Whether the input is batched or not.
    Returns:
        dict: A dictionary with the same keys as input, but with 'tokens' key added.
    """
    if not is_batched:
        for key, val in batch.items():
            batch[key] = [val]
    
    batch['tokens'] = [
        nltk.word_tokenize(el.lower()) 
        for el in batch['text']
    ]
    return batch

def extract_pos_tags(tokens):
    """
    Extracts part-of-speech tags from a list of tokens.
    Args:
        tokens (list): A list of tokens.
    Returns:
        list: A list of tuples containing tokens and their POS tags.
    """
    # Map POS tags to WordNet format
    pos_map = {
        'N': 'n',  # Noun
        'V': 'v',  # Verb
        'R': 'r',  # Adverb
        'J': 'a'   # Adjective
    }
    pos_tags = [pos_map.get(pos[1][0], 'n') for pos in nltk.pos_tag(tokens)]
    return pos_tags

def lemmatizer_func(batch_tokens,
        lemmatizer=LEMMATIZER,
        is_batched=True):
    """
    Lemmatizes a list of tokens using the WordNet lemmatizer.
    Args:
        tokens (list): A list of tokens to lemmatize.
        lemmatizer (nltk.WordNetLemmatizer): An instance of the WordNet lemmatizer.
    Returns:
        list: A list of lemmatized tokens.
    """
    if not is_batched:
        for key, val in batch_tokens.items():
            batch_tokens[key] = [val]
    # Get POS tags for each token
    batch_tokens['pos_tags'] = [
        extract_pos_tags(tokens) 
        for tokens in batch_tokens['tokens']
    ]
    # Lemmatize each token with its corresponding POS tag
    batch_tokens['cleaned_tokens'] = [
        [lemmatizer.lemmatize(token, pos) for token, pos in zip(tokens, pos_tags)]
        for tokens, pos_tags in zip(batch_tokens['tokens'], batch_tokens['pos_tags'])
    ]
    return batch_tokens
