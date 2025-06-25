from html import unescape
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Load the spaCy model
nlp_model = spacy.load(
    "en_core_web_sm", 
    disable=["parser"]
    ) 

# Global constants
FILTERED_ENTS = {
    'PERSON', 'DATE', 'TIME', 
    }
CUSTOM_STOP_WORDS = STOP_WORDS.copy()

# Updating the stop words with custom ones
# These are common words that may not be useful for analysis
# or may skew results, such as names of days, news organizations, etc.
day_names = [
    "monday", 
    "tuesday", 
    "wednesday", 
    "thursday", 
    "friday", 
    "saturday", 
    "sunday",
    "today",
    "yesterday"
    ]
news_org_names = [
    'ap', 
    'afp',
    'reuters',
]
other_stop_words = [
    'quickinfo',
    'newyork',
    'amp', # unescape html did not clean this well
]
additional_stop_words = day_names \
    + news_org_names \
    + other_stop_words
CUSTOM_STOP_WORDS.update(additional_stop_words)

# definitions of functions
def remove_html(text):
    """Escapes HTML entities in the text.
    This function uses the `html.unescape` method to convert HTML entities
    back to their corresponding characters. For example, it converts
    &amp; to &, &lt; to <, and &gt; to >.

    Args:
        text (str): The input text containing HTML entities.
    Returns:
        str: The text with HTML unescaped.
    """
    return unescape(text)

def remove_boilerplate(text):
    """Removes text patterns for stock quotes such as 
        (AEOS.O: Quote, Profile, Research) as this is an easy giveaway 
        for business articles.

    Args:
        text (str): The input text
    
    Returns:
        str: The cleaned text.
    """
    text = re.sub(
        r'\([^)]*(quote|profile|research)[^)]*\)', 
        '', 
        text, 
        flags=re.IGNORECASE
    )
    return text

def normalize_text(text):
    """
    Normalizes known patterns, such as 'new york' -> 'newyork',
    and removes boilerplate content.
    
    Args:
        text (str): The input text.
    
    Returns:
        str: The normalized text.
    """
    # Remove boilerplate first
    text = remove_boilerplate(text)
    
    # Collapse location names
    text = re.sub(r'\bnew york\b', 'newyork', text, flags=re.IGNORECASE)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    """
    Full preprocessing on raw text: remove HTML, remove boilerplate, normalize.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    text = remove_html(text)
    text = normalize_text(text)
    return text

def tokenize_lemmatize_filter_ents(
        batch, 
        nlp_model=nlp_model,
        stop_words=list(STOP_WORDS),
        filtered_ents=list(FILTERED_ENTS)):
    """
    Tokenizes, lemmatizes, and removes tokens with specified entity labels
      using spaCy.
    
    Args:
        batch (dict): A dictionary with a 'text' key and a list of strings.
        nlp_model (spacy.lang.*): A loaded spaCy language model.
        stop_words (list): A list of stop words to exclude. A list is passed 
            instead of a set so that it is serializable by dataset.map

    Returns:
        dict: Updated dictionary with 'tokens', 'lemmas', and 'cleaned_tokens'.
    """
    stop_words = set(stop_words)
    filtered_ents = set(filtered_ents)
    cleaned_texts = [preprocess_text(t) for t in batch['text']]
    batch['tokens'] = []
    batch['lemmas'] = []
    batch['cleaned_tokens'] = []
    for doc in nlp_model.pipe(cleaned_texts, batch_size=1000):
        tokens = [token.text for token in doc]
        lemmas = [token.lemma_ for token in doc]
        cleaned_tokens = [
            token.lemma_.lower() 
            for token in doc 
            if token.is_alpha 
            and token.lemma_.lower() not in stop_words
            and token.ent_type_ not in filtered_ents
            ]
        batch['tokens'].append(tokens)
        batch['lemmas'].append(lemmas)
        batch['cleaned_tokens'].append(cleaned_tokens)
    return batch