import os
import re
import json
import string
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

# NLP & Readability
import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
from langdetect import DetectorFactory, LangDetectException, detect_langs
import textstat
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
DetectorFactory.seed = 0

# ==========================================
# 1. 33 DUMMY CATEGORIES (Mancino et al., Table 4)
# ==========================================
CATEGORIES_33 = [
    "art", "bitcoin", "blog", "busy", "christianity", "colorchallenge", 
    "contest", "crypto", "dlive", "dsound", "dtube", "esteem", "food", 
    "funny", "gaming", "health", "indonesia", "introduceyourself", "life", 
    "love", "money", "music", "nature", "news", "photography", "poetry", 
    "science", "steem", "steemit", "story", "travel", "utopian-io", "zappl"
]


# ==========================================
# 2. ASCII FILTER & TEXT PREPROCESSING
# ==========================================
def is_pure_ascii(text: str) -> bool:
    """Returns True if text contains only valid ASCII characters (0-127)."""
    if not isinstance(text, str) or not text.strip():
        return False
    return text.isascii()


def is_english_post(title: object, body: object, min_confidence: float = 0.85) -> bool:
    """Returns True when the combined post text is confidently detected as English."""
    text = clean_post_text(f"{title or ''}. {body or ''}")
    alphabetic_chars = sum(character.isalpha() for character in text)
    if alphabetic_chars < 20:
        return False

    try:
        language_scores = detect_langs(text)
    except LangDetectException:
        return False

    english_score = next(
        (result.prob for result in language_scores if result.lang == "en"),
        0.0,
    )
    return english_score >= min_confidence


def clean_post_text(text: str) -> str:
    """Preprocesses text: removes HTML/markdown artifacts, links, and converts emojis."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    emoji_dict = {
        r':-\)|:\)': ' smiling face ',
        r':-\(|:\(': ' frowning face ',
        r':-D|:D': ' laughing face ',
        r';-\)|;\)': ' winking face ',
        r':-P|:P': ' playful face ',
        r'<3': ' love heart '
    }
    for emo, rep in emoji_dict.items():
        text = re.sub(emo, rep, text)
        
    text = re.sub(r'!\[.*?\]\(.*?\)', ' ', text)
    text = re.sub(r'\[.*?\]\(.*?\)', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    
    return re.sub(r'\s+', ' ', text).strip()


# ==========================================
# 3. OPINION & BERT EXTRACTORS
# ==========================================
class BatchFeatureExtractors:
    """Handles batched GPU/CPU inference for both TweetEval RoBERTa and 768-d BERT embeddings."""
    def __init__(self, device: str = None, batch_size: int = 64):
        if device is None:
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipe_device = 0 if torch.cuda.is_available() else -1
        else:
            self.device_str = device
            self.pipe_device = 0 if "cuda" in device else -1
            
        self.batch_size = batch_size
        print(f"Loading Models on: {self.device_str.upper()}...")
        
        # 1. TweetEval RoBERTa Opinion Pipelines (9 features)
        self.sentiment_pipe = pipeline(
            "sentiment-analysis", 
            model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
            top_k=None, 
            device=self.pipe_device,
            truncation=True,
            max_length=512
        )
        self.emotion_pipe = pipeline(
            "text-classification", 
            model="cardiffnlp/twitter-roberta-base-emotion", 
            top_k=None, 
            device=self.pipe_device,
            truncation=True,
            max_length=512
        )
        self.offensive_pipe = pipeline(
            "text-classification", 
            model="cardiffnlp/twitter-roberta-base-offensive", 
            top_k=None, 
            device=self.pipe_device,
            truncation=True,
            max_length=512
        )
        
        # 2. DistilRoBERTa Sentence Transformer (768-d Dense Vector)
        # As used in Mancino et al. (2025) [Ref 55]
        self.bert_model = SentenceTransformer("all-distilroberta-v1", device=self.device_str)

    def extract_opinions(self, texts: list) -> list:
        safe_texts = [t if t.strip() else "neutral text" for t in texts]
        sent_outs = self.sentiment_pipe(safe_texts, batch_size=self.batch_size)
        emo_outs = self.emotion_pipe(safe_texts, batch_size=self.batch_size)
        off_outs = self.offensive_pipe(safe_texts, batch_size=self.batch_size)
        
        results = []
        for s_out, e_out, o_out in zip(sent_outs, emo_outs, off_outs):
            s_dict = {item['label'].lower(): item['score'] for item in s_out}
            e_dict = {item['label'].lower(): item['score'] for item in e_out}
            o_dict = {item['label'].lower(): item['score'] for item in o_out}
            
            off_val = o_dict.get('offensive', o_dict.get('label_1', 0.0))
            not_off_val = o_dict.get('non-offensive', o_dict.get('not-offensive', o_dict.get('label_0', 1.0 - off_val)))
            
            results.append({
                "positive": s_dict.get('positive', 0.0),
                "neutral": s_dict.get('neutral', 0.0),
                "negative": s_dict.get('negative', 0.0),
                "anger": e_dict.get('anger', 0.0),
                "joy": e_dict.get('joy', 0.0),
                "sadness": e_dict.get('sadness', 0.0),
                "optimism": e_dict.get('optimism', 0.0),
                "offensive": off_val,
                "not_offensive": not_off_val
            })
        return results

    def extract_bert_embeddings(self, texts: list) -> np.ndarray:
        """Generates (N, 768) dimensional dense vector representations."""
        safe_texts = [t if t.strip() else "empty" for t in texts]
        embeddings = self.bert_model.encode(
            safe_texts, 
            batch_size=self.batch_size, 
            show_progress_bar=False, 
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        return embeddings


# ==========================================
# 4. CHUNK PROCESSOR
# ==========================================
def process_chunk(chunk_df: pd.DataFrame, extractors: BatchFeatureExtractors, include_bert: bool = True) -> pd.DataFrame:
    """Processes English ASCII posts and extracts 74 features plus 768 BERT dimensions."""
    
    # 1. Strict ASCII and English-language filters
    ascii_mask = chunk_df['body'].apply(is_pure_ascii)
    english_mask = chunk_df.apply(
        lambda row: is_english_post(row.get('title', ''), row.get('body', '')),
        axis=1,
    )
    filtered_df = chunk_df[ascii_mask & english_mask].copy().reset_index(drop=True)
    
    if filtered_df.empty:
        return pd.DataFrame()

    # Retain Author and Title identifiers
    id_df = filtered_df[['author', 'title']].copy()
    id_df['author'] = id_df['author'].fillna('')
    id_df['title'] = id_df['title'].fillna('')

    cleaned_texts = []
    metadata_list = []
    raw_texts = []

    for _, row in filtered_df.iterrows():
        raw_title = str(row.get('title', ''))
        raw_body = str(row.get('body', ''))
        raw_full = f"{raw_title}. {raw_body}".strip()
        raw_texts.append(raw_full)
        cleaned_texts.append(clean_post_text(raw_full))
        
        meta_str = str(row.get('json_metadata', '{}'))
        try:
            meta = json.loads(meta_str)
        except Exception:
            meta = {}
        metadata_list.append(meta)

    # 2. Batch inferences
    opinion_features = extractors.extract_opinions(cleaned_texts)
    
    if include_bert:
        bert_embeddings = extractors.extract_bert_embeddings(cleaned_texts)  # Shape: (N, 768)

    # 3. Tabular Features Extraction
    rows = []
    for i, clean_text in enumerate(cleaned_texts):
        raw_full = raw_texts[i]
        meta = metadata_list[i]
        opin = opinion_features[i]

        words = word_tokenize(clean_text) if clean_text else []
        sentences = sent_tokenize(clean_text) if clean_text else []
        num_words = len(words)
        num_sentences = len(sentences)
        text_len = len(clean_text)
        syllables_total = textstat.syllable_count(clean_text) if clean_text else 0

        # Stylistic (1-22)
        n_spaces = clean_text.count(' ')
        characters_word = (sum(len(w) for w in words) / num_words) if num_words > 0 else 0.0
        n_punctuation = sum(1 for c in clean_text if c in string.punctuation)
        
        cap_valid = 0
        if num_sentences > 0:
            valid_caps = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
            cap_valid = 1 if (valid_caps / num_sentences) >= 0.8 else 0

        # Readability (23-24)
        gunning_fog = textstat.gunning_fog(clean_text) if clean_text and num_words > 5 else 0.0
        flesch = textstat.flesch_reading_ease(clean_text) if clean_text and num_words > 5 else 0.0

        # Grammar (25-30)
        pos_tags = pos_tag(words) if words else []
        n_adverbs = sum(1 for _, tag in pos_tags if tag.startswith('RB'))
        n_verbs = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
        n_adjectives = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
        n_pronouns = sum(1 for _, tag in pos_tags if tag in ['PRP', 'PRP$', 'WP', 'WP$'])
        n_nouns = sum(1 for _, tag in pos_tags if tag.startswith('NN'))

        # Metadata & Categories (40-74)
        tags = [str(t).lower() for t in meta.get('tags', [])] if isinstance(meta.get('tags'), list) else []
        images = meta.get('image', []) if isinstance(meta.get('image'), list) else []
        links = meta.get('links', []) if isinstance(meta.get('links'), list) else []
        
        n_img = len(images) if images else len(re.findall(r'!\[.*?\]\(.*?\)', raw_full))
        n_link = len(links) if links else len(re.findall(r'https?://\S+', raw_full))
        
        cat_features = {f"cat_{cat}": 0 for cat in CATEGORIES_33}
        for tag in tags:
            if f"cat_{tag}" in cat_features:
                cat_features[f"cat_{tag}"] = 1

        feature_dict = {
            # Stylistic (1-22)
            "length": text_len,
            "words": num_words,
            "sentences": num_sentences,
            "syllables_count": syllables_total,
            "word_sentence": (num_words / num_sentences) if num_sentences > 0 else 0.0,
            "syllables_word": (syllables_total / num_words) if num_words > 0 else 0.0,
            "characters_word": characters_word,
            "unique_word": len(set([w.lower() for w in words])),
            "n_spaces": n_spaces,
            "n_consonants": len(re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]', clean_text)),
            "n_numbers": len(re.findall(r'\d+', clean_text)),
            "contains_?": 1 if '?' in raw_full else 0,
            "contains_!": 1 if '!' in raw_full else 0,
            "multiple_!": 1 if raw_full.count('!') > 1 else 0,
            "multiple_?": 1 if raw_full.count('?') > 1 else 0,
            "multiple_...": 1 if re.search(r'\.{4,}', raw_full) else 0,
            "spaces_length": (n_spaces / text_len) if text_len > 0 else 0.0,
            "avg_l_words": characters_word,
            "avg_l_sentences": (text_len / num_sentences) if num_sentences > 0 else 0.0,
            "n_punctuation": n_punctuation,
            "punctuation_length": (n_punctuation / text_len) if text_len > 0 else 0.0,
            "correct_capitalization": cap_valid,
            
            # Readability (23-24)
            "gunning_fog": gunning_fog,
            "flesch": flesch,
            
            # Grammar (25-30)
            "n_adverbs": n_adverbs,
            "n_verbs": n_verbs,
            "n_adjectives": n_adjectives,
            "n_pronouns": n_pronouns,
            "n_nouns": n_nouns,
            "adjectives_adverbs": n_adjectives + n_adverbs,
            
            # Opinion-based (31-39)
            "positive": opin["positive"],
            "neutral": opin["neutral"],
            "negative": opin["negative"],
            "anger": opin["anger"],
            "joy": opin["joy"],
            "sadness": opin["sadness"],
            "optimism": opin["optimism"],
            "offensive": opin["offensive"],
            "not_offensive": opin["not_offensive"],
            
            # Metadata (40-41)
            "n_img": n_img,
            "n_link": n_link,
            
            # Categories (42-74)
            **cat_features
        }
        rows.append(feature_dict)

    tabular_df = pd.DataFrame(rows)

    # 4. Assemble: [author, title] + [74 features] + [768 BERT embeddings]
    if include_bert:
        bert_col_names = [f"bert_dim_{k}" for k in range(768)]
        bert_df = pd.DataFrame(bert_embeddings, columns=bert_col_names)
        combined_df = pd.concat([id_df, tabular_df, bert_df], axis=1)
    else:
        combined_df = pd.concat([id_df, tabular_df], axis=1)
        
    return combined_df


# ==========================================
# 5. STREAMING CHUNK PIPELINE
# ==========================================
def convert_dataset_to_features(
    input_csv_path: str, 
    output_csv_path: str, 
    chunk_size: int = 1000, 
    batch_size: int = 64, 
    include_bert: bool = True
):
    """Streams a large CSV file in chunks, applies ASCII and English filtering,

    retains author & title, extracts the feature vector, and saves to CSV.
    """
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)

    extractors = BatchFeatureExtractors(batch_size=batch_size)
    
    first_chunk = True
    total_processed = 0
    total_retained = 0

    print(f"Reading {input_csv_path} in chunks of {chunk_size}...")
    reader = pd.read_csv(input_csv_path, chunksize=chunk_size, on_bad_lines='skip')

    for chunk_idx, chunk in enumerate(tqdm(reader, desc="Processing Chunks")):
        total_processed += len(chunk)
        
        features_chunk = process_chunk(chunk, extractors, include_bert=include_bert)
        
        if features_chunk.empty:
            continue
            
        total_retained += len(features_chunk)
        
        # Append chunk to output file
        features_chunk.to_csv(
            output_csv_path, 
            mode='w' if first_chunk else 'a', 
            index=False, 
            header=first_chunk
        )
        first_chunk = False

    expected_cols = 2 + (74 + 768 if include_bert else 74)
    print("\n" + "="*45)
    print(f"Processing Complete!")
    print(f"Total raw posts processed : {total_processed}")
    print(f"Valid English ASCII posts retained: {total_retained}")
    print(f"Total Columns Saved       : {expected_cols} (author, title, + {expected_cols - 2} features)")
    print(f"Output written to         : {output_csv_path}")
    print("="*45)


# ==========================================
# 6. EXECUTION
# ==========================================
if __name__ == "__main__":
    INPUT_FILE = "src/cleaned.csv"
    OUTPUT_FILE = "src/mancino_features_with_ids.csv"
    
    convert_dataset_to_features(
        input_csv_path=INPUT_FILE,
        output_csv_path=OUTPUT_FILE,
        chunk_size=1000,
        batch_size=64,
        include_bert=True  # 844 columns total (author, title, 74 tabular, 768 BERT)
    )