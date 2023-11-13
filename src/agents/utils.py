import logging
import os
import random
import string

import torch
from openai import OpenAI
from text2vec import semantic_search


def setup_logging():
    logger = logging.getLogger('chat_logger')
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()

    if os.path.exists('chat.log'):
        os.remove('chat.log')

    file_handler = logging.FileHandler('chat.log')

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_embedding(sentence):
    client = OpenAI(api_key=os.environ.get("API_KEY"))

    text = sentence.replace("\n", " ")
    embedding = client.embeddings.create(
        input=[text], model="text-embedding-ada-002").data[0].embedding
    embedding = torch.tensor(embedding, dtype=torch.float32)

    if len(embedding.shape) == 1:
        embedding = embedding.unsqueeze(0)
    return embedding


def get_code():
    return "".join(random.sample(string.ascii_letters + string.digits, 8))


def get_content_between_a_b(start_tag, end_tag, text):
    extracted_text = ""
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag):end_index] + " "
            extracted_text += text[start_index + len(start_tag):end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break

    return extracted_text.strip()


def extract(text, type):
    target_str = get_content_between_a_b(f"<{type}>", f"</{type}>", text)
    return target_str


def get_relevant_history(query, history, embeddings):
    try:
        top_k = int(os.getenv("TOP_K", 0))
    except ValueError:
        raise ValueError("Environment variable TOP_K must be an integer.")

    relevant_history = []
    query_embedding = get_embedding(query)

    top_k = min(top_k, len(embeddings))

    try:
        hits = semantic_search(query_embedding, embeddings, top_k=top_k)[0]
    except IndexError:
        return []

    for hit in hits:
        matching_idx = hit.get("corpus_id")
        if matching_idx is not None and 0 <= matching_idx < len(history):
            relevant_history.append(history[matching_idx])

    return relevant_history
