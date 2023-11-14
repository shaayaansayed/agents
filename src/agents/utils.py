import logging
import os
import random
import string

import faiss
import torch
from langchain.docstore import InMemoryDocstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS
from openai import OpenAI
from text2vec import semantic_search

VECTOR_STORE_EMBEDDING_SIZE = 1536


def setup_logging(debug=False):
    logger = logging.getLogger('chat_logger')
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()

    if os.path.exists('chat.log'):
        os.remove('chat.log')

    file_handler = logging.FileHandler('chat.log')

    if debug:
        console_handler.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def init_memory_vector_store(top_k):
    index = faiss.IndexFlatL2(VECTOR_STORE_EMBEDDING_SIZE)
    embedding_fn = OpenAIEmbeddings().embed_query
    vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
    retriever = vectorstore.as_retriever(search_kwargs=dict(k=top_k))
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    return vectorstore


def append_memory_buffer(memory, user_message=None, ai_message=None):
    if user_message and not ai_message:
        memory.chat_memory.add_user_message(user_message)
    elif ai_message and not user_message:
        memory.chat_memory.add_ai_message(ai_message)
    else:
        raise ValueError(
            "Exactly one of user_message or ai_message must be provided, but not both."
        )

    memory.prune()


def extract_formatted_chat_messages(memory):
    formatted_messages = []
    for message in memory.buffer:
        formatted_messages.append(message.content)

    return '\n'.join(formatted_messages)


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
    if len(embeddings) == 0:
        return []

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
