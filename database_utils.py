import pickle
import uuid
from typing import List, Union, Dict
import pandas as pd
import yake
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance


def split_texts_for_db(text: str) -> List[str]:
    """
    Splits the input text into chunks for database storage.

    Args:
        text (str): The text to be split.

    Returns:
        List[str]: A list of text chunks.
    """
    c_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return c_splitter.split_text(text)


def calculate_embedding(input_data: Union[str, List[str]]) -> Union[List[float], float]:
    """
    Calculates embeddings for the input data.

    Args:
        input_data (Union[str, List[str]]): A string or a list of strings to embed.

    Returns:
        Union[List[float], float]: The embedding(s) for the input data.
    
    Raises:
        ValueError: If input_data is neither a string nor a list of strings.
    """
    embeddings = FastEmbedEmbeddings()

    if isinstance(input_data, str):
        return embeddings.embed_documents(input_data)[0]
    elif isinstance(input_data, list):
        return [embeddings.embed_documents(sentence)[0] for sentence in input_data]
    else:
        raise ValueError("Input must be a string or a list of strings")


def create_collection_if_not_exists(client: QdrantClient, collection_name: str) -> None:
    """
    Creates a Qdrant collection if it does not already exist.

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The name of the collection to create.
    """
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )


def add_article_to_qdrant(row: Dict, qdrant_key: str, qdrant_url: str, collection_name: str = "bbc_news_articles") -> None:
    """
    Adds a single article to the Qdrant database.

    Args:
        row (Dict): A dictionary containing article data.
        qdrant_key (str): The API key for Qdrant.
        qdrant_url (str): The URL for the Qdrant instance.
        collection_name (str): The name of the collection to add the article to.
    """
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    create_collection_if_not_exists(client, collection_name)

    metadata = {'label': row['labels'], 'keywords': row['keywords'], 'source': row['data_source']}
    for embedding, text in zip(row['embeddings'], row['splitted_text']):
        vector = embedding
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "page_content": text,
                        **metadata
                    }
                )
            ]
        )


def add_articles_to_qdrant(df: pd.DataFrame, qdrant_key: str, qdrant_url: str, collection_name: str = "bbc_news_articles") -> None:
    """
    Adds multiple articles to the Qdrant database.

    Args:
        df (pd.DataFrame): A DataFrame containing article data.
        qdrant_key (str): The API key for Qdrant.
        qdrant_url (str): The URL for the Qdrant instance.
        collection_name (str): The name of the collection to add the articles to.
    """
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    create_collection_if_not_exists(client, collection_name)

    for _, row in df.iterrows():
        metadata = {'label': row['labels'], 'keywords': row['keywords'], 'source': row['data_source']}
        for embedding, text in zip(row['embeddings'], row['splitted_text']):
            vector = embedding
            client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "page_content": text,
                            **metadata
                        }
                    )
                ]
            )


def assign_keywords(text: str) -> List[str]:
    """
    Extracts keywords from the input text using YAKE.

    Args:
        text (str): The text to extract keywords from.

    Returns:
        List[str]: A list of extracted keywords.
    """
    language = 'en'
    ngram_size = 2
    num_of_keywords = 5
    deduplication_threshold = 0.2

    kw_extractor = yake.KeywordExtractor(lan=language, n=ngram_size, dedupLim=deduplication_threshold, top=num_of_keywords)
    keywords = kw_extractor.extract_keywords(text)
    return [keyword[0] for keyword in keywords]


def classify_text(text: str) -> str:
    """
    Classifies the input text using a pre-trained model.

    Args:
        text (str): The text to classify.

    Returns:
        str: The predicted class label.
    """
    with open("models/BBC_articles_vectorizer.pkl", 'rb') as file:
        vectorizer = pickle.load(file)
    with open("models/BBC_articles_classifier.pkl", 'rb') as file:
        classifier = pickle.load(file)
    with open("models/BBC_articles_label_encoder.pkl", 'rb') as file:
        label_encoder = pickle.load(file)

    text_vectorized = vectorizer.transform([text])
    y_pred = classifier.predict(text_vectorized)
    return label_encoder.classes_[y_pred][0]
