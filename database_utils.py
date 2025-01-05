from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import uuid
import yake

def split_texts_for_db(text):
    c_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    return c_splitter.split_text(text)

def calculate_embedding(input_data):

    embeddings = FastEmbedEmbeddings()

    if isinstance(input_data, str):
        # Encode a single sentence
        return embeddings.embed_documents(input_data)[0]
    elif isinstance(input_data, list):
        # Encode a list of sentences
        return [embeddings.embed_documents(sentence)[0] for sentence in input_data]
    else:
        raise ValueError("Input must be a string or a list of strings")

def add_article_to_qdrant(row, qdrant_key, qdrant_url, collection_name = "bbc_news_articles"):
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    metadata = { 'label': row['labels'], 'keywords': row['keywords'], 'source': row['data_source']}
    
    for embedding, text in zip(row['embeddings'], row['splitted_text']):
        
        vector = embedding
    
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
    
        # Add data to collection
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                    "page_content": text,  # Include the text content as `page_content`
                    **metadata  # Add additional metadata
                }
                )
            ]
        )

def add_articles_to_qdrant(df, qdrant_key, qdrant_url, collection_name = "bbc_news_articles"):
    
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    

    for _, row in df.iterrows():
        metadata = {'label': row['labels'], 'keywords': row['keywords'], 'source': row['data_source']}
        
        for embedding, text in zip(row['embeddings'], row['splitted_text']):

            vector = embedding
            
            # Add data to collection
            client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                    "page_content": text,  # Include the text content as `page_content`
                    **metadata  # Add additional metadata
                }
                    )
                ]
            )

def assign_keywords(text):
    language = 'en'
    ngram_size = 2
    num_of_keywords = 5
    deduplication_threshold = 0.2
    
    kw_extractor = yake.KeywordExtractor(lan = language, n = ngram_size, dedupLim = deduplication_threshold, top = num_of_keywords)
    
    keywords = kw_extractor.extract_keywords(text)
    keywords = [keyword[0] for keyword in keywords]
    return keywords
