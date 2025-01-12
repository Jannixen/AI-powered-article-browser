# AI-powered-article-browser

The aim of the project is to create an API that allows the QA of the OpenAI LLM model with the support of a vector database search engine. The database is populated with BBC articles obtained using a few datasets (listed below), but the project is generic and can be easily duplicated with other documents. The API allows sending a question to the model and getting an answer based on related texts from the database, retrieving only related documents from the database (without LLM) and adding new articles to the database.

## Tech stack
- langchain
- Qdrant vector database
- OpenAI models
- langfuse
- Flask


## Project milestones 

### 1. Decide on the structure of the database. Create a text classification model

The first BBC News dataset was analysed. It had two summaries, category labels, Flesch Reading Ease Score and Dale-Chall Readability Score for each article. It was decided to keep only the category and additional keywords generated from the text in the database. Summaries may be useful in some cases, but here the texts are not so long that we cannot store them in their full form. Also, generating summaries will be time consuming later when adding new articles via API.

Classification model to determine one of the text category (business, entertainment, politics, sports, technology) was created using term frequency inverse document frequency vectorisation with ngram_range of (1,3) so using unigrams, bigrams and trigrams. Four classifiers (MultinomialNB, LogisticRegression, LinearSVC, RandomForestClassifier) commonly used for text classification were tested and MultinomialNB was chosen as the final classifier with an accuracy of 97.2%.

![curl-ask](https://github.com/Jannixen/AI-powered-article-browser/blob/main/images/cm.png)

### 2. Prepare Qdrant database

This step consisted of adding articles from the collected datasets to a Qdrant collection named "bbc_news_articles". When creating the collection, the Qdrant FastEmbedding model is used along with cosine as the distance function. Articles are also appended with the 5 keywords each using KeywordExtractor from Python yake library.

![curl-ask](https://github.com/Jannixen/AI-powered-article-browser/blob/main/images/drant2.png)
![curl-ask](https://github.com/Jannixen/AI-powered-article-browser/blob/main/images/qdrant1.png)

### Create conversational retrieval chain

Test database connection and retrieval. Adding memory to the conversation, comparing `map_reduce`, `refine` and `stuff` chains in RetrievalQA. All the time using the gpt-4o-mini OpenAI model.

### Building the API

The API is built using Flask and has 3 main functions
- ask/ - get LLM's answer to a question about articles
- add/ - add new articles to the database (text only) 
- retrieve/ - retrieve only documents related to questions with a specified number of documents as a limit

In this step, Langfuse integration is also added to be able to monitor query costs.

## Sample API usage

### 1. Ask question about articles

```bash
curl -s -X POST http://127.0.0.1:5000/ask -H "Content-Type: application/json" -d '{"question": "How much ipod with one gigabyte storage costs that was unveiled by Steve Jobs during annual MacWorld speech?"}'
```

![curl-ask](https://github.com/Jannixen/AI-powered-article-browser/blob/main/images/curl1.png)

### 2. Add new article

```bash
 curl-s -X POST http://127.0.0.1:5000/add -H "Content-Type: application/json" -d '{"text": "The first full-sized digital scan of the Titanic, which lies 3,800m (12,500ft) down in the Atlantic, has been created using deep-sea mapping.Read more: Scans of Titanic reveal wreck as never seen before"}'
```

![curl-add](https://github.com/Jannixen/AI-powered-article-browser/blob/main/images/curl2.png)

### 3. Retrieve articles related to question

```bash
curl -s -X POST http://127.0.0.1:5000/retrieve -H "Content-Type: application/json" -d '{"question": "How much ipod with one gigabyte storage costs that was unveiled by Steve Jobs during annual MacWorld speech?", "limit":1}'
```

![curl-retrieve](https://github.com/Jannixen/AI-powered-article-browser/blob/main/images/curl3.png)

## Usage costs

Simple QA consisting of two pairs of questions and answers costed $0.001:

```txt
{"question": "How much ipod with one gigabyte storage costs that was unveiled by Steve Jobs during annual MacWorld speech?"}'
{"answer":"The iPod shuffle with one gigabyte of storage costs $149 as unveiled by Steve Jobs during the annual MacWorld speech."}
{"question": "Is there a cheaper option?"}
{"answer":"Yes, Apple has released the iPod shuffle, which is a cheaper option compared to other iPod models. The iPod shuffle uses flash memory instead of hard drives and is available in two versions: one with 512MB of storage for $99 and a second with 1GB of storage for $149."}
```

![image](https://github.com/user-attachments/assets/6c8c9336-af8c-4304-8439-3a136119597c)


## Sources

1. https://www.kaggle.com/datasets/jacopoferretti/bbc-articles-dataset
2. https://www.kaggle.com/datasets/bhavikjikadara/bbc-news-articles
3. https://huggingface.co/datasets/SetFit/bbc-news
4. http://mlg.ucd.ie/datasets/bbc.html
5. https://learn.deeplearning.ai/courses/langchain-chat-with-your-data/lesson/4/vectorstores-and-embedding
6. https://www.bbc.com/
  
