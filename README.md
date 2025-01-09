# AI-powered-article-browser

The aim of the project is to create an API allowing the QA of OpenAI LLM model with the support of vector database search engine. The database is populated with BBC articles obtained using few datasets (listed below), but the project is generic and can be easily duplicated with other documents. API allows sending question to model and getting answer based on related texts from database, retrieving only related documents from database (without LLM) and adding new articles to database.

Tech stack:
- langchain
- Qdrant vector database
- OpenAI models
- langfuse

  

  
