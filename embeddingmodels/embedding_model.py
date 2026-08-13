from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    dimension= 64
)

vector = embeddings.embed_query("What is data science")

'''
text = [
    "What is data science?",
    "What is the difference between data science and machine learning?",
    "What is the difference between data science and artificial intelligence?",
    "What is the difference between data science and deep learning?"
]

vector = embeddings.embed_documents(text)
'''

print(vector)