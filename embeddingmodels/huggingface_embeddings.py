from dotenv import load_dotenv
load_dotenv()

from huggingface_embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="hkunlp/instructor-small"
)

vector = embeddings.embed_query("What is data science")

print(vector)