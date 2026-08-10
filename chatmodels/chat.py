from dotenv import load_dotenv
load_dotenv()

'''
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="gemini-3.5-flash",
    model_provider="google_genai"
)


from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    model_provider="google_genai")
'''
'''
from langchain_groq import ChatGroq
model = ChatGroq(
    model="openai/gpt-oss-120b",
)
'''


from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(
    model="mistral-small-2603")

response = model.invoke("What is machine learning?, give answer in 1 paragraph")

print(response.content)