from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

model = ChatMistralAI(model="mistral-small-2603")

print("Choose a mode:")
print("Press 1 for Angry Bot")
print("Press 2 for Funny Bot")
print("Press 3 for Sad Bot")

choice = input("Enter your choice (1, 2, or 3): ")

if choice == "1":
    mode = "You are an angry bot. Respond to the user in a very angry and aggressive manner."
elif choice == "2":
    mode = "You are a funny bot. Respond to the user with humor and wit."
elif choice == "3":
    mode = "You are a sad bot. Respond to the user in a very sad and melancholic manner."

message = [
    SystemMessage(content=mode),

]

print("------- Ask me anything (type 'exit' to quit) -------")
while True:
    
    prompt = input("You : ")
    message.append(HumanMessage(content=prompt))
    if prompt == "exit":
        break
    response = model.invoke(message)
    message.append(AIMessage(content=response.content))
    print("Bot : " + response.content)

print(message)
















'''
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model="mistral-small-2603")

message = []

print("------- Ask me anything (type 'exit' to quit) -------")
while True:
    
    prompt = input("You : ")
    message.append(prompt)
    if prompt == "exit":
        break
    response = model.invoke(message)
    message.append(response.content)

    print("Bot : " + response.content)

# print(message)
'''