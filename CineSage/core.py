from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

class MovieInfo(BaseModel):
    title: str = Field(..., description="The title of the movie")
    release_year: Optional[int] = Field(None, description="The release year of the movie")
    genre: Optional[str] = Field(None, description="The genre of the movie")
    main_characters: Optional[List[str]] = Field(None, description="List of main characters in the movie")
    director: Optional[str] = Field(None, description="The director of the movie")
    themes_or_message: Optional[str] = Field(None, description="The themes or message conveyed by the movie")
    rating: Optional[float] = Field(None, description="The rating of the movie (e.g., PG, R)")

parser = PydanticOutputParser(pydantic_object=MovieInfo)

model = ChatMistralAI(model="mistral-small-2603")


'''
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert movie analyst.

Your task is to read the following movie paragraph and extract all the important information.

Movie Paragraph:
{paragraph}

Instructions:
- Summarize the paragraph in 5-8 concise bullet points.
- Keep only the important information.
- Ignore unnecessary descriptions and repeated details.
- Mention:
  - Movie title (if available)
  - Genre
  - Main characters
  - Setting (time/place)
  - Plot overview
  - Main conflict
  - Important events
  - Ending (if mentioned)
  - Themes or message

Output:
"""),
(
    'human',
    """
Extract information from this paragraph:
{paragraph}
"""
)
]
)
'''

prompt = ChatPromptTemplate.from_messages([
    ("system", """ 
        Extract movie information from the following paragraph {format_instructions}"""),
        (
            'human',
            """
            {paragraph}
            """
        )
    ]
    )

para = input("Enter the movie paragraph: ")

final_prompt = prompt.invoke({
    "paragraph": para,
    "format_instructions": parser.get_format_instructions()
})

response = model.invoke(final_prompt)
movie_data = parser.parse(response.content)

print(movie_data)