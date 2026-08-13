import streamlit as st
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


# ---------------- Streamlit UI ---------------- #

st.set_page_config(page_title="Movie Info Extractor", page_icon="🎬", layout="centered")

st.title("🎬 Movie Info Extractor")
st.write("Paste a paragraph describing a movie, and this app will extract structured information from it using Mistral AI.")

para = st.text_area("Enter the movie paragraph:", height=200, placeholder="e.g. Inception is a 2010 sci-fi thriller directed by Christopher Nolan...")

if st.button("Extract Movie Info", type="primary"):
    if not para.strip():
        st.warning("Please enter a movie paragraph first.")
    else:
        with st.spinner("Extracting movie information..."):
            try:
                final_prompt = prompt.invoke({
                    "paragraph": para,
                    "format_instructions": parser.get_format_instructions()
                })

                response = model.invoke(final_prompt)

                st.subheader("Raw Model Output")
                st.code(response.content, language="json")

                # Try to parse into structured MovieInfo for a nicer display
                try:
                    movie_info = parser.parse(response.content)
                    st.subheader("Parsed Movie Info")
                    st.json(movie_info.model_dump())
                except Exception as parse_err:
                    st.info(f"Could not auto-parse structured output: {parse_err}")

            except Exception as e:
                st.error(f"Something went wrong: {e}")