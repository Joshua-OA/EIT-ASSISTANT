import csv
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key=api_key)
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key,
                model_name="text-embedding-3-small",
            )
        elif model_type == "chroma":
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        elif model_type == "nomic":
            #using ollama nomic-embed-text model
            self.embedding_fn = embedding_functions.OllamaEmbeddingFunction(
                api_key="ollama",
                api_base="http://localhost:11434",
                model_name="nomic-embed-text"
            )

class LLMModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI( api_key=api_key )
            self.model_name = "gpt-4o-mini"
        else:
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self.model_name = "llama3.2"

    def generate_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0, #0.0 is deterministic.
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating completion: {e}")
            return None
        
    def select_models(self):
        # select LLM Model
        print("\nSelect a model:")
        print("1. gpt-4o-mini")
        print("2. Ollama")

        while True:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice in ["1", "2"]:
                llm_type = "openai" if choice == "1" else "ollama"
                break
            print("Please enter either 1 or 2.")

        # select embedding model
        print("\nSelect an embedding model:")
        print("1. OpenAI text-embedding-3-small")
        print("2. Chroma default embedding")
        print("3. Ollama nomic-embed-text")

        while True:
            choice = input("Enter choice (1, 2 or 3): ").strip()
            if choice in ["1", "2", "3"]:
                embedding_type = "openai" if choice == "1" else "chroma" if choice == "2" else "nomic"
                break
            print("Please enter either 1, 2 or 3.")

        return llm_type, embedding_type
    
    def generate_csv():

        facts = [
        {"id": 1, "fact": "The first human to orbit Earth was Yuri Gagarin in 1961."},
        {
            "id": 2,
            "fact": "The Apollo 11 mission landed the first humans on the Moon in 1969.",
        },
        {
            "id": 3,
            "fact": "The Hubble Space Telescope was launched in 1990 and has provided stunning images of the universe.",
        },
        {
            "id": 4,
            "fact": "Mars is the most explored planet in the solar system, with multiple rovers sent by NASA.",
        },
        {
            "id": 5,
            "fact": "The International Space Station (ISS) has been continuously occupied since November 2000.",
        },
        {
            "id": 6,
            "fact": "Voyager 1 is the farthest human-made object from Earth, launched in 1977.",
        },
        {
            "id": 7,
            "fact": "SpaceX, founded by Elon Musk, is the first private company to send humans to orbit.",
        },
        {
            "id": 8,
            "fact": "The James Webb Space Telescope, launched in 2021, is the successor to the Hubble Telescope.",
        },
        {"id": 9, "fact": "The Milky Way galaxy contains over 100 billion stars."},
        {
            "id": 10,
            "fact": "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
        },
    ]
