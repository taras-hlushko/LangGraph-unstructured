import aiohttp
import os
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from typing import Annotated, List, Dict, Any
from openai import AsyncOpenAI
from astrapy import DataAPIClient
from datetime import datetime
import uuid

from my_agent.utils.state import AgentState

client = UnstructuredClient(
    api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
    server_url=os.getenv("UNSTRUCTURED_API_URL"),
)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Astra DB client
astra_client = DataAPIClient(os.getenv("ASTRA_DB_TOKEN"))

# Get the database instance
db = astra_client.get_database(os.getenv("ASTRA_DB_ENDPOINT"))

collection_name = "vectorized_summaries" 
try:
    # Create collection with vector search enabled and correct dimensions
    collection = db.create_collection(
        collection_name,
        dimension=1536,  # text-embedding-3-small produces 1536-dimensional vectors
        metric="cosine"  # similarity metric for vector search
    )
    print(f"Created new collection: {collection_name}")
except Exception as e:
    collection = db.get_collection(collection_name)
    print(f"Using existing collection: {collection_name}")

async def fetch_and_process(state: Annotated[AgentState, "state"]) -> AgentState:
    """Fetch content from URL and process with Unstructured.io API"""
    try:
        # Fetch the file from URL
        async with aiohttp.ClientSession() as session:
            async with session.get(state["url"]) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch URL: {response.status}")
                content = await response.read()

        # Prepare the request for Unstructured.io API
        req = {
            "partition_parameters": {
                "files": {
                    "content": content,
                    "file_name": state["url"].split("/")[-1],  # Extract filename from URL
                },
                "strategy": shared.Strategy.HI_RES,
                "chunk_strategy": "by_title",
                "max_characters": 2000,              
                "multipage_sections": True,            
                "combine_text_under_n_chars": 500, 
            }
        }

        # Process with Unstructured.io API asynchronously
        result = await client.general.partition_async(request=req)
        
        # Process elements and create summaries
        summarized_elements = []
        for elem in result.elements:
            if isinstance(elem, dict) and "text" in elem:
                response = await openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Please provide a concise summary of the following text:"},
                        {"role": "user", "content": elem["text"]}
                    ],
                    max_tokens=300
                )
                
                summarized_element = {
                    "content": response.choices[0].message.content,
                    "file_name": elem.get("metadata", {}).get("filename", "unknown")
                }
                summarized_elements.append(summarized_element)
        
        # Return only url and summarized_elements
        return {"url": state["url"], "summarized_elements": summarized_elements}
    
    except Exception as e:
        return {"url": state["url"], "error": str(e)}

async def summarize_elements(state: Annotated[AgentState, "state"]) -> AgentState:
    """Summarize each element individually using OpenAI"""
    if "error" in state:
        return state
    
    try:
        summarized_elements = []
        print("!!!!!!Saving elements:", state["summarized_elements"])
        for elem in state["elements"]:
            if isinstance(elem, dict) and "text" in elem:
                response = await openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Please provide a concise summary of the following text:"},
                        {"role": "user", "content": elem["text"]}
                    ],
                    max_tokens=300
                )
                
                # Only include content and file_name
                summarized_element = {
                    "content": response.choices[0].message.content,
                    "file_name": elem.get("metadata", {}).get("filename", "unknown")
                }
                summarized_elements.append(summarized_element)
        
        state["summarized_elements"] = summarized_elements
        return state
        
    except Exception as e:
        state["error"] = f"Summarization error: {str(e)}"
        return state

async def display_results(state: Annotated[AgentState, "state"]) -> AgentState:
    """Display only the summarized elements"""
    if "summarized_elements" in state:
        summaries = [
            {
                "content": elem["content"],
                "file_name": elem["file_name"]
            }
            for elem in state["summarized_elements"]
        ]
        print("!!!!!!Summarized elements:", summaries)
    return state

async def save_to_astra(state: Annotated[AgentState, "state"]) -> AgentState:
    """Save vectorized summarized elements to Astra DB"""
    try:
        if "summarized_elements" not in state:
            raise ValueError("No summarized elements to save.")
        
        for summary in state["summarized_elements"]:
            # Generate embedding for the content
            embedding_response = await openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=summary["content"]
            )
            embedding_vector = embedding_response.data[0].embedding

            # Prepare document with vector and metadata
            document = {
                "_id": str(uuid.uuid4()),
                "content": summary["content"],
                "metadata": {
                    "source": state["url"]
                },
                "$vector": embedding_vector  # This is the special field name for vectors in Astra DB
            }
            print("!!!!!!Saving document:", document)
            collection.insert_one(document)
        
        print("Successfully saved vectorized elements to Astra DB")
        return state
    
    except Exception as e:
        print(f"Error saving to Astra DB: {str(e)}")
        state["error"] = str(e)
        return state

async def fetch_context_from_astra(url: str) -> str:
    """Fetch context from Astra DB based on URL"""
    try:
        # Fetch context from Astra DB based on URL
        query = {
            "metadata.source": url
        }
        
        # Get all documents matching the URL
        documents = collection.find(query)
        
        # Combine all content from matching documents to form the context
        context = ""
        if documents:
            context = "\n".join([doc.get("content", "") for doc in documents])
        
        if not context:
            raise ValueError(f"No content found in database for URL: {url}")
            
        return context

    except Exception as e:
        raise Exception(f"Error fetching context: {str(e)}")

async def generate_curriculum_topics(state: Annotated[AgentState, "state"]) -> AgentState:
    """Process curriculum generation request using OpenAI"""
    try:
        # Fetch context using the dedicated function
        context = await fetch_context_from_astra(state["url"])

        # Extract other parameters from the state
        curriculum = state.get("curriculum", "")
        main_topic = state.get("main_topic", "")
        user_input = state.get("user_input", "")

        # Construct the prompt
        prompt = f"""# Objective

You are a specialized assistant designed to help with curriculum development tasks. Your primary responsibility is to assist in expanding half-complete curriculums by suggesting additional subtopics for each specified topic, thus helping users create a more comprehensive curriculum structure.

## Task

Your main task is to suggest candidates for subtopics that can be included in an existing curriculum or that can form the basis for it. For this, you will be provided with next details from the user:
- Context: content of the document (book, article, etc.) user generates curriculum for. You must take it as ground truth and use only it while generating curriculum expansion
- USE ONLY CONTEXT PROVIDED BELOW AS A GROUND SOURCE!
- Curriculum: current state of curriculum (can be empty)
- Main topic: user can ask us to expand some specific topic from the curriculum. So, if "Main topic" is provided - your task is to generate additional subtopics specifically for this topic
- User requirements: user can provide some specification how to do task correctly (for example, how much subtopics to generate)

## Response format

Your response must be in JSON format, structured as follows:
{{
  "topics": [
    {{
      "title": "Topic Title",
      "description": "A clear description of what this topic covers"
    }}
  ]
}}

## Notes

- The suggested subtopics should be relevant and logically connected to the main topic.
- Each subtopic must include a clear description explaining its content and relevance.
- Ensure that all suggestions are clear and concise.
- Maintain consistency with the existing curriculum style and depth.
- The descriptions should be informative yet concise, typically 1-3 sentences.
- Maintain hierarchical relationships between topics and subtopics.

## Instructions

1. Ensure clarity and relevance in all suggested topics/subtopics.
2. You must take into account current user curriculum to better complete it based on given topics.

## Context (USE ONLY CONTEXT PROVIDED BELOW AS A GROUND SOURCE!):

{context}

## Current curriculum

{curriculum}

## Main Topic:

{main_topic}

## User requirements

{user_input}
"""

        response = await openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a curriculum development assistant that always responds in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        
        content = response.choices[0].message.content
        
        return {
            "url": state["url"],
            "topic_summaries": content
        }

    except Exception as e:
        return {
            "url": state["url"],
            "error": f"Error processing curriculum: {str(e)}"
        }
