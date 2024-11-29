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

# Initialize the client with proper configuration
client = UnstructuredClient(
    api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
    server_url=os.getenv("UNSTRUCTURED_API_URL"),
)

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Astra DB client
astra_client = DataAPIClient(os.getenv("ASTRA_DB_TOKEN"))

# Get the database instance
db = astra_client.get_database(os.getenv("ASTRA_DB_ENDPOINT"))

# Update the collection name and creation
collection_name = "vectorized_summaries"  # New collection name
try:
    # Create collection with vector search enabled and correct dimensions
    collection = db.create_collection(
        collection_name,
        dimension=1536,  # text-embedding-3-small produces 1536-dimensional vectors
        metric="cosine"  # similarity metric for vector search
    )
    print(f"Created new collection: {collection_name}")
except Exception as e:
    # Collection might already exist
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
            }
        }

        # Process with Unstructured.io API asynchronously
        result = await client.general.partition_async(request=req)
        
        # Immediately process elements and create summaries
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

            # Prepare document with vector
            document = {
                "_id": str(uuid.uuid4()),
                "content": summary["content"],
                "file_name": summary["file_name"],
                "created_at": datetime.utcnow().isoformat(),
                "$vector": embedding_vector  # This is the special field name for vectors in Astra DB
            }
            
            collection.insert_one(document)
        
        print("Successfully saved vectorized elements to Astra DB")
        return state
    
    except Exception as e:
        print(f"Error saving to Astra DB: {str(e)}")
        state["error"] = str(e)
        return state
