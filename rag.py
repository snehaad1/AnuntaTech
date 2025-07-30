import nltk
import fitz  # PyMuPDF for PDF extraction
from nltk.tokenize import sent_tokenize
import openai
import numpy as np
import faiss
import os

# Initialize OpenAI client
from dotenv import load_dotenv
import os
import openai

# Load environment variables from .env
load_dotenv()

# Get the key from the environment
api_key = os.getenv("OPENAI_API_KEY")

# Use the key with OpenAI
openai.api_key = api_key

# Ensure the sentence tokenizer is available
nltk.download("punkt")

# -------------------------------
# Function: Extract text from PDF
# -------------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# -------------------------------
# Function: Chunk text by sentence and token count
# -------------------------------
def chunk_text(text, max_tokens=300, overlap=50):
    # Tokenize the input text into sentences using NLTK's sentence tokenizer
    sentences = sent_tokenize(text)
    
    # Initialize variables to store the chunks and the current chunk being built
    chunks = []  # List to store all the chunks of text
    current_chunk = []  # The current chunk that is being built
    current_length = 0  # The current length (in tokens) of the current_chunk

    # Loop through each sentence in the tokenized text
    for sentence in sentences:
        # Split the sentence into tokens (words) using a simple space separator
        tokens = sentence.split()
        token_len = len(tokens)  # Number of tokens in the current sentence

        # If adding the current sentence to the chunk does not exceed the max tokens allowed
        if current_length + token_len <= max_tokens:
            # Add the tokens to the current chunk
            current_chunk.extend(tokens)
            # Update the current length of the chunk
            current_length += token_len
        else:
            # If the chunk has reached the maximum token limit, store it in the chunks list
            chunks.append(" ".join(current_chunk))

            # If 'overlap' is greater than 0, keep the last 'overlap' number of tokens from the previous chunk
            if overlap > 0:
                # Take the last 'overlap' number of tokens from the current_chunk
                overlap_tokens = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
                # Start a new chunk with these overlapping tokens
                current_chunk = list(overlap_tokens)
                current_length = len(current_chunk)  # Update the length of the new chunk
            else:
                # If no overlap is required, reset the current_chunk
                current_chunk = []
                current_length = 0

            # Add the tokens from the current sentence to the new chunk
            current_chunk.extend(tokens)
            # Update the current length with the length of the new sentence added
            current_length += token_len

    # After all sentences have been processed, if there's any leftover text in current_chunk, add it as a final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# -------------------------------
# Function: Embed and Save to JSON
# -------------------------------
def embed_and_save_json(text, json_path="embeddings.json"):
    chunks = chunk_text(text)

    data = []  # List to hold embeddings and text data

    for chunk in chunks:
        # Embed the chunk using OpenAI's embedding API
        response = openai.Embedding.create(
            input=chunk,
            model="text-embedding-ada-002"  # You can use other embedding models
        )
        embedding = response['data'][0]['embedding']  # Get the embedding

        # Store the chunk text and embedding
        data.append({
            "text": chunk,
            "embedding": embedding
        })

    # Save the data (chunks and embeddings) into a JSON file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"Saved {len(data)} chunks and embeddings to {json_path}")

# -------------------------------
# Function: Build the FAISS index from embeddings
# -------------------------------
def build_faiss_index(embeddings):
    dimension = len(embeddings[0])  # The embedding dimension
    index = faiss.IndexFlatL2(dimension)  # Initialize the FAISS index
    index.add(np.array(embeddings))  # Add the embeddings to the index
    return index

# -------------------------------
# Function: Query the system and return the most relevant chunk
# -------------------------------
def query_system(user_query, index, chunks, top_k=3):
    # Embed the user query
    query_embed = openai.Embedding.create(
        input=[user_query],
        model="text-embedding-ada-002"
    )["data"][0]["embedding"]

    query_embed = np.array(query_embed).reshape(1, -1).astype("float32")

    # Search the FAISS index for the most similar chunks
    D, I = index.search(query_embed, top_k)
    relevant_chunks = [chunks[i] for i in I[0]]  # Get the corresponding chunks

    # Combine the relevant chunks into a single context
    combined_context = " ".join(relevant_chunks)

    return combined_context

# -------------------------------
# Main pipeline to process the script and query the system
# -------------------------------
if __name__ == "__main__":
    # Step 1: Extract the text from the Superbad PDF
    pdf_path = "superbad_script.pdf"  # Replace with your PDF file
    script_text = extract_text_from_pdf(pdf_path)

    # Step 2: Embed the script and save embeddings to JSON
    embed_and_save_json(script_text, json_path="superbad_embeddings.json")

    # Step 3: Load the embeddings and chunks from JSON
    with open("superbad_embeddings.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = [item["text"] for item in data]  # Extract chunks
    embeddings = [item["embedding"] for item in data]  # Extract embeddings

    # Step 4: Build the FAISS index
    index = build_faiss_index(embeddings)

    # Step 5: Interact with the system
    while True:
        user_query = input("\nAsk a question (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        # Query the system to get the most relevant chunks
        relevant_context = query_system(user_query, index, chunks)

        # Step 6: Use GPT to generate the response based on the context
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Or use gpt-3.5 or any other model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the Superbad script."},
                {"role": "user", "content": f"Here is the script context:\n\n{relevant_context}\n\nQuestion: {user_query}"}
            ],
            max_tokens=300,
            temperature=0.7,
        )

        # Get and print the answer from GPT
        answer = response.choices[0].message['content'].strip()
        print(f"\nAnswer: {answer}") 
