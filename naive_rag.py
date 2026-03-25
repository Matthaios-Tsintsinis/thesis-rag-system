import os
import ollama
import chromadb
import PyPDF2

def get_text_chunks_from_pdfs(folder_path, chunk_size=200):
    """Reads all PDFs in a folder and splits them into word chunks."""
    chunks = []
    
    if not os.path.exists(folder_path):
        print(f"Directory '{folder_path}' not found. Please create it and add PDFs.")
        return chunks

    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            filepath = os.path.join(folder_path, filename)
            print(f"   Reading: {filename}")
            
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                pdf_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pdf_text += text + " "
                
                words = pdf_text.split()
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i + chunk_size])
                    chunks.append(chunk)
                    
    return chunks

def main():
    print("--- Starting Dynamic Naive RAG Pipeline ---\n")

    # ==========================================
    # STAGE 1: INDEXING (Happens only ONCE)
    # ==========================================
    print("1. Indexing Stage: Reading PDFs and embedding documents...")
    
    client = chromadb.Client()
    try:
        client.delete_collection(name="naive-rag-docs")
    except Exception:
        pass
        
    collection = client.create_collection(name="naive-rag-docs")

    pdf_folder = "data"
    documents = get_text_chunks_from_pdfs(pdf_folder, chunk_size=200)

    if not documents:
        print("No documents found. Please add PDFs to the 'data' folder and run again.")
        return

    print(f"   Created {len(documents)} document chunks. Embedding them now...")

    for i, chunk in enumerate(documents):
        response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
        embedding = response["embedding"]
        
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk]
        )
    print("   Indexing complete.\n")

    # ==========================================
    # INTERACTIVE CHAT LOOP
    # ==========================================
    print("========================================================")
    print("Ready! You can now ask questions about your documents.")
    print("Type 'exit' or 'quit' to stop the program.")
    print("========================================================\n")

    while True:
        # Get the user's input dynamically
        user_query = input("Your Question: ")
        
        # Give the user a way to break the loop and exit
        if user_query.lower() in ['exit', 'quit']:
            print("Shutting down. Goodbye!")
            break
            
        # Ignore empty inputs
        if not user_query.strip():
            continue

        # ==========================================
        # STAGE 2: RETRIEVAL (Happens for EVERY question)
        # ==========================================
        query_response = ollama.embeddings(model="nomic-embed-text", prompt=user_query)
        query_embedding = query_response["embedding"]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )

        retrieved_context = "\n".join(results['documents'][0])

        # ==========================================
        # STAGE 3: GENERATION (Happens for EVERY question)
        # ==========================================
        prompt = f"""
        Answer the user's question using ONLY the provided context below. If you cannot answer based on the context, state that you do not have enough information.

        Context:
        {retrieved_context}

        Question:
        {user_query}
        """

        print("Thinking...")
        response = ollama.chat(model='llama3', messages=[
          {
            'role': 'user',
            'content': prompt,
          },
        ])

        print("\nAnswer:")
        print(response['message']['content'])
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()