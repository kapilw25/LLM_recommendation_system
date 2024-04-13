import pandas as pd
import os
import json
import boto3
import streamlit as st
import warnings
import asyncio

from typing import List
from langchain.vectorstores import Docstore  # If needed

# Import necessary components from langchain
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 

# Bedrock clients
bedrock = boto3.client(service_name='bedrock-runtime')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Grocery_and_Gourmet_Food_path = "/Users/kapilwanaskar/Downloads/AWS_BEDROCK/recommendation_system/dataset/meta_Grocery_and_Gourmet_Food.jsonl"
Grocery_and_Gourmet_Food_path = "recommendation_system/dataset/meta_Grocery_and_Gourmet_Food.jsonl"

def load_and_preprocess_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    df = pd.DataFrame(data)
    
    # Filter the dataframe to include only specific columns
    df = df[['title', 'features', 'description', 'categories']]
    
    # Check each column for list data type and convert to single string if necessary
    for column in df.columns:
        if df[column].apply(type).eq(list).all():  # Check if every cell in the column is a list
            df[column] = df[column].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    
    # Convert each column to string format, necessary if any column wasn't a list and hence not converted above
    df = df.astype(str)
    
    # Concatenate the filtered and converted columns for a comprehensive text dataset for embedding
    df['text'] = df.apply(lambda x: ' '.join(x), axis=1)
    
    return df



async def generate_embeddings(reviews: pd.DataFrame) -> List[List[float]]:
    # Asynchronously generate embeddings for the list of review texts
    embeddings = await bedrock_embeddings.aembed_documents(reviews['text'].tolist())
    return embeddings

async def create_faiss_vector_store(embeddings: List[List[float]], index_file: str):
    # Assuming `Docstore` and a method to initialize or load `index` is available
    # Example initialization, adjust based on actual usage and requirements
    docstore = Docstore()  # Initialize as per actual requirement
    index = None  # Initialize or load an index as per actual FAISS usage
    
    # Creating the FAISS vector store with embeddings
    faiss_store = FAISS(
        embedding_function=bedrock_embeddings.embed_documents,  # This might need adjustment
        index=index,
        docstore=docstore,
        index_to_docstore_id={},  # Adjust as necessary
        normalize_L2=False  # Based on your normalization preference
    )
    
    # Populate FAISS store with embeddings
    # Note: This step depends on how you plan to use embeddings to populate the FAISS store.
    # The following is a placeholder for the concept:
    for embedding in embeddings:
        faiss_store.add_embeddings([embedding])  # Adjust based on actual method signatures

    # Save the FAISS store locally
    faiss_store.save_local(index_file)  # Adjust according to actual method to save the FAISS store

def get_vector_store(reviews: pd.DataFrame, index_file: str):
    # Generate embeddings and create FAISS vector store asynchronously
    embeddings = asyncio.run(generate_embeddings(reviews))
    asyncio.run(create_faiss_vector_store(embeddings, index_file))




def get_llama2_llm():
    return Bedrock(
        model_id="meta.llama2-70b-chat-v1",
        client=bedrock,
        model_kwargs={'max_gen_len': 512} 
    )

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but usse atleast summarize with 250 words with detailed explaantions. If you don't know the answer, just say that you don't know, don't try to make up an answer.

<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template = prompt_template,
    input_variables = ["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, interest):
    # Assuming the 'interest' is the user's query about what they're looking for
    # In a real application, 'context' would likely come from retrieved documents or specific information about the query domain
    context = "User is exploring various grocery products to find items that match their interest."
    question = interest[:2048]  # Ensuring the interest query does not exceed token limits for the question part

    # Prepare the input for the LLM based on the updated prompt template
    input_for_llm = {
        "context": context,
        "question": question
    }

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="recommendation",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    # Use the structured input in the RetrievalQA chain
    recommendations = qa(input_for_llm)
    return recommendations['result']


def setup_vector_store(dry_run=True):
    index_file = "dry_run_index.faiss" if dry_run else "complete_index.faiss"
    
    if not os.path.exists(index_file):
        print("Loading and Preprocessing Data...")
        reviews = load_and_preprocess_data(Grocery_and_Gourmet_Food_path)
        
        if dry_run:
            # Limit to first 1000 rows for a dry run
            reviews = reviews.head(1000)
        
        print(f"Creating Vector Store ({'Dry Run' if dry_run else 'Complete'})...")
        get_vector_store(reviews, index_file)  # Note: get_vector_store function needs to be adjusted to accept index_file as a parameter
        print(f"Vector Store ({'Dry Run' if dry_run else 'Complete'}) is set up and ready to use!")
    else:
        warnings.warn(f"Vector Store ({'Dry Run' if dry_run else 'Complete'}) already exists!")

        
def main():
    st.set_page_config("Amazon Grocery Product Recommendations")
    st.header("Find Grocery Products 🛒")

    # Toggle for Dry Run Mode
    dry_run_mode = st.checkbox("Dry Run Mode", value=True)

    user_interest = st.text_input("What type of grocery products are you interested in?")
    
    if user_interest and st.button("Get Recommendations"):
        # Check and setup vector store if not already done, based on the dry run mode
        setup_vector_store(dry_run=dry_run_mode)

        with st.spinner("Finding Recommendations..."):
            # Load the vector store, adjusting filename based on dry run mode
            index_file = "dry_run_index.faiss" if dry_run_mode else "complete_index.faiss"
            
            try:
                faiss_index = FAISS.load_local(index_file, bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_llama2_llm()
                
                recommendations = get_response_llm(llm, faiss_index, user_interest)
                if recommendations:
                    st.write(recommendations)
                else:
                    st.write("No recommendations found.")
                st.success("Recommendations Generated")
            except Exception as e:
                st.error(f"Failed to load vector store ({'Dry Run' if dry_run_mode else 'Complete'}). Make sure the vector store is properly set up.")
                st.error(str(e))

if __name__ == "__main__":
    main()
