from transformers import pipeline
import re

from langchain.tools import tool

import agents.docpool as docpool

# Instantiate the document retriever
document_retriever_instance = docpool.DocumentRetriever()
embedding_retriever_instance = docpool.EmbeddingBasedRetriever()

@tool
def retrieve_function(query):
    """
    Retrieves documents relevant to a given query using the document retriever instance.

    This function interfaces with a DocumentRetriever instance to find and return documents
    that are relevant to the specified query. The documents are returned in a format
    that is suitable for use by the agent, facilitating integration with the agent's
    response generation process.

    Parameters:
    - query (str): The search query for which relevant documents are to be retrieved.

    Returns:
    - list of tuples: A list where each tuple contains the path to a document and its
      relevance score with respect to the query. The format is [(document_path, score), ...].
    """
    documents = document_retriever_instance.retrieve_documents(query)
    # Optionally, format the output as needed for your agent here
    return documents


from transformers import pipeline

def summarize_text(text, model="sshleifer/distilbart-cnn-12-6", chunk_size=1024):
    summarizer = pipeline("summarization", model=model)
    # Choose either to truncate or split and summarize based on your preference
    # summary = truncate_and_summarize(text, ...)
    # or
    summary = split_and_summarize(text, model=model, chunk_size=chunk_size)
    return summary

def split_and_summarize(text, model="sshleifer/distilbart-cnn-12-6", chunk_size=1024, min_length=30, max_output_length=130):
    summarizer = pipeline("summarization", model=model)
    # Split the text into manageable parts
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    i = 0
    summaries = []
    print( f"> Nr of chunks {len( chunks)}" )
    for chunk in chunks:
        summary = summarizer(chunk, max_length=max_output_length, min_length=min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])
        i = i + 1
        print( i )

    print( "-->" )
    # Combine summaries
    combined_summary = ' '.join(summaries)
    return combined_summary



def extract_numbers(text):
    numbers = re.findall(r'\d+', text)
    return numbers

@tool
def process_documents(query):
    """
    Processes documents relevant to a given query by summarizing their content and extracting numbers.

    This function first retrieves documents semantically related to the input query. It then summarizes the content.
    It extracts any numerical information present in the text.

    Parameters:
    - query (str): The query string used to find relevant documents.

    Returns:
    - tuple of lists: A tuple containing two lists. The first list contains summaries of each relevant document.
                      Format: (summaries, numbers_list)
    """
    retriever = embedding_retriever_instance
    relevant_docs = retriever.retrieve_documents(query)

    summaries = [summarize_text(doc) for doc, _ in relevant_docs]
    numbers = [extract_numbers(doc) for doc, _ in relevant_docs]

    return summaries, numbers