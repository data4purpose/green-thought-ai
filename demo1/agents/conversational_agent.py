from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai.chat_models import ChatOpenAI

from agents.tools.countries_image_generator import countries_image_generator
from agents.tools.get_countries_by_name import get_countries_by_name
from agents.tools.google_search import google_search

import agents.tools.document_processing as document_processing

from langchain.tools import tool


from langchain.agents import AgentExecutor
from agents.tools.llama_index_tool import LLaMAIndexTool


index_path = './llama_index.faiss'
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
#model_name = 'allenai/llama'
llama_index_tool = LLaMAIndexTool(index_path=index_path, model_name=model_name)

@tool
def llama_index_query(query):
    """Performs a document search in a defeined corpus using the provided query string. Choose this tool when you need to find documents and data."""
    # This wrapper will be called by the agent with the query
    return llama_index_tool.search(query)

def create_agent():

    tools = [countries_image_generator, get_countries_by_name, google_search]

    # Initialize the LLaMA-Index Tool if parameters are provided
    #if index_path and model_name:


    tools.append(llama_index_query)  # Add the LLaMA indexer query function to tools

    # Add `retrieve_function` to your tools
    tools.append(document_processing.retrieve_function)
    tools.append(document_processing.process_documents)

    functions = [convert_to_openai_function(f) for f in tools]

    model = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0).bind(functions=functions)

    prompt = ChatPromptTemplate.from_messages([("system", "You are helpful but sassy assistant"),
                                               MessagesPlaceholder(variable_name="chat_history"), ("user", "{input}"),
                                               MessagesPlaceholder(variable_name="agent_scratchpad")])

    memory = ConversationBufferWindowMemory(return_messages=True, memory_key="chat_history", k=5)

    chain = RunnablePassthrough.assign(agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
                                      ) | prompt | model | OpenAIFunctionsAgentOutputParser()

    agent_executor = AgentExecutor(agent=chain, tools=tools, memory=memory, verbose=True)

    return agent_executor


