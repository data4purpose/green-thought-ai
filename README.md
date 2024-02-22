# ESG Report Assistant

This AI agent helps consumers to find and navigate through ESG data.

_But hey: What is it? What is an ESG Report Assistant?_

In this project we develop the prototype of an intelligent companion for consumers
who are interested in ESG data. First, we look into a company focus. 
As and second area of this work we selected supply chain data for specific products.

For this purpose, we envision a supportive research assistant, which helps users
to find and understand reported ESG data. Researchers will be able to get 
valuable input to their research faster and ideally with accurate sources.

## Our Approach
### A Multi-Modal Chatbot Powered by LangChain Agents, OpenAI's Function Calling, and Streamlit

This work is inspired by the [Medium blog](https://medium.com/cyberark-engineering/a-developer-guide-for-creating-a-multi-modal-chatbot-using-langchain-agents-9003ba0ffb4d) and this
[Github Repository](https://github.com/nirbar1985/country-compass-ai). Thanks, to _Nir Bar_ for breaking that ice for us.

#### Overview
The multi-modal chatbot we are crafting is backed by an agent that uses "self made" tools:
- REST Countries API Chain: enables retrieving information on countries, invoking [Rest countries API](https://restcountries.com/)
- DALL·E 3 Image Generator: Generates an image of countries based on the country name
- Google Search Tool: Useful for fetching information from the web

In the first iteration we plan to switch to "DuckDuck Search" instead of the costly Google Search API tool.
Next, we plan to add some better suited APIs for context retrieval, but in the befinning, we are already able to retrieve country
specific information from the _Countries API_, this is realy good starting point.

## Getting Started
Clone the repository, set up the virtual environment, and install the required packages.

1. git clone https://github.com/data4purpose/green-thought-ai.git

1. ( In case you have python version 3.11.4 installed in pyenv)
   ```shell script
   cd green-thought-ai
   pip3 install -r requirements.txt
   ```

1. Install dependencies
    ```shell script
    poetry install
    ```

1. Enter virtual env by:
    ```shell script
    poetry shell
    ```

## Store your API keys
- Create .env file
- Place your OPENAI_API_KEY into .env file
- Place your SERPAPI_API_KEY into .env file


## Running the Multi-Modal ChatBot
### Kick of the chatbot by running:
```
streamlit run chatbot_ui.py
```
### Pose queries for instance -  
- Create images of countries

  ![Screenshot 2024-02-10 at 18 16 30](https://github.com/nirbar1985/country-compass-ai/assets/19358731/9f5d57f4-9936-453b-a48a-b7600a8ced1e)

- Retrieve information on countries

  <img width="1518" alt="Screenshot 2024-02-10 at 0 44 48" src="https://github.com/nirbar1985/country-compass-ai/assets/19358731/0c8cfac0-484f-4ae8-9f47-1dee95eb217b">

- Get current data

![Screenshot 2024-02-09 at 23 02 00](https://github.com/nirbar1985/country-compass-ai/assets/19358731/b49af9b4-424c-417c-b4ed-c4a903d36f6f)

## License
Distributed under the MIT License. See LICENSE.txt for more information.
