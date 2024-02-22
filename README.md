
![img.png](img.png)

# ESG Report Assistant

_What is an ESG Report Agent doing?_
This AI agent helps consumers to find and navigate through ESG data.

_*But hey:* What is this project for? What is an ESG Report Assistant, from a technical point of view?_

In this project we develop the prototype of an intelligent companion for consumers
who are interested in ESG data. First, we look into the ESG reports, published by multiple companies. 
For the second part of this work we selected supply chain data for specific products.

For this purpose, we envision a supportive research assistant, which helps users
to find and understand already reported ESG data. Researchers will be able to get 
valuable input to their research faster and ideally linked to accurate sources.

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
   cd green-thought-ai/demo1
   pip3 install -r requirements.txt
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
- Find ESG report for company XYZ ...

- Retrieve information on product life cycles ...

- Get supply chain data ...

## License
Distributed under the MIT License. See LICENSE.txt for more information.
