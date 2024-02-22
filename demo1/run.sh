#!/bin/bash

source ~/.zsh

########################################################################################
#  This script starts the project's Flask application on the host it is started on.
#

. /Users/mkaempf/opt/anaconda3/bin/activate && conda activate /Users/mkaempf/opt/anaconda3/envs/ofcc;

# pip3 install -r requirements.txt

#pip3 install pdfplumber python-docx scikit-learn
#pip3 install joblib
#pip3 install transformers
#pip3 install sentence-transformers
#pip3 install faiss-cpu

# Create the index ...
#python3 agents/docpool.py


#python3 llama-index-tool.py

streamlit run chatbot_ui.py