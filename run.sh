#!/bin/bash

source ~/.zsh

########################################################################################
#  This script starts the project's Flask application on the host it is started on.
#

. /Users/mkaempf/opt/anaconda3/bin/activate && conda activate /Users/mkaempf/opt/anaconda3/envs/ofcc;

#pip3 install watchdog
#pip3 install -r requirements.txt

streamlit run chatbot_ui.py