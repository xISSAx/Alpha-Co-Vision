# Store Your API keys. #Use Production key for best expereince. Not sure but you might receive upto $60-$70 worth of credits initially, checkout Cohere for more info.
# cohere_api_key = "" #USE COHERE_AI PRODUCTION API FIND FOUT MORE --> : https://dashboard.cohere.ai/api-keys
#
# #Should you wish to use OpenAI GPT-3 API, please uncomment the following, and change response_generation to openai_response_generation in main.py.
# #Warning high $$$ spend on OpenAI due to the use of GPT-3 instruct davinci - 003, please change the model in response_generation if you want less $ spend before proceeding.
# openai_api_key = "" # GET YOUR OPENAI API KEY HERE: #https://platform.openai.com/account/api-keys
# enable_openai = True
# enable_mps = False

import argparse
import json

# Create a configuration container
class Settings:
    pass

settings = None
def load_config(config_file):
    global settings

    # Load the JSON config file
    with open(config_file, "r") as f:
        config_data = json.load(f)

    # Save the values to the configuration variables
    settings = Settings()
    settings.cohere_api_key = config_data["cohere_api_key"]
    settings.openai_api_key = config_data["openai_api_key"]
    settings.enable_openai = config_data["enable_openai"]
    settings.enable_mps = config_data["enable_mps"]
    settings.edge_tts_enable = config_data["edge_tts_enable"]
    settings.edge_tts_voice = config_data["edge_tts_voice"]

    return settings
