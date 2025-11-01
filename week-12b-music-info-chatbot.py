import re
import random as random
from chatbot_base import ChatbotBase

# Imports for local data retrieval (part 1)
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Imports for web data retrieval (part 2)
import time
import requests
import urllib.request
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup


"""
Retrieval based chatbot that recommends songs and finds biographical info about musicians.

See Week-6a-Retrieval-based-chatbots.ipynb for the tasks that guide you through developing this bot. 
"""

class MusicBot(ChatbotBase):
    def __init__(self, name="MusicBot"):
        ChatbotBase.__init__(self,name)
        # Code for tasks 3 & 7 goes here
    
    def greeting(self):
        print(f"Hello I am {self.name}, I am here to give you song recommendations and information about your favourite artists!")

    def process_input(self, user_input):
        processed_input = re.sub(r'[^\x00-\x7f]',r'', user_input) 
        processed_input = processed_input.lower()
        return processed_input
    
    # Add your functions from tasks 2, 6 & 8 here
    #
    #
    #
    
    def generate_response(self, processed_input):
        # Code for tasks 4 & 8 goes here
    
if __name__ == "__main__":
    music_bot = MusicBot()
    music_bot.greeting()

    response = 'How can I help you?'

    while music_bot.conversation_is_active:
        response = music_bot.respond(response)

