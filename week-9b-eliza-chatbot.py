import re
import random as random
from chatbot_base import ChatbotBase

"""
Eliza-esque chatbot that performs therapy in the style of the original Eliza chatbot: https://web.njit.edu/~ronkowit/eliza.html

See Week-5a-Parsing-text.ipynb for the tasks that guide you through developing this bot. 
"""

class ELIZA(ChatbotBase):
    def __init__(self, name="ELIZA"):
        ChatbotBase.__init__(self,name)
        self.conversation_is_active = True
        
        self.default_responses = ["Please tell me more.",
                            "Let's change focus a bit... Tell me about your family.",
                            "Can you elaborate on that?",
                            "Why do you say that?",
                            "I see.",
                            "Very interesting.",
                            "I see.  And what does that tell you?",
                            "How does that make you feel?",
                            "How do you feel when you say that?"]
    
    def greeting(self):
        print(f"Hello I am {self.name}, your therapist for today.")

    
if __name__ == "__main__":
    
    eliza = ELIZA()
    eliza.greeting()

    response = 'Tell me how are you feeling.'

    while eliza.conversation_is_active:
        response = eliza.respond(response)

