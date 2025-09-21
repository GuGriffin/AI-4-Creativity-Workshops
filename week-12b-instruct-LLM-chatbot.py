# Surpress huggingface warnings being printed to terminal
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'False'

# Import ChatbotBase class
from chatbot_base import ChatbotBase

# Misc
import re
import random as random

# LLM imports
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
Instruct-LLM Chatbot that uses generative AI to respond to users.

You can take this code and adapt it for your projects.
Try altering the system prompt, ammending text to the user prompt, or change the hyperparameters listed in the constructor to customise the output.
"""

class InstructLLMChatbot(ChatbotBase):
    def __init__(self, name="LLMBot", device='cpu'):
        ChatbotBase.__init__(self,name)
        # Use either a cpu or gpu -- for now it's best to keep this as 'cpu' 
        self.device = device

        # Hyperparameters for text generation
        self.max_tokens = 100
        self.temperature = 0.3
        self.top_p = 0.99
        self.min_p = 0.1

        # System prompt for Instruct-LLM
        self.system_prompt = {
            "role": "system", 
            "content": "You are a friendly chatbot that answers questions in single sentences.",
            }

        # Checkpoint for our LLM
        checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)


    # Clean and process input text
    def process_input(self, user_input):
        # Remove leading and trailing whitespace characters
        processed_input = user_input.strip()
        return processed_input
    
    # Greeting function
    def greeting(self):
        print(f"Hello I am {self.name}, I utilise an Instruct-LLM to answer any question you wish!")
 
    # Respond to user farewell
    def farewell(self):
        self.conversation_is_active = False
        responses = ['Goodbye',
                    'Have a nice day',
                    'See you later!',
                    'Take care, bye!']           
        
        print(random.choice(responses))
        return "Conversation has terminated"

    # Take user input and add it the instruct prompt format alongside the system_prompt defined in the constructor
    def prepare_instruct_str(self, processed_input):
        user_prompt = {
            "role": "user", 
            "content": processed_input
        }
        messages = [self.system_prompt, user_prompt]
        input_str = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return input_str

    def respond_with_LLM(self, processed_input):
        # Prepare and tokenize input
        input_str = self.prepare_instruct_str(processed_input)
        input_tokens = self.tokenizer.encode(input_str, return_tensors="pt").to(self.device)
        input_token_len = input_tokens.shape[-1]
        
        # Generate, decode, clean and return output
        output = self.model.generate(input_tokens, max_new_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, min_p=self.min_p, do_sample=True)
        out_str = self.tokenizer.decode(output[0][input_token_len:])
        out_str = re.sub(r'(<\|im_start\|>assistant\n)|(<\|im_end\|>)','',out_str)
        return out_str

    # Generate response for the user 
    def generate_response(self, processed_input):
        # Use ^ and $ to match the start and end of the string
        bye_regex = r'^(bye|goodbye|end|exit|quit)$'
        
        # Check for a farewell message to end the program
        if re.search(bye_regex, processed_input.lower()):
            return self.farewell()
        else:
            return self.respond_with_LLM(processed_input)


if __name__ == "__main__":
    llm_chatbot = InstructLLMChatbot()
    llm_chatbot.greeting()

    response = 'How can I help you?'

    while llm_chatbot.conversation_is_active:
        response = llm_chatbot.respond(response)

