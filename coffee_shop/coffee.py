"""
Coffee base class
All coffee drinks should inherit from this class
"""

class Coffee:

    # Constructor 
    def __init__(self, size, milk_type):
        # Values that need to be passed into class when created
        self.size = size
        self.milk_type = milk_type

        # Default values for each coffee drink 
        self.base_price = 0
        self.drink_name = None

        # Dicts for prices that should be the same for all different coffee drinks 
        self.size_prices = {'small': 0, 'medium': 50, 'large': 90}
        self.milk_prices = {'whole': 0, 'skimmed': 0, 'oat': 20, 'almond': 50, 'soya': 20}

    # Function that returns price of drink as an Int in pence
    def get_price(self):
        raise NotImplementedError('get_price() not implemented in base class Coffee')

    # Function that gives a descriptive name of the drink as a String
    def to_str(self):
        raise NotImplementedError('to_str() not implemented in base class Coffee')
    
    # Function that returns a Dictionary containing info about the drink in a structured foramt
    def to_dict(self):
        raise NotImplementedError('to_dict() not implemented in base class Coffee')