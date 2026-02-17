from uuid import uuid4

class Firm:
    """
    A class for a specific firm.
    """
    
    def __init__(self
                 , market_cap: float
                 , shares_outstanding: float
                 , stock_price: float
                 , seed: int = None
                 ):

        self.id = str(uuid4())
        self.market_cap = market_cap
        self.shares_outstanding = shares_outstanding
        self.stock_price = stock_price
        self.seed = seed
        
    
    ### Variables ###
    class PROJECTS:
        LOW_RISK = "low-risk"
        HIGH_RISK = "high-risk"
        
        
        
    
    
    ### Update functions ####
    def update_market_cap(self, new_market_cap: float):
        self.market_cap = new_market_cap
    
    def update_shares_outstanding(self, new_shares_outstanding: float):
        self.shares_outstanding = new_shares_outstanding
        
    def update_stock_price(self, new_stock_price: float):
        self.stock_price = new_stock_price
        
        
        
        
        
        
        
        
        
        

if __name__ == "__main__":
    # Example usage
    firm = Firm(market_cap=1000000, shares_outstanding=10000, stock_price=100)
    print(f"Firm ID: {firm.id}")
    print(f"Market Cap: {firm.market_cap}")
    print(f"Shares Outstanding: {firm.shares_outstanding}")
    print(f"Stock Price: {firm.stock_price}")
    
    # Update values
    firm.update_market_cap(1200000)
    firm.update_shares_outstanding(12000)
    firm.update_stock_price(110)
    
    print("\nAfter updates:")
    print(f"Market Cap: {firm.market_cap}")
    print(f"Shares Outstanding: {firm.shares_outstanding}")
    print(f"Stock Price: {firm.stock_price}")