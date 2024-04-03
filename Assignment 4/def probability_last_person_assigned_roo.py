import numpy as np

def simulate(n, m, num_simulations=100000):

    successful_attempts = 0
    
    for _ in range(num_simulations):
        rooms = np.zeros(m) 
        
        person = np.random.randint(0, m) 
        rooms[person] = 1
        
        for friend in range(1, n-1):  
            if rooms[friend] == 0:
                rooms[friend] = 1  
            else:
                emptyrooms = np.where(rooms == 0)[0]
                randomroom = np.random.choice(emptyrooms)
                rooms[randomroom] = 1
        
        if rooms[n-1] == 0: 
            successful_attempts += 1
    
    return successful_attempts / num_simulations
n = 5  
m = 10
prob = simulate(n, m)
print(prob)
