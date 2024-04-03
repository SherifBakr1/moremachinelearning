# def probability_last_person_own_room(n):
#     # Base case for recursion
#     if n == 1:
#         return 1
#     elif n == 2:
#         return 1/2
#     else:
#         # Recurrence relation: p(n, n) = 1/n + (n-2)/n * p(n-1, n-1)
#         return 1/n + (n-2)/n * probability_last_person_own_room(n-1)

# n = 500 
# probability = probability_last_person_own_room(n)
# print(f"The probability that the last person ends up in their own room for n={n} is {probability:.2f}")
import random

def probability_last_person_own_room_m_equals_n(n):
    # Initialize variables for simulation
    total_trials = 10000  # Number of trials for simulation
    success_count = 0  # Count of trials where nth person ends up in their own room
    
    for _ in range(total_trials):
        rooms = list(range(n))  # List of room numbers
        assigned_rooms = list(range(n))  # Person i's assigned room is rooms[i]
        random.shuffle(assigned_rooms)  # Randomize room assignments
        
        # Simulate each person choosing a room
        for i in range(n - 1):  # Iterate over the first n-1 people
            if assigned_rooms[i] in rooms:
                rooms.remove(assigned_rooms[i])  # Person chooses their own room
            else:
                rooms.remove(random.choice(rooms))  # Person chooses a random available room
        
        # Check if the last person ends up in their own room
        if assigned_rooms[-1] in rooms:
            success_count += 1

    # Calculate probability
    probability = success_count / total_trials
    return probability

# Example usage
n = 5  # Number of people and rooms, change this value to test different scenarios
probability = probability_last_person_own_room_m_equals_n(n)
print(f"The probability that the last person ends up in their own room for n={n} is approximately {probability:.2f}")
