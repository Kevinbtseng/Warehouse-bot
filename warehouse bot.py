import numpy as np

env_rows = 11
env_cols = 11

#Creates a 3d array, 11 rows, 11 cols, 4 values in each [row][col]
q_values = np.zeros((env_rows, env_cols, 4))

#up = 0, right = 1, down = 2, left = 3
actions = ['up','right','down','left']

rewards = np.full((env_rows, env_cols), -100)
rewards[0, 5] = 100

aisles = {}
aisles[1] = [i for i in range(1,10)]
aisles[2] = [1,7,9]
aisles[3] = [i for i in range(1,8)]
aisles[3].append(9)
aisles[4] = [3,7]
aisles[5] = [i for i in range(11)]
aisles[6] = [5]
aisles[7] = [i for i in range(1, 10)]
aisles[8] = [3,7]
aisles[9] = [i for i in range(11)]

for row_index in range(1, 10):
    for col_index in aisles[row_index]:
        rewards[row_index, col_index] = -1

#for row in rewards:
#    print(row)
#define a function that determines if the specified location is a terminal state
def is_terminal_state(current_row_index, current_column_index):
  #if the reward for this location is -1, then it is not a terminal state (i.e., it is a 'white square')
  if rewards[current_row_index, current_column_index] == -1:
    return False
  else:
    return True


#finds a random, non terminal starting location
def get_starting_location():
    curr_row_index = np.random.randint(env_rows)
    curr_col_index = np.random.randint(env_cols)

    while is_terminal_state(curr_row_index, curr_col_index):
        curr_row_index = np.random.randint(env_rows)
        curr_col_index = np.random.randint(env_cols)
    return curr_row_index, curr_col_index

def get_next_action(curr_row_index, curr_col_index, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[curr_row_index, curr_col_index])
    else:
        return np.random.randint(4)

def get_next_location(curr_row_index, curr_col_index, action_index):
    new_row_index = curr_row_index
    new_col_index = curr_col_index
    if actions[action_index] == 'up' and curr_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and curr_col_index < env_cols - 1:
        new_col_index += 1
    elif actions[action_index] == 'down' and curr_row_index <= env_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and curr_col_index > 0:
        new_col_index -= 1
    return new_row_index, new_col_index

#Define a function that will get the shortest path between any location within the warehouse that 
#the robot is allowed to travel and the item packaging location.
def get_shortest_path(start_row_index, start_col_index):
    if is_terminal_state(start_row_index, start_col_index):
        return[]
    else:
        curr_row_index, curr_col_index = start_row_index, start_col_index
        shortest = []
        shortest.append([curr_row_index, curr_col_index])
        while not is_terminal_state(curr_row_index, curr_col_index):
            action_index = get_next_action(curr_row_index, curr_col_index, 1.)
            curr_row_index, curr_col_index = get_next_location(curr_row_index, curr_col_index, action_index)
            shortest.append([curr_row_index, curr_col_index])
        return shortest
    
    


epsilon = .9
discount_factor = .9
learning_rate = .9


for episode in range(1000):
    row_index, col_index = get_starting_location()

    while not is_terminal_state(row_index, col_index):
        action_index = get_next_action(row_index, col_index, epsilon)

        old_row_index, old_col_index = row_index, col_index
        row_index, col_index = get_next_location(row_index, col_index, action_index)

        reward = rewards[row_index, col_index]
        old_q_val = q_values[old_row_index, old_col_index, action_index]
        temportal_diference = reward + (discount_factor * np.max(q_values[row_index,col_index])) - old_q_val

        new_q_val = old_q_val + (learning_rate * temportal_diference)
        q_values[old_row_index, old_col_index, action_index] = new_q_val

print("Done")

print(get_shortest_path(3, 9))
print(get_shortest_path(5, 0)) 
print(get_shortest_path(9, 5))