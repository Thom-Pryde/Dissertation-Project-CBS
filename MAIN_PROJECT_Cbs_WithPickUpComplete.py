import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import numpy as np
import heapq


#HOW TO RUN
#run 'pip3 install matplotlib'
#swappable grid exmaples below, simply uncomment the single grid and single agent set you want to test
#and comment out the previous grid and agent set.






import time
start = time.time()

##1 example

# grid = [
#     [0, 0, 0, 0, 0],
#     [0, 1, 1, 0, 0],
#     [0, 1, 0, 1, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0]
# ]



# agents = {
#     0: {'start': (0, 0, 0), 'goal': [(2, 2, 0), (4, 4, 0)]},
#     1: {'start': (1, 3, 0), 'goal': [(3, 1, 0), (1, 0, 0)]},
#     2: {'start': (0, 2, 0), 'goal': [(3, 3, 0), (0, 2, 0)]},
# }




##example 2, slightly larger grid with 2 agents sets to choose from (comment one at a time to individually test), they are the same but in different order


# grid = [
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],  
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],     
# ]

# agents = {
#     0: {'start': (0, 0, 0), 'goal': [(1, 6, 0), (9, 9, 0)]},      
#     1: {'start': (1, 3, 0), 'goal': [(3, 3, 0), (2, 0, 0)]},     
#     2: {'start': (2, 5, 0), 'goal': [(0, 3, 0), (7, 0, 0)]}, 
#     3: {'start': (5, 5, 0), 'goal': [(3, 8, 0), (0, 2, 0)]},          
#     4: {'start': (0, 4, 0), 'goal': [(1, 8, 0), (8, 4, 0)]},     
#     5: {'start': (1, 0, 0), 'goal': [(3, 4, 0), (5, 6, 0)]},       
#     6: {'start': (0, 2, 0), 'goal': [(7, 3, 0), (0, 8, 0)]},       
#     7: {'start': (7, 8, 0), 'goal': [(0, 1, 0), (5, 1, 0)]}, 
#     8: {'start': (0, 6, 0), 'goal': [(4, 8, 0), (0, 5, 0)]}      
# }


# agents = {
#         0: {'start': (1, 0, 0), 'goal': [(3, 4, 0), (5, 6, 0)]},
#         1: {'start': (0, 2, 0), 'goal': [(7, 3, 0), (0, 8, 0)]},
#         2: {'start': (0, 0, 0), 'goal': [(1, 6, 0), (9, 9, 0)]},
#         3: {'start': (1, 3, 0), 'goal': [(3, 3, 0), (2, 0, 0)]}, 
#         4: {'start': (0, 6, 0), 'goal': [(4, 8, 0), (0, 5, 0)]},
#         5: {'start': (7, 8, 0), 'goal': [(0, 1, 0), (5, 1, 0)]},
#         6: {'start': (0, 4, 0), 'goal': [(1, 8, 0), (8, 4, 0)]},  
#         7: {'start': (5, 5, 0), 'goal': [(3, 8, 0), (0, 2, 0)]},
#         8: {'start': (2, 5, 0), 'goal': [(0, 3, 0), (7, 0, 0)]},
# }










##example 3 where I show it working with multiple pickups 

grid = [
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],  
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],     
]





agents = {
    0: {'start': (0, 0, 0), 'goal': [(1, 6, 0), (0, 5, 0), (9, 9, 0)]},      
    1: {'start': (1, 3, 0), 'goal': [(3, 3, 0), (2, 0, 0)]},     
    2: {'start': (2, 5, 0), 'goal': [(0, 3, 0), (7, 0, 0)]}, 
    3: {'start': (5, 5, 0), 'goal': [(3, 8, 0), (0, 2, 0)]},          
    4: {'start': (0, 4, 0), 'goal': [(1, 8, 0), (9, 0, 0), (8, 4, 0)]},     
    5: {'start': (1, 0, 0), 'goal': [(3, 4, 0), (5, 6, 0)]},       
    6: {'start': (0, 2, 0), 'goal': [(7, 3, 0), (0, 8, 0)]},       
    7: {'start': (7, 8, 0), 'goal': [(0, 1, 0), (5, 1, 0)]}, 
    8: {'start': (0, 6, 0), 'goal': [(4, 8, 0), (0, 5, 0)]}      
}






##example 4 with large grid 


# grid = [
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1 ,0, 0],#0
#   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ,1, 0, 0, 0, 0, 1, 0, 0 ,0, 0],#1
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0],#2
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ,0, 0],#3
#   [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ,0, 0],#4
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ,0, 0],#5
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1 ,0, 0],#6
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ,0, 0],#7
#   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1 ,0, 0],#8 
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0 ,0, 0],#9
#   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0],#10    
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ,0, 0],#11 
#   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0],#12    
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ,0, 0],#13
#   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0],#14   
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ,0, 0],#15   
#   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ,0, 0],#16   
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1 ,0, 0],#17   
#   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ,0, 0],#18   
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1 ,0, 0] #19   
# ]# 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19



# agents = {
#     0: {'start': (0, 0, 0), 'goal': [(1, 6, 0), (9, 9, 0)]},      
#     1: {'start': (1, 3, 0), 'goal': [(3, 3, 0), (2, 0, 0)]},     
#     2: {'start': (2, 5, 0), 'goal': [(0, 3, 0), (7, 0, 0)]}, 
#     3: {'start': (5, 5, 0), 'goal': [(3, 8, 0), (0, 2, 0)]},          
#     4: {'start': (0, 4, 0), 'goal': [(1, 8, 0), (8, 4, 0)]},     
#     5: {'start': (1, 0, 0), 'goal': [(3, 4, 0), (5, 6, 0)]},       
#     6: {'start': (0, 2, 0), 'goal': [(7, 3, 0), (0, 8, 0)]},       
#     7: {'start': (7, 8, 0), 'goal': [(0, 1, 0), (5, 1, 0)]}, 
#     8: {'start': (0, 6, 0), 'goal': [(4, 8, 0), (0, 5, 0)]},
#     9: {'start': (9, 8, 0), 'goal': [(13, 1, 0), (10, 18, 0)]},
#    10: {'start': (10, 17, 0), 'goal': [(0, 8, 0), (11, 5, 0)]},
#    11: {'start': (14, 4, 0), 'goal': [(7, 8, 0), (12, 7, 0)]},
#    12: {'start': (13, 2, 0), 'goal': [(13, 19, 0), (13, 5, 0)]},
#    13: {'start': (3, 18, 0), 'goal': [(13, 4, 0), (14, 16, 0)]},
#    14: {'start': (7, 19, 0), 'goal': [(14, 19, 0), (15, 18, 0)]},
#    15: {'start': (18, 0, 0), 'goal': [(10, 8, 0), (16, 5, 0)]},
#    16: {'start': (15, 6, 0), 'goal': [(13, 7, 0), (18, 1, 0)]},
#    17: {'start': (12, 16, 0), 'goal': [(19, 0, 0), (19, 19, 0)]},
# }




























#start is the current node
def heuristic_cost_estimate(start, goal):#as you can only move 4 directions (left right up down)
    return abs(goal[0] - start[0]) + abs(goal[1] - start[1])

# return list of neighbour
def get_neighbors(node, grid):
    x, y, t = node
    neighbors = []
    Possible_Directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]
    for dx, dy in Possible_Directions:
        neighbor_x, neighbor_y = x + dx, y + dy
        if 0 <= neighbor_x < len(grid) and 0 <= neighbor_y < len(grid[0]) and grid[neighbor_x][neighbor_y] != 1:
            t2 = t + 1
            neighbors.append((neighbor_x, neighbor_y, t2))
    return neighbors

#reconstruct path, currrent node is the last node, and goes backwards
def reconstruct_path_from_node(came_from, start_node, current_node):
    path = [(current_node[0], current_node[1])]
    while current_node != start_node:
        if current_node in came_from:
            current_node = came_from[current_node]
            path.append((current_node[0], current_node[1]))
        else:
            break  #Break incase current node isnt in came_from
    return path[::-1]#reverse the path




    #f(n) = g(n) + h(n)       - n is the current vertex
    #where g(n) is the cost of the path from start to next
    #where h(n) is the heuristic estimated cost from vertex n to goal
def a_star(grid, agent_id, start_node, goals, constraints=[]):
    open_set = [(0, start_node)]
    came_from = {}
    g_score = {start_node: 0}
    goal_index = 0
    final_path = []

    while open_set:
        current_cost, (x,y,t)= heapq.heappop(open_set) 
        current_node = (x,y,t)  
        goal_x, goal_y = goals[goal_index][0], goals[goal_index][1]#from first set of goal break down x y 

        if  (x,y) == (goal_x, goal_y):
            #print(f"Camefrom: {came_from}")
            path_to_goal = reconstruct_path_from_node(came_from, start_node, current_node)
            if goal_index != 0:
                path_to_goal = path_to_goal[1:]

            final_path.extend(path_to_goal)#use extend to make it one full list
            start_node = current_node  # Update the start node for the next segment
            
            goal_index += 1
            if goal_index >= len(goals):
                return final_path  
            open_set = [(0, current_node)]
            g_score = {current_node: 0}
            continue

        for neighbor in get_neighbors(current_node, grid):
            x, y, t = neighbor
            if any(constraint[0] == agent_id and constraint[1] == x and constraint[2] == y and constraint[3] == t for constraint in constraints):
                continue

            tentative_g_score = g_score[current_node] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic_cost_estimate((x, y), (goal_x, goal_y))
                heapq.heappush(open_set, (f_score, neighbor))
                came_from[neighbor] = current_node

    return final_path  # Return the path if all goals are reached


#function to generate solution given a list of agents from low level a star search
#Root.solution = find individual paths by the low level()
def Solution_maker(agent_list,constraints):
    path_list = []
    for agent_number, agent_info in agent_list.items(): #loop through the two keys as its dictionary inside of dictionary
        key = agent_number  
        start_value = agent_info['start']#retreaves the value associated with the key
        end_value = agent_info['goal']

        temp_path = a_star(grid,agent_number,start_value,end_value,constraints)
        path_list.append(temp_path)
    return path_list


def detect_conflicts(solution):#conflict detection 

    #this is a list of lists [   [(2,2),(2,3),(3,3)],   [(4,4),(3,4),(3,3 ]  ]
    #collision in timestep 3
    #the max is used to only loop the lenghth of the longest one as another more the values dont need to be checked
    #is should really be the second shortests but for simplicity reasons max is used
    #edge and vertex conflcits 
    
    #VERTEX COLLSION
    for time_step in range (len (max(solution))): #individual timestep of the list, this to vertcially slice and check downwards
        for i in range (len(solution)): #i is the the first path in the list of lists
            for j in range (i+1,len(solution)):#j is another path so when i and j are all searched all coords are checked
                #we start at i+1 as we dont wish to compare the same node
                if time_step < len(solution[i]) and time_step < len(solution[j]):
                    if solution[i][time_step] == solution[j][time_step]:
                        x, y = solution[i][time_step]
                        return[i,j,x,y,time_step]
                        
                    
                #EDGE COLLISION 
                if time_step + 1 < len(solution[i]) and time_step + 1 < len(solution[j]):
                    if solution[i][time_step] == solution[j][time_step + 1] and solution[i][time_step + 1] == solution[j][time_step]:
                        x, y = solution[i][time_step]
                        x2, y2 = solution[j][time_step]
                        return [i,j,x,y,x2,y2,time_step+1]
                    # means theres an edge collision at with two agents at that timestep

    return None


def calculate_cost (solution):
    total = 0 
    for path in solution:
        total+=len(path)
    return total



class CBSNode:
    def __init__(self, solution,constraints, cost):
        self.solution= solution #a dictionary {agent_id: path}
        self.constraints = constraints#a dictionary {agent_id: [(x, y, time_step)]}
        self.cost = cost
    def __lt__(self, other):
        return self.cost < other.cost

def add_constraint_if_new(constraints, new_constraint):
    if new_constraint not in constraints:
        constraints.append(new_constraint)

def cbs(grid, agents): # agents is a dictionary of agents and their start and end goal
    #make new cbs node with {agent id: path}
    print("Starting CBS...")
    initial_empty_constraints = []
    root_solution = (Solution_maker(agents,initial_empty_constraints))
    root_cost = calculate_cost (root_solution)
    root_node = CBSNode(root_solution,initial_empty_constraints,root_cost)
    
    open_set = []
    count_for_heap = 0

    heapq.heappush(open_set,(root_node.cost, count_for_heap,root_node))
    while open_set:
        count_for_heap +=1
        # print("Processing node...")
        current_cost, count , current_node = heapq.heappop(open_set)
        #the current cost n count  is never used but I still need to pop it
        conflict = detect_conflicts(current_node.solution)
        #print("the confliccts : ", conflict)
        if conflict is None:
            return current_node.solution  # No conflicts, solution found
        
        if len(conflict) == 5:   #vertex collision, split conflcit up
            for i in range(2):
                new_constraint = (conflict[i], conflict[2], conflict[3], conflict[4])
                new_constraints_list = current_node.constraints.copy()
                add_constraint_if_new(new_constraints_list, new_constraint)

                new_node = CBSNode(
                        current_node.solution,
                        new_constraints_list,
                        current_node.cost)
                # print("this is the combined contraints",new_node.constraints)
                new_node.solution = Solution_maker(agents, new_node.constraints)

                # for path in new_node.solution:
                #     print("this is path",i, "made from solution maker   ", path)
                # print("new node solution-> ",new_node.solution)
                new_node.cost = calculate_cost(new_node.solution)
                heapq.heappush(open_set, (new_node.cost, count_for_heap, new_node))

        else:#edge conflict 
                #return [i,j,x,y,x2,y2,time_step+1]
                    new_constraint1 = (conflict[0], conflict[4], conflict[5], conflict[6])
                    first_new_constraints_list = current_node.constraints.copy()
                    add_constraint_if_new(first_new_constraints_list, new_constraint1)

                    new_constraint2 = (conflict[1], conflict[2], conflict[3], conflict[6])
                    second_new_constraints_list = current_node.constraints.copy()
                    add_constraint_if_new(second_new_constraints_list, new_constraint1)

                    new_node1 = CBSNode(
                            current_node.solution,
                            first_new_constraints_list,
                            current_node.cost)
                    
                    # print("this is the combined contraints",new_node1.constraints)
                    new_node1.solution = Solution_maker(agents, new_node1.constraints)

                    # for path in new_node.solution:
                    #     print("this is path",i, "made from solution maker   ", path)
                    # print("new node solution-> ",new_node.solution)
                    new_node1.cost = calculate_cost(new_node1.solution)
                    heapq.heappush(open_set, (new_node1.cost, count_for_heap, new_node1))




                    new_node2 = CBSNode(
                            current_node.solution,
                            second_new_constraints_list,
                            current_node.cost)
                    
                    # print("this is the combined contraints",new_node2.constraints)
                    new_node2.solution = Solution_maker(agents, new_node2.constraints)

                    # for path in new_node.solution:
                    #     print("this is path",i, "made from solution maker   ", path)
                    # print("new node solution-> ",new_node.solution)
                    new_node2.cost = calculate_cost(new_node2.solution)
                    heapq.heappush(open_set, (new_node2.cost, count_for_heap, new_node2))




def animate_solution(grid, solutions):
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(right=0.7)
    num_agents = len(solutions)  #solutions array  defined
    cmap = plt.get_cmap('tab20')  #A colormap with 20 indivual colors
    agent_colors = [cmap(i) for i in np.linspace(0, 1, num_agents)]
    # agent_colors = ['blue', 'green', 'red', 'yellow']
    
    # Plot the grid
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == 1:  # 1 represents an obstacle
                ax.add_patch(Rectangle((x, y), 1, 1, color='gray'))
    # Plotting parameters
    ax.set_xlim(-0.5, len(grid[0]))
    ax.set_ylim(-0.5, len(grid))
    ax.invert_yaxis()  # Invert the y-axis so (0,0) is at the top left as matlab requires x y and not the usual y x format in code.


    ax.set_xticks([x for x in range(len(grid[0]))])
    ax.set_yticks([y for y in range(len(grid))])
    ax.grid(which='both')












##this is for normal one before i added 
    
# # Highlight edges of pickup locations
#     for agent_id, agent_info in agents.items():
#         pickup_location = agent_info['goal'][0][:2]  # Pickup location (discard the time component)
#         color = agent_colors[agent_id]
#         # Draw a rectangle with a thick border to highlight the edge
#         ax.add_patch(Rectangle((pickup_location[1], pickup_location[0]), 1, 1, fill=False, edgecolor=color, linewidth=2))




    #this is for multi pickup colours!!!
    
    # Highlight edges of all but the last goal locations for each agent
    for agent_id, agent_info in agents.items():
        # Iterate through all goals except the last one
        for pickup_location in agent_info['goal'][:-1]:  # Exclude the last goal
            color = agent_colors[agent_id]
            # Draw a rectangle with a thick border around the pickup location
            ax.add_patch(Rectangle((pickup_location[1], pickup_location[0]), 1, 1, fill=False, edgecolor=color, linewidth=2))








    # for x in range(len(grid[0])):
    #     for y in range(len(grid)):
    #          ax.text(x + 0.5, y + 0.5, f'({x},{y})', ha='center', va='center')
    marker_size = max(10, 750 / (len(grid) + len(grid[0])))
    #make goals coloured
    for i, path in enumerate(solutions): 
        goal_x, goal_y = path[-1][1], path[-1][0]  # Swap x and y for matplotlib's coordinate system
        ax.add_patch(Rectangle((goal_x, goal_y), 1, 1, color=agent_colors[i], alpha=0.5))  # Semi-transparent

    
    #makke a dot for each agent with increased size and specified color
    dots = [plt.plot([], [], 'o', markersize=marker_size, color=agent_colors[i], label=f'Agent {i}')[0] for i in range(len(solutions))]

    #a dot for each agent with increased size
    # dots = [plt.plot([], [], 'o', markersize=10, label=f'Agent {i}')[0] for i in range(len(solutions))]

    #Animation update function
    def update(frame):
        for dot, path in zip(dots, solutions):
            if frame < len(path):
                # Add 0.5 to both coordinates to center the dot in the square
                dot.set_data([path[frame][1] + 0.5], [path[frame][0] + 0.5])
                dot.set_visible(True)
            else:
                # Keep the dot at the final position if the path is complete
                # dot.set_data([path[-1][1] + 0.5], [path[-1][0] + 0.5])
                dot.set_visible(False)
        return dots





    legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    for handle in legend.legendHandles:
        handle.set_markersize(6)

    plt.get_current_fig_manager().full_screen_toggle() 

    
    # Create animation with slower interval
    anim = FuncAnimation(fig, update, frames=range(max(len(path) for path in solutions)), interval=500, blit=True)  # Slower animation

    
    plt.show()
# animate_solution(grid, solutions)





#since this problem is like the halting problem where
#we dont know whether the program will finish running, or continue to run forever
#must create functions to validate the postions..
    
#check if coords are same as an obstacle or outside of grid
def valid_position_check(position, grid):
    y,x, _ = position
    #check if position is on x and y, and also its not an obstacle 
    valid =  0 <= y < len(grid) and 0 <= x < len(grid[0]) and grid[y][x] == 0
    print(f"Validating position {position}: {valid}")
    return valid


#fucntion to check all agents' positions, assume all are valid to begin with..
def validate_agents_before_cbs(agents, grid):
    all_valid = True
    for agent_id, info in agents.items():
        # Check if both start and goal positions are valid
        start_valid = valid_position_check(info['start'], grid)
        goals_valid = True  # Assume all goals are valid initially

        for goal in info['goal']:
            if  valid_position_check(goal, grid) == False:  
                goals_valid = False  #if one fo the goals fails make it false now 
                break

        if start_valid == False or goals_valid == False:
            all_valid = False  # Mark as invalid if any check fails
            if start_valid == False:
                print(f"Agent {agent_id}'s start position {info['start']} is invalid.")
            if goals_valid == False:
                print(f"One or more of Agent {agent_id}'s goal positions are invalid.")
    return all_valid

#check is any agent have the same start coords
def check_start_positions(agents):
    start_positions = []  
    for agent_id, info in agents.items():
        found_duplicate = False
        for position in start_positions:
            if position == info['start']:
                found_duplicate = True
                break
        if found_duplicate:
            print(f"Conflict detected: Agent {agent_id}'s start position {info['start']} is shared with another agent.")
            return False
        else:
            start_positions.append(info['start'])


    return True  


if check_start_positions(agents) and validate_agents_before_cbs(agents,grid): 
    solutions = cbs(grid, agents)
    end = time.time() 
    duration = end - start 
    print("Time: {} seconds".format(round(duration, 3)))
    animate_solution(grid, solutions)
    if solutions is not None:
        print("Solutions found:")
        for i, path in enumerate(solutions):
            print(f"Agent {i}: {path}")
    else:
        print("No solutions found.")

# end = time.time() 
# duration = end - start 
# print("Time: {} seconds".format(round(duration, 3)))