import heapq
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import numpy as np

#pip3 install matplotlib



def heuristic_cost_estimate(start, goal): #as you can only move 5 directions (left right up down and sametile)
    return abs(goal[0] - start[0]) + abs(goal[1] - start[1])
#((1-0) , (5,4)) = (5-1) + (4-0) = 8

def get_neighbors(node, grid): # return list of neighbours
    x, y, t= node #changed
    neighbors = []
    Possible_Directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (0,0)] # make sure we dotn allow diagonals
    for dx, dy in Possible_Directions:
        neighbor_x, neighbor_y = x + dx, y + dy
        #if node is 2,2 it add in the first loop 0,1 so first neighbour is 2,3...

        if 0 <= neighbor_x < len(grid) and 0 <= neighbor_y < len(grid[0]) and grid[neighbor_x][neighbor_y] != 1:
            t2 = t +1
            neighbors.append((neighbor_x, neighbor_y,t2))

    return neighbors



def reconstruct_path_from_node(came_from, start_node, goal_node):
    # Start with the last node's spatial coordinates
    #print(f"Camefrom: {came_from}")
    last_node = None
    for node in came_from:
        if (node[0], node[1]) == (goal_node[0], goal_node[1]):
            last_node = node
            break
    #might reach (4, 4) at timestep 8, but goal_node could start at like (4, 4, 0)
    if not last_node:
        raise ValueError("Goal node not found in came_from.")

    path = [(last_node[0], last_node[1])]

    current_node = last_node
    while current_node != start_node:
        if current_node not in came_from:
            raise ValueError(f"Node {current_node} not found in came_from.")
        
        predecessor = came_from[current_node]
        path.append((predecessor[0], predecessor[1]))
        current_node = predecessor

    return path[::-1]




def a_star(grid,agent_id ,start_node, goal_node, constraints=[]):

    #f(n) = g(n) + h(n)       - n is the current vertex
    #where g(n) is the cost of the path from start to next
    #where h(n) is the heuristic estimated cost from vertex n to goal


    open_set = [(0, start_node)]
    came_from = {}#node and timestep
    g_score = {start_node: 0}
    time_step = 0  # Keep track of the current time step
    #maintain min element andfirst one 
    #so in  if i heapq push 5 2 9 i get [2,5,9] and if i push i alwyas
    #the smallest element always goes to the front when you push but rest of them arent ordered
    while open_set:
        
        current_cost, (x,y,t)= heapq.heappop(open_set) 
        current_node = (x,y,t)

        #print(f"Current node: {current_node}")
        if (x,y)  == (goal_node[0],goal_node[1]):
            print("Goal reached")
            path = reconstruct_path_from_node(came_from, start_node,goal_node)
            return path

        for neighbor in get_neighbors(current_node, grid):    #neighboru format is n1,n1 timestep
            x, y, t = neighbor  #x is agent, y coords and t timstep 
            # print(neighbor)
            #if (x, y, t) in constraints:
            if any(constraint[0] == agent_id and constraint[1] == x and constraint[2] == y and constraint[3] == t for constraint in constraints):
                continue  #Skip the neighbors that are in the constraints at the current time step
            # print(neighbor)       
            tentative_g_score = g_score[current_node] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score

                neighbor_x, neighbor_y = neighbor[0], neighbor[1]
                goal_x, goal_y = goal_node[0], goal_node[1]
                heuristic_cost = heuristic_cost_estimate((neighbor_x, neighbor_y), (goal_x, goal_y))
                f_score = tentative_g_score + heuristic_cost


                heapq.heappush(open_set, (f_score, neighbor))
                came_from[neighbor] = current_node

    return None  #No path
        
        
# grid = [
#     [0, 0, 0, 0, 0],
#     [0, 1, 1, 0, 0],
#     [1, 1, 0, 1, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0]
# ]
# agents = {
#     0: {'start': (0, 0, 0), 'goal': (4, 4, 0)},
#     1: {'start': (1, 3, 0), 'goal': (0, 1, 0)},

# }





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
# (9, 9, 0)
# (2, 0, 0)
# (7, 0, 0)
# (0, 2, 0)
# (8, 4, 0)
# (5, 6, 0)
# (0, 8, 0)
# (5, 1, 0)
# (0, 5, 0)

agents = {
    0: {'start': (0, 0, 0), 'goal': (1, 6, 0)},      
    1: {'start': (1, 3, 0), 'goal': (3, 3, 0)},     
    2: {'start': (2, 5, 0), 'goal': (0, 3, 0)}, 
    3: {'start': (5, 5, 0), 'goal': (3, 8, 0)},          
    4: {'start': (0, 4, 0), 'goal': (1, 8, 0)},     
    5: {'start': (1, 0, 0), 'goal': (3, 4, 0)},       
    6: {'start': (0, 2, 0), 'goal': (7, 3, 0)},       
    7: {'start': (7, 8, 0), 'goal': (0, 1, 0)}, 
    8: {'start': (0, 6, 0), 'goal': (4, 8, 0)}      
}




# agents = { #fine working goals
#     1: {'start': (0, 0, 0), 'goal': (2, 2, 0)},
#     2: {'start': (4, 4, 0), 'goal': (0, 0, 0)},
# }



# agent_id = 1  
# agent = agents[1]
# # path = a_star(grid,agent_id ,agent['start'], agent['goal'], [(1, 0,1, 1),(1, 1,0, 2)])
# path = a_star(grid,agent_id ,agent['start'], agent['goal'],[(1, 1,0, 1)])
#  #works by blocking into corner 
# path2 = a_star(grid,agent_id ,agent['start'], agent['goal'], [])
# print(path)
# print(path2)




def Solution_maker(agent_list,constraints):
    path_list = []
    for agent_number, agent_info in agent_list.items(): #loop thru the two keys as its dictionary inside of dictionary
        key = agent_number  
        start_value = agent_info['start']#retreaves the value associated with the key
        end_value = agent_info['goal']

        temp_path = a_star(grid,agent_number,start_value,end_value,constraints)
        # print(temp_path)
        path_list.append(temp_path)

    # for path in path_list:

        # print("full listfrom mak:",path)
    return path_list

#(0,0,2,2,),(0,0,2,3),(0,0,1,3)]))


# root = (Solution_maker(agents,[(1, 1,0, 1)]))
# root = (Solution_maker(agents,[(0,0,2,2,),(0,0,2,3),(0,0,1,3)]))

# #root = (Solution_maker(agents,[(1, 0, 2, 2)]))
# for path in root:

#     print(path)












def calculate_cost (solution):
    total = 0 
    for path in solution:
        total+=len(path)
    return total



class CBSNode:
    def __init__(self, solution,constraints, cost):
        self.solution= solution #a dictionary {agent_id: path}
        self.constraints = constraints#a dictionary {agent_id: [(x, y, time_step)]}
        #agent X cannot be (x,y) at timestep Z
        #it is set as empty for now as we add constraints as we go for the inididiual agents 
        self.cost = cost
    def __lt__(self, other):
        return self.cost < other.cost






def detect_conflicts(solution):#conflict detection 

    #this is a list of lists [   [(2,2),(2,3),(3,3)],   [(4,4),(3,4),(3,3 ]  ]
    #collision in timestep 3
    #theh max is used to only loop the lenghth of the longest one as aniother more the values dont need to be checked
    #is should really be the second shortests but for simplicity reasons max is used
    #edge and vertex conflcits 
    for time_step in range (len (max(solution))): #individual timestep of the list, this to vertcially slice and check downwards
        for i in range (len(solution)): #i is the the first path in the list of lists
            for j in range (i+1,len(solution)):#J is another path so when i and j are all searched all coords are checked
                #we start at i+1 as we dont wish to compare the same node
                #could also do the step below as
                #path = solution[i]
                #node1 = path[time_step]
                #same for node 2 and compare
                if time_step < len(solution[i]) and time_step < len(solution[j]):
                    if solution[i][time_step] == solution[j][time_step]:
                        x, y = solution[i][time_step]
                        #this means get the first path at postion time_step and comapre....
                        return[i,j,x,y,time_step]
                        
                    
                #edge collsions 
                if time_step + 1 < len(solution[i]) and time_step + 1 < len(solution[j]):
                    if solution[i][time_step] == solution[j][time_step + 1] and solution[i][time_step + 1] == solution[j][time_step]:
                        x, y = solution[i][time_step]
                        x2, y2 = solution[j][time_step]
                        return [i,j,x,y,x2,y2,time_step+1]
                    


                        # x, y = solution[i][time_step]
                        # return [i, j, x,y, time_step]
                    # means theres an ege collision at with two agents at that timestep

    return None



def add_constraint_if_new(constraints, new_constraint):
    if new_constraint not in constraints:
        constraints.append(new_constraint)


#[]solo cost
#
def cbs(grid, agents): # agents is a dictionary of agents and their start and end goal
    #make new cbs node with {agent id: path}
    print("Starting CBS...")
    initial_empty_constraints = []
    root_solution = (Solution_maker(agents,initial_empty_constraints))
    root_cost = calculate_cost (root_solution)
    root_node = CBSNode(root_solution,initial_empty_constraints,root_cost)
    # print("this the path of the first node", root_node)
    #put the list of paths from astar inside root
    
    open_set = []
    count_for_heap = 0
    heapq.heappush(open_set,(root_node.cost, count_for_heap,root_node))
    while open_set:
        count_for_heap +=1
        print("Processing node...")

        
        current_cost, count , current_node = heapq.heappop(open_set)
        #the current cost n count  is never used but i still need to pop it

        #return[i,j,solution[i][time_step],time_step]
        conflict = detect_conflicts(current_node.solution)
        print("the confliccts : ", conflict)
        # print("length of copnflict",len(conflict))
        # comflcits = [agent1, agent2, the node its at , timestep]
        # conflict_data = detect_conflicts(solution)
        # first_path_index = conflict_data[0]
        # second_path_index = conflict_data[1]
        # conflict_coords = conflict_data[2]
        # conflict_time = conflict_data[3]
        if conflict is None:
            return current_node.solution  # No conflicts, solution found
        
        if len(conflict) == 5:   #vertex collision
            for i in range(2):
                # def add_constraint_if_new(constraints, new_constraint):
                #     if new_constraint not in constraints:
                #     constraints.append(new_constraint)
                
                
                # if len(conflict) == 5:       
                    #return[i,j,x,y,time_step]
                    
                new_constraint = (conflict[i], conflict[2], conflict[3], conflict[4])
                new_constraints_list = current_node.constraints.copy()
                add_constraint_if_new(new_constraints_list, new_constraint)

                new_node = CBSNode(
                        current_node.solution,
                        new_constraints_list,
                        current_node.cost)
                
                print("this is the combined contraints",new_node.constraints)
                new_node.solution = Solution_maker(agents, new_node.constraints)

                for path in new_node.solution:
                    print("this is path",i, "made from solution maker   ", path)
                # print("new node solution-> ",new_node.solution)
                new_node.cost = calculate_cost(new_node.solution)
                heapq.heappush(open_set, (new_node.cost, count_for_heap, new_node))




        else:#edge conflict 
                    print("this is in edge part")
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
                    
                    print("this is the combined contraints",new_node1.constraints)
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
                    
                    print("this is the combined contraints",new_node2.constraints)
                    new_node2.solution = Solution_maker(agents, new_node2.constraints)

                    # for path in new_node.solution:
                    #     print("this is path",i, "made from solution maker   ", path)
                    # print("new node solution-> ",new_node.solution)
                    new_node2.cost = calculate_cost(new_node2.solution)
                    heapq.heappush(open_set, (new_node2.cost, count_for_heap, new_node2))




# solutions = cbs(grid, agents)
# [(0,0,2,2,),(0,0,2,3),(0,0,1,3)]
# root = (Solution_maker(agents,[(0,0,2,2,),(0,0,2,3),(0,0,1,3)]))

# root2 = (Solution_maker(agents,[]))
# test = detect_conflicts(root2)
# print("the detected conflicts---" , test)

# for path in root:

#     print("manually add ny conflicts",path)


solutions = cbs(grid, agents)
if solutions is not None:
    print("Solutions found:")
    for i, path in enumerate(solutions):
        print(f"Agent {i}: {path}")
else:
    print("No solutions found.")






    # root = (Solution_maker(agents,[(1, 1,0, 1)]))

    # root = (Solution_maker(agents,[(1, 0, 2, 2)]))
# root = (Solution_maker(agents,[(0,0,2,2,),(0,0,2,3),(0,0,1,3),(1,0,3,1)]))


# for path in root:

#     print(path)



    #root = (Solution_maker(agents,[(1, 0, 2, 2)]))



def animate_solution(grid, solutions):
    # fig, ax = plt.subplots(figsize=(10, 10))
    grid_height, grid_width = len(grid), len(grid[0])


    # Calculate figure size to fit the grid
    fig_width = grid_width 
    fig_height = grid_height 

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))








    num_agents = len(solutions)
    cmap = plt.get_cmap('tab20')
    agent_colors = [cmap(i) for i in np.linspace(0, 1, num_agents)]

    # Plot the grid
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == 1:
                ax.add_patch(Rectangle((x, y), 1, 1, color='gray'))

    ax.set_xlim(-0.5, len(grid[0]))
    ax.set_ylim(-0.5, len(grid))
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(grid[0])))
    ax.set_yticks(np.arange(len(grid)))
    ax.set_aspect('equal')

    ax.grid(True)

    # Plot goals as colored squares
    for i, agent_info in agents.items():
        goal_x, goal_y = agent_info['goal'][1], agent_info['goal'][0]
        ax.add_patch(Rectangle((goal_x, goal_y), 1, 1, color=agent_colors[i], alpha=0.5))

    # Create a dot for each agent
    dots = [ax.plot([], [], 'o', markersize=20, color=agent_colors[i], label=f'Agent {i}')[0] for i in range(num_agents)]

    def update(frame):
        for dot, path in zip(dots, solutions):
            if frame < len(path):
                x, y = path[frame][1], path[frame][0]
                dot.set_data(x+0.5, y+0.5)
                dot.set_visible(True)
            else:
                dot.set_visible(False)
        return dots
    
    legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    for handle in legend.legendHandles:
        handle.set_markersize(6)

    plt.get_current_fig_manager().full_screen_toggle() 

    
    anim = FuncAnimation(fig, update, frames=range(max(len(path) for path in solutions)+1), interval=500, blit=True, repeat=True)

    plt.show()
animate_solution(grid, solutions)










def valid_position_check(position, grid):
    y,x, _ = position
    #check if position is on x and y, and also its not an obstacle 
    valid =  0 <= y < len(grid) and 0 <= x < len(grid[0]) and grid[y][x] == 0
    print(f"Validating position {position}: {valid}")
    return valid

def validate_agents_before_cbs(agents, grid):
    all_valid = True
    for agent_id, info in agents.items():
        # Check if both start and goal positions are valid
        start_valid = valid_position_check(info['start'], grid)
        goal_valid = valid_position_check(info['goal'], grid)

        if start_valid == False or goal_valid == False:
            all_valid = False  # Mark as invalid if any check fails
            if start_valid == False:
                print(f"Agent {agent_id}'s start position {info['start']} is invalid.")
            if goal_valid == False:
                print(f"One or more of Agent {agent_id}'s goal positions are invalid.")
    return all_valid


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
