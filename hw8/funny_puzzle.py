import heapq



def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    from_dic = {}
    for i in from_state:
        if i == 0:
            continue
        idx = from_state.index(i)
        x = idx // 3
        y = idx % 3
        from_dic[i] = [x,y]
    if to_state == [1, 2, 3, 4, 5, 6, 7, 0, 0]:        
        to_dic = {1:[0,0], 2:[0,1], 3:[0,2],
         4:[1, 0], 5:[1,1], 6:[1,2],
         7:[2,0]}
    else:
        for i in to_state:
            if i == 0:
                continue
            idx = to_state.index(i)
            x = idx // 3
            y = idx % 3
            to_dic[i] = [x,y]
         
    dist_sum = 0
    too_state = [1, 2, 3, 4, 5, 6, 7] 
    for i in too_state:
        dist_sum += abs(from_dic[i][0] - to_dic[i][0]) + abs(from_dic[i][1] - to_dic[i][1])
    return dist_sum 



def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.
    INPUT: 
        A state (list of length 9)
    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))

        

def get_succ(state):
    #find index of two blanks 
    zero_i = [state.index(0), 8 - state[::-1].index(0)]
    zero_2d_idx = []
    for i in zero_i:
        x = i // 3
        y = i % 3
        zero_2d_idx.append([x,y])
    
    #Move blank squares
    poss = set()
    poss.add(tuple(state))
    
    for i,j in zero_2d_idx:
        if i > 0: #up
            i_poss = i - 1
            j_poss = j
            one_idx = i_poss*3 + j_poss
            state_copy = state.copy()
            element = state[one_idx]
            state_copy[one_idx] = 0
            state_copy[i*3 + j] = element
            if tuple(state_copy) not in poss: #add state in a set
                poss.add(tuple(state_copy))
        if i < 2: #down
            i_poss = i + 1
            j_poss = j
            one_idx =  i_poss*3 + j_poss
            state_copy = state.copy()
            element = state[one_idx]
            state_copy[one_idx] = 0
            state_copy[i*3 + j] = element
            if tuple(state_copy) not in poss:
                poss.add(tuple(state_copy))
        if j > 0: #left
            i_poss = i
            j_poss = j - 1
            one_idx = i_poss*3 + j_poss
            state_copy = state.copy()
            element = state[one_idx]
            state_copy[one_idx] = 0
            state_copy[i*3 + j] = element
            if tuple(state_copy) not in poss:
                poss.add(tuple(state_copy))
        if j < 2: #right
            i_poss = i
            j_poss = j + 1
            one_idx = i_poss*3 + j_poss
            state_copy = state.copy()
            element = state[one_idx]
            state_copy[one_idx] = 0
            state_copy[i*3 + j] = element
            if tuple(state_copy) not in poss:
                poss.add(tuple(state_copy))
     
    poss.remove(tuple(state))
    succ_states = list(poss)
    
    for i in range(len(succ_states)):
        succ_states[i] = list(succ_states[i])
    return sorted(succ_states)



def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.
    INPUT: 
        An initial state (list of length 9)
    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along h values, number of moves, and max queue number in the format specified in the pdf.
    """
    pq = []
    states = [] #store all the visited states
    ctr = -1 #keep track of latest parent idx in states list
    visited = set(tuple(state))
    h = get_manhattan_distance(state)
    g = 0
    heapq.heappush(pq,(h+g,state,(g,h,ctr)))
    
    while len(pq) > 0:
        current = heapq.heappop(pq)
        g = current[2][0]
        current_state = current[1]
        visited.add(tuple(current_state))
        
        if current_state == goal_state:
            break
        
        states.append(current)
        ctr += 1
        succs = get_succ(current_state)
        g += 1
        for i in succs:
            if tuple(i) in visited:
                continue
            else:
                ##visited.add(tuple(current_state))
                h = get_manhattan_distance(i)
            heapq.heappush(pq,(h+g,i,(g,h,ctr)))
            
    if current_state == goal_state:
        path = [current_state]
        info = current[2]
        parent_idx = info[2]
        while parent_idx != -1:
            current = states[parent_idx]
            path.append(current[1])
            info = current[2]
            parent_idx = info[2]
    
    for idx,succ_state in enumerate(path[::-1]):
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)),"moves: {}".format(idx))
    print("Max queue length: "+str(len(pq)+1))  #str(len(states)))
    
    

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    #print_succ([2,5,1,4,0,6,7,0,3])

    #print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))

    #solve([4,3,0,5,1,6,7,2,0])
    #solve([3,4,6,0,0,1,7,2,5])
    #solve([0,4,7,1,3,0,6,2,5])
    solve([0,1,5,2,6,4,3,7,0])