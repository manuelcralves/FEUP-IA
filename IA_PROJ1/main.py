import numpy as np
import pandas as pd
import copy, random, math
import functools

#Make random seed 
from datetime import datetime
random.seed(datetime.now().timestamp())

#Constants
infinity = float("inf")
initial_hour = 9 

#Globals
fEvaluate = (lambda: print("undefined fEvaluate"))
def set_fEvaluate(func): 
    global fEvaluate
    fEvaluate = func
    print("fEvaluate: i was set")
    
fGenSolution = (lambda: print("undefined fGenSolution"))
def set_fGenSolution(func): 
    global fGenSolution
    fGenSolution = func
    print("set_fGenSolution: i was set")

def print_options():
    global fEvaluate, fGenSolution, establishments, n_cars
    print("fEvaluate name: ", fEvaluate.__name__)
    print("fGenSolution name: ", fGenSolution.__name__)
    print("len(establishments: )", str(len(establishments)))
    print("n_cars: ", str(n_cars))



#---------------------------------IMPORT---------------------------------
'''Return list that include root (id = 0) and {length} extra establishments
Each node --> index: id [0: utility, 1: inpection time, 2: opening hours list ]'''
def import_establishments(length):
    establishments_df = pd.read_csv(r'establishments.csv')

    size = min(establishments_df['Id'].count(),length +1)

    establishments_df_short = pd.DataFrame(establishments_df, columns=['Inspection Utility', 'Inspection Time', 'Opening Hours', 'Latitude','Longitude', 'Id']).head(size)
    establishments_df_short['Opening Hours'] = establishments_df_short['Opening Hours'].apply(lambda x: list(map(int,x.strip('[]').split(','))))

    #establishments = establishments_df_short.values.tolist()
    return establishments_df_short

def import_establishments_remove_utility0(length):
    establishments_df = pd.read_csv(r'establishments.csv')

    establishments_df = establishments_df[(establishments_df['Inspection Utility'] != 0) | (establishments_df['Id'] == 0)]

    size = min(establishments_df['Id'].count(),length +1)

    establishments_df_short = pd.DataFrame(establishments_df, columns=['Inspection Utility', 'Inspection Time', 'Opening Hours', 'Latitude','Longitude', 'Id']).head(size)
    establishments_df_short['Opening Hours'] = establishments_df_short['Opening Hours'].apply(lambda x: list(map(int,x.strip('[]').split(','))))

    #establishments = establishments_df_short.values.tolist()
    return establishments_df_short

'''Return 2D list that include root (id = 0) and {length} extra establishments
Each node --> index1: destiny index2: origin'''
def import_distances(length):
    distances_df = pd.read_csv(r'distances.csv')

    size = min(len(distances_df.index), len(distances_df.columns), length + 1)
    
    distances = distances_df.iloc[0:size, :size].values.tolist()

    return distances

'''Main import function'''
def import_data(set_size):
    global establishments, distances, n_est, n_cars
    establishments_df = import_establishments(set_size)
    establishments= establishments_df.values.tolist()
    distances = import_distances(set_size)
    n_est = len(establishments) #Number of establishments
    n_cars = math.floor(0.1 * n_est) #Number of vehicles
    return establishments_df

def import_data_remove_utility0(set_size):
    global establishments, distances, n_est, n_cars
    establishments_df = import_establishments_remove_utility0(set_size)
    establishments = establishments_df.values.tolist()
    distances = import_distances(set_size)
    n_est = len(establishments) #Number of establishments
    n_cars = math.floor(0.1 * n_est) #Number of vehicles
    return establishments_df

'''TEST<-----------------------REMOVE AFTER'''
#import_data(50)


#---------------------------------Constraints tests---------------------------------
'''Check if, in solution, every establishment is inspected and only once'''
def only_all_inspection_check(solution: dict):
    found = [False] * n_est
    found[0] = True

    for i_cars in range(len(solution)):
        for i_establishment in range(len(solution[i_cars])):
            establishment = solution[i_cars][i_establishment]
            if(establishment == 0): continue
            if(found[establishment]): 
                #print(":( There are repeated establishments)")
                return False 
            else:
                found[establishment] = True
    
    #print(":) Each establishment was only inspected once")

    if(functools.reduce(lambda x,y: x and y, found)):
        #print(":) All establishments were inspected")
        return True
    else: 
        #print(":( There are missing establishments")
        return False

'''Check if, in solution, every car stars and ends in root'''
def all_start_end_root_check(solution: dict):
    for cars in solution.values():
        if(cars[0] != 0 or cars[-1] != 0): return False
    return True

'''Check if exceeds car limit'''
def max_car_check(solution: dict):
    return (len(solution) <= n_cars)

'''Check if solution uses all cars'''
def all_cars_check(solution: dict):
    if(len(solution) != n_cars): return False
    for car_list in solution.values():
        if(car_list == [] or car_list == [0,0]): return False
    return True


#---------------------------------Evaluate Functions---------------------------------
#Return minutes
def evaluate_solution_without_timewindow(solution: dict):
    values = []
    for cars in solution.values():
        time = 0
        for i in range(1, len(cars)):
            origin = cars[i-1]
            destiny = cars[i]
            time += (distances[destiny][origin])/60 #trip time (in minutes)
            if(destiny != 0): #dont inspect origin
                time += establishments[destiny][1] #inspection time (in minutes)
        values.append(time)

    # print("VALUES: ", values)
    return max(values)

# end_time = start_time +  travel_time + waiting_time + inpection_time

#arrival time at the establishment-> in hours (float) 
#return wainting time -> in hours (float) 
def calc_wainting(arrival_time, establishment):
    timetable = establishments[establishment][2]
    arrival_time_floor = int(arrival_time)

    wainting_time = 0
    if(timetable[arrival_time_floor] == 1):
        return 0
    else:
        wainting_time += (arrival_time_floor + 1) - arrival_time

    initial_time = (arrival_time_floor +1)%24
    while(True):
        if(timetable[initial_time] == 1):
            break
        else:
            initial_time = (initial_time +1)%24
            wainting_time += 1

    return wainting_time
#Tests -> all working

#Tests done
def evaluate_car_with_timewindow(car: int, route: list) -> tuple:
    total_time = 0 #in minutes
    trip_time = 0
    insp_time = 0
    hour = 9 #in hours
    waiting_time = 0

    for i in range(1, len(route)):
        origin = route[i-1]
        destiny = route[i]

        #trip time
        trip_time_min = (distances[destiny][origin])/60
        trip_time_hour = trip_time_min/60

        trip_time += trip_time_min
        total_time += trip_time_min #trip time (in minutes)
        hour = (hour + trip_time_hour)%24

        #waiting time
        waiting_time_hour = calc_wainting(hour, destiny)
        waiting_time_min = waiting_time_hour * 60

        waiting_time += waiting_time_min #waiting time (in minutes)
        hour = (hour + waiting_time_hour)%24

        #inspection time
        if(destiny != 0): #dont inspect origin
            inspection_time_min = establishments[destiny][1]
            inspection_time_hour = inspection_time_min /60
            insp_time += inspection_time_min
            total_time += inspection_time_min #inspection time (in minutes)
            hour = (hour + inspection_time_hour)%24

    return (trip_time, waiting_time, insp_time, total_time)


def evaluate_solution_with_timewindow(solution: dict):
    values = []
    for cars in solution.values():
        total_time = 0 #in minutes
        hour = 9 #in hours
        for i in range(1, len(cars)):
            origin = cars[i-1]
            destiny = cars[i]

            #trip time
            trip_time_min = (distances[destiny][origin])/60
            trip_time_hour = trip_time_min/60

            total_time += trip_time_min #trip time (in minutes)
            hour = (hour + trip_time_hour)%24

            #waiting time
            waiting_time_hour = calc_wainting(hour, destiny)
            waiting_time_min = waiting_time_hour * 60

            total_time += waiting_time_min #waiting time (in minutes)
            hour = (hour + waiting_time_hour)%24

            #inspection time
            if(destiny != 0): #dont inspect origin
                inspection_time_min = establishments[destiny][1]
                inspection_time_hour = inspection_time_min /60

                total_time += inspection_time_min #inspection time (in minutes)
                hour = (hour + inspection_time_hour)%24

        values.append(total_time)

    #print("VALUES: ", values)
    return max(values)

#---------------------------------Neighbour Functions---------------------------------
#experimentar trocar o >2 por >3
def change_establishment_car(solution: dict):
    cars_with_establishments = [car for car in solution.keys() if len(solution[car]) > 3]
    all_cars = [car for car in solution.keys() if len(solution[car]) > 1]
    if not cars_with_establishments:
        return "Error: No cars with establishments to change"
    
    # Select a random car to change
    car_i = random.randint(0, len(cars_with_establishments)-1)
    car = cars_with_establishments[car_i]
    #print("Car: {0}".format(car))

    # Select a random establishment to change its car
    establishment_idx = random.randint(1, len(solution[car]) - 2) # exclude first and last establishments


    # Select a random car to move the establishment
    while True:
        new_car = random.choice(all_cars)
        if new_car != cars_with_establishments[car_i]:
            break
    #print("NEW Car: {0}".format(new_car))

    # Move the establishment to the new car
    establishment = solution[car].pop(establishment_idx)
    #print("Establishment: {0}".format(establishment))
    new_pos = random.randint(1, len(solution[new_car]) - 2) # exclude first position and last
    #print("NEW pos: {0}".format(new_pos))
    solution[new_car].insert(new_pos, establishment)
    
    return solution

def switch_establishment_dif_cars(solution: dict):
    cars_with_establishments = [car for car in solution.keys() if len(solution[car]) > 2]
    if len(cars_with_establishments) < 2:
        return "Error: Not enough cars with establishments to switch"
    
    # select two different cars randomly
    car1_i = random.randrange(0, len(cars_with_establishments))
    car1 = cars_with_establishments[car1_i]
    #print("Car 1: {0}".format(car1))

    cars_with_establishments.pop(car1_i)

    car2_i = random.randrange(0, len(cars_with_establishments))
    car2 = cars_with_establishments[car2_i]
    #print("Car 2: {0}".format(car2))

    # select two different establishments randomly in each car's route
    est1_index = random.randint(1, len(solution[car1])-2)
    est2_index = random.randint(1, len(solution[car2])-2)
    est1, est2 = solution[car1][est1_index], solution[car2][est2_index]

    #print("Est 1: {0}".format(est1))
    #print("Est 2: {0}".format(est2))

    # switch the establishments
    solution[car1][est1_index], solution[car2][est2_index] = est2, est1

    return solution

def switch_establishment_same_car(solution: dict):
    cars_with_establishments = [car for car in solution.keys() if len(solution[car]) > 3]
    if len(cars_with_establishments) < 1:
        return "Error: Not enough cars with establishments to switch"
    
    #Choose random car
    car_i = random.randrange(0, len(cars_with_establishments))
    car = cars_with_establishments[car_i]
    #print("Car: {0}".format(car))

    #Chose 2 random establishment
    car_solution = copy.copy(solution[car])
    establishment1_idx = random.randint(1, len(car_solution) - 2) # exclude first and last establishments
    est1 = solution[car][establishment1_idx]

    car_solution.pop(establishment1_idx)
    establishment2_idx = random.randint(1, len(car_solution) - 2)
    est2 = solution[car][establishment2_idx]

    #print("Est 1: {0}".format(est1))
    #print("Est 2: {0}".format(est2))

    # switch the establishments
    solution[car][establishment1_idx], solution[car][establishment2_idx] = est2, est1
    
    return solution

def mash_up_neighbour_solutions(solution: dict):
    if len(solution) != 1:
        i = random.randint(0,2) #generate number 1, 2 or 3
    elif len(solution) == 1:
        i = 2
    else:
        raise Exception("Invalid number of cars")
        
    if (i==0):
        return change_establishment_car(solution)
    elif (i==1):
        return switch_establishment_dif_cars(solution)
    else:
        return switch_establishment_same_car(solution)

#---------------------------------Random solution---------------------------------

def generate_random_solution():
    solution = {}

    #get a list with all ids of the establishments (excluding root)
    list_est = list(range(1, n_est))
    
    #shuffle the establishments list
    random.shuffle(list_est)  
    
    for car in range(n_cars):
        number_est=len(list_est)
        
        #get a random number of establishments to assign to each car  
        if(car == (n_cars -1)): #last iteration
            left_est = list_est[:] #rest of the list
            print("Car " + str(car))
        else:
            num_assigned = random.randint(0,number_est)
            print("Car " + str(car) + ", num_assigned: " + str(num_assigned))
            left_est = list_est[:num_assigned] #get the list with the establishments assigned
            #remove the establishments assigned to the car from the list of establishments available
            list_est = list_est[num_assigned:]
        
        #assign the random establishments to the car beggining and finishing in establishment 0         
        solution[car] = [0] +left_est + [0]
        
    return solution  

def generate_random_solution_all_cars():
    solution = {}

    #get a list with all ids of the establishments (excluding root)
    list_est = list(range(1, n_est))
    #shuffle the establishments list
    random.shuffle(list_est)  

    #each car has at least 1 establishment
    for car in range(n_cars):
        solution[car] = [0] + [list_est[0]]
        list_est = list_est[1:]
    
    for car in range(n_cars):
        number_est=len(list_est)
        
        #get a random number of establishments to assign to each car  
        if(car == (n_cars -1)): #last iteration
            left_est = list_est.copy() #rest of the list
            print("Car " + str(car))
        else:
            num_assigned = random.randint(0,number_est)
            print("Car " + str(car) + ", num_assigned: " + str(num_assigned))
            left_est = list_est[:num_assigned] #get the list with the establishments assigned
            #remove the establishments assigned to the car from the list of establishments available
            list_est = list_est[num_assigned:]
        
        #assign the random establishments to the car beggining and finishing in establishment 0         
        solution[car] += left_est + [0]
        
    return solution  

def generate_random_solution_all_cars_normalized():
    solution = {}

    list_est = list(range(1, n_est))
    #shuffle the establishments list
    random.shuffle(list_est)  

    #calculate the average number of establishments per vehicle
    avg_est_per_vehicle = len(list_est) // n_cars
    
    #each car has at least 1 establishment
    for car in range(n_cars):
        solution[car] = [0] + [list_est[0]]
        list_est = list_est[1:]
    
    for car in range(n_cars):
        #determine the number of establishments to assign to this vehicle
        if car == n_cars - 1:
            num_assigned = len(list_est)
        else:
            num_assigned = random.randint(avg_est_per_vehicle-2, avg_est_per_vehicle)
            
        #assign the random establishments to the car beginning and finishing in establishment 0
        left_est = list_est[:num_assigned]
        solution[car] += left_est + [0]
        
        #remove the establishments assigned to the car from the list of establishments available
        list_est = list_est[num_assigned:]
        
    return solution 

#--------------------------------Algorithms------------------------------------
#Hill climbing
def get_hc_solution(max_no_solution_iterations, max_iterations=infinity, log=False):
    bestSolutionEvol = [] #plot 

    total_iterations = 0
    no_sol_iteration = 0
    best_solution = fGenSolution() # Best solution after 'num_iterations' iterations without improvement
    best_score = fEvaluate(best_solution)
    bestSolutionEvol.append((total_iterations, best_score)) #plot 
    
    print(f"Init Solution:  {best_solution}, score: {best_score}")
    
    while no_sol_iteration < max_no_solution_iterations and total_iterations<max_iterations:
        no_sol_iteration += 1
        total_iterations += 1
        new_solution = copy.deepcopy(best_solution)
        new_solution = mash_up_neighbour_solutions(new_solution)
        new_score = fEvaluate(new_solution)
        
        if(new_score < best_score):
            best_solution = copy.deepcopy(new_solution)
            best_score = new_score
            no_sol_iteration = 0
            bestSolutionEvol.append((total_iterations, best_score)) #plot 
            if log:
                (print(f"Solution:       {best_solution}, score: {best_score}"))
        
    print(f"Final Solution: {best_solution}, score: {best_score}")
    return best_solution, bestSolutionEvol

#Tabu search
def tabu_search(max_no_solution_iterations, max_iterations=infinity, tabu_size=30, num_candidates=5, log=False):
    bestSolutionEvol = [] #plot 
    currentSolutionEvol = [] #plot 
    
    total_iterations = 0
    no_sol_iteration = 0

    #Global best solution
    best_solution = fGenSolution() 
    best_score = fEvaluate(best_solution)
    bestSolutionEvol.append((total_iterations, best_score)) #plot 

    #Global tabu list
    tabu_list = []
    tabu_list.append(best_solution)

    #Current solution
    current_solution = copy.deepcopy(best_solution)
    current_score = best_score
    currentSolutionEvol.append((total_iterations, current_score)) #plot 

    while no_sol_iteration < max_no_solution_iterations and total_iterations<max_iterations:
        no_sol_iteration += 1
        total_iterations += 1
        best_candidate = {}
        best_candidate_score = float("inf") #+INF

        #generate list of candidates out of current solution
        for i in range(num_candidates):
            new_candidate = copy.deepcopy(current_solution)
            new_candidate = mash_up_neighbour_solutions(new_candidate)
            new_candidate_score = fEvaluate(new_candidate)

            if((new_candidate_score < best_candidate_score) and (new_candidate not in tabu_list)):
                best_candidate = copy.deepcopy(new_candidate)
                best_candidate_score = new_candidate_score

        
        if(best_candidate != {}): 
            tabu_list.append(copy.deepcopy(best_candidate))#add best candidate to tabu list
            current_solution = copy.deepcopy(best_candidate)  #set current solution to best candidate
            current_score = best_candidate_score
            currentSolutionEvol.append((total_iterations, current_score)) #plot 

        if len(tabu_list) > tabu_size:
                tabu_list.pop(0)
        
        #update best solution
        if(current_score < best_score):
            best_solution = copy.deepcopy(current_solution)
            best_score = current_score
            no_sol_iteration = 0
            bestSolutionEvol.append((total_iterations, best_score)) #plot 
        
        if log:
            (print(f"Current solution: {current_solution}, score: {current_score}"))
            (print(f"Best solution: {best_solution}, score: {best_score}\n"))


    print(f"Final Solution: {best_solution}, score: {best_score}")
    return best_solution, bestSolutionEvol, currentSolutionEvol

#Simulated annealing
def get_sa_solution(max_no_solution_iterations, max_iterations=infinity, log=False):
    bestSolutionEvol = [] #plot 
    currentSolutionEvol = [] #plot 
    total_iterations = 0
    no_sol_iteration = 0
    temperature = 1000

    #Global best solution
    best_solution = fGenSolution()
    best_score = fEvaluate(best_solution)
    bestSolutionEvol.append((total_iterations, best_score)) #plot 

    #Current solution
    current_solution = copy.deepcopy(best_solution)
    current_score = best_score
    currentSolutionEvol.append((total_iterations, current_score)) #plot 
    

    while no_sol_iteration < max_no_solution_iterations and total_iterations<max_iterations:
        temperature = temperature * 0.999
        no_sol_iteration += 1
        total_iterations += 1
        
        new_current_solution = copy.deepcopy(current_solution)
        new_current_solution = mash_up_neighbour_solutions(new_current_solution)
        new_current_score = fEvaluate(new_current_solution)
        
        if(new_current_score < best_score): #better than global
            #update best solution
            best_solution = copy.deepcopy(new_current_solution)
            best_score = new_current_score
            no_sol_iteration = 0
            bestSolutionEvol.append((total_iterations, best_score)) #plot 
            #update current solution
            current_solution = copy.deepcopy(new_current_solution)
            current_score = new_current_score
            currentSolutionEvol.append((total_iterations, current_score)) #plot 
        elif(new_current_score < current_score): #better than current and not global
            #update current solution
            current_solution = copy.deepcopy(new_current_solution)
            current_score = new_current_score
            currentSolutionEvol.append((total_iterations, current_score)) #plot 
        else: #worse than current and global
            delta = - (new_current_score - current_score) #<0
            p = np.exp(- delta / temperature)
            random_p = random.random()
            if(p > random_p):
                #update current solution
                current_solution = copy.deepcopy(new_current_solution)
                current_score = new_current_score   
                currentSolutionEvol.append((total_iterations, current_score)) #plot  
        if log:
            (print(f"Current solution: {current_solution}, score: {current_score}"))
            (print(f"Best solution: {best_solution}, score: {best_score}\n"))

        
    print(f"Final Solution: {best_solution}, score: {best_score}")
    return best_solution, bestSolutionEvol, currentSolutionEvol


#--------------------------------Genetic------------------------------------

def mutation(sol, prob_mut):
    """Performs a mutation operation on a solution with a given probability"""
    # Iterate over each route in the solution
    for car in sol:
        if (len(sol[car])>=4):
        # Check if the mutation should be performed for this route
            if random.random() < prob_mut:
            # Generate two random indices to swap within the route
              i, j = random.sample(range(1, len(sol[car])-1), 2)
            # Swap the establishments at the two indices
              sol[car][i], sol[car][j] = sol[car][j], sol[car][i]
    return sol

def create_population(population_size):
    solutions = []
    for i in range(population_size):
        solutions.append(fGenSolution())
    
    return solutions    

def evaluate_fitness(population):
    fitness = []
    for i in range(len(population)):
        fitness.append(fEvaluate(population[i])) #se for mais util usar a funçao pra avaliar soluçao sem timewindow
        
    return fitness    

def tournament_selection(population, tournament_size):
    population_copy=copy.deepcopy(population)
    best_solution = population_copy[0]
    best_value = fEvaluate(population_copy[0])
    
    for i in range(tournament_size):
        index = random.randrange(0,len(population_copy))
        score = fEvaluate(population_copy[index])
        if score>best_value:
            best_value = score
            best_solution=population_copy[index]
        del population_copy[index]

    return best_solution


def roullete_selection(population):
    total_sum = 0
    sel_prob = []
    for solution in population:
        total_sum=total_sum + fEvaluate(solution)

    for solution in population:
        sel_prob.append(fEvaluate(solution)/total_sum)

    return population[np.random.choice(len(population), p = sel_prob)]


def crossover(sol1, sol2):
    """
    Performs crossover between two solutions.
    """

    # Get the number of cars and establishments
    num_cars = len(sol1)
    attempts = 1

    offspring1 = {}
    offspring2 = {}

    while((only_all_inspection_check(offspring1) and only_all_inspection_check(offspring2))==False and attempts>0):
    # Choose a random crossover point
        crossover_point = random.randint(1, num_cars-1)

    # Create the offspring solutions
       # offspring1 = {}
       # offspring2 = {}

    # Perform the crossover for each car
        offspring1 = dict(list(sol1.items())[0:crossover_point])
        offspring2 = dict(list(sol2.items())[0:crossover_point])
    
        offspring1.update(dict(list(sol2.items())[crossover_point:]))
        offspring2.update(dict(list(sol1.items())[crossover_point:]))
    
        attempts=attempts-1
  
    if (attempts==0):
        return sol1,sol2
    else:
        return offspring1,offspring2
        
       

def get_index_leats_fittest(population):
    least_fittest_idx = 0
    least_fittest_value = fEvaluate(population[0])
    for i in range(1, len(population)):
        value = fEvaluate(population[i])
        if (value<least_fittest_value):
            least_fittest_idx = i
            least_fittest_value= fEvaluate(population[i])
    return least_fittest_idx        


def genetic_algorithm(population_size, max_no_solution_iterations, max_iterations=infinity, crossover_func=crossover, mutation_func=mutation, log=False):
    bestSolutionEvol = [] #plot 
    currentSolutionEvol = [] #plot 
    
    total_iterations=0
    no_sol_iterations=0
    population = create_population(population_size)
    
    best_sol = population[0]
    best_value = fEvaluate(population[0])
    best_sol_generation = 0
    
    generation_number=0
    
    print(f"Initial solution: {best_sol}, score: {best_value}")
    bestSolutionEvol.append((total_iterations, best_value)) #plot 
    
    while no_sol_iterations<max_no_solution_iterations and total_iterations<max_iterations:
        no_sol_iterations+=1
        total_iterations+=1
        generation_number += 1
        
        tournament_winner = tournament_selection(population,4)
        roullete_winner = roullete_selection(population)
        
        crossover_1, crossover_2 = crossover_func(tournament_winner, roullete_winner)
        
        if random.randint(0,10)<5:
            population[get_index_leats_fittest(population)] = mutation_func(tournament_winner, random.random())
            population[get_index_leats_fittest(population)] = mutation_func(roullete_winner, random.random())
        else:
            population[get_index_leats_fittest(population)]=crossover_1
            population[get_index_leats_fittest(population)]=crossover_2
            
        #Finding best solution of population
        
        greatest_fittest = population[0]
        greatest_fittest_value = fEvaluate(population[0])
        
        for i in range(1,len(population)):
            score = fEvaluate(population[i])
            if (score < greatest_fittest_value):
                greatest_fittest_value=fEvaluate(population[i])
                greatest_fittest=population[i]
                best_sol_generation=generation_number

        currentSolutionEvol.append((total_iterations, greatest_fittest_value)) #plot 

        if(greatest_fittest_value < best_value):
            best_value = greatest_fittest_value
            best_sol = greatest_fittest
            bestSolutionEvol.append((total_iterations, best_value)) #plot 
            

        
    print(f"  Final solution: {best_sol}, score: {best_value}")
    print(f"  Found on generation {best_sol_generation}")        
             
    return best_sol, bestSolutionEvol, currentSolutionEvol    

'''
print("GENETIC ALGORITHM")
genetic_algorithm(10, 30, infinity, crossover, mutation, True)
'''





