from main import *

'''
TO DO
DONE generate_random_solution() -> respect restrictions
DONE generate_random_solution_all_cars -> using all cars 
Restrictions:
    DONE Start and end in 0
    DONE Max k cars
    DONE Inspect only 1 time each establishment -> only_inspection_check()
    DONE Inspect all establishments 
    DONE ??? -> DONE usar sempre todos os carros

DONE mash_up_neighbour_solutions -> Neighbour 1,2, 3, ... with {p} probability each

DONE evaluate_solution() ->Evaluation function

DONE Hill Climbing -> get_hc_solution(num_iterations, log=False)
Simulated Annealing -> get_sa_solution(num_iterations, log=False)
DONE Tabu search 
Genetic Algorithms  -> !! investigar !!
'''

'''#Example solution
routes_all_wrong = {
    0: [7,8,4],
    1: [1,2,3,4,5],
    2: [6,0,8,9], 
    3: [10,11,12],
    4: [1,2,3,4],
    5: [5,6,7,8],
    6: [9,10,11,12],
    7: [1,2,3,5],
    8: [6,7,8],
    9: [4,9,10,11,12],
    10: [1,2,4,5]
}
routes_all_well_10 = {
    0: [0,7,8,4,9,10,0],
    1: [0,1,2,11,0],
    2: [0,3,12,6,0], 
    3: [0,15,0],
    4: [0,14,13,5,0]
}
routes_all_well = {
    0: [0,7,8,4,9,10,0],
    1: [0,1,2,3,5,6,0]
}
routes_repetition = {
    0: [0,7,8,4,9,10,0],
    1: [0,1,2,3,5,10,6,0]
}
routes_missing = {
    0: [0,7,8,4,9,10,0],
    1: [0,1,3,5,6,0]
}
routes_no_start = {
    0: [0,7,8,4,9,10,0],
    1: [1,3,5,6,0]
}
routes_no_end = {
    0: [0,7,8,4,9,10],
    1: [0,1,3,5,6,0]
}
'''

'''#Tests calc_wainting
print("wait arrival=9 est= 1: ",calc_wainting(9,1)) #=0
print("wait arrival=9.6 est= 1: ",calc_wainting(9.6,1)) #=0
print("wait arrival=0 est= 1: ",calc_wainting(0,1)) #=5
print("wait arrival=0.3 est= 1: ",calc_wainting(0.3,1)) #=4.7
print("wait arrival=20 est =1: ", calc_wainting(20.2,1)) #=0
print("wait arrival=20 est =1: ", calc_wainting(21,1)) #=8
print("wait arrival=20 est =1: ", calc_wainting(21.5,1)) #=7.5
'''

'''
#TESTE
print("change_establishment_car")
for j in range (1000):
    test = copy.deepcopy(routes_all_well_10)
    result = change_establishment_car(test)
    print(test)
    print(result)
    if(only_all_inspection_check(result) and all_start_end_root_check(result) and max_car_check(result)):
        print("\n")
    else:
        raise Exception("solution does not respect restriction in j={0}".format(j))
'''

'''
#TESTE
print("switch_establishment_dif_cars")
for j in range (1000):
    test = copy.deepcopy(routes_all_well_10)
    result = switch_establishment_dif_cars(test)
    print(test)
    print(result)
    if(only_all_inspection_check(result) and all_start_end_root_check(result) and max_car_check(result)):
        print("\n")
    else:
        raise Exception("solution does not respect restriction in j={0}".format(j))
'''

'''
#TESTE
print("switch_establishment_same_car")
for j in range (1000):
    test = copy.deepcopy(routes_all_well_10)
    result = switch_establishment_same_car(test)
    print(test)
    print(result)
    if(only_all_inspection_check(result) and all_start_end_root_check(result) and max_car_check(result)):
        print("\n")
    else:
        raise Exception("solution does not respect restriction in j={0}".format(j))
'''

'''
#TESTE 
for j in range(100):
    print("RANDOM SOLUTION")
    a = generate_random_solution()
    print(a)
    b = only_all_inspection_check(a)  
    if(not b):
       raise Exception("not only_all_inspection_check")
'''

'''
#TESTE random all cars
for j in range(100):
    print("RANDOM SOLUTION")
    a = generate_random_solution_all_cars()
    print(a)
    b = only_all_inspection_check(a)  
    c = all_cars_check(a)
    if(not b):
       raise Exception("not only_all_inspection_check")
    if(not c):
        raise Exception("dont use all cars")
'''

'''
test
l = 0
for i in range(5):
    bs, bv = tabu_search(routes_all_well_10, 100, 1000, 1000, 10)
    print(bs, bv)
    l += bv
print(l/5)
to do: check str error
'''

'''
print("Hill climbing:\n")
sol = get_hc_solution(10000, log=True)
if(only_all_inspection_check(sol) and all_start_end_root_check(sol) and max_car_check(sol)):
        print("\n")
else:
    raise Exception("solution does not respect restriction")
'''

'''
print("\Tabu search:\n")
sol = tabu_search(10000, max_iterations=10000, log=True)
if(only_all_inspection_check(sol) and all_start_end_root_check(sol) and max_car_check(sol)):
        print("\n")
else:
    raise Exception("solution does not respect restriction")
'''

'''
print("Simulated annealing:\n")
sol = get_sa_solution(10000, log=True)
if(only_all_inspection_check(sol) and all_start_end_root_check(sol) and max_car_check(sol)):
        print("\n")
else:
    raise Exception("solution does not respect restriction")
'''