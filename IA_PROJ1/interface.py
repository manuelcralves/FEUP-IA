from main import *
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import gui
import matplotlib.pyplot as plt
import timeit

def disableChildren(parent):
    for child in parent.winfo_children():
        wtype = child.winfo_class()
        if wtype not in ('Frame','Labelframe','TFrame','TLabelframe'):
            child.configure(state='disable')
        else:
            disableChildren(child)

def enableChildren(parent):
    for child in parent.winfo_children():
        wtype = child.winfo_class()
        if wtype not in ('Frame','Labelframe','TFrame','TLabelframe'):
            child.configure(state='normal')
        else:
            enableChildren(child)

n_establishments_dict = {"Very Small - 20": 20, "Small - 100": 100, "Medium - 500":500, "Large - 1000": 1000}
sol_type_dict = {"All cars used - random distribution": generate_random_solution_all_cars, "All cars used - equal distribution": generate_random_solution_all_cars_normalized}
eval_func_dict = {"With time_window": evaluate_solution_with_timewindow, "Without time_window": evaluate_solution_without_timewindow}
utility_dispose_dict = {'1': import_data_remove_utility0, '0': import_data}

def get_data():
    n_est = n_establishments_dict.get(n_establishments_drop.get(), 0)
    sol_type = random_solution_drop.get()
    eval_func = eval_func_drop.get()
    utility_dispose = utility_check_var.get()
    alg = alg_drop.get()

    max_iter = int(max_iter_box.get())
    if(max_iter == -1): max_iter=infinity
    max_no_iter = int(max_no_iter_box.get())
    if(max_no_iter == -1): max_no_iter=infinity

    tabu_size = int(tabu_size_box.get())
    num_candidates = int(tabu_cand_box.get())
    if(n_est == 0 or sol_type=="" or eval_func=="" or alg==""):
        tk.messagebox.showwarning(title= "ERROR", message="You didn't fill all the boxes")
        return
    
    population_size = int(population_size_box.get())

    #Import data
    data = utility_dispose_dict[utility_dispose](n_est)
    
    #Evaluate function
    set_fEvaluate(eval_func_dict[eval_func])

    #Solution generator
    set_fGenSolution(sol_type_dict[sol_type])

    init = timeit.default_timer()
    
    #Call algorithm
    if(alg == "Hill climbing"):
        solution, bestSolutionEvol = get_hc_solution(max_no_iter, max_iter, True)
        finish = round(timeit.default_timer() - init, 3)
        gui.import_establishments(data, eval_func, finish)

        #print("bestSolutionEvol: ", bestSolutionEvol)
        xb, yb = zip(*bestSolutionEvol)
        gui.plot_best_evol(xb, yb, solution=solution)

        print_options()
    elif(alg == "Simulated Annealing"):
        solution, bestSolutionEvol, currentSolutionEvol = get_sa_solution(max_no_iter, max_iter, True)
        finish = round(timeit.default_timer() - init, 3)
        gui.import_establishments(data, eval_func, finish)

        x, y = zip(*bestSolutionEvol)
        xc, yc = zip(*currentSolutionEvol)
        gui.plot_best_evol(x, y, xc, yc, solution=solution)

        #gui.plot(solution)
        print_options()
    elif(alg == "Tabu search"):
        solution, bestSolutionEvol, currentSolutionEvol = tabu_search(max_no_iter, max_iter, tabu_size, num_candidates, True)
        finish = round(timeit.default_timer() - init, 3)
        gui.import_establishments(data, eval_func, finish)

        x, y = zip(*bestSolutionEvol)
        xc, yc = zip(*currentSolutionEvol)
        gui.plot_best_evol(x, y, xc, yc, solution=solution)

        #gui.plot(solution)
        print_options()
    elif(alg == "Genetic"):
        solution, bestSolutionEvol, currentSolutionEvol = genetic_algorithm(population_size, max_no_iter, max_iter, log=True)
        finish = round(timeit.default_timer() - init, 3)
        gui.import_establishments(data, eval_func, finish)

        x, y = zip(*bestSolutionEvol)
        xc, yc = zip(*currentSolutionEvol)
        gui.plot_best_evol(x, y, xc, yc, solution=solution)

        #gui.plot(solution)
        
    

def calc_n_car():
    ne = n_establishments_dict.get(n_establishments_drop.get(), 0)
    nc = math.floor(0.1 * ne)
    n_cars_value.configure(text=str(nc))


root = tk.Tk()

#root.geometry("800x500")
root.title("ASAE Problem - Minimize Travel Time (3A)")

frame = tk.Frame(root)
frame.pack()
frame.columnconfigure(0, weight=1)
frame.columnconfigure(1, weight=1)
frame.columnconfigure(2, weight=1)


#Frame 1
problem_frame = tk.LabelFrame(frame, text="Problem")
problem_frame.grid(row=0, column=0, pady= 10, padx = 20, columnspan=3, sticky="news")
problem_frame.columnconfigure(1, weight=1)


n_establishments_label = tk.Label(problem_frame, text="Number establishments:")
n_establishments_label.grid(row = 0, column= 0)
##
n_establishments_drop = ttk.Combobox(problem_frame, values= list(n_establishments_dict.keys()))
n_establishments_drop.grid(row = 1, column= 0)
n_establishments_drop.bind('<<ComboboxSelected>>', lambda x: calc_n_car())

n_cars_label = tk.Label(problem_frame, text="Number cars:")
n_cars_label.grid(row = 0, column= 1)
##
n_cars_value = tk.Label(problem_frame, text="0")
n_cars_value.grid(row = 1, column= 1)

random_solution_label = tk.Label(problem_frame, text="Random solution generator:")
random_solution_label.grid(row = 0, column = 2)
##
random_solution_drop = ttk.Combobox(problem_frame, values=["All cars used - random distribution", "All cars used - equal distribution"])
random_solution_drop.grid(row = 1, column = 2)

eval_func_label = tk.Label(problem_frame, text="Evaluation function:")
eval_func_label.grid(row = 2, column= 0)
##
eval_func_drop = ttk.Combobox(problem_frame, values=["With time_window", "Without time_window"])
eval_func_drop.grid( row= 3, column = 0)

utility_check_var = tk.StringVar(value=False)
##
utility_check = tk.Checkbutton(problem_frame, text="Don't inspect establishments with utility equal to 0", variable=utility_check_var, onvalue=True, offvalue=False)
utility_check.grid(row=3, column = 1, columnspan=2)

#Frame 2
algorithm_frame = tk.LabelFrame(frame, text="Algorithm")
algorithm_frame.grid(row=1, column=0, padx = 20, columnspan=3, sticky="news")

alg_label = tk.Label(algorithm_frame, text="Algorithm name:")
alg_label.grid(row = 0, column= 0)
##
alg_drop = ttk.Combobox(algorithm_frame, values=["Hill climbing", "Tabu search", "Simulated Annealing", "Genetic"])
alg_drop.grid( row= 1, column = 0)


max_iter_label = tk.Label(algorithm_frame, text="Max total iteration:")
max_iter_label.grid(row = 0, column = 1)
##
max_iter_box = tk.Spinbox(algorithm_frame, from_=-1, to=infinity)
max_iter_box.grid(row = 1, column=1)

max_no_iter_label = tk.Label(algorithm_frame, text="Max iteration without improvement:")
max_no_iter_label.grid(row = 0, column = 2)
##
max_no_iter_box = tk.Spinbox(algorithm_frame, from_=-1, to=infinity)
max_no_iter_box.grid(row = 1, column=2)

#Frame 3
tabu_frame = tk.LabelFrame(frame, text="Tabu search")
tabu_frame.grid(row=2, column=0, pady= 10, padx = 20, columnspan=1)

#tabu_frame.columnconfigure(0, weight=1)

tabu_size_label = tk.Label(tabu_frame, text="Tabu list size:")
tabu_size_label.grid(row = 0, column = 0)
##
tabu_size_box = tk.Spinbox(tabu_frame, from_=1, to=infinity)
tabu_size_box.grid(row= 1, column = 0)

tabu_cand_label = tk.Label(tabu_frame, text="Number candidates:")
tabu_cand_label.grid(row = 2, column = 0)
##
tabu_cand_box = tk.Spinbox(tabu_frame, from_=1, to=infinity)
tabu_cand_box.grid(row= 3, column = 0)

for child in tabu_frame.winfo_children():
    child.configure(state='disable')

#Frame 4
genetic_frame = tk.LabelFrame(frame, text="Genetic algorithm")
genetic_frame.grid(row=2, column=1, pady= 10, padx = 20, columnspan=1)

population_size_label = tk.Label(genetic_frame, text="Population size:")
population_size_label.grid(row = 0, column = 0)
##
population_size_box = tk.Spinbox(genetic_frame, from_=1, to=infinity)
population_size_box.grid(row= 1, column = 0)

for child in genetic_frame.winfo_children():
    child.configure(state='disable')

for widget in (problem_frame.winfo_children() + algorithm_frame.winfo_children() + tabu_frame.winfo_children() + genetic_frame.winfo_children()):
    widget.grid_configure(padx=10, pady=5)


#Button
button = tk.Button(frame, text="Start", command=get_data)
button.grid(row = 3, column= 0, columnspan= 3,sticky="news", padx= 20, pady = 10)

def comboselected(x):
    if alg_drop.get()=="Tabu search":
        enableChildren(tabu_frame) 
        disableChildren(genetic_frame)
    elif alg_drop.get()=="Genetic":
        enableChildren(genetic_frame) 
        disableChildren(tabu_frame)
    else:
        disableChildren(genetic_frame)
        disableChildren(tabu_frame)

alg_drop.bind('<<ComboboxSelected>>', comboselected) 

root.mainloop()