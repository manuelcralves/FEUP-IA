import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import main
import matplotlib.patches as patches
import tkinter.ttk as ttk

eval = ""
time = 0

def import_establishments(data, eval_func, finish):
    # iterate over the rows of the dataframe and add the latitude and longitude of each establishment to the coords dictionary
    for index, row in data.iterrows():
        coords[row['Id']] = (row['Latitude'], row['Longitude'])
    #establishments = data.values.tolist()
    #return establishments
    global eval
    eval = eval_func
    global time
    time = finish

# define the coordinates of each establishment
coords = {}

#solution = tabu_search(200,500,10,10,True)

# keep track of the colors used for each establishment
def plot_routes(solution, selected_car=None):
    # keep track of the colors used for each establishment
    establishment_colors = {0: 'black'}

    car_arrows = {}
    for car, route in solution.items():
        if selected_car is not None and car != selected_car:
            continue
        color = 'C' + str(car)
        arrows = []
        for i in range(len(route)-1):
            start = coords[route[i]]
            end = coords[route[i+1]]
            dx = (end[1] - start[1])
            dy = (end[0] - start[0])
            arrow = plt.arrow(start[1], start[0], dx, dy, color=color,
                              length_includes_head=True, head_width=0.006)
            arrows.append(arrow)
            # update the establishment_colors dictionary
            if i > 0:
                establishment_colors[route[i]] = color
        car_arrows[car] = arrows

    # plot the establishments with their assigned colors and IDs
    for i, coord in coords.items():
        color = establishment_colors.get(i, 'black') # default color is black
        if selected_car is None or i in solution[selected_car]:
            plt.scatter(coord[1], coord[0], color=color)
            plt.text(coord[1] + 0.01, coord[0] + 0.01, str(i), fontsize=8, color=color)
        #plt.scatter(coord[1], coord[0], color=color)
        #plt.text(coord[1] + 0.01, coord[0] + 0.01, str(i), fontsize=8, color=color)

    # add a legend
    '''
    legend_elements = []
    for car in solution:
        color = 'C' + str(car)
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=f'Car {car}'))

    plt.legend(handles=legend_elements)
    '''
    
    # show the plot
    # plt.show()

def plot(solution):
    # create a Tkinter window
    window = tk.Tk()
    window.title("Vehicle Routing Problem")

    # create a Figure object and add a subplot
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)

    # create a canvas and pack it into the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack()

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # create a drop-down menu with the car numbers
    car_options = list(solution.keys())
    car_options = ["All cars"] + car_options
    var = tk.StringVar(value=car_options[0])
    var.trace_add('write', lambda *args: update_plot())
    drop_down = tk.OptionMenu(window, var, *car_options)
    drop_down.pack()

    # create a button to update the plot
    def update_plot():
        # clear the previous plot
        ax.clear()

        # get the selected car number
        selected_car = var.get()
        if selected_car == "All cars":
            plot_routes(solution)
        else:
            car = int(selected_car)
            # plot the route for the selected car
            plot_routes(solution, car)

        # redraw the canvas
        canvas.draw()

    def show_stats():
        # create a new Tkinter window
        stats_window = tk.Toplevel()
        stats_window.title("Statistics")

        # create a Treeview widget to display the table
        if eval == "With time_window":
            table = ttk.Treeview(stats_window, columns=("establishments", "trip_time", "wait_time", "insp_time", "total_time"))
            table.heading("#0", text="Car")
            table.heading("establishments", text="Establishments visited")
            table.heading("trip_time", text="Trip Time")
            table.heading("wait_time", text="Wait Time")
            table.heading("insp_time", text="Inspection Time")
            table.heading("total_time", text="Total Time")
            table.column("establishments", width=500)  # set the width of the column
            table.pack()
        else:
            table = ttk.Treeview(stats_window, columns=("establishments", "trip_time", "insp_time", "total_time"))
            table.heading("#0", text="Car")
            table.heading("establishments", text="Establishments visited")
            table.heading("trip_time", text="Trip Time")
            table.heading("insp_time", text="Inspection Time")
            table.heading("total_time", text="Total Time")
            table.column("establishments", width=500)  # set the width of the column
            table.pack()

        # insert the data into the table
        max_time = 0
        for car, route in solution.items():
            car_trip_time, car_wait_time, car_insp_time, total_time = main.evaluate_car_with_timewindow(car, route)
            establishments = [str(e) for e in route]
            tags = ()
            if eval == "With time_window":
                total_time += car_wait_time
                table.insert("", "end", text=f"Car {car}", values=(", ".join(establishments), car_trip_time, car_wait_time, car_insp_time, total_time), tags=tags)
            else:
                table.insert("", "end", text=f"Car {car}", values=(", ".join(establishments), car_trip_time, car_insp_time, total_time), tags=tags)
            if total_time > max_time:
                max_time = total_time

        time_label = tk.Label(stats_window, text=f"Total time to run the algorithm: {time} seconds")
        time_label.pack()
        
        solution_label = tk.Label(stats_window, text=f"Best solution: {max_time}")
        solution_label.pack()

        # create a Label widget to show the number of cars used
        num_cars = len(solution)
        cars_label = tk.Label(stats_window, text=f"Number of cars used: {num_cars}")
        cars_label.pack()

        # create a Label widget to show the total number of establishments inspected
        all_establishments = set()
        for route in solution.values():
            all_establishments.update(route)
        num_establishments = len(all_establishments) - 1
        establishments_label = tk.Label(stats_window, text=f"Number of establishments inspected: {num_establishments}")
        establishments_label.pack()



    # create a button to exit the program
    def close():
        window.destroy()

    # create a button to show the statistics table
    button_stats = tk.Button(window, text="Statistics", command=show_stats)
    button_stats.pack()
    
    button_exit = tk.Button(window, text="Exit", command=close)
    button_exit.pack()

    # plot the initial routes of all cars
    plot_routes(solution)

    # start the Tkinter event loop
    tk.mainloop()

def plot_best_evol(xb, yb, xc=[], yc=[], solution={}):
    gui_root = tk.Tk()
    # create a Tkinter window
    window1 = tk.Frame(gui_root)
    window1.pack(side="left", padx=10, pady=10)
    gui_root.title("Solution evaluation")

    plt.switch_backend('agg')
    fig = plt.figure(figsize=(6, 6), dpi=100)
    #ax = fig.add_subplot(111)

    canvas = FigureCanvasTkAgg(fig, master=window1)
    canvas.get_tk_widget().pack()

    plt.xlabel("Iterations")
    plt.ylabel("Solution evaluation")

    plt.plot(xb, yb, 'ro')

    if(xc != [] and yc !=[]):
        plt.plot(xc, yc, 'b')

    if(solution!={}):
        plot(solution)
    
    #plt.show()
    tk.mainloop()


#plot(solution)