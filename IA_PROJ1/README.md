# IA_PROJ1

# Libraries

Install the following libraries if you don't have them:

|     Name    |   Command   |
| ----------- | ----------- |
| matplotlib | `pip install matplotlib`|
| numpy       | `pip install numpy` |
| pandas       | `pip install pandas` |
| tkinter       | `pip install tkinter` |

# Compilation and Execution

Run `python3 interface.py` on the terminal.

# Usage

After running the command above, you will be presented with a new window and there is the meaning of each input:

1. Number establishments - choose the size of the dataset you want to use (the number of establishments implies the number of cars)
2. Random solution generator - choose the random solution generator (equal distribution or random distribution)
3. Evaluation function - choose the evaluation function with or without timewindow
4. Algorithm name - choose the algorithm that you want to test
5. Max total iteration - choose the maximum number of total iterations that can be done by the algorithm
6. Max iteration without improvement - choose the maximum number of iterations without any improvement until the algorithm stops

In 4 if you choose tabu search:

8. Tabu list size - size of the tabu list with the solutions already visited
9. Number candidates - nuber of candidates to consider in the evaluation in each iteration
