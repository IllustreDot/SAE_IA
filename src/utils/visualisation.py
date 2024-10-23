import matplotlib.pyplot as plt

def plot_2D(graph):
    for point in graph:
        plt.plot(point[0], point[1], 'bo-', markersize=1)  # 'bo-' means blue circles for points and solid line

    plt.title("Graph Visualization")
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")

    ax = plt.gca()

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    
    ax.spines['left'].set_linewidth(2)  
    ax.spines['bottom'].set_linewidth(2)
    
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    plt.show()

# graph=[(1,2),(3,5),(-2,-3),(-1,4)]
# visualize_graph_two_dimensions(graph)


def plot_2D_with_label(graph,labels):
    for i in range(len(graph)):
        if labels[i] == "1":
            plt.plot(graph[i][0], graph[i][1], 'bo-', markersize=1)  
        elif labels[i] == "2":
            plt.plot(graph[i][0], graph[i][1], 'ro-', markersize=1)  
        elif labels[i] == "3":
            plt.plot(graph[i][0], graph[i][1], 'go-', markersize=1) 
        else:
            plt.plot(graph[i][0], graph[i][1], 'yo-', markersize=1)  

    plt.title("Graph Visualization")
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")

    ax = plt.gca()

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    
    ax.spines['left'].set_linewidth(2)  
    ax.spines['bottom'].set_linewidth(2)
    
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    plt.show()