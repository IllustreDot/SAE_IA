import matplotlib.pyplot as plt

def plot_2D(graph):
    for point in graph:
        plt.plot(point[0], point[1], 'bo-', markersize=10)  # 'bo-' means blue circles for points and solid line

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