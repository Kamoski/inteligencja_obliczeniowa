import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_route(route, env):
    points = env.points

    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], s=200, c='red', marker='o', label='Locations')
    plt.scatter(points[0, 0], points[0, 1], s=200, c='blue', marker='s', label='Depot')

    for i, txt in enumerate(range(env.num_points)):
        plt.annotate(txt, (points[i, 0] + 1, points[i, 1] + 1))

    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define as many colors as needed

    for car_id, car_route in route.items():
        car_route_points = [points[point] for point in car_route]
        car_route_lines = np.array(car_route_points)
        plt.plot(car_route_lines[:, 0], car_route_lines[:, 1], '--', color=colors[car_id % len(colors)], label=f'Route {car_id}')
    
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.title('Vehicle Routing')
    plt.show()

def animate_route(route, env):
    points = env.points

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(points[:, 0], points[:, 1], s=200, c='red', marker='o', label='Locations')
    ax.scatter(points[0, 0], points[0, 1], s=200, c='blue', marker='s', label='Depot')

    for i, txt in enumerate(range(env.num_points)):
        ax.annotate(txt, (points[i, 0] + 1, points[i, 1] + 1))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define as many colors as needed

    # Initialize empty lines for each car
    lines = [ax.plot([], [], '--', color=colors[car_id % len(colors)], label=f'Route {car_id}')[0] for car_id in range(env.max_vehicles)]

    def animate(i):
        for car_id in range(env.max_vehicles):
            if car_id in route.keys():
                car_route_points = [points[point] for point in route[car_id][:i+1]]  # Get points visited by the car up to i
                car_route_lines = np.array(car_route_points)
                lines[car_id].set_data(car_route_lines[:, 0], car_route_lines[:, 1])
        return lines

    ani = animation.FuncAnimation(fig, animate, frames=len(route[0]), interval=500, blit=True)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.title('Vehicle Routing')
    plt.show()