import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import cycle
import random
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyArrow

def animate(env, actions):
    fig, ax = plt.subplots()

    # Generate distinct colors
    colors = list(plt.cm.colors.CSS4_COLORS.values())
    color_cycler = cycle(colors)
    
    line_color = {k: next(color_cycler) for k in actions}
    
    def init():
        ax.scatter(env[:, 0], env[:, 1])
        for i, coord in enumerate(env):
            ax.text(coord[0] + random.uniform(-0.5, 0.5), coord[1] + random.uniform(-0.5, 0.5), str(i), fontsize=12, ha='right')
        ax.scatter([0], [0], color='red', label='Depot')  # depot point at (0,0)
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 10)
        
        # Adding legend
        for k, color in line_color.items():
            ax.plot([], [], color=color, label=f'Line {k}')
        ax.legend(loc='upper right')

    # flatten the action points for sequential animation
    flat_actions = [(k, pt) for k, v in actions.items() for pt in zip(v, v[1:])]

    def update(frame):
        key, (pt1, pt2) = flat_actions[frame]
        ax.set_title(f'Animating for key: {key}')
        pt1_coords = [0, 0] if pt1 == -1 else env[pt1]
        pt2_coords = [0, 0] if pt2 == -1 else env[pt2]
        ax.plot([pt1_coords[0], pt2_coords[0]], [pt1_coords[1], pt2_coords[1]], color=line_color[key])

        # Draw arrow on line
        arrow = FancyArrow(pt1_coords[0], pt1_coords[1], pt2_coords[0] - pt1_coords[0], pt2_coords[1] - pt1_coords[1],
                           color=line_color[key], width=0.02, length_includes_head=True, head_width=0.1)
        arrow_collection = PatchCollection([arrow])
        ax.add_collection(arrow_collection)

    ani = animation.FuncAnimation(fig, update, frames=len(flat_actions), init_func=init, interval=1000, repeat=False)
    plt.show()