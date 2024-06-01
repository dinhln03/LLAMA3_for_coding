#!/usr/bin/env python3
from matplotlib.patches import Circle, Rectangle, ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from math import floor

Colors = ['green', 'purple', 'orange', 'red', 'blue', 'yellow']


class Animation:
    def __init__(self, my_map, starts, goals, paths, predictions):
        self.my_map = np.flip(np.transpose(my_map), 1)
        self.predictions = predictions
        self.starts = []
        for start in starts:
            self.starts.append((start[1], len(self.my_map[0]) - 1 - start[0]))
        self.goals = []
        for goal in goals:
            self.goals.append((goal[1], len(self.my_map[0]) - 1 - goal[0]))
        self.paths = []
        if paths:
            for path in paths:
                self.paths.append([])
                for loc in path:
                    self.paths[-1].append((loc[1], len(self.my_map[0]) - 1 - loc[0]))

        aspect = len(self.my_map) / len(self.my_map[0])

        self.fig = plt.figure(frameon=False, figsize=(4 * aspect, 4))
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
        # self.ax.set_frame_on(False)

        self.patches = []
        self.artists = []
        self.agents = dict()
        self.agent_names = dict()
        self.goal_predictions = dict()
        self.agent_goal_connections = dict()
        # create boundary patch

        x_min = -0.5
        y_min = -0.5
        x_max = len(self.my_map) - 0.5
        y_max = len(self.my_map[0]) - 0.5
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(np.arange(x_min, x_max, 1))
        plt.yticks(np.arange(y_min, y_max, 1))
        plt.grid(color='0.85')

        self.patches.append(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, facecolor='none', edgecolor='gray'))
        for i in range(len(self.my_map)):
            for j in range(len(self.my_map[0])):
                if self.my_map[i][j]:
                    self.patches.append(Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor='gray', edgecolor='gray'))

        self.T = 0
        # draw goals
        for i, goal in enumerate(self.goals):
            goal_color = Colors[i % len(Colors)]
            self.patches.append(Rectangle((goal[0] - 0.25, goal[1] - 0.25), 0.5, 0.5, facecolor=goal_color,
                                          edgecolor='black', alpha=0.5))
        
        # create agents
        for a in range(len(self.paths)):
            name = str(a)
            self.agents[a] = Circle((starts[a][0], starts[a][1]), 0.3, facecolor=Colors[a % len(Colors)],
                                    edgecolor='black')
            self.agents[a].original_face_color = Colors[a % len(Colors)]
            self.patches.append(self.agents[a])
            self.T = max(self.T, len(paths[a]) - 1)
            self.agent_names[a] = self.ax.text(starts[a][0], starts[a][1] + 0.25, name)
            self.agent_names[a].set_horizontalalignment('center')
            self.agent_names[a].set_verticalalignment('center')
            self.artists.append(self.agent_names[a])

            # connections & predictions
            self.goal_predictions[a] = dict()
            self.agent_goal_connections[a] = dict()
            for i, goal in enumerate(self.goals):
                goal_color = Colors[i % len(Colors)]
                self.goal_predictions[a][i] = self.ax.text(goal[0], goal[1], str(i))
                self.goal_predictions[a][i].set_horizontalalignment('center')
                self.goal_predictions[a][i].set_verticalalignment('center')
                self.artists.append(self.goal_predictions[a][i])
                self.agent_goal_connections[a][i] = plt.Line2D((start[1], goal[0]), (len(self.my_map[0]) - 1 - start[0], goal[1]), lw=2.5, color = goal_color)
                self.artists.append(self.agent_goal_connections[a][i])

        self.animation = animation.FuncAnimation(self.fig, self.animate_func,
                                                 init_func=self.init_func,
                                                 frames=int(self.T + 1) * 10,
                                                 interval=100,
                                                 blit=True)

    def save(self, file_name, speed):
        self.animation.save(
            file_name,
            fps=10 * speed,
            dpi=200,
            savefig_kwargs={"pad_inches": 0})

    @staticmethod
    def show():
        plt.show()

    def init_func(self):
        for p in self.patches:
            self.ax.add_patch(p)
        for a in self.artists:
            self.ax.add_artist(a)
        return self.patches + self.artists

    def animate_func(self, t):
        # per ogni agente
        for a in range(len(self.paths)):
            pos = self.get_state(t / 10, self.paths[a])
            self.agents[a].center = (pos[0], pos[1])
            self.agent_names[a].set_position((pos[0], pos[1] + 0.5))
            # per ogni goal
            for i in self.agent_goal_connections[a]:
                timestep = floor(t/10)
                if timestep not in self.predictions[a]:
                    continue

                prediction = self.predictions[a][timestep][i]
                # Linee
                self.agent_goal_connections[a][i].set_data([pos[0], self.goals[i][0]], [pos[1], self.goals[i][1]])
                self.agent_goal_connections[a][i].set_alpha(prediction)
                # Percentuali
                self.goal_predictions[a][i].set_text("{:.2f}".format(prediction*100))
                self.goal_predictions[a][i].set_position([(pos[0] + self.goals[i][0])/2, (pos[1] + self.goals[i][1])/2])
                self.goal_predictions[a][i].set_alpha(prediction)


        # reset all colors
        for _, agent in self.agents.items():
            agent.set_facecolor(agent.original_face_color)

        # check drive-drive collisions
        agents_array = [agent for _, agent in self.agents.items()]
        for i in range(0, len(agents_array)):
            for j in range(i + 1, len(agents_array)):
                d1 = agents_array[i]
                d2 = agents_array[j]
                pos1 = np.array(d1.center)
                pos2 = np.array(d2.center)
                if np.linalg.norm(pos1 - pos2) < 0.7:
                    d1.set_facecolor('red')
                    d2.set_facecolor('red')
                    print("COLLISION! (agent-agent) ({}, {}) at time {}".format(i, j, t/10))

        return self.patches + self.artists

    @staticmethod
    def get_state(t, path):
        if int(t) <= 0:
            return np.array(path[0])
        elif int(t) >= len(path):
            return np.array(path[-1])
        else:
            pos_last = np.array(path[int(t) - 1])
            pos_next = np.array(path[int(t)])
            pos = (pos_next - pos_last) * (t - int(t)) + pos_last
            return pos
