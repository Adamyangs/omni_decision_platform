from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np


class DoorGoalEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8, rand=False):
        self.random = rand
        super().__init__(
            grid_size=size,
            # width = 7,
            # height = 5,
            max_steps=5 * size * size
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # random_idx >= 0.5: right goal; random_idx < 0.5: left goal
        random_idx = np.random.random()
        # print("\n Random Index", random_idx)
        if random_idx >= 0.5:
            # Place a goal in the bottom-right corner
            splitIdx = 3
            goal_width = self._rand_int(splitIdx + 2, width - 1)
            goal_height = self._rand_int(1, height - 1)
            agent_height = height - 1
            self.place_agent(
                top=(0, 0),
                size=(splitIdx, agent_height))
            self.place_obj(
                obj=Key('yellow'),
                top=(0, 0),
                size=(splitIdx, height))
        else:
            splitIdx = 4
            goal_width = self._rand_int(1, splitIdx - 1)
            goal_height = self._rand_int(1, height - 1)
            agent_height = height - 1
            self.place_agent(
                top=(splitIdx + 1, 0),
                size=(width, agent_height))
            self.place_obj(
                obj=Key('yellow'),
                top=(splitIdx + 1, 0),
                size=(width, height))
        self.put_obj(Goal(), goal_width, goal_height)

        # Create a vertical splitting wall
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall

        # Place a door in the wall
        doorIdx = self._rand_int(1, height - 2)
        # print(splitIdx, doorIdx)
        self.put_obj(Door('yellow', is_locked=False), splitIdx, doorIdx)

        self.mission = "to open the door and then get to the goal"


class SimpleDoorGoalEnv(DoorGoalEnv):
    def __init__(self, size=8):
        super().__init__(size=size, rand=True)

