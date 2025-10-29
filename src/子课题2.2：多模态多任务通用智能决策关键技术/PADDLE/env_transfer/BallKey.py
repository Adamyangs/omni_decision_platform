from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np


class BallKey(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8, rand=True):
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
                size=(splitIdx, height)
            )
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
        # print(goal_width, goal_height)
        self.put_obj(Ball(color='green'), goal_width, goal_height)
        # Place a yellow key on the left side

        # Create a vertical splitting wall
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall

        # Place a door in the wall
        doorIdx = self._rand_int(1, height - 2)
        # print(splitIdx, doorIdx)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        self.mission = "use the key to open the door and then pick up the ball"

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    if fwd_cell.type == 'ball':
                        done = True
                        reward = self._reward()
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}


class SimpleBallKeyEnv(BallKey):
    def __init__(self, size=8):
        super().__init__(size=size, rand=True)

