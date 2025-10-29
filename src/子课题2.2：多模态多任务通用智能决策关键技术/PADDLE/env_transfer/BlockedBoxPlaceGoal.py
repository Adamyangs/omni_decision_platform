from gym_minigrid.minigrid import Ball, Key, Box, Wall, Floor
from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.register import register
import numpy as np


class BlockedBoxPlaceGoalEnv(RoomGrid):
    """
    Unlock a door blocked by a ball, then pick up a box
    in another room
    """

    def __init__(self, seed=0, room_size=6):
        room_size = room_size
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=5*room_size**2,
            seed=seed
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        color = "yellow"

        random_idx = np.random.random()

        if random_idx >= 0.5:
            # Add a box to the room on the right
            obj, _ = self.add_object(1, 0, kind="ball", color="green")
            # Make sure the two rooms are directly connected by a locked door
            door, pos = self.add_door(0, 0, 0, locked=True, color=color)
            floor_obj, self.pos_goal = self.add_object(1, 0, kind='floor', color="red")
            # for i in range(int(self.room_size / 2)):
            #     _, _ = self.add_object(1, 0, kind='wall')
            #     _, _ = self.add_object(0, 0, kind='wall')
            # Block the door with a ball
            self.blocked_obj = Ball(color)
            self.grid.set(pos[0] - 1, pos[1], self.blocked_obj)
            # Add a key to unlock the door
            self.add_object(0, 0, 'box', door.color)
            blocked_place_obj, self.blocked_place_pos_goal = self.add_object(0, 0, kind='floor', color=color)
            self.place_agent(0, 0)
        else:
            # Add a box to the room on the right
            obj, _ = self.add_object(0, 0, kind="ball", color="green")
            # Make sure the two rooms are directly connected by a locked door
            door, pos = self.add_door(0, 0, 0, locked=True, color=color)
            floor_obj, self.pos_goal = self.add_object(0, 0, kind='floor', color="red")
            # for i in range(int(self.room_size / 2)):
            #     _, _ = self.add_object(1, 0, kind='wall')
            #     _, _ = self.add_object(0, 0, kind='wall')
            # Block the door with a ball
            self.blocked_obj = Ball(color)
            self.grid.set(pos[0] + 1, pos[1], self.blocked_obj)
            # Add a key to unlock the door
            self.add_object(1, 0, 'box', door.color)
            blocked_place_obj, self.blocked_place_pos_goal = self.add_object(1, 0, kind='floor', color=color)
            self.place_agent(1, 0)

        self.obj = obj
        self.floor_obj = floor_obj
        self.blocked_place_obj = blocked_place_obj
        self.place_blocked = False
        self.mission = "pick up the %s %s" % (obj.color, obj.type)
        # print(self.mission)

    def add_object(self, i, j, kind=None, color=None):
        """
        Add a new object to room (i, j)
        """

        if kind == None:
            kind = self._rand_elem(['key', 'ball', 'box', 'wall', 'floor'])

        if color == None:
            color = self._rand_color()

        # TODO: we probably want to add an Object.make helper function
        assert kind in ['key', 'ball', 'box', 'wall', 'floor']
        if kind == 'key':
            obj = Key(color)
        elif kind == 'ball':
            obj = Ball(color)
        elif kind == 'box':
            obj = Box(color, contains=Key(color=color))
        elif kind == 'wall':
            obj = Wall()
        elif kind == 'floor':
            obj = Floor(color)

        return self.place_in_room(i, j, obj)

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
            if fwd_cell == None or (fwd_cell.can_overlap() and fwd_cell.type != 'floor'):
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.type != 'box' and fwd_cell.can_pickup():
                if self.carrying is None and (fwd_cell != self.blocked_obj or not self.place_blocked):
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
            # if fwd_cell and fwd_cell.can_pickup():
            #     if self.carrying is None:
            #         self.carrying = fwd_cell
            #         self.carrying.cur_pos = np.array([-1, -1])
            #         self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None
            if fwd_cell == self.floor_obj and self.carrying == self.obj:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None
                reward = self._reward()
                done = True
            if fwd_cell == self.blocked_place_obj and self.carrying == self.blocked_obj:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None
                self.place_blocked = True


        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell and fwd_cell.type == 'door' and self.place_blocked:
                fwd_cell.toggle(self, fwd_pos)
            elif fwd_cell and fwd_cell.type != 'door':
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

