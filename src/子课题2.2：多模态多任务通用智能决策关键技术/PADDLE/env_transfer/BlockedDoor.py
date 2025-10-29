from gym_minigrid.minigrid import Ball, Key, Box
from gym_minigrid.roomgrid import RoomGrid, reject_next_to
from gym_minigrid.register import register
import numpy as np


class BlockedDoorEnv(RoomGrid):
    """
    Unlock a door, then pick up a box in another room
    """

    def __init__(self, room_size=6, seed=0):
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=5*room_size**2,
            seed=seed,
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        random_idx = np.random.random()

        if random_idx >= 0.5:
            # Make sure the two rooms are directly connected by a locked door
            door, door_pos = self.add_door(0, 0, 0, color='yellow', locked=True)
            # Block the door with a ball
            self.grid.set(door_pos[0] + 1, door_pos[1], Ball(color='yellow'))
            # Add a key to unlock the door
            obj, _ = self.add_object(1, 0, kind="key", color='yellow')

            self.place_agent(1, 0)
        else:
            # Make sure the two rooms are directly connected by a locked door
            door, door_pos = self.add_door(0, 0, 0, color='yellow', locked=True)
            # Block the door with a ball
            self.grid.set(door_pos[0] - 1, door_pos[1], Ball(color='yellow'))
            # Add a key to unlock the door
            obj, key_pos = self.add_object(0, 0, kind="key", color='yellow')

            self.place_agent(0, 0)

        self.obj = obj
        self.mission = "pick up the %s %s" % (obj.color, obj.type)

    def add_object(self, i, j, kind=None, color=None):
        """
        Add a new object to room (i, j)
        """

        if kind == None:
            kind = self._rand_elem(['key', 'ball', 'box'])

        if color == None:
            color = self._rand_color()

        # TODO: we probably want to add an Object.make helper function
        assert kind in ['key', 'ball', 'box']
        if kind == 'key':
            obj = Key(color)
        elif kind == 'ball':
            obj = Ball(color)
        elif kind == 'box':
            obj = Box('yellow')

        return self.place_in_room(i, j, obj, kind)

    def place_in_room(self, i, j, obj, kind=None):
        """
        Add an existing object to room (i, j)
        """
        # print("\n Using new place in room function")
        room = self.get_room(i, j)
        if kind == 'box':
            if i == 1:
                room_tops = [room.top[0]+2, room.top[1]+2]
                room_size = room.size
            elif i==0:
                room_tops =  room.top
                room_size = [room.size[0]-2, room.size[1]-2]
        else:
            room_tops = room.top
            room_size = room.size
        pos = self.place_obj(
            obj,
            #
            room_tops,
            room_size,
            reject_fn=reject_next_to,
            max_tries=1000
        )

        room.objs.append(obj)

        return obj, pos

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
                if fwd_cell.type == 'door' and fwd_cell.is_open:
                    done = True
                    reward = self._reward()

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}
