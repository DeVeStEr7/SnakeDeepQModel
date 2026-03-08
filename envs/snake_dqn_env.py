import pygame
import numpy as np
import random
from collections import deque

class SnakeEnv:
    #sets up sizing for grid
    TILE = 20
    GRID = 36
    W = GRID * TILE + TILE
    H = GRID * TILE + TILE

    #size of patch window
    PATCH = 5
    #38 dimensions from 2 apple location (x/y), 4 directions, 4 apple direction flags, 5x5 grid, 3 floods to see reachability in moveable directoins
    STATE_DIM = 38     # 2 + 4 + 4 + 25 + 3

    #base 20 fps and rendering for visual
    def __init__(self, fps=20, render=True):
        self.fps = fps
        self.do_render = render
        self.backgroundColor = (0, 0, 0)

        self.screen = None
        self.clock = None
        if self.do_render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption("Snake DQN")
            self.clock = pygame.time.Clock()

        self.reset()

    #reseting game upon death 
    def reset(self):
        center = (self.GRID // 2) * self.TILE
        self.rectData = [center, center]

        self.trail = []
        self.board = np.zeros((self.GRID, self.GRID), dtype=np.int32)
        self.appleLoc = np.zeros(2, dtype=np.int32)
        self.gameOver = False
        self.direction = 3

        #starvation counter to penalize looping
        self.steps_since_apple = 0
        self.apples_eaten = 0

        self._spawn_apple()
        return self._get_state()

    #renders board for game
    def _createBoard(self):
        for i in range(self.GRID):
            for j in range(self.GRID):
                value = self.board[i, j]
                if value == 1:
                    pygame.draw.rect(self.screen, (0, 255, 0),
                                     pygame.Rect(i * self.TILE, j * self.TILE, self.TILE, self.TILE), 0)
                elif value == 4:
                    pygame.draw.rect(self.screen, (0, 0, 255),
                                     pygame.Rect(i * self.TILE, j * self.TILE, self.TILE, self.TILE), 0)

    #renders barriers of game (bright red)
    def _createBarriers(self):
        barriers = [
            pygame.Rect(0, 0, self.W, self.TILE),
            pygame.Rect(0, self.H - self.TILE, self.W, self.TILE),
            pygame.Rect(0, 0, self.TILE, self.H),
            pygame.Rect(self.W - self.TILE, 0, self.TILE, self.H),
        ]
        for wall in barriers:
            pygame.draw.rect(self.screen, (255, 0, 0), wall, 0)

    #creates head of snake
    def _createHead(self):
        pygame.draw.rect(self.screen, (0, 255, 0),
                         pygame.Rect(self.rectData[0], self.rectData[1], self.TILE, self.TILE), 0)

    #draws grid for snake to traverse
    def _drawGrid(self):
        for x in range(self.TILE, self.W - self.TILE, self.TILE):
            for y in range(self.TILE, self.H - self.TILE, self.TILE):
                pygame.draw.rect(self.screen, (255, 0, 0),
                                 pygame.Rect(x, y, self.TILE, self.TILE), 2)

    def render(self):
        if not self.do_render:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.gameOver = True
        self.screen.fill(self.backgroundColor)
        self._createBoard()
        self._createBarriers()
        self._drawGrid()
        self._createHead()
        pygame.display.update()
        self.clock.tick(self.fps)



    #randomly spawns apple within tiles
    def _spawn_apple(self):
        while True:
            rand1 = random.randint(1, self.GRID - 2)
            rand2 = random.randint(1, self.GRID - 2)
            if self.board[rand1, rand2] == 0 and (rand1 * self.TILE, rand2 * self.TILE) not in self.trail:
                self.board[rand1, rand2] = 4
                self.appleLoc[0] = rand1
                self.appleLoc[1] = rand2
                return

    #moves snake head in one of 4 directions
    def _move(self, direction):
        if direction == 0:
            self.rectData[1] -= self.TILE
        elif direction == 1:
            self.rectData[1] += self.TILE
        elif direction == 2:
            self.rectData[0] -= self.TILE
        else:
            self.rectData[0] += self.TILE

    #checks to see if collision w/wall or self to end game
    def _catchCollision(self):
        grow = False
        headx = self.rectData[0] // self.TILE
        heady = self.rectData[1] // self.TILE

        if headx < 0 or headx >= self.GRID or heady < 0 or heady >= self.GRID:
            self.gameOver = True
            return grow

        if (self.rectData[0] <= 0 or self.rectData[1] <= 0 or
                self.rectData[0] >= self.W - self.TILE or self.rectData[1] >= self.H - self.TILE):
            self.gameOver = True
            return grow

        head_pos = (self.rectData[0], self.rectData[1])
        if head_pos in self.trail:
            self.gameOver = True
            return grow

        if self.appleLoc[0] == headx and self.appleLoc[1] == heady:
            self.board[self.appleLoc[0], self.appleLoc[1]] = 0
            grow = True

        return grow

    #updates snake within the board + grows it if necessary
    def _updateSnake(self, grow):
        head = (self.rectData[0], self.rectData[1])
        self.trail.insert(0, head)

        if not grow and len(self.trail) > 0:
            tail = self.trail.pop()
            tx, ty = tail[0] // self.TILE, tail[1] // self.TILE
            self.board[tx, ty] = 0

        self.board[self.board == 1] = 0
        for segment in self.trail:
            sx, sy = segment[0] // self.TILE, segment[1] // self.TILE
            if 0 <= sx < self.GRID and 0 <= sy < self.GRID:
                self.board[sx, sy] = 1

    #small local 5x5 vision patch to see around head of snake to try to minimize body collisions
    #checks for body parts or walls in area to plan for avoiding
    #0 = empty space, 1 = collidable object, 0.5 = apple
    def _get_patch(self):
        half = self.PATCH // 2
        headx = self.rectData[0] // self.TILE
        heady = self.rectData[1] // self.TILE

        patch = np.zeros((self.PATCH, self.PATCH), dtype=np.float32)

        for pi in range(self.PATCH):
            for pj in range(self.PATCH):
                gx = headx + (pi - half)
                gy = heady + (pj - half)

                #wall incoming
                if gx < 0 or gx >= self.GRID or gy < 0 or gy >= self.GRID:
                    patch[pi, pj] = 1.0
                    continue

                #border tiles for wall
                if gx == 0 or gx == self.GRID - 1 or gy == 0 or gy == self.GRID - 1:
                    patch[pi, pj] = 1.0
                #snake tiles for body
                elif self.board[gx, gy] == 1:
                    patch[pi, pj] = 1.0
                #apple spotted
                elif self.board[gx, gy] == 4:
                    patch[pi, pj] = 0.5

        return patch.flatten()

    #apple direction flags to show if apple is up/down/left/right
    def _apple_dir_flags(self):
        headx = self.rectData[0] // self.TILE
        heady = self.rectData[1] // self.TILE
        ax, ay = int(self.appleLoc[0]), int(self.appleLoc[1])
        return (float(ax < headx), float(ax > headx),
                float(ay < heady), float(ay > heady))

    #flood fill using BFS to quickly check if snake will be trapped or not
    #counts al reachable cells from start position
    def _flood_fill(self, start_gx, start_gy):
        # playable tiles = inner grid excluding border walls
        playable = (self.GRID - 2) * (self.GRID - 2)

        #checkes if start cell blocked
        if (start_gx <= 0 or start_gx >= self.GRID - 1 or
                start_gy <= 0 or start_gy >= self.GRID - 1):
            return 0.0
        #snake body blockage
        if self.board[start_gx, start_gy] == 1:
            return 0.0

        visited = set()
        queue   = deque()
        queue.append((start_gx, start_gy))
        visited.add((start_gx, start_gy))

        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited:
                    continue
                # border wall so stop
                if nx <= 0 or nx >= self.GRID - 1 or ny <= 0 or ny >= self.GRID - 1:
                    continue
                # snake body means blocked in
                if self.board[nx, ny] == 1:
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))

        return len(visited) / playable

    #returns the values for flooding ahead left and right based on where the snake is currently going
    #1 means it can go and be safe, 0 means it will crash
    def _flood_fill_features(self):

        headx = self.rectData[0] // self.TILE
        heady = self.rectData[1] // self.TILE

        # absolute delta for each action
        deltas = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

        #relative directions: straight, left-turn, right-turn
        left_of  = {0: 2, 1: 3, 2: 1, 3: 0}
        right_of = {0: 3, 1: 2, 2: 0, 3: 1}

        def fill_for(action):
            dx, dy = deltas[action]
            return self._flood_fill(headx + dx, heady + dy)

        flood_ahead = fill_for(self.direction)
        flood_left  = fill_for(left_of[self.direction])
        flood_right = fill_for(right_of[self.direction])

        return flood_ahead, flood_left, flood_right

    #each step of the game based on an action (main driver of the actual game)
    def step(self, action):
        if self.gameOver:
            return self._get_state(), 0.0, True

        #stops snake from going 180 and opposite direction
        opposites = {0: 1, 1: 0, 2: 3, 3: 2}
        if action == opposites[self.direction]:
            action = self.direction

        old_dist = self._manhattan_to_apple()
        self.direction = action
        self._move(self.direction)
        grow = self._catchCollision()

        if self.gameOver:
            return self._get_state(), -1.0, True

        self._updateSnake(grow)
        self.steps_since_apple += 1

        if grow:
            reward = 1.0
            self.steps_since_apple = 0
            self.apples_eaten += 1
            self._spawn_apple()
        else:
            new_dist = self._manhattan_to_apple()
            reward = 0.005 if new_dist < old_dist else -0.005

            # Starvation penalty: ramps up after 100 idle steps
            # Scales with snake length so a long snake can't just loop
            max_idle = max(100, len(self.trail) * 3)
            if self.steps_since_apple > max_idle:
                overage = self.steps_since_apple - max_idle
                reward -= 0.002 * overage  # gentle ramp, not a cliff

        return self._get_state(), float(reward), False

    #checks manhattan distance to apple
    def _manhattan_to_apple(self):
        headx = self.rectData[0] // self.TILE
        heady = self.rectData[1] // self.TILE
        return abs(int(self.appleLoc[0]) - headx) + abs(int(self.appleLoc[1]) - heady)

    #grabs current state of snake in position
    def _get_state(self):
        applex = int(self.appleLoc[0]) / (self.GRID - 1)
        appley = int(self.appleLoc[1]) / (self.GRID - 1)

        d = np.zeros(4, dtype=np.float32)
        d[self.direction] = 1.0

        #grabs flags (aL=apple left, aR = right, etc.), patch around head and the flood areas of forward,left,right
        aL, aR, aU, aD = self._apple_dir_flags()
        patch = self._get_patch()
        flood_ahead, flood_left, flood_right = self._flood_fill_features()

        return np.array(
            [applex, appley, *d, aL, aR, aU, aD, *patch,
             flood_ahead, flood_left, flood_right],
            dtype=np.float32
        )

    #closes game
    def close(self):
        if self.do_render:
            pygame.quit()