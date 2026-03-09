import pygame
import numpy as np
import random
import heapq
from collections import deque

pygame.init()
FPS = 30
clock = pygame.time.Clock()

TILE_SIZE = 20
BOARD_SIZE = 36
WALL_OFFSET = 20
TOP_MARGIN = 60
SCREEN_WIDTH = BOARD_SIZE * TILE_SIZE + 2 * WALL_OFFSET
SCREEN_HEIGHT = BOARD_SIZE * TILE_SIZE + 2 * WALL_OFFSET + TOP_MARGIN

rectData = [5, 5, pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]
trail = []
board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
appleAvailable = False
appleLoc = np.zeros(2, dtype=int)
gameOver = False
direction = 4
backgroundColor = (0, 0, 0)
score = 0

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake")
font = pygame.font.SysFont("arial", 24)

def board_to_screen(x_idx, y_idx):
    return x_idx * TILE_SIZE + WALL_OFFSET, y_idx * TILE_SIZE + WALL_OFFSET + TOP_MARGIN

def displayApple():
    while True:
        rand1 = random.randint(0, BOARD_SIZE - 1)
        rand2 = random.randint(0, BOARD_SIZE - 1)
        if board[rand1, rand2] == 0 and (rand1, rand2) not in trail:
            board[rand1, rand2] = 4
            appleLoc[0] = rand1
            appleLoc[1] = rand2
            return True

def createBoard():
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            value = board[i, j]
            x, y = board_to_screen(i, j)
            if value == 1:
                pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(x, y, TILE_SIZE, TILE_SIZE))
            elif value == 4:
                pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(x, y, TILE_SIZE, TILE_SIZE))

def createBarriers():
    pygame.draw.rect(screen,(255,0,0),(0,0,SCREEN_WIDTH,TOP_MARGIN + WALL_OFFSET))
    pygame.draw.rect(screen,(255,0,0),(0,SCREEN_HEIGHT-WALL_OFFSET,SCREEN_WIDTH,WALL_OFFSET))
    pygame.draw.rect(screen,(255,0,0),(0,TOP_MARGIN,WALL_OFFSET,SCREEN_HEIGHT))
    pygame.draw.rect(screen,(255,0,0),(SCREEN_WIDTH-WALL_OFFSET,TOP_MARGIN,WALL_OFFSET,SCREEN_HEIGHT))

def drawGrid():
    for x in range(WALL_OFFSET, WALL_OFFSET + BOARD_SIZE * TILE_SIZE, TILE_SIZE):
        for y in range(WALL_OFFSET + TOP_MARGIN, WALL_OFFSET + TOP_MARGIN + BOARD_SIZE * TILE_SIZE, TILE_SIZE):
            pygame.draw.rect(screen, (100, 100, 100), pygame.Rect(x, y, TILE_SIZE, TILE_SIZE), 1)

def createSquares():
    x, y = board_to_screen(rectData[0], rectData[1])
    pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(x, y, TILE_SIZE, TILE_SIZE))

def checkInput(direction):
    pressed = pygame.key.get_pressed()
    if pressed[rectData[2]] and direction != 2:
        direction = 1
    elif pressed[rectData[3]] and direction != 1:
        direction = 2
    elif pressed[rectData[4]] and direction != 4:
        direction = 3
    elif pressed[rectData[5]] and direction != 3:
        direction = 4
    return direction

def catchCollision(appleAvailable, gameOver):
    grow = False
    headx, heady = rectData[0], rectData[1]
    # check out-of-bounds
    if headx < 0 or headx >= BOARD_SIZE or heady < 0 or heady >= BOARD_SIZE:
        gameOver = True
        return appleAvailable, gameOver, False
    # self collision
    if (headx, heady) in trail:
        gameOver = True
    # apple collision
    if appleLoc[0] == headx and appleLoc[1] == heady:
        board[appleLoc[0], appleLoc[1]] = 0
        appleAvailable = False
        grow = True
    return appleAvailable, gameOver, grow

def updateSnake(grow):
    head = (rectData[0], rectData[1])
    trail.insert(0, head)

    if not grow and len(trail) > 0:
        tail = trail.pop()
        board[tail[0], tail[1]] = 0

    for segment in trail:
        board[segment[0], segment[1]] = 1

def move(direction):
    if direction == 1:
        rectData[1] -= 1
    elif direction == 2:
        rectData[1] += 1
    elif direction == 3:
        rectData[0] -= 1
    elif direction == 4:
        rectData[0] += 1

def drawScore():
    scoreText = font.render(f"Score: {score}", True, (255,255,255))
    rect = scoreText.get_rect(center=(SCREEN_WIDTH//2, 40))
    screen.blit(scoreText, rect)

# return manhattan distance
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# return opposite direction
def oppositeDirection(d):
    return {1:2,2:1,3:4,4:3}[d]

# convert snake data into snake grid
def getSnakeGrid():
    snake = []
    head = (rectData[0], rectData[1])
    snake.append(head)
    for segment in trail:
        snake.append(segment)
    return snake

# find valid neighbor
def getNeighbors(node):
    x, y = node
    moves = [(0,-1),(0,1),(-1,0),(1,0)]
    neighbors = []
    for dx, dy in moves:
        nx = x + dx
        ny = y + dy
        if nx < 0 or nx >= 36 or ny < 0 or ny >= 36:
            continue
        if board[nx, ny] == 1:
            continue
        neighbors.append((nx, ny))
    return neighbors

# check if move is safe
def moveIsSafe(direction):
    x = rectData[0]
    y = rectData[1]
    if direction == 1: y -= 1
    elif direction == 2: y += 1
    elif direction == 3: x -= 1
    elif direction == 4: x += 1
    if x < 0 or x >= 36 or y < 0 or y >= 36:
        return False
    if board[x, y] == 1:
        return False
    return True

# A* search
def astar(start, goal):
    openSet = []
    heapq.heappush(openSet, (0, start))
    cameFrom = {}
    gScore = {start:0}
    while openSet:
        _, current = heapq.heappop(openSet)
        if current == goal:
            path = []
            while current in cameFrom:
                path.append(current)
                current = cameFrom[current]
            path.reverse()
            return path
        for neighbor in getNeighbors(current):
            tentative = gScore[current] + 1
            if neighbor not in gScore or tentative < gScore[neighbor]:
                gScore[neighbor] = tentative
                f = tentative + manhattan(neighbor, goal)
                heapq.heappush(openSet,(f + random.random()*0.001, neighbor))
                cameFrom[neighbor] = current
    return None

# convert path to direction
def pathToDirection(path):
    if not path:
        return None
    headx = rectData[0]
    heady = rectData[1]
    nx, ny = path[0]
    if nx == headx and ny == heady-1:
        return 1
    if nx == headx and ny == heady+1:
        return 2
    if nx == headx-1 and ny == heady:
        return 3
    if nx == headx+1 and ny == heady:
        return 4
    return None

# simulate snake following path
def simulateSnake(snake, path):
    snake = list(snake)
    for step in path:
        snake.insert(0, step)
        if step == (appleLoc[0], appleLoc[1]):
            break
        else:
            snake.pop()
    return snake

# BFS path existence check from start to goal
def pathExists(start, goal):
    openSet = deque([start])
    visited = set()
    while openSet:
        node = openSet.popleft()
        if node == goal:
            return True
        if node in visited:
            continue
        visited.add(node)
        for n in getNeighbors(node):
            if n not in visited:
                openSet.append(n)
    return False

# flood fill
floodCache = {}
def floodFill(start):
    if start in floodCache:
        return floodCache[start]
    queue = deque([start])
    visited = set()
    count = 0
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        count += 1
        for n in getNeighbors(node):
            if n not in visited:
                queue.append(n)
    floodCache[start] = count
    return count

# count free neighbors
def freeNeighbors(pos):
    x, y = pos
    count = 0
    moves = [(0,-1),(0,1),(-1,0),(1,0)]
    for dx, dy in moves:
        nx = x + dx
        ny = y + dy
        if nx < 0 or nx >= 36 or ny < 0 or ny >= 36:
            continue
        if board[nx, ny] != 1:
            count += 1
    return count

# safe apple path
def safeApplePath(snake):
    head = snake[0]
    food = (appleLoc[0], appleLoc[1])
    path = astar(head, food)
    if not path:
        return None
    # always go to apple in early game
    if len(snake) < 10:
        return path
    # avoid apples in small pockets
    if floodFill(food) < 3:
        return None
    futureSnake = simulateSnake(snake, path)
    newHead = futureSnake[0]
    tail = futureSnake[-1]
    # allow if tail is far away
    if manhattan(newHead, tail) > 10:
        return path
    # check tail safety
    if pathExists(newHead, tail):
        return path
    # check if following tail 1 step opens path
    tailPath = astar(head, tail)
    if not tailPath:
        return path
    testSnake = simulateSnake(snake, tailPath[:1])
    testHead = testSnake[0]
    newApplePath = astar(testHead, food)
    if newApplePath:
        return None
    # allow apple anyway as last resort
    return path

# find safest move based on flood fill
def longestSafeMove(currentDirection):
    head = (rectData[0], rectData[1])
    bestDir = None
    bestScore = -1
    for d in [1,2,3,4]:
        if d == oppositeDirection(currentDirection):
            continue
        if not moveIsSafe(d):
            continue
        x, y = head
        if d == 1: y -= 1
        elif d == 2: y += 1
        elif d == 3: x -= 1
        elif d == 4: x += 1
        space = floodFill((x, y))
        if space > bestScore:
            bestScore = space
            bestDir = d
    if bestDir:
        return bestDir
    return currentDirection

# smart AI decision to determine move
def getSmartAIMove(currentDirection):
    snake = getSnakeGrid()
    head = snake[0]
    floodCache.clear()
    # safe apple path
    path = safeApplePath(snake)
    if path:
        direction = pathToDirection(path)
        if direction and direction != oppositeDirection(currentDirection) and moveIsSafe(direction):
            return direction
    # follow tail if cannot safely reach apple
    tail = snake[-1]
    path = astar(head, tail)
    if path:
        direction = pathToDirection(path)
        if direction and direction != oppositeDirection(currentDirection) and moveIsSafe(direction):
            return direction
    # maximize free space if previous two are invalid
    return longestSafeMove(currentDirection)

# dumb AI decision to determine move
def getDumbAIMove(currentDirection):
    headx, heady = rectData[0], rectData[1]
    foodx, foody = appleLoc
    if foodx > headx and currentDirection != 3:
        return 4
    if foodx < headx and currentDirection != 4:
        return 3
    if foody > heady and currentDirection != 1:
        return 2
    if foody < heady and currentDirection != 2:
        return 1
    return currentDirection

smart = False
appleAvailable = displayApple()
print(f"Apple Score: {score}", end="\r")
running = True
while running:
    screen.fill(backgroundColor)

    # ALWAYS process events first
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not gameOver:
        if smart:
            direction = getSmartAIMove(direction)
        else:
            direction = getDumbAIMove(direction)
        move(direction)
        appleAvailable, gameOver, grow = catchCollision(appleAvailable, gameOver)

        if not gameOver:
            updateSnake(grow)

            if not appleAvailable:
                score += 1
                print(f"Apple Score: {score}", end="\r")
                appleAvailable = displayApple()

    # draw (still draw after game over if you want frozen screen)
    createBoard()
    createBarriers()
    drawGrid()
    createSquares()
    drawScore()

    pygame.display.update()
    clock.tick(FPS)
pygame.quit()