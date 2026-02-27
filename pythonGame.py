# import pygame
# import numpy as np
# import random

# pygame.init()
# clock = pygame.time.Clock()
# FPS = 12
#     #        x    y        up        down          left      right
# rectData = [100, 100, pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT] 
# trail = []
# backgroundColor = (0,0,0)
# board = np.zeros((35,35))
# appleAvailable = False
# appleLoc = np.zeros(2, dtype=int)
# gameOver = False

# def displayApple():
#     rand1 = random.randint(0,34)
#     rand2 = random.randint(0,34)
#     while(board[rand1][rand2] != 0):
#         rand1 = random.randint(0,34)
#         rand2 = random.randint(0,34)
#     board[rand1][rand2] = 4
#     appleLoc[0] = rand1
#     appleLoc[1] = rand2
#     appleAvailable = True
#     return appleAvailable

# def createBoard():
#     for i in range(0, len(board)):
#         for j in range(0, len(board[i])):
#             value = board[i][j] 
#             if(value == 1):
#                 pygame.draw.rect(screen, (0,255,0), pygame.Rect(i*20,j*20,20,20), 0)
#             elif(value == 4):
#                 pygame.draw.rect(screen, (0,0,255), pygame.Rect(i*20,j*20,20,20), 0)

# def createBarriers():
#     barriers = [pygame.Rect(0,0,755,20),
#                 pygame.Rect(0,720,755,20),
#                 pygame.Rect(0,0,20,755),
#                 pygame.Rect(720,0,20,755)]
#     for wall in barriers:
#         pygame.draw.rect(screen, (255,0,0), wall, 0)

# def createSquares():
#     pygame.draw.rect(screen, (0,255,0), pygame.Rect(rectData[0],rectData[1],20,20), 0)

# def drawGrid():
#     for x in range(20,720,20):
#         for y in range(20,720,20):
#             tile = pygame.Rect(x,y,20,20)
#             pygame.draw.rect(screen, (255,0,0), tile, 2)

# def checkInput(direction):
#     oldx = rectData[0]
#     oldy = rectData[1]
#     pressed = pygame.key.get_pressed()
#     if pressed[rectData[2]] and direction != 2:
#         direction = 1
#     elif pressed[rectData[3]] and direction != 1:
#         direction = 2
#     elif pressed[rectData[4]] and direction != 4:
#         direction = 3
#     elif pressed[rectData[5]] and direction != 3:
#         direction = 4
#     return direction

# def catchCollision(appleAvailable, gameOver):
#     if(rectData[0] == 0 or rectData[1] == 0 or rectData[0] == 740 or rectData[1] == 740):
#         gameOver = True
#     if(appleLoc[0] == rectData[0]/20 and appleLoc[1] == rectData[1]/20):
#         xApple = appleLoc[0]
#         yApple = appleLoc[1]
#         board[xApple][yApple] = 1
#         appleAvailable = False
#     return appleAvailable, gameOver

# def followTrail():
#     prev = (rectData[0],rectData[1])
#     lastTile = np.array([rectData[0]/20,rectData[1]/20],dtype=int)
#     if len(trail) != 0:
#         lastIndex = trail[len(trail)-1]
#         lastTile = np.array([lastIndex[0]/20,lastIndex[1]/20],dtype=int)
#     for tile in trail:
#         xVal = int(prev[0]/20)
#         yVal = int(prev[1]/20)
#         temp = tile
#         board[xVal][yVal] = 1
#         tile = prev
#         prev = temp
#     board[lastTile[0]][lastTile[1]] = 0

# def move(direction):
#     if(direction == 1):
#         rectData[1] -= 20
#     elif(direction == 2):
#         rectData[1] += 20
#     elif(direction == 3):
#         rectData[0] -= 20
#     else:
#         rectData[0] += 20
        
# screen = pygame.display.set_mode((740, 740))

# pygame.display.set_caption("Pygame Test")

# running = True
# direction = 4
# appleAvailable = displayApple()
# while running:
#     while not gameOver:
#         screen.fill(backgroundColor)
#         createBoard()
#         createBarriers()
#         drawGrid()
#         followTrail()
#         appleAvailable, gameOver = catchCollision(appleAvailable, gameOver)
#         if not appleAvailable:
#             trail.append((appleLoc[0]*20,appleLoc[1]*20))
#             appleAvailable = displayApple()
#         move(direction)
#         createSquares()
#         direction = checkInput(direction)
#         pygame.display.update()

#         for event in pygame.event.get():     
#             if event.type == pygame.QUIT:
#                 running = False
#         clock.tick(FPS)
       

# pygame.quit()

import pygame
import numpy as np
import random

pygame.init()
clock = pygame.time.Clock()
FPS = 10

#        x    y        up        down          left      right
rectData = [100, 100, pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]

trail = []
backgroundColor = (0, 0, 0)
board = np.zeros((35, 35), dtype=int)
appleAvailable = False
appleLoc = np.zeros(2, dtype=int)
gameOver = False


def displayApple():
    while True:
        rand1 = random.randint(0, 34)
        rand2 = random.randint(0, 34)

        # check BOTH board and trail (extra safety)
        if board[rand1, rand2] == 0 and (rand1*20, rand2*20) not in trail:
            board[rand1, rand2] = 4
            appleLoc[0] = rand1
            appleLoc[1] = rand2
            return True


def createBoard():
    for i in range(len(board)):
        for j in range(len(board[i])):
            value = board[i, j]
            if value == 1:
                pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(i * 20, j * 20, 20, 20), 0)
            elif value == 4:
                pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(i * 20, j * 20, 20, 20), 0)


def createBarriers():
    barriers = [
        pygame.Rect(0, 0, 755, 20),
        pygame.Rect(0, 720, 755, 20),
        pygame.Rect(0, 0, 20, 755),
        pygame.Rect(720, 0, 20, 755),
    ]
    for wall in barriers:
        pygame.draw.rect(screen, (255, 0, 0), wall, 0)


def createSquares():
    pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(rectData[0], rectData[1], 20, 20), 0)


def drawGrid():
    for x in range(20, 720, 20):
        for y in range(20, 720, 20):
            tile = pygame.Rect(x, y, 20, 20)
            pygame.draw.rect(screen, (255, 0, 0), tile, 2)


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

    headx = rectData[0] // 20
    heady = rectData[1] // 20

    # OUT OF BOUNDS check (robust)
    if headx < 0 or headx >= 35 or heady < 0 or heady >= 35:
        gameOver = True
        return appleAvailable, gameOver, False
    
    # wall collision
    if rectData[0] == 0 or rectData[1] == 0 or rectData[0] == 740 or rectData[1] == 740:
        gameOver = True

    headx = rectData[0] // 20
    heady = rectData[1] // 20

    # self collision
    head_pos = (rectData[0], rectData[1])
    if head_pos in trail:
        gameOver = True

    # apple collision
    if appleLoc[0] == headx and appleLoc[1] == heady:
        board[appleLoc[0], appleLoc[1]] = 0
        appleAvailable = False
        grow = True

    return appleAvailable, gameOver, grow


def updateSnake(grow):
    head = (rectData[0], rectData[1])

    # add new head
    trail.insert(0, head)

    # remove tail if not growing
    if not grow and len(trail) > 0:
        tail = trail.pop()
        tx, ty = tail[0] // 20, tail[1] // 20
        board[tx, ty] = 0

    # mark snake body on board
    for segment in trail:
        sx, sy = segment[0] // 20, segment[1] // 20
        if 0 <= sx < 35 and 0 <= sy < 35:
            board[sx, sy] = 1


def move(direction):
    if direction == 1:
        rectData[1] -= 20
    elif direction == 2:
        rectData[1] += 20
    elif direction == 3:
        rectData[0] -= 20
    else:
        rectData[0] += 20


screen = pygame.display.set_mode((740, 740))
pygame.display.set_caption("Snake")
running = True
direction = 4
appleAvailable = displayApple()

while running:
    screen.fill(backgroundColor)

    # ALWAYS process events first
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not gameOver:
        direction = checkInput(direction)
        move(direction)
        appleAvailable, gameOver, grow = catchCollision(appleAvailable, gameOver)

        if not gameOver:
            updateSnake(grow)

            if not appleAvailable:
                appleAvailable = displayApple()
                print(appleLoc)

    # draw (still draw after game over if you want frozen screen)
    createBoard()
    createBarriers()
    drawGrid()
    createSquares()

    pygame.display.update()
    clock.tick(FPS)

pygame.quit()