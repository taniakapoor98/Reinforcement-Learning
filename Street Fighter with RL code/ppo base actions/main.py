import time
from io import BytesIO

import numpy as np
from PIL import Image
from py4j.java_gateway import JavaGateway
from matplotlib import pyplot as plt
# Connect to the Java GatewayServer
gateway = JavaGateway()
# Get the Game instance through the entry point
game = gateway.entry_point.getGame()
# game.resetGame()
print('gateway created')
game.menus()
# game.setRoundStart(True)
# game.setRound(1)
print('ii')
# game.setP2Wins(1)
# print(game.getP1Wins())
# # game.setTimeLeft(99)
# print((game.getTimeLeft()))
# print(f'player1 health {game.getPlayer1().getHealth()}')
# print(f'player2 health {game.getPlayer2().getHealth()}')
# i=-1
# while (i<90):
#     obs = game.getPlayer1().getObs()
#
#     print(np.array(obs, dtype=np.float32))
#     time.sleep(20)
#     i+=1
#
# # print(f'controls p1:{game.getPlayer1Controls()}')
# player1_controls = game.getPlayer1Controls()
#
# #---------------capture frame
# def get_image_bytes():
#     return game.getImageAsByteArray()
#
# def image_preprocessing(byte_data, target_size=(100, 100)):
#     # Convert byte data to an image
#     image = Image.open(BytesIO(byte_data))
#     grayscale_image = image.convert("L")
#     resized_image = grayscale_image.resize(target_size, Image.Resampling.LANCZOS)
#
#     image_array = np.array(resized_image).reshape(*target_size, 1)
#     print(image_array.shape)
#     return image_array
#
# def plot_image_from_bytes(byte_data):
#     # Preprocessing  image
#     preprocessed_image = image_preprocessing(byte_data)
#     plt.imshow(preprocessed_image.squeeze(), cmap="gray")
#     plt.axis('off')  # Hide the axis
#     plt.show()  # Display the image
#
# # image_bytes = get_image_bytes()
# # plot_image_from_bytes(image_bytes)
# #-------------------- gameplay
#
# player1Controls = game.getPlayer1Controls()  # Retrieve the control array once
#
# player1KeyMap = {
#     'left': player1Controls[2],   # Left action
#     'right': player1Controls[3],  # Right action
#     # 'up': player1Controls[0],     # Up action
#     # 'down': player1Controls[1],   # Down action
#     'light_attack': player1Controls[4],  # Light attack
#     'medium_attack': player1Controls[5], # Medium attack
#     'heavy_attack': player1Controls[6],  # Heavy attack
# }
#
# def player1_action(action):
#     time.sleep(0.5)
#     game.player1Move(player1KeyMap[action])  # light attack
#     time.sleep(0.5)
#
# #
# # player1_action('light_attack')
# # player1_action('right')
# # player1_action('left')
# # player1_action('right')
# # player1_action('heavy_attack')
# # player1_action('medium_attack')
#
#
#
# # public static final int[] PLAYER1_DEFAULT = {
# #     87, // KeyEvent.VK_W (Up - Move Up)
# #     83, // KeyEvent.VK_S (Down - Move Down)
# #     65, // KeyEvent.VK_A (Left - Move Left)
# #     68, // KeyEvent.VK_D (Right - Move Right)
# #     72, // KeyEvent.VK_H (Light Attack)
# #     74, // KeyEvent.VK_J (Medium Attack)
# #     75  // KeyEvent.VK_K (Heavy Attack)
# # };