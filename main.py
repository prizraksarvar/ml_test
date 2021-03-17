import sys
import time

import matplotlib
from PyQt5.QtCore import Qt

from Game import Game
from ManualGame import ManualGame
from PolicyEstimator import PolicyEstimator
from experienceReplayBuffer import experienceReplayBuffer
from game_funct import manual_teach_net, game_funct, teach_net, paint_mean_score

import numpy as np
import random
import copy

import matplotlib.pyplot as plt
# %matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.utils import shuffle
from IPython.display import clear_output, display, HTML
from collections import namedtuple, deque

from matplotlib import animation, rc
from matplotlib import pyplot as plt
from celluloid import Camera

from PyQt5 import QtCore, QtWidgets, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

matplotlib.use('Qt5Agg')

DIR = '/home/prizrak/Загрузки/'

higth_weel_g = 32  # глубина колодца
width_weel_g = 16  # ширина  колодца
width_racet_g = 4  # ширина  ракетки
max_point_weel_g = 3  # максимальное одновременное кол-во точек в колодце
min_len_between_point = 17  # минимальное расстояние между точками
kol_point = 5  # количество точек падающих за игру

# Одноканальная Игра

paint_game = False  # True #        # Отрисовывать игру или нет (True - отрисовывать)
step = 10  # Шаг усреднения при Анализе обученности
kol_game = 3  # количество игр в цикле обучения


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=5, dpi=100):
        # fig = Figure(figsize=(width, height), dpi=dpi)
        fig = Figure(figsize=(width, height))
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.manual_game = None
        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        sc = MplCanvas(self, width=5, height=5, dpi=100)
        # sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])

        self.start_time = time.time()

        self.canvas = sc
        self.setCentralWidget(sc)

        self.show()

        self.iteration = 0
        lr_in = 1e-4

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

        self.policy_estimator = PolicyEstimator(statement_size=higth_weel_g * width_weel_g, action_size=9,
                                                device_in=self.device)

        self.manual_buffer = experienceReplayBuffer(memory_size=5000)

        learning_rate = 1e-4
        # Осуществляем оптимизацию путем стохастического градиентного спуска
        self.optimizerSGD = optim.SGD(self.policy_estimator.network.parameters(), lr=learning_rate, momentum=0.9)
        # Создаем функцию потерь
        self.criterionSGD = nn.NLLLoss()

        self.global_buffer = experienceReplayBuffer(memory_size=5000)
        # self.optimizer_adam = optim.Adam(self.policy_estimator.network.parameters(), lr=lr_in)  # , weight_decay=wd_in)
        self.optimizer_adam = optim.Adamax(self.policy_estimator.network.parameters(), lr=lr_in)

        self.list_mean_score = [0]  # Список усредненных значений
        self.list_rolout_score = []

        self.timer = QtCore.QTimer()

        # self.start_manual()
        self.start_autogame()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key_0:
            self.manual_game.handle_action('0')
        if event.key() == Qt.Key_1:
            self.manual_game.handle_action('1')
        if event.key() == Qt.Key_2:
            self.manual_game.handle_action('2')
        if event.key() == Qt.Key_3:
            self.manual_game.handle_action('3')
        if event.key() == Qt.Key_4:
            self.manual_game.handle_action('4')
        if event.key() == Qt.Key_5:
            self.manual_game.handle_action('5')
        if event.key() == Qt.Key_6:
            self.manual_game.handle_action('6')
        if event.key() == Qt.Key_7:
            self.manual_game.handle_action('7')
        if event.key() == Qt.Key_8:
            self.manual_game.handle_action('8')
        if event.key() == Qt.Key_9:
            self.manual_game.handle_action('9')

    def start_manual(self):
        for x in range(0, 1):
            print('Цикл ручного обучения № ', x)
            self.manual_game = ManualGame(self.canvas, higth_weel_g, width_weel_g, width_racet_g, max_point_weel_g,
                                          kol_point,
                                          min_len_between_point, self.manual_net_learn, list_col_point=[0.5, 0.5])
            self.manual_game.start()

    def manual_net_learn(self):
        images, actions, scores = self.manual_game.get_results()
        self.manual_buffer.append_butch(images, actions, scores)

        sel_pr = manual_teach_net(self.manual_buffer.all_sample_batch(), self.policy_estimator, self.optimizerSGD,
                                  self.criterionSGD,
                                  self.device)
        self.start_autogame()

    def start_autogame(self):
        self.canvas.figure.clf()
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.iteration = 0
        self.timer.singleShot(10, self.start_autogame_iteration)

    def start_autogame_iteration(self):
        x = self.iteration
        data_set_images, data_set_actions, data_set_reward = game_funct(kol_game, self.policy_estimator, higth_weel_g,
                                                                        width_weel_g, width_racet_g,
                                                                        max_point_weel_g, kol_point,
                                                                        min_len_between_point, self.list_rolout_score)
        self.global_buffer.append_butch(data_set_images, data_set_actions, data_set_reward)
        sel_pr = teach_net(self.global_buffer.sample_batch(batch_size=128), self.policy_estimator, self.optimizer_adam,
                           self.device)
        self.global_buffer.clear()

        # lr_in = 1e-2 * 0.95 * (x / 10)
        # self.optimizer_adam = optim.Adam(self.policy_estimator.network.parameters(), lr=lr_in)  # , weight_decay=wd_in)

        # if self.iteration % 500 == 0:
        #     sel_pr = manual_teach_net(self.manual_buffer.sample_batch(batch_size=32), self.policy_estimator, self.optimizerSGD,
        #                               self.criterionSGD, self.device)

        # Анализ степени обученности

        if x % 100 == 0:
            # self.canvas.axes.cla()
            paint_mean_score(x, self.list_rolout_score, self.list_mean_score, self.canvas)
            # self.optimizer_adam = optim.Adam(self.policy_estimator.network.parameters(),
            #                                  lr=1e-2 * 0.95 * (x / 100))  # , weight_decay=wd_in)
            self.list_rolout_score = []

        print('Цикл обучения № ', x)
        mean_score=0
        if len(self.list_rolout_score) >= step:
            mean_score = np.array(self.list_rolout_score[-step:]).mean()
        else:
            if len(self.list_mean_score) > 0:
                mean_score = self.list_mean_score[-1]
        mean_score = round(mean_score, 2)
        if mean_score >= kol_point * kol_game or x > 50000:
            print('Обучение Закончено')
            self.save_model()
            self.print_last_game(self.policy_estimator)
            self.save_loss()

            exec_time = round(time.time() - self.start_time, 2)
            print("--- %s seconds ---" % exec_time)
            self.canvas.axes.set_title(
                'Обучение закончено\nДанные сохранены' + "\n--- %s seconds ---" % exec_time,
                fontsize=12)
            self.canvas.draw()
            return

        if x % 20 == 0:
            print('Точность на последних циклах ', str(step), ' = ', mean_score)
            exec_time = round(time.time() - self.start_time, 2)
            print("--- %s seconds ---" % exec_time)
            self.canvas.axes.set_title(
                'Цикл обучения № ' + str(x) + '\nТочность на последних циклах ' + str(step) + ' = ' + str(
                    mean_score) + "\n--- %s seconds ---" % exec_time,
                fontsize=12)
            self.canvas.draw()
            self.iteration = self.iteration + 1
            self.timer.singleShot(10, self.start_autogame_iteration)
        else:
            # Отрисовка не нужна
            self.iteration = self.iteration + 1
            self.start_autogame_iteration()

    def save_model(self):
        torch.save(self.policy_estimator.network.state_dict(), DIR+'smart_2.pth')

    def save_loss(self):
        self.canvas.figure.savefig(DIR + 'smart_2_loss.png')

    def print_last_game(self, policy_estimator):
        fig = plt.figure(figsize=(5, 5))
        plt.axis('off')
        camera = Camera(fig)

        G = Game(higth_weel_g, width_weel_g, width_racet_g, max_point_weel_g, kol_point, min_len_between_point,
                 list_col_point=[0.5, 0.5])

        # Основной Цикл Игры
        done = False
        while done == False:
            # Блок выбора действия для Агента
            image = G.get_weel_state()
            image_tensor = torch.Tensor(np.expand_dims([image], axis=0)).float()
            prediction = policy_estimator.predict(image_tensor)  # Предсказываем
            if policy_estimator.device_in == 'cpu':
                action = np.argmax(prediction.cpu().detach().numpy())  # Получаем решение, что делать 'cpu'
            else:
                action = torch.max(prediction.detach(), 1)[1].item()  # Получаем решение, что делать 'cuda'
            # выполняем действие
            done = G.act_pg(action)
            # Отрисовка игры
            plt.imshow(G.get_weel_state())
            camera.snap()

        anim = camera.animate()

        anim.save(DIR+'smart_2.gif', writer='PillowWriter', fps=50)


# rc('animation', html='jshtml')


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec_()
