import time

import numpy as np
import torch
from torch import optim

from PolicyEstimator import PolicyEstimator
from experienceReplayBuffer import experienceReplayBuffer
from game_funct import game_funct, teach_net

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


class NetworkTrainer(object):
    def __init__(self, policy_estimator: PolicyEstimator, buffer: experienceReplayBuffer, optimizer: optim.Optimizer,
                 batch_size: int, device: torch.device, history2, name: str, color: str, loss_fn):
        self.policy_estimator = policy_estimator
        self.buffer = buffer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.history2 = history2
        self.device = device
        self.list_mean_score = [0]  # Список усредненных значений
        self.list_mean_score_y = [0]
        self.list_mean_loss = [0]
        self.min_x = 0
        self.max_x = 1
        self.min_y = -1
        self.max_y = 1
        self.y_len = 1
        self.list_rollout_score = []
        self.list_rollout_loss = []
        self.start_time = time.time()
        self.name = name
        self.color = color
        self.done = False
        self.plot_rf = None
        self.plot_loss_rf = None
        self.loss_fn = loss_fn

    def start_autogame_iteration(self, x):
        data_set_images, data_set_actions, data_set_reward = game_funct(kol_game, self.policy_estimator, higth_weel_g,
                                                                        width_weel_g, width_racet_g,
                                                                        max_point_weel_g, kol_point,
                                                                        min_len_between_point, self.list_rollout_score)
        self.buffer.append_butch(data_set_images, data_set_actions, data_set_reward)
        loss = teach_net(self.buffer.sample_batch(batch_size=self.batch_size), self.policy_estimator, self.optimizer,
                           self.loss_fn, self.device)

        self.list_rollout_loss.append(loss)

        self.buffer.clear()

        # Анализ степени обученности
        if x > 0 and x % 100 == 0:
            self.paint_mean_score(x)
            self.list_rollout_score = []
            self.list_rollout_loss = []

        mean_score = 0
        if len(self.list_rollout_score) >= step:
            mean_score = np.array(self.list_rollout_score[-step:]).mean()
        else:
            if len(self.list_mean_score) > 0:
                mean_score = self.list_mean_score[-1]
        mean_score = round(mean_score, 2)
        if mean_score >= kol_point * kol_game or x > 50000:
            self.done = True
            return True

        # if x % 20 == 0:
        # print('Точность на последних циклах ', str(step), ' = ', mean_score)
        # exec_time = round(time.time() - self.start_time, 2)
        # print("--- %s seconds ---" % exec_time)
        # self.axes.set_title(
        #    'Цикл обучения № ' + str(x) + '\nТочность на последних циклах ' + str(step) + ' = ' + str(
        #        mean_score) + "\n--- %s seconds ---" % exec_time,
        #    fontsize=12)
        return False

    # отрисовка процесса обучения - средний результат серии из 10 игр
    def paint_mean_score(self, x):
        local_step = 10  # Шаг усреднения

        # if len(self.list_rollout_score) >= local_step:
        #     Усредняем в обратном порядке
        #     for num in range(int(len(self.list_rollout_score) / local_step), 0, -1):
        #         mean_score = np.array(self.list_rollout_score[num * local_step - local_step:num * local_step]).mean()
        #         mean_loss = np.array(self.list_rollout_loss[num * local_step - local_step:num * local_step]).mean()
        #         self.history2.log(self.y_len * 10, loss=mean_loss, accuracy=mean_score)
        #         self.y_len += 1

        if len(self.list_rollout_score) > 0:
            mean_score = np.array(self.list_rollout_score).mean()
            mean_loss = np.array(self.list_rollout_loss).mean()
            self.history2.log((0, x), loss=mean_loss, accuracy=mean_score)

