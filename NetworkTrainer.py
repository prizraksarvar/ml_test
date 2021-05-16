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
                 batch_size: int, device: torch.device, axes, name: str, color: str, loss_fn):
        self.policy_estimator = policy_estimator
        self.buffer = buffer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.axes = axes
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
        if x % 100 == 0:
            self.paint_mean_score()
            self.list_rollout_score = []
            self.list_rollout_loss = []
            mean_score = 0
            if len(self.list_mean_score) > 0:
                mean_score = self.list_mean_score[-1]
            mean_score = round(mean_score, 2)
            history2.log(step, loss=loss, accuracy=mean_score, image=image)

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
    def paint_mean_score(self):
        local_step = 10  # Шаг усреднения

        if len(self.list_rollout_score) >= local_step:
            # Усредняем в обратном порядке
            for num in range(int(len(self.list_rollout_score) / local_step), 0, -1):
                mean_score = np.array(self.list_rollout_score[num * local_step - local_step:num * local_step]).mean()
                mean_loss = np.array(self.list_rollout_loss[num * local_step - local_step:num * local_step]).mean()
                self.list_mean_score.append(mean_score)
                self.list_mean_score_y.append(self.y_len * 10)
                if mean_score > self.max_y:
                    self.max_y = mean_score
                if mean_score < self.min_y:
                    self.min_y = mean_score
                self.max_x = self.y_len * 10
                self.y_len += 1
                self.list_mean_loss.append(mean_loss)

            # list_mean_score = list_mean_score[::-1]  # разворачиваем список
            mn_sc = np.array(self.list_mean_score)
            mn_sc_y = np.array(self.list_mean_score_y)
            # fig, subplot = plt.subplots()  # доступ к Figure и Subplot
            # subplot.plot(mn_sc)  # построение графика функции
            # gr_dir_name = './Grafics/'
            # gr_file_name = str(num_step) + '_' + str(mn_sc[-1]) + '_' + str(kol_point) + '_point_' + 'Conv2D' + '.png'
            # plt.title(gr_file_name, fontsize=12)
            # plt.savefig(gr_dir_name + gr_file_name)
            # plt.show()

            if self.plot_rf is None:
                self.plot_rf = self.axes.plot(mn_sc_y, mn_sc, color=self.color, label=self.name)[0]
                self.axes.legend()
            else:
                self.plot_rf.set_data(mn_sc_y, mn_sc)

            if self.plot_loss_rf is None:
                self.plot_loss_rf = self.axes.plot(mn_sc_y, np.array(self.list_mean_loss), color='light'+self.color, label=self.name+'_loss')[0]
                self.axes.legend()
            else:
                self.plot_loss_rf.set_data(mn_sc_y, np.array(self.list_mean_loss))

