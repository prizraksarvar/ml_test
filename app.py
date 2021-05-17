import time
import matplotlib
from Game import Game
from NetworkTrainer import NetworkTrainer
from PolicyEstimator import PolicyEstimator
from experienceReplayBuffer import experienceReplayBuffer

import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from celluloid import Camera

import torchvision.models
from torchvision import datasets, transforms
import hiddenlayer as hl

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


class App(object):
    def __init__(self):
        self.manual_game = None

        self.history2 = hl.History()
        self.canvas2 = hl.Canvas()

        self.start_time = time.time()

        self.iteration = 0
        lr_in = 1e-4

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

        learning_rate = 1e-4
        # Осуществляем оптимизацию путем стохастического градиентного спуска
        # self.optimizerSGD = optim.SGD(self.policy_estimator.network.parameters(), lr=learning_rate, momentum=0.9)
        # Создаем функцию потерь
        # self.criterionSGD = nn.NLLLoss()

        self.trainers = []

        self.max_x = 0
        self.min_x = 0
        self.max_y = 0
        self.min_y = 0

    def start(self):
        self.start_autogame()

    def activations_hook1(self, rself, inputs, output):
        """Intercepts the forward pass and logs activations.
        """
        batch_ix = self.iteration
        if batch_ix > 0 and batch_ix % 100 == 0:
            # The output of this layer is of shape [batch_size, 16, 32, 32]
            # Take a slice that represents one feature map
            self.history2.log((0, batch_ix), layer1=output.data[0, 0])

    def activations_hook2(self, rself, inputs, output):
        """Intercepts the forward pass and logs activations.
        """
        batch_ix = self.iteration
        if batch_ix > 0 and batch_ix % 100 == 0:
            # The output of this layer is of shape [batch_size, 16, 32, 32]
            # Take a slice that represents one feature map
            self.history2.log((0, batch_ix), layer2=output.data[0, 0])

    def start_autogame(self):
        loss_fn = nn.CrossEntropyLoss()

        policy_estimator = PolicyEstimator(statement_size=higth_weel_g * width_weel_g, action_size=9,
                                           device_in=self.device)

        pe1 = PolicyEstimator(statement_size=higth_weel_g * width_weel_g, action_size=9,
                              device_in=self.device)
        pe1.network.load_state_dict(copy.deepcopy(policy_estimator.network.state_dict()))

        pe2 = PolicyEstimator(statement_size=higth_weel_g * width_weel_g, action_size=9,
                              device_in=self.device)
        pe2.network.load_state_dict(copy.deepcopy(policy_estimator.network.state_dict()))

        pe3 = PolicyEstimator(statement_size=higth_weel_g * width_weel_g, action_size=9,
                              device_in=self.device)
        pe3.network.load_state_dict(copy.deepcopy(policy_estimator.network.state_dict()))

        pe4 = PolicyEstimator(statement_size=higth_weel_g * width_weel_g, action_size=9,
                              device_in=self.device)
        pe4.network.load_state_dict(copy.deepcopy(policy_estimator.network.state_dict()))

        pe5 = PolicyEstimator(statement_size=higth_weel_g * width_weel_g, action_size=9,
                              device_in=self.device)
        pe5.network.load_state_dict(copy.deepcopy(policy_estimator.network.state_dict()))

        pe1.network.layer1[0].register_forward_hook(self.activations_hook1)
        pe1.network.layer2[0].register_forward_hook(self.activations_hook2)

        self.trainers.append(NetworkTrainer(
            pe1,
            experienceReplayBuffer(memory_size=5000),
            optim.Adamax(pe1.network.parameters(), lr=1e-3),
            128,
            self.device,
            self.history2,
            'net1',
            'blue',
            None
        ))

        self.iteration = 0
        for i in range(50000):
            self.autogame_loop_func()

    def autogame_loop_func(self):
        all_done = True
        need_draw = False

        for trainer in self.trainers:
            if trainer.done:
                continue
            all_done = False
            done = trainer.start_autogame_iteration(self.iteration)
            if self.iteration % 100 == 0:
                if self.min_x > trainer.min_x:
                    self.min_x = trainer.min_x
                if self.max_x < trainer.max_x:
                    self.max_x = trainer.max_x
                if self.min_y > trainer.min_y:
                    self.min_y = trainer.min_y
                if self.max_y < trainer.max_y:
                    self.max_y = trainer.max_y
            if done:
                need_draw = True
                self.autogame_save_results(trainer)
        if all_done:
            self.save_loss()
        if self.iteration % 20 == 0:
            need_draw = True
        if self.iteration > 0 and self.iteration % 100 == 0:
            c = self.canvas2
            with c:
                c.draw_plot([self.history2["loss"], self.history2["accuracy"]])
                c.draw_image(self.history2["layer1"])
                c.draw_image(self.history2["layer2"])
        self.iteration = self.iteration + 1

    def autogame_save_results(self, trainer: NetworkTrainer):
        print('Обучение Закончено')
        self.save_model(trainer)
        self.print_last_game(trainer)

        exec_time = round(time.time() - self.start_time, 2)
        print("--- %s seconds ---" % exec_time)

    def save_model(self, trainer: NetworkTrainer):
        torch.save(trainer.policy_estimator.network.state_dict(), DIR + 'smart_' + trainer.name + '.pth')

    def save_loss(self):
        self.canvas2.figure.savefig(DIR + 'smart_all_loss.png')

    def print_last_game(self, trainer: NetworkTrainer):
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
            prediction = trainer.policy_estimator.predict(image_tensor)  # Предсказываем
            if trainer.policy_estimator.device_in == 'cpu':
                action = np.argmax(prediction.cpu().detach().numpy())  # Получаем решение, что делать 'cpu'
            else:
                action = torch.max(prediction.detach(), 1)[1].item()  # Получаем решение, что делать 'cuda'
            # выполняем действие
            done = G.act_pg(action)
            # Отрисовка игры
            plt.imshow(G.get_weel_state())
            camera.snap()

        anim = camera.animate()

        anim.save(DIR + 'smart_' + trainer.name + '.gif', writer='PillowWriter', fps=50)

# app = App()
# app.start()
