import numpy as np
from collections import namedtuple, deque
import random
from turtle import pd

import torch.nn as nn
import torch.nn.functional as F

import time

import torch
from torch import optim

import torchvision.models
from torchvision import datasets, transforms
import hiddenlayer as hl


class experienceReplayBuffer:

    def __init__(self, memory_size=5000):
        self.memory_size = memory_size
        self.Buffer = namedtuple('Buffer', field_names=['state', 'action', 'reward'])
        self.replay_memory = deque(maxlen=memory_size)

    # Получаем Батч случайных примеров из буфера памяти
    def sample_batch(self, batch_size=64):
        if batch_size>len(self.replay_memory):
            batch_size = len(self.replay_memory)
        samples = np.random.choice(len(self.replay_memory), batch_size, replace=False)
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    # Получаем Батч случайных примеров из буфера памяти
    def all_random_sample_batch(self):
        samples = np.random.choice(len(self.replay_memory), len(self.replay_memory), replace=False)
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    # Получаем Батч случайных примеров из буфера памяти
    def all_sample_batch(self):
        batch = zip(*self.replay_memory)
        return batch

    # Добавление одиночного примера в буфер памяти
    def append(self, state, action, reward):
        self.replay_memory.append(self.Buffer(state, action, reward))

    # Добавление Батча примеров в буфер памяти
    def append_butch(self, state_batch, action_batch, reward_batch):
        for i in range(state_batch.shape[0]):
            self.append(state_batch[i], action_batch[i], reward_batch[i])

    # Очистка буфера памяти
    def clear(self):
        self.replay_memory.clear()
        return

    # Расчет длины буфера памяти с датасетом
    def print_len(self):
        return len(self.replay_memory)


class Point(object):
    """Точка"""

    def __init__(self, y_in=0, x_in=0, color_in=0.5):
        self.color = color_in  # Цвет точки
        self.y = y_in
        self.x = x_in

    # Функция Установки точки в произвольных координатах
    def set_point_position(self, y_in, x_in):
        self.y = y_in
        self.x = x_in

    # Функция получения координат точки
    def get_point_position(self):
        return self.y, self.x, self.color

    # Функция движения точки вниз
    def move_point_down(self):
        self.y += 1


class Racket(object):
    """Ракетка Игрока"""

    def __init__(self, width_r_in, color_r_in):
        self.width_r = width_r_in  # Ширина ракетки
        self.color_r = color_r_in  # Цвет ракетки
        self.x_rlc = 0  # Левый угол ракетки х-координата
        self.y_rlc = 0  # Левый угол ракетки х-координата

    # Функция Установки левого угла ракетки в произвольных координатах
    def set_position(self, y_rlc_in, x_rlc_in):
        self.y_rlc = y_rlc_in
        self.x_rlc = x_rlc_in

    # Функция движения ракетки влево
    def move_left(self):
        if self.x_rlc >= 1:
            self.x_rlc -= 1  # Left

    # Функция движения ракетки вправо
    def move_right(self):
        if self.x_rlc + self.width_r < self.width_w:
            self.x_rlc += 1  # Right

    # Функция движения ракетки вверх
    def move_up(self):
        if self.y_rlc >= 1:
            self.y_rlc -= 1  # Up

    # Функция движения ракетки вниз
    def move_down(self):
        if self.y_rlc < self.high_w - 1:
            self.y_rlc += 1  # Down

    # Функция движения ракетки вверх и лево
    def move_up_left(self):
        if (self.y_rlc >= 1) & (self.x_rlc >= 1):  # Если вверх и влево в границах колодца
            self.move_up()  # Up
            self.move_left()  # left

    # Функция движения ракетки вверх и вправо
    def move_up_right(self):
        if (self.y_rlc >= 1) & (self.x_rlc + self.width_r < self.width_w):  # Если вверх и вправо в границах колодца
            self.move_up()  # Up
            self.move_right()  # Right

    # Функция движения ракетки вниз и лево
    def move_down_left(self):
        if (self.y_rlc < self.high_w - 1) & (self.x_rlc >= 1):  # Если вниз и влево в границах колодца
            self.move_down()  # Down
            self.move_left()  # Left

    # Функция движения ракетки вниз и вправо
    def move_down_right(self):
        if (self.y_rlc < self.high_w - 1) & (
                self.x_rlc + self.width_r < self.width_w):  # Если вниз и вправо в границах колодца
            self.move_down()  # Down
            self.move_right()  # Right

    # Функция возвращающая список координат блоков ракетки
    def get_list_position(self):
        list_out = []
        for i in range(0, self.width_r):
            racet_block = self.y_rlc, self.x_rlc + i, self.color_r
            list_out.append(racet_block)
        return list_out


class Weel(Point, Racket):
    """Колодец"""

    def __init__(self, high_w_in, width_w_in, width_racet_g, color_w_in, kol_point, min_len_between_point):
        self.high_w = high_w_in  # Высота колодца
        self.width_w = width_w_in  # Ширина колодца
        self.color_w = color_w_in  # Цвет пустого колодца
        self.width_racket_in = width_racet_g  # Ширина ракетки

        self.kol_point = kol_point
        self.min_len_between_point = min_len_between_point

        self.reset_game()

    # Функция сбрасывающая игру в нулевое начальное состояние
    def reset_game(self):
        self.field = np.zeros((self.high_w, self.width_w))  # Пустой Двумерный массив пикселей колодца
        self.list_point = []  # Список Точек

        self.count_point = 0
        self.count_yellow_catch_point = 0
        self.count_yellow_uncatch_point = 0
        self.count_blue_catch_point = 0
        self.count_blue_uncatch_point = 0

        # Устанавливаем признак конца игры и конца эпизода в False
        self.done_game = False

        # Создание ракетки
        Racket.__init__(self, self.width_racet_g, color_r_in=1)
        # Установка ракетки в начальное положение в центре колодца
        self.set_position(self.high_w - 1, (self.width_w - self.width_racket_in) // 2)

        # Создание точки
        self.list_point.append(0)
        self.list_point[0] = Point(y_in=0, x_in=5, color_in=self.list_color_point[0])
        # Установка точки в координаты
        new_x = random.randint(0, self.width_w - 1)
        self.list_point[0].set_point_position(0, new_x)
        self.count_point += 1

    #        print ('Игра сброшена','*'*50)

    # Функция добавляющая еще одну точку на пустое поле вверху колодца, min_len - миниммальное растояние между точками
    def add_point_on_top_weel(self, min_len):
        # Проверка на то что бы новая точка была не ближе расстояния min_len
        list_y_point_position = []
        need_add_point = 0

        # Определяем, что точку нужно довалять если список точек пустой
        if len(self.list_point) == 0:
            need_add_point = 1
        # Определяем, что точку нужно довалять если превышено мин.раастояние между точками
        if len(self.list_point) > 0:
            for num_point in self.list_point:
                point_position = np.array(num_point.get_point_position())
                list_y_point_position.append(point_position[0])
            list_y_point_position.sort()
            if list_y_point_position[0] > min_len:
                need_add_point = 1
        # Если нужно, то Добавление новой точки
        if need_add_point == 1:
            if len(self.list_point) < self.max_point_weel_g:
                flag_add = 0
                while flag_add == 0:
                    new_x = random.randint(0, self.width_w - 1)
                    if self.field[0, new_x] == 0:
                        # Случайное Определение цвета для будущей точки
                        point_color = self.list_color_point[0]
                        rnd = random.random()
                        if (rnd < 0.5):
                            point_color = self.list_color_point[1]
                        # Создание и установка точки
                        new_num = len(self.list_point)
                        self.list_point.append(new_num)
                        self.list_point[new_num] = Point(y_in=0, x_in=new_x, color_in=point_color)
                        self.list_point[new_num].set_point_position(0, new_x)
                        self.count_point += 1
                        flag_add = 1

    # Функция двигающая все точки в колодце вниз на 1 шаг
    def move_all_point_down(self):
        # self.impact_point_in_racet()        # Считаем точки точки попавшие в ракетку
        # self.impact_point_on_bootom_weel()  # Считаем точки точки упавшие до дна колодца, пойманные и не пойманные ракеткой
        for num_point in self.list_point:
            num_point.move_point_down()

    # Функция считающая все точки достигшие попавшие в ракетку
    def impact_point_in_racet(self):
        # Получаем список координат ракетки по x
        list_racetki = np.array(self.get_list_position())
        list_racetki = list_racetki[:, 0:2].tolist()
        # print()
        # print(list_racetki)

        for num_point in self.list_point:
            point_position = np.array(num_point.get_point_position())
            short_point_position = point_position[0:2].tolist()
            # print(short_point_position)

            if short_point_position in list_racetki:
                # print('Попадание в ракетку')
                color_impact = point_position[2]
                if color_impact == self.list_color_point[0]:
                    self.count_yellow_catch_point += 1  # поймана точка желтого цвета
                else:
                    self.count_blue_catch_point += 1  # поймана точка синего цвета
                #                print('Правильных попаданий = ', self.count_right_catch_point, \
                #                      '   Неправильных попаданий = ',self.count_wrong_catch_point)
                # Удаление попавшей точки
                # self.list_point.remove(num_point)

    # Функция удаляющая все точки попавшие в ракетку
    def delete_impact_point_in_racet(self):
        # Получаем список координат ракетки по x
        list_racetki = np.array(self.get_list_position())
        list_racetki = list_racetki[:, 0:2].tolist()

        for num_point in self.list_point:
            point_position = np.array(num_point.get_point_position())
            short_point_position = point_position[0:2].tolist()

            # Удаление попавшей в ракетку точки
            if short_point_position in list_racetki:
                self.list_point.remove(num_point)

    # Функция считающая все точки достигшие дна колодца и не пойманные ракеткой
    def impact_point_on_bootom_weel(self):
        # Получаем список координат ракетки по x
        for num_point in self.list_point:
            if num_point.get_point_position()[0] == self.high_w:  # self.high_w-1:
                #                print('Попадание в Дно колодца')
                color_impact = num_point.get_point_position()[2]
                if color_impact == self.list_color_point[1]:
                    self.count_blue_uncatch_point += 1  # не поймана точка синего цвета
                else:
                    self.count_yellow_uncatch_point += 1  # не поймана точка желтого цвета
                #                print('Правильных попаданий в Дно колодца = ', self.count_right_uncatch_point, \
                #                      '   Правильных попаданий в Дно колодца = ',self.count_wrong_uncatch_point)
                # Удаление попавшей точки
                self.list_point.remove(num_point)

    # Функция считающая все точки достигшие дна колодца и не пойманные ракеткой
    def delete_impact_point_on_bootom_weel(self):
        # Получаем список координат ракетки по x
        for num_point in self.list_point:
            if num_point.get_point_position()[0] == self.high_w:  # self.high_w-1:
                # Удаление уппавшей точки и не пойманной ракеткой
                self.list_point.remove(num_point)

    # Функция печатающая состояние колодца (матрицу точек)
    def print_weel_state(self):
        # Обнуляем массив
        self.field = np.zeros((self.high_w, self.width_w))

        # Получаем данные по точкам, заносим их в массив
        for num_point in self.list_point:
            y_p, x_p, c_p = num_point.get_point_position()  # получаем координаты точки и ее цвет
            # print(y_p,x_p,c_p)
            self.field[y_p, x_p] = c_p  # заполням массив колодца

        # Получаем данные по блокам ракетки, заносим их в массив
        list_rpos = self.get_list_position()  # получаем список координаткоординаты блоков ракетки
        for num_block in list_rpos:
            y_r, x_r, c_r = num_block  # заполням массив колодца
            self.field[y_r, x_r] = c_r
        return (pd.DataFrame(self.field).astype(float))

    # Функция возвращающаяся состояние колодца (матрицу точек)
    def get_weel_state(self):
        # Обнуляем массив
        self.field = np.zeros((self.high_w, self.width_w))
        # Получаем данные по точкам, заносим их в массив
        for num_point in self.list_point:
            y_p, x_p, c_p = num_point.get_point_position()  # получаем координаты точки и ее цвет
            # print(y_p,x_p,c_p)
            self.field[y_p, x_p] = c_p  # заполням массив колодца
        # Получаем данные по блокам ракетки, заносим их в массив
        list_rpos = self.get_list_position()  # получаем список координаткоординаты блоков ракетки
        # print(list_rpos)
        for num_block in list_rpos:
            y_r, x_r, c_r = num_block  # заполням массив колодца
            self.field[y_r, x_r] = c_r
            # Поворачиваем массив изображения для Pytorch - работает с ошибкой
        # self.field[:, 0], self.field[:, 1] = self.field[:, 1], self.field[:, 0].copy()
        return (self.field.astype(float))

    # Функция выдающая кол-во созданных точек
    def get_count_point(self):
        return (self.count_point)

    # Функция возвращающая список имеющихся точек
    def get_list_of_point(self):
        return (self.list_point)

    # Функция выдающая кол-во правильно пойманных точек ракеткой
    def get_count_yellow_catch_point(self):
        return (self.count_yellow_catch_point)

    # Функция выдающая кол-во неправильно пойманных точек ракеткой
    def get_count_blue_catch_point(self):
        return (self.count_blue_catch_point)

    # Функция выдающая кол-во правильно пойманных точек ракеткой
    def get_count_yellow_uncatch_point(self):
        return (self.count_yellow_uncatch_point)

    # Функция выдающая кол-во неправильно пойманных точек ракеткой
    def get_count_blue_uncatch_point(self):
        return (self.count_blue_uncatch_point)

    def act_pg(self, action):
        # Проверяем на окончание игры (кол-во пойманных и непойманных точек равно кол-ву точек на игру)
        sum_catch_point = self.get_count_yellow_catch_point() + self.get_count_blue_catch_point()
        sum_uncatch_point = self.get_count_yellow_uncatch_point() + self.get_count_blue_uncatch_point()
        if sum_catch_point + sum_uncatch_point != self.kol_point:
            # Запоминаем счет игры
            old_count_game = self.get_game_count_dqn()
            # Запоминаем нулевое состояние среды и счет игры
            # s_0 = self.get_weel_state()

            # Движения в Среде
            self.move_all_point_down()  # Двигаем точки вниз
            self.impact_point_in_racet()  # Считаем точки попавшие в ракетку
            self.impact_point_on_bootom_weel()  # Считем точки упавшие на дно
            self.delete_impact_point_in_racet()  # Удаляем все точки попавшие в ракетку
            self.delete_impact_point_on_bootom_weel()  # Удаляем все точки со дна колодца и не пойманные ракеткой

            self.do_what_NN_say(action)  # Делаем предсказание нейросетью
            self.impact_point_in_racet()  # Считаем точки попавшие в ракетку
            self.impact_point_on_bootom_weel()  # Считем точки упавшие на дно
            self.delete_impact_point_in_racet()  # Удаляем все точки попавшие в ракетку
            self.delete_impact_point_on_bootom_weel()  # Удаляем все точки со дна колодца и не пойманные ракеткой

            # Добавляем еще одну точку
            if self.get_count_point() < self.kol_point:
                self.add_point_on_top_weel(self.min_len_between_point)

                # Запоминаем новое состояние среды после все действий
            s_1 = self.get_weel_state()
            # Запоминаем новый счет игры
            new_count_game = self.get_game_count_dqn()

            # Получаем Награду за действие
            reward = new_count_game - old_count_game
            # print('reward =', reward)

        # Действия если достигнут конец игры
        else:
            self.done_game = True

        return self.done_game

    def get_game_count_dqn(self):
        game_count = 0
        sum_drope_point = self.get_count_yellow_catch_point() + self.get_count_blue_catch_point() + self.get_count_yellow_uncatch_point() + self.get_count_blue_uncatch_point()
        if sum_drope_point > 0:
            sum_catch_point = self.get_count_yellow_catch_point() + self.get_count_blue_catch_point()
            sum_uncatch_point = self.get_count_yellow_uncatch_point() + self.get_count_blue_uncatch_point()
            game_count = (sum_catch_point - sum_uncatch_point)  # / sum_drope_point
        return game_count

    def get_str_game_count_dqn(self):
        sum_drop_point = self.get_count_yellow_catch_point() + self.get_count_blue_catch_point() + self.get_count_yellow_uncatch_point() + self.get_count_blue_uncatch_point()

        string_count = str(sum_drop_point) + ' = ' \
                       + str(self.get_count_yellow_catch_point()) + ':' + str(
            self.get_count_yellow_uncatch_point()) + ' /  ' \
                       + str(self.get_count_blue_catch_point()) + ':' + str(self.get_count_blue_uncatch_point())
        return string_count

    # Функция выдающая кол-во непойманных точек
    def do_what_NN_say(self, what_do):
        if what_do == 0:
            pass

        elif what_do == 4:
            self.move_left()

        elif what_do == 6:
            self.move_right()

        elif what_do == 8:
            self.move_up()

        elif what_do == 2:
            self.move_down()

        elif what_do == 7:
            self.move_up_left()

        elif what_do == 9 or what_do == 5:
            self.move_up_right()

        elif what_do == 1:
            self.move_down_left()

        elif what_do == 3:
            self.move_down_right()
        else:
            raise NameError('Ошибка в указании действия')

        return what_do

    # Функция переводящая номер действия в текст
    def translate_what_do(self, what_do):
        if what_do == 0:
            return 'Wait'
        elif what_do == 4:
            return 'Left'
        elif what_do == 6:
            return 'Right'
        elif what_do == 8:
            return 'Up'
        elif what_do == 2:
            return 'Down'
        elif what_do == 7:
            return 'Up_Left'
        elif what_do == 9 or what_do == 5:
            return 'Up_Right'
        elif what_do == 1:
            return 'Down_Left'
        elif what_do == 3:
            return 'Down_Right'
        else:
            raise NameError('Ошибка записанного действия')

    # Функция случайно двигающая ракетку
    def deside_what_to_do(self):
        return (self.move_random_racet_left_right())


class Game(Weel):
    """ Игра """

    def __init__(self, higth_weel_g, width_weel_g, width_racet_g, max_point_weel_g, kol_point, min_len_between_point, list_col_point=[0.4]):
        self.list_color_point = list_col_point  # Максимальное кол-во цветов точек)
        self.width_racet_g = width_racet_g  # Ширина ракетки
        self.max_point_weel_g = max_point_weel_g  # Максимальное одновременное кол-во точек в колодце
        self.higth_weel_g = higth_weel_g  # Высота колодца
        self.width_weel_g = width_weel_g  # Глубина колодца
        self.buffer = experienceReplayBuffer(memory_size=5000)  # , burn_in=1000)

        # Создание колодца
        Weel.__init__(self, higth_weel_g, width_weel_g, width_racet_g, 0, kol_point, min_len_between_point)

    def get_score(self):
        game_count = 0
        sum_drope_point = self.get_count_yellow_catch_point() + self.get_count_blue_catch_point() + self.get_count_yellow_uncatch_point() + self.get_count_blue_uncatch_point()
        if sum_drope_point > 0:
            sum_catch_point = self.get_count_yellow_catch_point() + self.get_count_blue_catch_point()
            sum_uncatch_point = self.get_count_yellow_uncatch_point() + self.get_count_blue_uncatch_point()
            game_count = (sum_catch_point - sum_uncatch_point)  # / sum_drope_point
        return game_count

    def get_str_score(self):
        sum_drop_point = self.get_count_yellow_catch_point() + self.get_count_blue_catch_point() + self.get_count_yellow_uncatch_point() + self.get_count_blue_uncatch_point()

        string_count = str(sum_drop_point) + ' = ' \
                       + str(self.get_count_yellow_catch_point()) + ':' + str(self.get_count_yellow_uncatch_point()) + ' /  ' \
                       + str(self.get_count_blue_catch_point()) + ':' + str(self.get_count_blue_uncatch_point())
        return string_count


# Одно входовая Сеть
# Convolutional neural network (two convolutional layers)
class MyCNNClassifier(nn.Module):
    def __init__(self, in_c, action_size):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(1024, 512)
        self.drop_out = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_0):
        out_0 = self.layer1(x_0)
        out_0 = self.layer2(out_0)
        out_0 = out_0.reshape(out_0.size(0), -1)

        out = F.relu(self.fc1(out_0))
        out = self.drop_out(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.softmax(out)
        return out


class PolicyEstimator(object):
    def __init__(self, statement_size, action_size, device_in='cuda'):
        self.n_inputs = statement_size
        self.n_outputs = action_size
        self.device_in = device_in

        # Define Conv network
        self.network = MyCNNClassifier(1, action_size).to(self.device_in)
        print(self.network)

    def predict(self, input_0):
        action_probs = self.network.forward(input_0.to(self.device_in))
        return action_probs



def game_funct(kol_game, policy_estimator, higth_weel_g, width_weel_g, width_racet_g, max_point_weel_g, kol_point,
               min_len_between_point, list_rolout_score):
    list_images = []  # Список массивов Изображений серии игр
    list_muvies = []  # Список Действий Агента
    list_scores = []  # Список Результатов серии игр
    data_set = {}  # Дата Сет для нейросети

    rollout_score = 0

    # Сама Игра
    for num_game in range(kol_game):

        game_list_images = []
        game_list_muvies = []
        game_list_scores = []

        episode_list_images = []
        episode_list_muvies = []
        episode_list_scores = []
        episode_list_game_count = []
        episode_list_str_count = []

        num_episode = 0
        len_episode = 0

        G = Game(higth_weel_g, width_weel_g, width_racet_g, max_point_weel_g, kol_point, min_len_between_point,
                 list_col_point=[0.5, 0.5])

        # Основной Цикл Игры
        done = False
        while done == False:

            # Запоминаем состояние награды на начало эпизода
            old_count_game = G.get_score()

            image = G.get_weel_state()

            # Блок выбора действия для Агента
            image_tensor = torch.Tensor(np.expand_dims([image], axis=0)).float()
            prediction = policy_estimator.predict(image_tensor)  # Предсказываем
            if policy_estimator.device_in == 'cpu':
                action = np.argmax(prediction.cpu().detach().numpy())  # Получаем решение, что делать 'cpu'
            else:
                action = torch.max(prediction.detach(), 1)[1].item()  # Получаем решение, что делать 'cuda'
            done = G.act_pg(action)  # выполняем действие

            episode_list_images.append(image)
            episode_list_muvies.append(action)  # Заносим в список действие агента в эпизоде
            episode_list_str_count.append(G.get_str_score())  # Заносим в список кол-во пойманных точек в эпизоде
            episode_list_game_count.append(G.get_score())  # Заносим в список кол-во счет игры на момент эпизода

            episode_list_scores.append(0)  # Добавляем нулевое значение в список наград эпизода
            len_episode += 1

            # Проверяем Изменеие Счета в Игре

            if old_count_game != G.get_score():
                # Формирование награды на конец эпизода
                new_count_game = G.get_score()

                # Заполнение списка наград за весь эпизод
                mas = np.zeros((len(episode_list_scores)))
                mas[:] = new_count_game - old_count_game
                mas[:] = discount_correct_rewards(mas[:])
                episode_list_scores = mas.tolist()  # присваиваем всем значения награды в эпизоде награду эпизода

                # Фиксируем списки эпизода в списке игры
                game_list_images.extend(episode_list_images)
                game_list_muvies.extend(episode_list_muvies)
                game_list_scores.extend(episode_list_scores)

                # Очищаем списки эпизода
                episode_list_images = []
                episode_list_muvies = []
                episode_list_scores = []
                episode_list_str_count = []
                episode_list_game_count = []

                len_episode = 0
                num_episode += 1

        # Оценка Серии результатов Игр
        rollout_score = rollout_score + G.get_score()

        list_images.extend(game_list_images)
        list_muvies.extend(game_list_muvies)
        list_scores.extend(game_list_scores)

    data_set_loc = {'image': np.array(list_images),
                    'muvie': np.array(list_muvies),
                    'score': np.array(list_scores)}

    list_rolout_score.append(rollout_score)

    return data_set_loc['image'], data_set_loc['muvie'], data_set_loc['score']


def discount_correct_rewards(r, gamma=0.98):  # Дисконтированная награда
    """ take 1D float array of rewards and compute discounted reward """
    r[:-1] = 0  # Обнуляем все значения в массиве кроме последней награды в эпизоде
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    # discounted_r -= discounted_r.mean()
    # discounted_r /- discounted_r.std()
    return discounted_r


# Одноканальная Игра
def manual_teach_net(batch, policy_estimator, optimizer, criterion, device_in='cuda'):
    optimizer.zero_grad()
    states, actions, rewards = [i for i in batch]

    # Преобразуем в тензоры
    image_tensor = torch.Tensor(np.expand_dims(states, axis=1)).float().to(device_in)
    action_tensor = torch.Tensor(actions).long().to(device_in)
    optimizer.zero_grad()
    net_out = policy_estimator.predict(image_tensor)
    loss = criterion(net_out, action_tensor)
    loss.backward()
    optimizer.step()


# Одноканальная Игра
def teach_net(batch, policy_estimator, optimizer, loss_fn, device_in='cuda'):
    optimizer.zero_grad()
    # Распаковываем батч данных
    # print(dir(batch))
    states, actions, rewards = [i for i in batch]

    # Преобразуем в тензоры
    image_tensor = torch.Tensor(np.expand_dims(states, axis=1)).float().to(device_in)
    score_tensor = torch.Tensor(rewards).float().to(device_in)

    action_tensor = torch.Tensor(actions).long().to(device_in)

    # Этап 1 - логарифмируем вероятности действий
    prob = torch.log(policy_estimator.predict(image_tensor)[np.arange(len(action_tensor)), action_tensor])
    # Этап 2 - отрицательное среднее произведения вероятностей на награду
    selected_probs = score_tensor * prob
    loss = -selected_probs.mean()
    loss.backward()
    optimizer.step()

    return loss.item()



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

class App(object):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.manual_game = None
        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        sc = MplCanvas(self, width=10, height=10, dpi=100)
        # sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])

        self.history2 = hl.History()
        self.canvas2 = hl.Canvas()

        self.start_time = time.time()

        self.canvas = sc
        self.setCentralWidget(sc)

        self.show()

        self.iteration = 0
        lr_in = 1e-4

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

        self.manual_buffer = experienceReplayBuffer(memory_size=5000)

        learning_rate = 1e-4
        # Осуществляем оптимизацию путем стохастического градиентного спуска
        # self.optimizerSGD = optim.SGD(self.policy_estimator.network.parameters(), lr=learning_rate, momentum=0.9)
        # Создаем функцию потерь
        # self.criterionSGD = nn.NLLLoss()

        self.trainers = []

        self.timer = QtCore.QTimer()

        self.max_x = 0
        self.min_x = 0
        self.max_y = 0
        self.min_y = 0
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

    def activations_hook1(self, inputs, output):
        """Intercepts the forward pass and logs activations.
        """
        batch_ix = step[1]
        if batch_ix and batch_ix % 100 == 0:
            # The output of this layer is of shape [batch_size, 16, 32, 32]
            # Take a slice that represents one feature map
            self.history2.log(step, layer1=output.data[0, 0])

    def activations_hook2(self, inputs, output):
        """Intercepts the forward pass and logs activations.
        """
        batch_ix = step[1]
        if batch_ix and batch_ix % 100 == 0:
            # The output of this layer is of shape [batch_size, 16, 32, 32]
            # Take a slice that represents one feature map
            self.history2.log(step, layer2=output.data[0, 0])

    def start_autogame(self):
        self.canvas.figure.clf()
        self.canvas.axes = self.canvas.figure.add_subplot(111)

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
            self.canvas.axes,
            'net1',
            'blue',
            None
        ))

        self.iteration = 0
        self.timer.singleShot(10, self.autogame_loop_func)

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
        if self.iteration % 100 == 0:
            self.canvas.axes.set_xlim(xmax=self.max_x + 100, xmin=self.min_x - 10)
            self.canvas.axes.set_ylim(ymax=self.max_y + 2, ymin=self.min_y - 2)
        self.iteration = self.iteration + 1
        if need_draw:
            exec_time = round(time.time() - self.start_time, 2)
            # print("--- %s seconds ---" % exec_time)
            self.canvas.axes.set_title(
                'Цикл обучения № ' + str(self.iteration)  + "\n--- %s seconds ---" % exec_time,
                fontsize=12)
            self.canvas.draw()
            self.timer.singleShot(10, self.autogame_loop_func)
        else:
            self.autogame_loop_func()

    def autogame_save_results(self, trainer: NetworkTrainer):
        print('Обучение Закончено')
        self.save_model(trainer)
        self.print_last_game(trainer)

        exec_time = round(time.time() - self.start_time, 2)
        print("--- %s seconds ---" % exec_time)
        trainer.axes.set_title(
            'Обучение закончено\nДанные сохранены' + "\n--- %s seconds ---" % exec_time,
            fontsize=12)

    def save_model(self, trainer: NetworkTrainer):
        torch.save(trainer.policy_estimator.network.state_dict(), DIR + 'smart_' + trainer.name + '.pth')

    def save_loss(self):
        self.canvas.figure.savefig(DIR + 'smart_all_loss.png')

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


# rc('animation', html='jshtml')

w = App()
w.start()
