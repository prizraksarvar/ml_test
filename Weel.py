import random
from turtle import pd

import numpy as np
from Point import Point
from Racket import Racket


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