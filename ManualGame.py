from time import sleep

import numpy as np
from Game import Game
from game_funct import discount_correct_rewards


class ManualGame(object):
    def __init__(self, canvas, higth_weel_g, width_weel_g, width_racet_g, max_point_weel_g, kol_point, min_len_between_point, finish_handler, list_col_point=[0.4]):
        self.list_images = []  # Список массивов Изображений серии игр
        self.list_muvies = []  # Список Действий Агента
        self.list_scores = []  # Список Результатов серии игр
        self.list_rolout_score = []  # Список Ролаутов Игр

        self.episode_list_images = []
        self.episode_list_muvies = []
        self.episode_list_scores = []
        self.episode_list_game_count = []
        self.episode_list_str_count = []

        self.num_episode = 0
        self.len_episode = 0

        self.canvas = canvas
        self.finish_handler = finish_handler

        self.game_done = False
        self.game = Game(higth_weel_g, width_weel_g, width_racet_g, max_point_weel_g, kol_point, min_len_between_point, list_col_point)

        self.rollout_score = 0

    def start(self):
        self.manual_game_draw()

    def finish(self):
        self.game_done = True
        print('game finished')
        # self.canvas.axes.fill()
        self.canvas.axes.set_title('Game Finished\nScore '+self.game.get_str_score(), fontsize=12)
        self.canvas.draw()
        timer = QtCore.QTimer()
        timer.singleShot(100,self.finish_handler)

    def get_results(self):
        return np.array(self.list_images), np.array(self.list_muvies), np.array(self.list_scores)

    def handle_action(self, key):
        if self.game_done:
            return
        c = key
        if not (c in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
            return
        if c == '9':
            c = '5'
        action = int(c)
        done = self.manual_game_player_action(action)
        self.manual_game_draw()
        if done:
            self.finish()
            return

    def manual_game_player_action(self, action):
        image = self.game.get_weel_state()
        old_count_game = self.game.get_score()
        done = self.game.act_pg(action)  # выполняем действие

        self.episode_list_images.append(image)
        self.episode_list_muvies.append(action)  # Заносим в список действие агента в эпизоде
        self.episode_list_str_count.append(self.game.get_str_score())  # Заносим в список кол-во пойманных точек в эпизоде
        self.episode_list_game_count.append(self.game.get_score())  # Заносим в список кол-во счет игры на момент эпизода

        self.episode_list_scores.append(0)  # Добавляем нулевое значение в список наград эпизода
        self.len_episode += 1

        # Проверяем Изменеие Счета в Игре
        if old_count_game != self.game.get_score():
            # Формирование награды на конец эпизода
            new_count_game = self.game.get_score()

            # Заполнение списка наград за весь эпизод
            mas = np.zeros((len(self.episode_list_scores)))
            mas[:] = new_count_game - old_count_game
            mas[:] = discount_correct_rewards(mas[:])
            episode_list_scores = mas.tolist()  # присваиваем всем значения награды в эпизоде награду эпизода

            # Фиксируем списки эпизода в списке игры
            self.list_images.extend(self.episode_list_images)
            self.list_muvies.extend(self.episode_list_muvies)
            self.list_scores.extend(self.episode_list_scores)

            # Очищаем списки эпизода
            self.episode_list_images = []
            self.episode_list_muvies = []
            self.episode_list_scores = []
            self.episode_list_str_count = []
            self.episode_list_game_count = []

            self.len_episode = 0
            self.num_episode += 1
        return done

    def manual_game_draw(self):
        image = self.game.get_weel_state()
        # Отрисовка игры
        self.canvas.axes.imshow(image)
        self.canvas.draw()

        #clear_output()
        #plt.imshow(self.game.get_weel_state())
        #plt.show()

        # ax.clear()
        # plt.title('Game', fontsize=12)
        # plt.imshow(image)
        # fig.canvas.draw()
        # plt.pause(0.02)  # pause
        # c = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'][random.randrange(0,9,1)]
