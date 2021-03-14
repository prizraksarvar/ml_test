from Weel import Weel
from experienceReplayBuffer import experienceReplayBuffer


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
