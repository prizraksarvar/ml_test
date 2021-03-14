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
