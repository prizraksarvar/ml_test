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