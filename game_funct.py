from time import sleep

import numpy as np
import copy
import torch
from Game import Game


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

    data_set_out = copy.deepcopy(data_set_loc)

    # print('Результат серии = ', rollout_score)
    # print('Кол-во примеров = ', len(list_images))
    list_rolout_score.append(rollout_score)

    data_set_loc.clear()  # Словарь с ДатаСетом
    list_images.clear()  # Список массивов Изображений серии игр
    list_muvies.clear()  # Список Действий Агента
    list_scores.clear()  # Список Результатов серии игр

    return data_set_out['image'], data_set_out['muvie'], data_set_out['score']


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
def teach_net(batch, policy_estimator, optimizer, device_in='cuda'):
    optimizer.zero_grad()
    # Распаковываем батч данных
    # print(dir(batch))
    states, actions, rewards = [i for i in batch]

    # Преобразуем в тензоры
    image_tensor = torch.Tensor(np.expand_dims(states, axis=1)).float().to(device_in)
    score_tensor = torch.Tensor(rewards).float().to(device_in)

    action_tensor = torch.Tensor(actions).long().to(device_in)
    # Этап 1 - логарифмируем вероятности действий
    prob = torch.log(policy_estimator.predict(image_tensor))
    # Этап 2 - отрицательное среднее произведения вероятностей на награду
    selected_probs = score_tensor * prob[np.arange(len(action_tensor)), action_tensor]
    loss = -selected_probs.mean()
    loss.backward()
    optimizer.step()
    return selected_probs


# отрисовка процесса обучения - средний результат серии из 10 игр
def paint_mean_score(num_step, list_rolout_score, list_mean_score, canvas):
    step = 10  # Шаг усреднения

    if len(list_rolout_score) >= step:
        # Усредняем в обратном порядке
        for num in range(int(len(list_rolout_score) / step), 0, -1):
            mean_score = np.array(list_rolout_score[num * step - step:num * step]).mean()
            list_mean_score.append(mean_score)

        #list_mean_score = list_mean_score[::-1]  # разворачиваем список
        mn_sc = np.array(list_mean_score)
        # fig, subplot = plt.subplots()  # доступ к Figure и Subplot
        # subplot.plot(mn_sc)  # построение графика функции
        # gr_dir_name = './Grafics/'
        # gr_file_name = str(num_step) + '_' + str(mn_sc[-1]) + '_' + str(kol_point) + '_point_' + 'Conv2D' + '.png'
        # plt.title(gr_file_name, fontsize=12)
        # plt.savefig(gr_dir_name + gr_file_name)
        # plt.show()

        canvas.axes.plot(mn_sc, color='blue')
        canvas.axes.set_title('Lern epoh', fontsize=12)
        canvas.draw()
