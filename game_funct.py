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
