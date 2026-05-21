import pandas as pd

def solve_binary_knapsack(weights, values, capacity):
    n = len(weights)
    # Создаем таблицу (n + 1) строк и (capacity + 1) столбцов, заполняем нулями
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    print("=== Метод динамического программирования: Бинарный рюкзак ===")
    print(f"Вместимость рюкзака: {capacity}")
    print(f"Веса предметов: {weights}")
    print(f"Ценности предметов: {values}\n")

    # Итерация 0 (Инициализация)
    print("--- Инициализация (0 предметов) ---")
    columns_labels = [f"Вес {w}" for w in range(capacity + 1)]
    df_init = pd.DataFrame([dp[0]], index=["Предмет 0"], columns=columns_labels)
    print(df_init.to_string())
    print("\n")

    # Заполнение таблицы DP с выводом каждой итерации
    for i in range(1, n + 1):
        w_i = weights[i - 1]
        v_i = values[i - 1]

        for w in range(capacity + 1):
            if w_i <= w:
                # Если предмет помещается, выбираем максимум между:
                # 1. Не брать предмет (остается предыдущая ценность)
                # 2. Взять предмет (добавляем его ценность к лучшей комбинации для оставшегося веса)
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - w_i] + v_i)
            else:
                # Предмет не помещается, переносим значение сверху
                dp[i][w] = dp[i - 1][w]

        # Вывод таблицы на текущей итерации
        print(f"--- Итерация {i}: Рассматриваем Предмет {i} (вес: {w_i}, ценность: {v_i}) ---")
        row_labels = [f"Предмет {k}" for k in range(i + 1)]
        df_iter = pd.DataFrame(dp[:i + 1], index=row_labels, columns=columns_labels)
        print(df_iter.to_string())
        print("\n")

    # Обратный проход (Backtracking) для поиска выбранных предметов
    result_value = dp[n][capacity]
    w = capacity
    selected_items = []

    for i in range(n, 0, -1):
        if result_value <= 0:
            break
        # Если значение пришло не сверху, значит мы взяли этот предмет
        if result_value != dp[i - 1][w]:
            selected_items.append(i) # Сохраняем номер предмета (индексация с 1)
            result_value -= values[i - 1]
            w -= weights[i - 1]

    selected_items.reverse() # Разворачиваем для порядка от меньшего к большему

    # Вывод финального решения
    print("=== Решение ===")
    print(f"Максимальная ценность: {dp[n][capacity]}")
    print(f"Номера выбранных предметов: {selected_items}")
    print(f"Суммарный вес выбранных предметов: {sum(weights[i-1] for i in selected_items)}")

weights = [2, 3, 2, 4, 1]
values = [1, 2, 3, 1, 1]
cap = 7

solve_binary_knapsack(weights, values, cap)
