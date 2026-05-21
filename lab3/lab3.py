import copy
import pandas as pd

class HungarianAlgorithm:
    def __init__(self, matrix, mode="min"):
        self.original_matrix = copy.deepcopy(matrix)
        self.matrix = copy.deepcopy(matrix)
        self.n = len(matrix)
        self.iteration = 0
        self.mode = mode

    def print_matrix(self, title, stars=None):
            if stars is None:
                stars = []
            print(f"--- {title} ---")

            # Собираем строковую матрицу
            str_matrix = []
            for i in range(self.n):
                row_str = []
                for j in range(self.n):
                    val = self.matrix[i][j]
                    if val == 0:
                        if (i, j) in stars:
                            row_str.append("0*")
                        else:
                            row_str.append("0/")
                    else:
                        row_str.append(str(val))
                str_matrix.append(row_str)

            # Используем DataFrame для красивого вывода
            df = pd.DataFrame(str_matrix)
            # Выводим без заголовков столбцов и индексов строк
            print(df.to_string(index=False, header=False))
            print()

    def _find_max_independent_zeros(self):
        """
        Ищет максимальное паросочетание (максимальное число независимых нулей).
        """
        match_r = [-1] * self.n
        match_c = [-1] * self.n

        def dfs(r, visited):
            for c in range(self.n):
                if self.matrix[r][c] == 0 and not visited[c]:
                    visited[c] = True
                    if match_c[c] == -1 or dfs(match_c[c], visited):
                        match_r[r] = c
                        match_c[c] = r
                        return True
            return False

        for i in range(self.n):
            visited = [False] * self.n
            dfs(i, visited)

        stars = []
        for i in range(self.n):
            if match_r[i] != -1:
                stars.append((i, match_r[i]))
        return stars

    def solve(self):
        self.print_matrix("Исходная матрица")

        # Предварительный этап
        if self.mode == 'max':
            for j in range(self.n):
                max_val = max(self.matrix[i][j] for i in range(self.n))
                for i in range(self.n):
                    self.matrix[i][j] = max_val - self.matrix[i][j]

            self.print_matrix(title="Предварительны этап, столбцы")

            for i in range(self.n):
                min_val = min(self.matrix[i])
                for j in range(self.n):
                    self.matrix[i][j] -= min_val

            self.print_matrix(title="Предварительны этап, строки")

        elif self.mode == 'min':
            for i in range(self.n):
                min_val = min(self.matrix[i])
                for j in range(self.n):
                    self.matrix[i][j] -= min_val

            self.print_matrix(title="Предварительны этап, строки")

            for j in range(self.n):
                min_val = min(self.matrix[i][j] for i in range(self.n))
                for i in range(self.n):
                    self.matrix[i][j] -= min_val

            self.print_matrix(title="Предварительны этап, столбцы")

        self.iteration += 1
        # Находим нули со звездочкой для вывода
        current_stars = self._find_max_independent_zeros()
        self.print_matrix(f"Итерация {self.iteration}: После предварительного этапа", current_stars)

        while True:
            # Этап 1: Поиск независимых нулей
            stars = self._find_max_independent_zeros()

            # Если независимых нулей n, решение найдено
            if len(stars) == self.n:
                print("Оптимальное решение получено!")
                self.print_matrix("Финальная матрица", stars)
                self.print_solution(stars)
                break

            # Этап 2: Выделение строк и столбцов (скрыто в логике множеств)
            marked_rows = set(range(self.n)) - set(r for r, c in stars)
            marked_cols = set()

            added = True
            while added:
                added = False
                for i in list(marked_rows):
                    for j in range(self.n):
                        if self.matrix[i][j] == 0 and j not in marked_cols:
                            marked_cols.add(j)
                            added = True

                for j in list(marked_cols):
                    for r, c in stars:
                        if c == j and r not in marked_rows:
                            marked_rows.add(r)
                            added = True

            # Этап 3: Модификация матрицы
            covered_rows = set(range(self.n)) - marked_rows
            covered_cols = marked_cols

            min_uncovered = float('inf')
            for i in range(self.n):
                if i not in covered_rows:
                    for j in range(self.n):
                        if j not in covered_cols:
                            if self.matrix[i][j] < min_uncovered:
                                min_uncovered = self.matrix[i][j]

            for i in range(self.n):
                for j in range(self.n):
                    if i not in covered_rows and j not in covered_cols:
                        self.matrix[i][j] -= min_uncovered
                    elif i in covered_rows and j in covered_cols:
                        self.matrix[i][j] += min_uncovered

            self.iteration += 1
            # Снова ищем звездочки для наглядного вывода
            current_stars = self._find_max_independent_zeros()
            self.print_matrix(f"Итерация {self.iteration}: Пересчет матрицы", current_stars)

    def print_solution(self, stars):
        total_cost = 0
        print("\n=== Ответ (Назначения) ===")
        # Сортируем для красивого вывода по номеру работника
        stars.sort(key=lambda x: x[0])
        for r, c in stars:
            cost = self.original_matrix[r][c]
            total_cost += cost
            #print(f"Механизм {r+1} -> Работа {c+1} (Производительность: {cost})")
            print(f"({r+1},{c+1}): {cost}")
        str = f"\nМинимальный суммарный эффект: {total_cost}" if self.mode == "min" else f"\nМаксимальный суммарный эффект: {total_cost}"
        print(str)

C = [
    [1, 4, 5, 8, 9, 4],
    [4, 6, 7, 8, 10, 11],
    [5, 18, 4, 7, 6, 7],
    [4, 4, 3, 6, 10, 4],
    [9, 10, 8, 9, 5, 13],
    [6, 8, 11, 12, 7, 8]
]

solver = HungarianAlgorithm(C, mode="max")
solver.solve()
