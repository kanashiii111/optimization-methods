import pandas as pd
import numpy as np

class State:
    def __init__(self, c, A, b, rules):
        self.c_orig = np.array(c, dtype=float)
        self.A_orig = np.array(A, dtype=float)
        self.b_orig = np.array(b, dtype=float)
        self.rules = list(rules)

        self.m, self.n_vars = self.A_orig.shape

        # Инвертируем строки с отрицательными b
        for i in range(self.m):
            if self.b_orig[i] < 0:
                self.b_orig[i] = -self.b_orig[i]
                self.A_orig[i] = -self.A_orig[i]
                if self.rules[i] == "<=":
                    self.rules[i] = ">="
                elif self.rules[i] == ">=":
                    self.rules[i] = "<="

        self.A = np.copy(self.A_orig)
        self.b = np.copy(self.b_orig)
        self.basis_vars = []
        self.art_indices = []

    def to_canonical(self):
        current_A = self.A.tolist()

        slack_count = 0
        for i, rule in enumerate(self.rules):
            if rule == "<=":
                for r in range(self.m):
                    current_A[r].append(1.0 if r == i else 0.0)
                slack_count += 1
            elif rule == ">=":
                for r in range(self.m):
                    current_A[r].append(-1.0 if r == i else 0.0)
                slack_count += 1

        self.A = np.array(current_A)
        self.c = np.concatenate([self.c_orig, np.zeros(slack_count)])

        self.basis_vars = [-1] * self.m
        for j in range(self.A.shape[1]):
            col = self.A[:, j]
            if np.sum(col == 1.0) == 1 and np.sum(col == 0.0) == self.m - 1:
                row = np.where(col == 1.0)[0][0]
                if self.basis_vars[row] == -1:
                    self.basis_vars[row] = j

        art_cols = []
        for i in range(self.m):
            if self.basis_vars[i] == -1:
                col = np.zeros((self.m, 1))
                col[i, 0] = 1.0
                art_cols.append(col)

                new_art_idx = self.A.shape[1] + len(art_cols) - 1
                self.art_indices.append(new_art_idx)
                self.basis_vars[i] = new_art_idx

        if art_cols:
            self.A = np.hstack([self.A] + art_cols)
            self.c = np.concatenate([self.c, np.zeros(len(art_cols))])

    def print_table(self, title=""):
        if title: print(f"\n{title}")

        total_vars = self.A.shape[1]
        cols = [f'x{i+1}' for i in range(total_vars)]

        # Строка Cj сверху таблицы
        cj_row = ["", "", "Cj"] + [round(val, 2) for val in self.c]

        data = []
        for i in range(self.m):
            b_idx = self.basis_vars[i]
            c_val = self.c[b_idx]
            row = [f"x{b_idx+1}", round(c_val, 2), round(self.b[i], 2)]
            row += [round(val, 2) for val in self.A[i]]
            data.append(row)

        # Расчет строки дельт
        cb = self.c[self.basis_vars]
        deltas = cb @ self.A - self.c
        obj_val = cb @ self.b

        delta_row = ["Δ", "", round(obj_val, 2)] + [round(d, 2) for d in deltas]

        df = pd.DataFrame([cj_row] + data + [delta_row], columns=["БП", "Ci", "Bi"] + cols)
        print(df.to_string(index=False))
        print("\n") # "-" * 60

def primal_simplex(state: State):
    while True:
        cb = state.c[state.basis_vars]
        deltas = cb @ state.A - state.c

        if np.all(deltas >= -1e-9):
            return True

        col = np.argmin(deltas)

        if np.all(state.A[:, col] <= 0):
            print("Целевая функция не ограничена!")
            return False

        ratios = []
        for i in range(state.m):
            if state.A[i, col] > 1e-9:
                ratios.append(state.b[i] / state.A[i, col])
            else:
                ratios.append(np.inf)

        row = np.argmin(ratios)

        pivot = state.A[row, col]
        state.A[row] /= pivot
        state.b[row] /= pivot
        for i in range(state.m):
            if i != row:
                factor = state.A[i, col]
                state.A[i] -= factor * state.A[row]
                state.b[i] -= factor * state.b[row]

        state.basis_vars[row] = col
        state.print_table() # f"Итерация симплекса (вход x{col+1} в базис)"

def dual_simplex(state: State):
    while np.any(state.b < -1e-9):
        row = np.argmin(state.b)

        cb = state.c[state.basis_vars]
        deltas = cb @ state.A - state.c

        ratios = []
        for j in range(state.A.shape[1]):
            if state.A[row, j] < -1e-9:
                ratios.append(abs(deltas[j] / state.A[row, j]))
            else:
                ratios.append(np.inf)

        if np.all(np.array(ratios) == np.inf):
            print("Целочисленного решения не существует!")
            return False

        col = np.argmin(ratios)

        pivot = state.A[row, col]
        state.A[row] /= pivot
        state.b[row] /= pivot
        for i in range(state.m):
            if i != row:
                factor = state.A[i, col]
                state.A[i] -= factor * state.A[row]
                state.b[i] -= factor * state.b[row]
        state.basis_vars[row] = col
    return True

def add_gomory_cut(state: State):
    fractions = state.b % 1
    fractions = np.where(fractions > 0.99999, 0, fractions)
    fractions = np.where(fractions < 1e-9, 0, fractions)

    if np.all(fractions == 0):
        return False

    row_idx = np.argmax(fractions)

    cut_row = -(state.A[row_idx] % 1)
    cut_b = -(state.b[row_idx] % 1)

    state.A = np.vstack([state.A, cut_row])
    state.b = np.append(state.b, cut_b)
    state.m += 1

    new_col = np.zeros((state.m, 1))
    new_col[-1, 0] = 1.0
    state.A = np.hstack([state.A, new_col])
    state.c = np.append(state.c, 0.0)
    state.basis_vars.append(len(state.c) - 1)

    return True

def solve(c, A, b, rules = None):
    if rules is None:
        rules = ["<="] * len(A)

    st = State(c, A, b, rules)
    st.to_canonical()

    if st.art_indices:
        # print("\n") # "\n=== ФАЗА 1: Удаление искусственных переменных ==="
        c_saved = np.copy(st.c)
        st.c = np.zeros(st.A.shape[1])
        for idx in st.art_indices:
            st.c[idx] = -1.0 # Максимизируем (-Искусственные), т.е. ведем их к 0

        st.print_table("Начальная таблица") # Начальная таблица Фазы 1
        if not primal_simplex(st):
            return

        st.c = c_saved
        for idx in st.art_indices:
            st.c[idx] = 0.0
            st.A[:, idx] = 0.0

    st.print_table("") # Начальная таблица
    if not primal_simplex(st):
        return

    iter_g = 1
    while add_gomory_cut(st):
        st.print_table() # f"Добавлено отсечение Гомори №{iter_g}"
        if not dual_simplex(st):
            break
        st.print_table() # f"После двойственного симплекса (Итерация {iter_g})"
        iter_g += 1
        if iter_g > 15:
            break

    print("ответ")
    final_x = np.zeros(st.n_vars)
    for i, b_idx in enumerate(st.basis_vars):
        if b_idx < st.n_vars:
            final_x[b_idx] = st.b[i]

    for i in range(st.n_vars):
        print(f"x{i+1} = {round(final_x[i])}")

    f_res = sum(st.c_orig[i] * final_x[i] for i in range(st.n_vars))
    print(f"макс F = {round(f_res)}")

# c: list[float] = [1.0, -1.0]
# A: list[list[float]] = [
#     [-1.0, 2.0],
#     [3.0, 2.0]
# ]
# b: list[float] = [4.0, 14.0]

# c: list[float] = [100, 250]
# A: list[list[float]] = [
#     [10, 30],
#     [25, 25],
#     [41, 90],
#     [90, 50]
# ]
# b: list[float] = [4500, 6250, 14100, 18000]

c: list[float] = [3.0, -1.0, -5]
A: list[list[float]] = [
    [5, 1, 4],
    [3.0, 0, 2.0],
    [1.0, 0, -3.0]
]
b: list[float] = [4.0, 4.0, 3]

solve(c, A, b)
