import pandas as pd
import numpy as np

class State:
    def __init__(self, c, A, b, rules):
        self.c_orig = np.array(c, dtype=float)
        self.A_orig = np.array(A, dtype=float)
        self.b_orig = np.array(b, dtype=float)
        self.rules = rules
        self.m, self.n_vars = self.A_orig.shape
        self.c = np.copy(self.c_orig)
        self.A = np.copy(self.A_orig)
        self.b = np.copy(self.b_orig)
        self.basis_vars = []
        
    def to_canonical(self):
        current_A = self.A.tolist()
        current_c = self.c.tolist()
        new_basis = [-1] * self.m
        
        for j in range(len(current_c)):
            col = [current_A[i][j] for i in range(self.m)]
            if col.count(1.0) == 1 and col.count(0.0) == self.m - 1:
                row_idx = col.index(1.0)
                if new_basis[row_idx] == -1:
                    new_basis[row_idx] = j
        
        for i in range(self.m):
            if new_basis[i] == -1:
                for r in range(self.m):
                    current_A[r].append(1.0 if r == i else 0.0)
                
                new_var_idx = len(current_A[0]) - 1
                new_basis[i] = new_var_idx
                
                if self.rules[i] == "<=":
                    current_c.append(0.0)
                else:
                    current_c.append(-1e6) 
        
        self.A = np.array(current_A)
        self.c = np.array(current_c)
        self.basis_vars = new_basis

    def print_table(self, title=""):
        if title: print(f"\n{title}")
        
        cols = [f'x{i+1}' for i in range(len(self.c))]
        
        data = []
        cj_row = ["", "", "Cj"] + [round(val, 2) for val in self.c]
        
        for i in range(self.m):
            b_idx = self.basis_vars[i]
            row = [
                f"x{b_idx+1}", 
                round(self.c[b_idx], 2), 
                round(self.b[i], 2)
            ]
            row += [round(val, 2) for val in self.A[i]]
            data.append(row)
            
        cb = self.c[self.basis_vars]
        deltas = cb @ self.A - self.c
        obj_val = cb @ self.b
        delta_row = ["Δ", "", round(obj_val, 2)] + [round(d, 2) for d in deltas]
        
        df = pd.DataFrame([cj_row] + data + [delta_row], 
                          columns=["БП", "Ci", "Bi"] + cols)
        print(df.to_string(index=False))
        print("-" * 60)

def primal_simplex(state: State):
    while True:
        cb = state.c[state.basis_vars]
        deltas = cb @ state.A - state.c
        
        if np.all(deltas >= -1e-9):
            return True # Оптимальное решение
        
        col = np.argmin(deltas)
        
        if np.all(state.A[:, col] <= 0):
            print("Целевая функция не ограничена")
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
        state.print_table(f"Итерация симплекса (вход x{col+1} в базис)")

def dual_simplex(state: State):
    while np.any(state.b < -1e-9):
        row = np.argmin(state.b)
        
        # Шаг 10
        cb = state.c[state.basis_vars]
        deltas = cb @ state.A - state.c
        
        ratios = []
        for j in range(state.A.shape[1]):
            if state.A[row, j] < -1e-9:
                ratios.append(abs(deltas[j] / state.A[row, j]))
            else:
                ratios.append(np.inf)
        
        if np.all(np.array(ratios) == np.inf):
            print("Целочисленного решения не существует (задача несовместна)")
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
    
    row_idx = np.argmax(fractions) # Шаг №3
    
    cut_row = -(state.A[row_idx] % 1) # Шаги №4-8
    cut_b = -(state.b[row_idx] % 1)
    
    state.A = np.vstack([state.A, cut_row]) # Шаг №9 добавление строки
    state.b = np.append(state.b, cut_b)
    state.m += 1
    
    new_col = np.zeros((state.m, 1))
    new_col[-1, 0] = 1.0
    state.A = np.hstack([state.A, new_col]) # Шаг №9 добавление столбца для переменной x_v
    state.c = np.append(state.c, 0.0)
    state.basis_vars.append(len(state.c) - 1)
    
    return True

def solve(c, A, b, rules):
    st = State(c, A, b, rules)
    st.to_canonical()
    st.print_table("Начальная таблица")
    
    if not primal_simplex(st): return # Шаг №1,2
    
    print("\n>>> Найдено оптимальное непрерывное решение. Начинаем метод Гомори.")
    
    iter_g = 1
    while add_gomory_cut(st):
        st.print_table(f"Добавлено отсечение Гомори №{iter_g}")
        if not dual_simplex(st):
            break
        st.print_table(f"Результат после двойственного симплекса (Гомори {iter_g})")
        iter_g += 1
        if iter_g > 10: break

    print("\nОтвет")
    final_x = np.zeros(st.n_vars)
    for i, b_idx in enumerate(st.basis_vars):
        if b_idx < st.n_vars:
            final_x[b_idx] = st.b[i]
    
    for i in range(st.n_vars):
        print(f"x{i+1} = {round(final_x[i])}")
    
    f_res = sum(st.c_orig[i] * final_x[i] for i in range(st.n_vars))
    print(f"Максимум F = {round(f_res)}")

# c = [3, -1, -5]
# A = [
#     [5, 1, 4],
#     [3, 0, 2],
#     [1, 0, -3]
# ]
# b = [4, 4, 3]
# rules = ["=", "<=", "<="]

# solve(c, A, b, rules)

c = [2.0, 3.0]
A = [
    [2.0, 5.0],
    [4.0, 2.0]
]
b = [16.0, 15.0]
rules = ["<=", "<="]

solve(c, A, b, rules)