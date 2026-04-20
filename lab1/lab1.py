import pandas as pd

class State:
    def __init__(self):
        self.n = 3
        self.m = 3
        self.x_n = [10.0, 14.0, 12.0]
        self.x_m_left = [[4.0, 2.0, 1.0], [3.0, 1.0, 3.0], [1.0, 2.0, 5.0]]
        self.x_m_right = [180.0, 210.0, 244.0]
        self.symplex_rels = []
        self.x_m_rules = ["<=", "<=", "<="]
        self.basis_vars_indexes = [] # 3 4 5
        self.free_vars_indexes = [] # 0 1 2
        self.table = pd.DataFrame()
        self.rel_scores = []
        self.e_column = -1
        self.e_row = -1
        self.e_element = -1

def calc_rel_scores(state: State):
    state.rel_scores = []
    for j in range(len(state.x_n)):
        delta_j = 0
        for i, index in enumerate(state.basis_vars_indexes):
            delta_j += state.x_n[index] * state.x_m_left[i][j]
        delta_j -=  state.x_n[j]
        state.rel_scores.append(delta_j)

def analyze_rel_scores(state: State) -> bool:
    is_solvable = False
    min_score = min(state.rel_scores)
    if min_score >= 0:
        print("\nРешение найдено")
        return False
    column_index = state.rel_scores.index(min_score)
    column_coefs = []
    for line in state.x_m_left:
        for i, number in enumerate(line):
            if i == column_index:
                column_coefs.append(number)
    for number in column_coefs:
        if number > 0:
            is_solvable = True
    if not is_solvable:
        print("Not solvable")
        return False
    state.e_column = column_index
    return True

def find_symplex_rels(state: State) -> bool:
    basis_min = state.x_m_right[0] / 2
    for i, number in enumerate(state.x_m_right):
        rel = number / 2
        if rel <= basis_min:
            basis_min = rel
            state.e_row = i
        state.symplex_rels.append(rel)

    if state.e_row == -1:
        print("Нет положительных коэффициентов в разрешающем столбце!")
        return False
    
    state.e_element = state.x_m_left[state.e_row][state.e_column]
    # ввод новой базисной переменной
    state.basis_vars_indexes[state.e_row] = state.e_column

    # делим строку на элемент
    for i, number in enumerate(state.x_m_left[state.e_row]):
        state.x_m_left[state.e_row][i] = number / state.e_element
    # делим b0 на элемент
    state.x_m_right[state.e_row] /= state.e_element
    
    # вычитаем из остальных строк
    for i in range(state.m):
        if i != state.e_row:
            multiplier = state.x_m_left[i][state.e_column]
            if multiplier != 0:
                for j in range(len(state.x_m_left[i])):
                    state.x_m_left[i][j] -= state.x_m_left[state.e_row][j] * multiplier
                state.x_m_right[i] -= state.x_m_right[state.e_row] * multiplier
    return True

def print_func(state: State):
    func_string = "f(x) = "
    for i, number in enumerate(state.x_n):
        func_string += f"{number}x_{i + 1} "
        func_string += "+ " if i < len(state.x_n) - 1 else ""
    print(func_string)

def print_restrictions(state: State):
    for equation in state.x_m_left:
        restr_string = ""
        for i, number in enumerate(equation):
            restr_string += f"{number}x_{i + 1}"
            restr_string += " + " if i < len(equation) - 1 else ""
        restr_string += " " + state.x_m_rules[state.x_m_left.index(equation)] + " " + str(state.x_m_right[state.x_m_left.index(equation)])
        print(restr_string)

def print_symplex_table(state: State):
    column_names = [f'x{i + 1}' for i in range(len(state.x_n))]
    basis_names = [f'x{i + 1}' for i in state.basis_vars_indexes]
    
    state.table = pd.DataFrame(index=range(state.m + 2), columns=['БП', 'Cδ', 'B₀'] + column_names)
    state.table = state.table.astype(object)
    
    state.table.loc[0:, 'БП'] = ''
    for i, name in enumerate(basis_names):
        state.table.loc[i + 1, 'БП'] = name
    
    state.table.loc[0:, 'Cδ'] = ''
    for i in range(state.m):
        state.table.loc[i + 1, 'Cδ'] = 0
    
    state.table.loc[0:, 'B₀'] = ''
    for i in range(state.m):
        state.table.loc[i + 1, 'B₀'] = state.x_m_right[i]
    
    for i, name in enumerate(column_names):
        state.table.loc[0, name] = state.x_n[i]
    
    for i, line in enumerate(state.x_m_left):
        for j, number in enumerate(line):
            state.table.loc[i + 1, column_names[j]] = number
    
    for i, name in enumerate(column_names):
        state.table.loc[state.m + 1, name] = state.rel_scores[i]
    
    print('\n')
    print(state.table)


# def get_func() -> list[int]:
#     x_n = []
#     n = int(input("Введите n: "))
#     for i in range(n):
#         x_n.append(int(input(f"Введите x_{i+1}: ")))
#     return n, x_n

# def get_restrictions(state: State):
#     m = int(input("Введите m: "))
#     x_m_left = []
#     x_m_right = []
#     x_m_rules = []
#     for i in range(m):
#         row = []
#         for j in range(n):
#             value = int(input(f"Введите x_{i + 1}_{j + 1}: "))
#             row.append(value)
#         x_m_left.append(row)
#         match input("Введите знак неравенства (>=, <=): "):
#             case ">=": x_m_rules.append(">=")
#             case "<=": x_m_rules.append("<=")
#         x_m_right.append(int(input(f"Введите правую часть неравенства {i + 1}: ")))
#     for i in range(m):
#         restr_string = ""
#         for j in range(n):
#             restr_string += f"{x_m_left[i][j]}x_{j + 1}"
#             restr_string += " + " if j < n - 1 else ""
#         restr_string += " " + x_m_rules[i] + " " + str(x_m_right[i])
#     return m, x_m_left, x_m_right, x_m_rules

def to_canonical(state: State):
    for i in range(len(state.x_m_rules)):
        state.x_n.append(0)
        state.basis_vars_indexes.append(i + state.m)
        state.free_vars_indexes.append(i)
    for i, rule in enumerate(state.x_m_rules):
        if rule == "<=":
            for j in range(state.m):
                state.x_m_left[i].append(1 if j == i else 0)
            state.x_m_rules[i] = "="
            continue
        if rule == ">=":
            for j in range(state.m):
                state.x_m_left[i].append(-1 if j == i else 0)
            state.x_m_rules[i] = "="
            continue
        
def symplex_iters(state: State):
    iteration = 1
    while True:
        calc_rel_scores(state)
        if not analyze_rel_scores(state):
            break
        if not find_symplex_rels(state):
            break
        print(f"\nИтерация {iteration}:")
        print_symplex_table(state)
        iteration += 1
        if iteration > 10:
            break
    calc_rel_scores(state)
    print_symplex_table(state)
    print("\n")
    for i, basis_idx in enumerate(state.basis_vars_indexes):
        if basis_idx < state.n:
            print(f"x{basis_idx + 1} = {state.x_m_right[i]:.2f}")

    # Вычисляем значение целевой функции
    optimal_value = 0
    for i, basis_idx in enumerate(state.basis_vars_indexes):
        if basis_idx < len(state.x_n):
            optimal_value += state.x_n[basis_idx] * state.x_m_right[i]
    print(f"F = {optimal_value:.2f}")


state = State()
print("\nbefore\n")
print_func(state)
print_restrictions(state)

to_canonical(state)

print("\nafter\n")
print_func(state)
print_restrictions(state)

symplex_iters(state)