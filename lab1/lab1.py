import pandas as pd

class State:
    def __init__(self):
        self.n = 2 # 3
        self.m = 1 # 3
        self.x_n = [1.0, 2.0] #[10.0, 14.0, 12.0]
        self.x_m_left = [[1.0, -1.0]] #[[4.0, 2.0, 1.0], [3.0, 1.0, 3.0], [1.0, 2.0, 5.0]]
        self.x_m_right = [5.0] #[180.0, 210.0, 244.0]
        self.symplex_rels = []
        self.x_m_rules = ["<="] # ["<=", "<=", "<="]
        self.basis_vars_indexes = [] # 3 4 5
        self.free_vars_indexes = [] # 0 1 2
        self.table = pd.DataFrame()
        self.rel_scores = []
        self.e_column = -1
        self.e_row = -1
        self.e_element = -1
        self.optimum = False

def calc_rel_scores(state: State):
    state.rel_scores = []
    for j in range(len(state.x_n)):
        delta_j = 0
        for i, index in enumerate(state.basis_vars_indexes):
            delta_j += state.x_n[index] * state.x_m_left[i][j]
        delta_j -=  state.x_n[j]
        state.rel_scores.append(delta_j)

def analyze_rel_scores(state: State) -> bool:
    # is_solvable = False
    # neg_scores = [score for score in state.rel_scores if score < 0]
    # for score in neg_scores:
    #     pos = 0
    #     # for line in state.x_m_left:
    #     #     for i, number in enumerate(line):
    #     #         if i == state.rel_scores.index(score):
    #     #             if number > 0:
    #     #                 pos += 1
    #     for j, score in enumerate(state.rel_scores):
    #         if score < 0:
    #             pos = sum(1 for i in range(state.m) if state.x_m_left[i][j] > 0)
    #     if pos == 0:
    #         print("\nЦелевая функция неограничена\n")
    #         return False
    for j, score in enumerate(state.rel_scores):
        if score < 0:
            pos = sum(1 for i in range(state.m) if state.x_m_left[i][j] > 0)
            if pos == 0:
                print("\nЦелевая функция неограничена\n")
                return False
    min_score = min(state.rel_scores)
    if min_score >= 0:
        # for i, var in enumerate(state.basis_vars_indexes): # [1, 4, 2]
        #     if var not in state.free_vars_indexes and state.x_m_right[i] != 0:
        #         print("\nОграничения задачи несовместны\n")
                # return False
        corner = "\n"
        zeros = [score for score in state.rel_scores if score == 0]
        if len(zeros) > len(state.x_n) / 2:
            corner = ". Ребро\n"
        print("\nНайдено оптимальное решение" + corner)
        state.optimum = True
        return False
    column_index = state.rel_scores.index(min_score)
    # column_coefs = []
    # for line in state.x_m_left:
    #     for i, number in enumerate(line):
    #         if i == column_index:
    #             column_coefs.append(number)
    # for number in column_coefs:
    #     if number > 0:
    #         is_solvable = True
    # if not is_solvable:
    #     print("Целевая функция неограничена")
    #     return False
    state.e_column = column_index
    return True

def find_symplex_rels(state: State) -> bool:
    basis_min = float('inf')
    state.e_row = -1
    state.symplex_rels = []

    for i, number in enumerate(state.x_m_right):
        if state.x_m_left[i][state.e_column] > 0:
            rel = number / state.x_m_left[i][state.e_column]
            if rel < basis_min: # <=
                basis_min = rel
                state.e_row = i
            state.symplex_rels.append(rel)
        else:
            state.symplex_rels.append(float('inf'))

    if state.e_row == -1:
        print("\nНет положительных коэффициентов в разрешающем столбце\n")
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
    for j, equation in enumerate(state.x_m_left):
        restr_string = ""
        for i, number in enumerate(equation):
            restr_string += f"{number}x_{i + 1}"
            restr_string += " + " if i < len(equation) - 1 else ""
        restr_string += " " + state.x_m_rules[j] + " " + str(state.x_m_right[j])
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
        state.table.loc[i + 1, 'Cδ'] = state.x_n[state.basis_vars_indexes[i]]
    
    state.table.loc[0:, 'B₀'] = ''
    for i in range(state.m):
        state.table.loc[i + 1, 'B₀'] = state.x_m_right[i]

    value = 0
    for i, basis_idx in enumerate(state.basis_vars_indexes):
        value += state.x_n[state.basis_vars_indexes[i]] * state.x_m_right[i]
    state.table.loc[state.m + 1, 'B₀'] = value
    
    for i, name in enumerate(column_names):
        state.table.loc[0, name] = state.x_n[i]
    
    for i, line in enumerate(state.x_m_left):
        for j, number in enumerate(line):
            state.table.loc[i + 1, column_names[j]] = number
    
    for i, name in enumerate(column_names):
        state.table.loc[state.m + 1, name] = state.rel_scores[i]
    
    print(state.table)

def to_canonical(state: State):
    for i in range(len(state.x_m_rules)):
        state.x_n.append(0)
        # state.basis_vars_indexes.append(i + state.m)
        # if i < state.n: state.free_vars_indexes.append(i)
        state.basis_vars_indexes.append(len(state.x_n) - 1)
    state.free_vars_indexes = list(range(state.n))
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
        if iteration == 1:
            calc_rel_scores(state)
            if not analyze_rel_scores(state):
                break
            print("\nБазисное решение\n")
            print_symplex_table(state)
            iteration += 1
            continue

        calc_rel_scores(state)
        if not analyze_rel_scores(state):
            break
        if not find_symplex_rels(state):
            break
        print(f"\nИтерация {iteration - 1}:\n")
        calc_rel_scores(state)
        print_symplex_table(state)
        iteration += 1
        if iteration > 10:
            break
    # print("\n")
    # for i, basis_idx in enumerate(state.basis_vars_indexes):
    #     if basis_idx < state.n:
    #         print(f"x{basis_idx + 1} = {state.x_m_right[i]:.2f}")

    # print("\n")
    # for i, basis_idx in enumerate(state.basis_vars_indexes):
    #     if basis_idx in state.free_vars_indexes:
    #         print(f"x{basis_idx + 1} = {state.x_m_right[i]:.2f}")
    # print(state.x_m_right)
    # print(state.basis_vars_indexes)
    # print(state.free_vars_indexes)
    if state.optimum:
        for free_idx in state.free_vars_indexes:
            if free_idx in state.basis_vars_indexes:
                print(f"x{free_idx + 1} = {state.x_m_right[state.basis_vars_indexes.index(free_idx)]:.2f}")
            else:
                print(f"x{free_idx + 1} = {0:.2f}")

        optimal_value = 0
        for i, basis_idx in enumerate(state.basis_vars_indexes):
            if basis_idx < len(state.x_n):
                optimal_value += state.x_n[basis_idx] * state.x_m_right[i]
        print(f"F = {optimal_value:.2f}")

state = State()
print("\nСтандартная форма\n")
print_func(state)
print_restrictions(state)

to_canonical(state)

print("\nКаноническая форма\n")
print_func(state)
print_restrictions(state)

symplex_iters(state)