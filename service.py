import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

labelMap = {'U1': 'Общее количество ДТП',
            'U2': 'Количество раненых',
            'U3': 'Количество погибших',
            'U4': 'Со стажем управления до 2 лет\n\n\n\n',
            'U5': 'Со стажем управления от 5 до 10 лет',
            'U6': 'Со стажем управления свыше 15 лет',
            'U7': 'ДТП при нарушении ПДД хотя бы одним водителем',
            'U8': 'ДТП из-за неудовлетворительного состояния дорог',
            'U9': 'ДТП с участием водителя в алкогольном и/или наркотическом опьянении',
            'U10': 'ДТП из-за неудовлетворительного дорожного покрытия (вода, лед, снег)',
            'U11': 'ДТП на дорогах федерального значения',
            'U12': '\n\n\n\n\nДТП на дорогах регионального или муниципального значения',
            'U13': '\n\nДТП на дорогах местного значения',
            'U14': 'ДТП на платных автомобильных дорогах',
            'U15': 'ДТП на железнодорожных переездах'}

class CharacteristicFromExcel:
    def __init__(self, *args, **kwargs):
        if len(kwargs) == 0 or kwargs["excel"] is None:
            raise ValueError("excel is None")

        self.max_m = [5 for i in range(15)]
        self.start_values = list(range(15))
        self.res = {}
        self.init_chars = {}
        self.characteristics_labels = [col_name.replace('\n', '') for col_name in kwargs["excel"].columns[1:]]
        self.char_faks_index = self.characteristics_labels.index('Fak1')
        self.chars = []
        excel_rows = kwargs["excel"].values
        # excel_head = [col_name.replace('\n', '') for col_name in kwargs["excel"].columns[1:]]
        for index_row, excel_row in enumerate(excel_rows):
            name = excel_row[0]
            self.init_chars[name] = {}
            self.start_values[index_row] = excel_row[26]

            for i, cell in enumerate(excel_row[1:25]):
                self.init_chars[name][self.characteristics_labels[i]] = cell

            char_val = list(self.init_chars[name].values())
            self.chars.append(self.Characteristic(index_row + 1, name, char_val[:self.char_faks_index],
                                                  char_val[self.char_faks_index:]))

        self.func_m = {}
        self.fak_f = {'FaK1': fak1, 'FaK2': fak2, 'FaK3': fak3, 'FaK4': fak4, 'FaK5': fak5, 'FaK6': fak6, 'FaK7': fak7, 'FaK8': fak8, 'FaK9': fak9, 'FaK10': fak10}
        for i in self.chars:
            for f in (i.b + i.d):
                name = 'f' + str(len(self.func_m.keys()) + 1)
                self.func_m[name] = lambda t: 1
                i.funcs[f] = name

    class Characteristic:
        def __init__(self, index, label, funcs, faks):
            self.index = index
            self.label = labelMap[label]
            self.b = []
            self.b_fak = []
            self.d_fak = []
            self.d = []
            self.funcs = {}
            for ind, f in enumerate(funcs):
                if f == 1:
                    self.b.append("m" + str(ind + 1))
                elif f == -1:
                    self.d.append("m" + str(ind + 1))

            for ind, fak in enumerate(faks):
                if fak == 1:
                    self.b_fak.append("FaK" + str(ind + 1))
                elif fak == -1:
                    self.d_fak.append("FaK" + str(ind + 1))

        def calculate(self, max_val, funcs, faks):
            res_b_f = list(map(lambda x: funcs[self.funcs[x]], self.b))
            res_d_f = list(map(lambda x: funcs[self.funcs[x]], self.d))

            res_b_fak = list(map(lambda x: faks[x], self.b_fak))
            res_d_fak = list(map(lambda x: faks[x], self.d_fak))

            return lambda y, t: 1 / max_val * (np.prod(list(map(lambda x: x(t), res_b_f))) * sum(
                list(map(lambda x: x(t), res_b_fak))) - np.prod(list(map(lambda x: x(t), res_d_f))) * sum(
                list(map(lambda x: x(t), res_d_fak))))

    def init_par(self, Q1,Q2,Q3,Q4,Q5,Q1k,Q2k,Q3k,Q4k,Q5k):
        self.func_m['f' + str(Q1)] = lambda t: Q_tempalte(t, Q1k)
        self.func_m['f' + str(Q2)] = lambda t: Q_tempalte(t, Q2k)
        self.func_m['f' + str(Q3)] = lambda t: Q_tempalte(t, Q3k)
        self.func_m['f' + str(Q4)] = lambda t: Q_tempalte(t, Q4k)
        self.func_m['f' + str(Q5)] = lambda t: Q_tempalte(t, Q5k)

    def calculate(self, init_params):

        for i, char in enumerate(self.chars):
            t = np.linspace(0, 1, 110)  # vector of time
            m_c = char.calculate(self.max_m[i], self.func_m, self.fak_f)
            init_m_param = float(init_params[char.index - 1])
            y = odeint(m_c, init_m_param, t)  # solve eq.
            y = np.array(y).flatten()
            self.res[char.label] = y

        return self.res

    def get_graphics(self):
        fig = plt.figure(figsize=(10, 5)) #todo тут поправить чтобы график на говно не был похож

        # создаём область, в которой будет
        # - отображаться график
        t = np.linspace(0, 1, 110)  # vector of time
        # рисуем график
        legend_labels = []
        for char in self.chars:
            plt.plot(t, self.res[char.label], linewidth=2)
            legend_labels.append(char.label)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('время')
        plt.ylabel('значение характеристик')
        plt.legend(legend_labels, bbox_to_anchor=(1, 1))
        plt.show()
        fig.tight_layout()
        fig.savefig('funcs.png')

    def get_diag(self, t,filename):
        labels = list(self.res.keys())

        t_110 = np.linspace(0, 1, 110)
        res_index = 0
        for i, t_i in enumerate(t_110):
            if t_i > t:
                break
            res_index = i

        start_stats = [i[0] for i in self.res.values()]
        stats = [i[res_index] for i in self.res.values()]
        make_radar_chart('T=' + str(t),filename, start_stats, stats, labels)




def make_radar_chart(name, filename, initialStats, stats, attribute_labels):
    labels = np.array(attribute_labels)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    initialStats = np.concatenate((initialStats, [initialStats[0]]))
    stats = np.concatenate((stats, [stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.plot(angles, initialStats, 'o-', linewidth=2, color='r')
    ax.fill(angles, initialStats, alpha=0.25, color='r')
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_title(name)
    ax.grid(True)
    plt.tight_layout
    fig.savefig(filename, bbox_inches='tight')

    return plt.show()


def Q_tempalte(t, k):
    return float(k[3]) * t ** 3 + k[2] * t ** 2 + float(k[1]) * t + float(k[0])


def fak1(t):
    return 2 * t + 0.1


def fak2(t):
    res = np.where(t >= 0, 0.8, t)
    res = np.where(t > 0.25, 0.6, res)
    res = np.where(t > 0.6, 0.4, res)
    res = np.where(t > 0.8, 0.15, res)

    return res

def fak3(t):
    return np.cos(t * 10) / 3 + 0.35

def fak4(t):
    return  t / 3 + 0.3

def fak5(t):
    return np.cos(t * 5) / 6 + 0.2

def fak6(t):
    return np.exp(t) / 2 - 0.5

def fak7(t):
    res = np.where(t >= 0, 0.08, t)
    res = np.where(t > 0.25, 0.17, res)
    res = np.where(t > 0.45, 0.26, res)
    res = np.where(t > 0.55, 0.38, res)
    res = np.where(t > 0.70, 0.46, res)

    return res

def fak8(t):
    res = np.where(t >= 0, 0.05, t)
    res = np.where(t > 0.15, 0.09, res)
    res = np.where(t > 0.24, 0.15, res)
    res = np.where(t > 0.60, 0.20, res)

    return res

def fak9(t):
    res = np.where(t >= 0, 0.1, t)
    res = np.where(t > 0.15, 0.2, res)
    res = np.where(t > 0.2, 0.24, res)
    res = np.where(t > 0.43, 0.3, res)
    res = np.where(t > 0.58, 0.34, res)
    res = np.where(t > 0.73, 0.39, res)
    res = np.where(t > 0.94, 0.5, res)

    return res

def fak10(t):
    return 1 / (t + 1) - 0.2

excel_file_path = 'tableKush.xlsx'

excel_source = pd.read_excel(excel_file_path)
chars = CharacteristicFromExcel(excel=excel_source)


def get_faks_image():
    fig = plt.subplots()

    # создаём область, в которой будет
    # - отображаться график
    t = np.linspace(0, 1, 100)

    # рисуем график
    plt.plot(t, fak1(t))
    plt.plot(t, fak2(t))
    plt.plot(t, fak3(t))
    plt.plot(t, fak4(t))
    plt.plot(t, fak5(t))
    plt.plot(t, fak6(t))
    plt.plot(t, fak7(t))
    plt.plot(t, fak8(t))
    plt.plot(t, fak9(t))
    plt.plot(t, fak10(t))

    plt.xlabel('время')
    plt.ylabel('значение характеристик')

    # показываем график

    plt.ylim([0, 1])
    plt.legend(['FaK1', 'FaK2', 'FaK3', 'FaK4', 'FaK5', 'FaK6', 'FaK7', 'FaK8', 'FaK9', 'Fak10'], bbox_to_anchor=(1, 1))
    
    fig[0].tight_layout()
    fig[0].savefig('fak.png')
