import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

matplotlib.rcParams['pdf.fonttype'] = 42


class Column:
    STEP = 'Step'
    VALUE = 'Value'
    ALGORITHM = 'Algorithm'
    CRITERION = 'Criterion'


class Algorithm:
    SAC = 'SAC'
    SAC_LAG = 'SAC-L'
    CPO = 'CPO'
    SAC_HC = 'SAC-HC'
    SAC_VBL = 'SAC-VBL'
    SAC_FSI = 'SAC-FSI (ours)'


class Criterion:
    COST = 'Average episode cost'
    RETURN = 'Average episode return'


class Env:
    CARTPOLE = 'Cart-pole'
    QUADROTOR = 'Quadrotor'
    POINT_GOAL = 'Point goal'
    CAR_GOAL = 'Car goal'


algorithm_to_filename_pattern = {
    Algorithm.SAC: 'sac_s',
    Algorithm.SAC_LAG: 'lag_s',
    Algorithm.CPO: 'cpo_s',
    Algorithm.SAC_HC: 'hc_s',
    Algorithm.SAC_VBL: 'vbl_s',
    Algorithm.SAC_FSI: 'fsi_s',
}

criterion_to_filename_pattern = {
    Criterion.COST: 'cost',
    Criterion.RETURN: 'return',
}

env_to_filename_pattern = {
    Env.CARTPOLE: 'cartpole',
    Env.QUADROTOR: 'quadrotor',
    Env.POINT_GOAL: 'point_goal',
    Env.CAR_GOAL: 'car_goal',
}

algorithm_to_color = {
    Algorithm.SAC: 'C4',
    Algorithm.SAC_LAG: 'C0',
    Algorithm.CPO: 'C2',
    Algorithm.SAC_HC: 'C5',
    Algorithm.SAC_VBL: 'C1',
    Algorithm.SAC_FSI: 'C3',
}


def load_data(path, algorithms):
    data = {alg: [] for alg in algorithms}
    filenames = os.listdir(path)
    for filename in filenames:
        for alg in algorithms:
            if algorithm_to_filename_pattern[alg] in filename:
                df = pd.read_csv(os.path.join(path, filename), sep=',')
                step = np.linspace(0, int(1e6), 101)
                value = np.interp(step, df[Column.STEP], df[Column.VALUE])
                df = pd.DataFrame({
                    Column.STEP: step,
                    Column.VALUE: value,
                })
                df[Column.VALUE] = smooth(df[Column.VALUE])
                df.insert(loc=len(df.columns), column=Column.ALGORITHM, value=alg)
                data[alg].append(df)
                break
    dfs = []
    for df in data.values():
        dfs.extend(df)
    data = pd.concat(dfs, ignore_index=True)
    return data


def smooth(x, width=2):
    y = np.ones(width)
    z = np.ones(len(x))
    return np.convolve(x, y, 'same') / np.convolve(z, y, 'same')


if __name__ == '__main__':
    env_name = Env.CAR_GOAL
    criterion = Criterion.COST
    algorithms = [
        Algorithm.SAC,
        Algorithm.SAC_LAG,
        Algorithm.CPO,
        Algorithm.SAC_HC,
        Algorithm.SAC_VBL,
        Algorithm.SAC_FSI,
    ]
    magnifier = criterion == Criterion.COST
    magnifier_range = (-2, 5)

    data_path = os.path.join(
        '../results',
        env_to_filename_pattern[env_name],
        criterion_to_filename_pattern[criterion]
    )
    data = load_data(data_path, algorithms)

    sns.set(style='darkgrid', font_scale=1.0)
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))
    sns.lineplot(data=data, x=Column.STEP, y=Column.VALUE, hue=Column.ALGORITHM, ci=95,
                 palette=algorithm_to_color, legend=False, seed=0)
    plt.xlim(0, data[Column.STEP].max() * 1.005)
    plt.ylim(-3, 100)
    plt.xlabel('Environment step')
    plt.ylabel(criterion)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title(env_name)
    plt.tight_layout()

    if magnifier:
        axins = inset_axes(ax, width='40%', height='30%', loc='lower left',
                           bbox_to_anchor=(0.5, 0.6, 1, 1),
                           bbox_transform=ax.transAxes)
        sns.lineplot(data=data, x=Column.STEP, y=Column.VALUE, hue=Column.ALGORITHM, ci=95,
                     palette=algorithm_to_color, legend=False)
        plt.xlim(data[Column.STEP].max() * 0.9, data[Column.STEP].max())
        plt.ylim(magnifier_range)
        plt.xlabel(None)
        plt.ylabel(None)
        mark_inset(ax, axins, loc1=3, loc2=4, fc='none', ec='k', lw=1)

    save_path = '../figures/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + env_name.replace(' ', '') + '_' + criterion.replace(' ', '') + '.pdf')

    plt.show()
