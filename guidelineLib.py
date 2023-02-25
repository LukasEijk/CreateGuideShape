import numpy as np
import matplotlib.pyplot as plt
import sympy as smp
from sympy import S
import scipy.linalg as lin
import itertools
import time
from tqdm import tqdm
import warnings
from tqdm import TqdmWarning


def calc_points(coord, x, funcs):
    """This function creates a set of four coordinates out of a given global x position"""
    coord = [coord, coord, 0]
    u1 = [0, 0, 0]
    u2 = [0, 0, 0]
    u3 = [0, 0, 0]
    u4 = [0, 0, 0]

    # x - coordinate
    u1[0] = u2[0] = u3[0] = u4[0] = coord[0]

    y, z, width, height = funcs
    # y koord für u
    y_ = y.evalf(subs={x: coord[0]})
    w_ = width.evalf(subs={x: coord[0]})
    u1[1] = u4[1] = y_ - w_ / 2
    u3[1] = u2[1] = y_ + w_ / 2

    # z- koordinate
    z_ = z.evalf(subs={x: coord[0]})
    h_ = height.evalf(subs={x: coord[0]})
    u1[2] = u2[2] = z_ - h_ / 2
    u3[2] = u4[2] = z_ + h_ / 2

    return u1, u2, u3, u4


def calc_extrema(funcs, x, start, stop):
    # calculate extremes

    critical_points = list()

    for entry in funcs:
        point = smp.solve(entry, x)

        for p in point:
            if start < p < stop:
                critical_points.append(p)
    critical_points.append(stop)

    break_points = critical_points.copy()
    break_points = np.array(break_points, dtype='float32')

    return break_points


def pad(input_array):
    max_len = max(len(sublst) for sublst in input_array)
    for ind, a in enumerate(input_array):
        d = [0 for i in range(max_len)]

        for i in range(len(a)):
            d[i] = a[i]
            input_array[ind] = d

    return input_array


def near_maximum(extremes, x0, delta_=0.2):
    filtered = extremes[(x0 + delta_ / 2 > extremes) & (x0 - delta_ / 2 < extremes)]
    filtered.sort()

    if len(filtered) > 0:
        return True
    else:
        return False


def get_xDifferences(x_list):
    test = list()
    for ind, entry in enumerate(x_list):
        if ind <= len(x_list) - 2:
            test.append(x_list[ind + 1] - entry)
    return test


def calc_new_x(funcs, x, x0, dfuncs, max_xStep, delta):
    """funcs = funcs,x=x,  x0 = xnew, dfuncs = dfuncs, max_xStep = max_xStep, delta = .1"""

    diffs = [funcs[i] - (funcs[i].evalf(subs={x: x0}) + dfuncs[i].evalf(subs={x: x0}) * (x - x0)) for i, entry in
             enumerate(funcs)]

    l = list()
    for diff in diffs:
        e = smp.solveset(smp.Abs(diff) - delta, x, domain=S.Reals)
        # e = smp.solveset(diff + delta, x, domain= S.Reals)
        # print(e)
        l.append(e)

    dummy = list()
    for i in range(len(l)):
        e = np.array(list(set(l[i])), dtype='float32')
        dummy.append(e)

    # entferne alle liste mit länge 0
    dummy = [entry for entry in dummy if len(entry) > 0]
    dummy = np.array(pad(dummy))

    d = dummy[dummy > x0]
    d.sort()

    pot_length = d[0] - x0
    # print('pot_length: ' , pot_length)
    assert pot_length > 0

    if pot_length > max_xStep:
        x_new = x0 + max_xStep

    else:
        x_new = d[0]

    return x_new


def create_xList(funcs, dfuncs, x, start, stop, max_xStep, delta):
    """Takes the functions as an input an calculates the x Points where the shapes are being created"""
    starttime = time.time()
    x0 = start
    x_list = [x0]

    cnt = 0
    while x0 < stop:
        cnt += 1
        x0 = calc_new_x(funcs, x, x0, dfuncs, max_xStep, delta)

        if x0 < stop:
            x_list.append(x0)
        else:
            x_list.append(stop)

        if cnt % 50 == 0:
            time_new = time.time()
            time_passed = np.round(starttime - time_new, 2)
            txt = 'sec'
            if time_passed > 60:
                txt = ' min'
                time_passed = np.round(time_passed / 60, 2)
            print(f'''Fortschritt:  {np.round(x0 / stop * 100, 2)}%        \nLaufzeit: {np.round(time_passed, 2)}''',
                  end="\r")

    print('                                                  ', end='\n')
    print('Fortschritt:  100.00%')
    print(f'Runtime: {time_passed}' + txt)
    print(f'Created {len(x_list) - 1} guide shapes, the longest one is: {np.round(max(get_xDifferences(x_list)), 2)}')

    return x_list


def create_xList_ProgressBar(funcs, dfuncs, x, start, stop, max_xStep, delta):
    """Takes the functions as an input an calculates the x Points where the shapes are being created"""
    starttime = time.time()
    x0 = start
    x_list = [x0]

    cnt = 0
    x_save_for_later = 0
    max_steps = 1000
    current_step = 0
    warnings.filterwarnings('ignore', category=TqdmWarning)
    with tqdm(total=(stop - start), unit_scale=True, leave=True) as pbar:
        while x0 < stop:
            cnt += 1
            x0 = calc_new_x(funcs, x, x0, dfuncs, max_xStep, delta)

            if x0 < stop:
                x_list.append(x0)
            else:
                x_save_for_later = x0
                x_list.append(stop)
                x0 = stop

            #pbar.update((x0 - start) / (stop - start) )
            update_val = max(min((x0 - start) / (stop - start), 1), 0)

            pbar.update(update_val)

            if x_save_for_later > stop:
                x0 = x_save_for_later

    time_new = time.time()
    time_passed = np.round(starttime - time_new, 2)
    print(f'Runtime: {time_passed}' + 'seconds')
    print(f'Created {len(x_list) - 1} guide shapes, the longest one is: {np.round(max(get_xDifferences(x_list)), 2)}')

    return x_list


def create_coordlist_for_off(xlist, x, funcs):
    c = list()
    for val in xlist:
        cs = calc_points(val, x, funcs)

        for entry in cs:
            c.append(entry)
    rects = np.reshape(c, (len(c) // 4, 4, 3))
    return rects


def str_point(coord_list):
    txt = ' '

    for entry in coord_list:
        txt += str(entry) + ' '
    return txt[:-1]


def str_point_obj(coord_list):
    txt = 'v '

    for entry in coord_list:
        txt += str(entry) + ' '
    return txt[:-1]


def write_blender_off(rects):
    """rects = rects
    Takes the rectangles list and turns it into a blender file"""
    object_cnt = 0

    edges_numbers = np.array([
        [1, 5, 8, 4],
        [2, 6, 7, 3],
        [4, 8, 7, 3],
        [1, 5, 6, 2]
    ])

    edges_text = '''f 1 5 8 4
    f 2 6 7 3
    f 4 8 7 3
    f 1 5 6 2'''
    output_text = ''
    for i in range(len(rects) - 1):

        object_text = f'o Leiter{object_cnt} \n'
        object_cnt += 1

        point_str_ = ''
        for point in rects[i]:
            point_str_ += str_point_obj(point)
            point_str_ += '\n'
        for point in rects[i + 1]:
            point_str_ += str_point_obj(point)
            point_str_ += '\n'
        point_str_ = point_str_[:-1]

        edges_text = ''
        for face in range(len(edges_numbers)):

            face_text = 'f '
            for eintrag in edges_numbers[face]:
                face_text = face_text + str(eintrag + i * 8) + ' '
            edges_text = edges_text + face_text + '\n'
            # print(edges_text)

        output_text += object_text + point_str_ + '\n' + edges_text + '\n' + '\n'

    with open(f'try_ges.obj', 'w') as file:
        file.write(output_text)
        print('File wurde erstellt')


def plot_results(funcs, x, x_list, start, stop):
    yvals_list = [np.array([func.evalf(subs={x: x0}) for x0 in x_list], dtype='float32') for func in funcs]
    xs = np.linspace(start, stop, 1000)
    ys_list = [np.array([func.evalf(subs={x: x0}) for x0 in xs], dtype='float32') for func in funcs]

    fig, ax = plt.subplots(figsize=(7, 5), nrows=2, ncols=2, sharex=True)

    col = 0
    row = 0
    titles = ['y', 'z', 'width', 'height']
    for ind, entry in enumerate(ys_list):
        ax[row, col].plot(xs, ys_list[ind], linewidth=10, label='y-func', alpha=.7)
        ax[row, col].plot(x_list, yvals_list[ind], marker='o', markersize=3, c='k')
        ax[row, col].set_title(titles[ind])
        col += 1

        if col > 1:
            row += 1
            col = 0

    plt.tight_layout()
    plt.show()
