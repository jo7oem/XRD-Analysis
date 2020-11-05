import argparse
import csv
import math

import numpy as np
from scipy import optimize
from typing import List

# parser init begin
VERSION = "0.1.0"
parser = argparse.ArgumentParser()
parser.add_argument("-w", "--window", action='append', help='Specify waveform window range. ex) \'-w 50-80\'')
parser.add_argument("-s", "--smoothing", default=7, type=int, help='Number of data points for smoothing.')
parser.add_argument("--maxfev", default=600, type=int, help='scipy curv_fit maxfev option')
parser.add_argument("--printSmooth", action='store_true', help='print post smoothing value.')
parser.add_argument('filename', metavar='file', type=str, help='read xrd csv file.')
parser.add_argument('peaks', metavar='Peak', type=str, nargs='+',
                    help='Use peak window and an xrd peak position x. ex) 0,66')
args = parser.parse_args()


# parser init end
class DataSet:
    def __init__(self):
        self.x = []
        self.y = []
        self.smooth = False
        return

    def append(self, T: tuple):
        if len(T) != 2:
            raise ValueError
        self.x.append(T[0])
        self.y.append(T[1])

    def pare_list(self) -> list:
        res = []
        for i in range(len(self.x)):
            res.append([self.x[i], self.y[i]])
        return res

    def deduce_func_x(self, perm, f, params, **kwargs):
        res = []
        for i in range(len(self)):
            val = self.y[i] - f(self.x[i], *params, **kwargs)
            if perm:
                self.y[i] = val
            else:
                res.append(val)
        if not perm:
            new = DataSet()
            new.x = self.x
            new.y = res
            return new
        return

    def search_x(self, x):
        for i in range(len(self.x)):
            if self.x[i] >= x:
                return self.y[i]
        return None

    def smoothing(self, point):
        if point <= 1:
            return
        elif point <= 5:
            return
        elif point <= 7:
            self.smoothing_7()
            return
        else:
            return

    def smoothing_5(self):
        if self.smooth:
            return
        ly = len(self.y)
        sw = [self.y[0]] * 5
        sw[4] = self.y[1]

        for i in range(ly):
            for j in range(4):
                sw[j] = sw[j + 1]
            if i >= ly - 2:
                sw[4] = self.y[ly - 1]
            else:
                sw[6] = self.y[i + 2]
            self.y[i] = (-3 * sw[0] + 12 * sw[1] + 17 * sw[2] + 12 * sw[3] + -3 * sw[4]) / 35
        self.smooth = True
        return

    def smoothing_7(self):
        if self.smooth:
            return
        ly = len(self.y)
        sw = [self.y[0]] * 7
        sw[5] = self.y[1]
        sw[6] = self.y[2]

        for i in range(ly):
            for j in range(6):
                sw[j] = sw[j + 1]
            if i >= ly - 3:
                sw[6] = self.y[ly - 1]
            else:
                sw[6] = self.y[i + 3]
            self.y[i] = (-2 * sw[0] + 3 * sw[1] + 6 * sw[2] + 7 * sw[3] + 6 * sw[4] + 3 + sw[5] - 2 * sw[6]) / 21
        self.smooth = True
        return

    def smoothing_9(self):
        if self.smooth:
            return
        ly = len(self.y)
        sw = [self.y[0]] * 9
        sw[6] = self.y[1]
        sw[7] = self.y[2]
        sw[8] = self.y[3]

        for i in range(ly):
            for j in range(8):
                sw[j] = sw[j + 1]
            if i >= ly - 4:
                sw[8] = self.y[ly - 1]
            else:
                sw[8] = self.y[i + 4]
            self.y[i] = (-21 * sw[0] + 14 * sw[1] + 39 * sw[2] + 54 * sw[3] + 59 * sw[4] + 54 + sw[5] + 39 * sw[
                6] + 14 * sw[7] - 21 * sw[8]) / 231
        self.smooth = True
        return

    def __len__(self):
        return len(self.x)


def read_csv(file_name: str, skip_line: int = 0) -> DataSet:
    line = 0
    csv_data = DataSet()
    with open(file_name, mode='r') as f:
        readr = csv.reader(f)
        for row in readr:
            if line < skip_line:
                line += 1
                continue
            if len(row) < 2:
                break
            xn = float(row[0])
            yn = float(row[1])
            csv_data.append((xn, yn))
    return csv_data


def separate_window(data: DataSet, window_list) -> (DataSet, List[DataSet]):
    if len(window_list) == 0:
        return data, []
    window_data = []
    not_window_data = DataSet()
    for _ in range(len(window_list)):
        window_data.append(DataSet())

    for x, y in data.pare_list():
        window_number = 0
        is_in_window = False
        for wl in window_list:
            if wl[0] <= x <= wl[1]:
                is_in_window = True
                window_data[window_number].append((x, y))
            window_number += 1

        if not is_in_window:
            not_window_data.append((x, y))

    return not_window_data, window_data


def gaussian(x: float, a: float, x0: float, sigma: float, BG: float) -> float:
    return a * gaussian_func(x, x0, sigma) + BG


def gaussian_func(x: float, x0: float, sigma: float) -> float:
    return np.exp(-1 * pow(x - x0, 2) / (2 * pow(sigma, 2))) / (sigma * math.sqrt(2 * math.pi))


def lorentzian(x: float, a: float, x0: float, sigma: float, bg: float) -> float:
    return a * lorentzian_func(x, x0, sigma) + bg


def lorentzian_func(x: float, x0: float, sigma: float) -> float:
    return (sigma / (pow(x - x0, 2) + pow(sigma, 2))) / math.pi


def pseudo_voigt(x: float, a: float, x0: float, sigma: float, eta: float, bg: float) -> float:
    return a * pseudo_voigt_func(x, x0, sigma, eta) + bg


def pseudo_voigt_func(x: float, x0: float, sigma: float, eta: float) -> float:
    return eta * lorentzian_func(x, x0, sigma) + (1 - eta) * gaussian_func(x, x0, sigma)


def multi_dim_func(x: float, *args) -> float:
    res = 0.0
    for i in range(len(args)):
        res = res + args[i] * pow(x, i)
    return res


def peak_fit(xy: DataSet, peaks, window):
    if len(peaks) < 1:
        return [], []
    p0 = []
    sigma = 0.2
    bounds_bottom = []
    bounds_top = []
    for peak in peaks:
        p0.append(xy.search_x(peak))
        bounds_bottom.append(10)
        bounds_top.append(np.inf)

        p0.append(peak)
        bounds_bottom.append(window[0])
        bounds_top.append(window[1])

        p0.append(sigma)
        bounds_bottom.append(0.005)
        bounds_top.append(np.inf)

        p0.append(0.5)
        bounds_bottom.append(0)
        bounds_top.append(1)

    p0.append(0.0)
    bounds_bottom.append(0)
    bounds_top.append(np.inf)
    bounds = (tuple(bounds_bottom), tuple(bounds_top))
    return optimize.curve_fit(sum_pseudo_voigt, xy.x, xy.y, p0=p0, bounds=bounds, maxfev=args.maxfev)


def sum_gaussians(x, *params):
    len_peak = int(len(params) / 3)
    res = 0.0

    for i in range(len_peak):
        res += gaussian(x=x, a=params[i * 3], x0=params[i * 3 + 1], sigma=params[i * 3 + 2], BG=0)
    return res + params[-1]


def sum_pseudo_voigt(x, *params):
    len_peak = int(len(params) / 4)
    res = 0.0

    for i in range(len_peak):
        res += pseudo_voigt(x=x, a=params[i * 4], x0=params[i * 4 + 1], sigma=params[i * 4 + 2], eta=params[i * 4 + 3],
                            bg=0)
    return res + params[-1]


def FWHM(sigma):
    return 2 * np.sqrt(2 * math.log(2, math.e)) * sigma


def main():
    target_file = args.filename
    window_list = []
    for window_str in args.window:
        x1, x2 = window_str.split(sep='-')
        if float(x1) < float(x2):
            bt = float(x1)
            up = float(x2)
        else:
            bt = float(x2)
            up = float(x1)
        window_list.append((bt, up))

    fittinglist = [[]]
    for i in range(len(window_list)):
        fittinglist.append([])
    for fitting_hint in args.peaks:
        separated = fitting_hint.split(sep=',')
        if len(separated) != 2:
            print("[ERROR]: " + fitting_hint + " is bad request.")
            return
        window_num = int(separated[0])
        peak_hint = float(separated[1])
        if not 0 <= window_num <= len(window_list):
            print("[ERROR]: " + str(window_num) + " in " + fitting_hint + " is bad request.")
            return

        fittinglist[window_num].append(peak_hint)

    try:
        xrd_orgine = read_csv(target_file)
    except FileNotFoundError:
        print(target_file + " is not found.")
        return

    xrd_orgine.smoothing(args.smoothing)
    BG_sumple, _ = separate_window(xrd_orgine, window_list)

    xs = np.array(BG_sumple.x)
    ys = np.array(BG_sumple.y)
    BG_fit, _ = optimize.curve_fit(pseudo_voigt, xs, ys, p0=[ys[0], xs[0], 1, ys[-1], 0.5], maxfev=args.maxfev,
                                   bounds=(
                                       (0, -100, 0.1, 0, 0),
                                       (np.inf, BG_sumple.x[0], np.inf, np.inf, 1)))

    noBG = xrd_orgine.deduce_func_x(False, pseudo_voigt, BG_fit)
    BG_adj = min(noBG.y)

    def adder(x, min_y):
        return 0 * x + min_y

    noBG.deduce_func_x(True, adder, (BG_adj,))

    if args.printSmooth:
        print(True)
        for i in range(len(noBG)):
            print(noBG.x[i], noBG.y[i], sep=',')
        return

    _, in_window_waves = separate_window(noBG, window_list)

    fitting_result = []
    for i in range(len(in_window_waves)):
        result, _ = peak_fit(in_window_waves[i], fittinglist[i], window_list[i])
        fitting_result.append((*result,))

    print("Fitting result by ", target_file)
    print("Pseudo-voigt,Amp,mu(peak),sigma,eta,BG,BG Adj")
    print("BG", BG_fit[0], BG_fit[1], BG_fit[2], BG_fit[3], BG_fit[4], BG_adj, sep=',')
    print("")

    print("Pseudo-voigt,Amp,mu(peak),sigma,eta,BG,FWHM")
    for w in range(len(fitting_result)):
        lres = int(len(fitting_result[w]) / 4)
        if lres == 0:
            continue
        for i in range(lres):
            print(("fit" + str(w) + "-" + str(i) + "(" + str(fittinglist[w][i]) + ")"), fitting_result[w][i * 4],
                  fitting_result[w][i * 4 + 1], fitting_result[w][i * 4 + 2], fitting_result[w][i * 4 + 3],
                  fitting_result[w][-1] / lres,
                  FWHM(fitting_result[w][i * 3 + 2]), sep=',')

    print("\n")


if __name__ == '__main__':
    main()
