import math
from itertools import repeat

import numpy as np
from pandas import DataFrame


class HBOS:
    def __init__(self, log_scale=True, ranked=False, bin_info_array=[], mode_array=[], nominal_array=[]):
        self.log_scale = log_scale
        self.ranked = ranked
        self.bin_info_array = bin_info_array
        self.mode_array = mode_array
        self.nominal_array = nominal_array
        self.histogram_list = []

    def fit(self, data):
        attr_size = len(data.columns)
        total_data_size = len(data)

        # init params if needed
        if len(self.bin_info_array) == 0:
            self.bin_info_array = list(repeat(-1, attr_size))

        if len(self.mode_array) == 0:
            self.mode_array = list(repeat('dynamic binwidth', attr_size))

        if len(self.nominal_array) == 0:
            self.nominal_array = list(repeat(False, attr_size))

        if self.ranked:
            self.log_scale = False

        normal = 1.0

        # calculate standard _bin size if needed
        for i in range(len(self.bin_info_array)):
            if self.bin_info_array[i] == -1:
                self.bin_info_array[i] = round(math.sqrt(len(data)))

        # initialize histogram
        self.histogram_list = []
        for i in range(attr_size):
            self.histogram_list.append([])

        # save maximum value for every attribute(needed to normalize _bin width)
        maximum_value_of_rows = data.apply(max).values

        # sort data
        sorted_data = data.apply(sorted)

        # create histograms
        for attrIndex in range(len(sorted_data.columns)):
            attr = sorted_data.columns[attrIndex]
            last = 0
            bin_start = sorted_data[attr][0]
            if self.mode_array[attrIndex] == 'dynamic binwidth':
                if self.nominal_array[attrIndex]:
                    while last < len(sorted_data) - 1:
                        last = self.create_dynamic_histogram(self.histogram_list, sorted_data, last, 1, attrIndex, True)
                else:
                    length = len(sorted_data)
                    binwidth = self.bin_info_array[attrIndex]
                    while last < len(sorted_data) - 1:
                        values_per_bin = math.floor(len(sorted_data) / self.bin_info_array[attrIndex])
                        last = self.create_dynamic_histogram(self.histogram_list, sorted_data, last, values_per_bin,
                                                             attrIndex, False)
                        if binwidth > 1:
                            length = length - self.histogram_list[attrIndex][-1].quantity
                            binwidth = binwidth - 1
            else:
                count_bins = 0
                binwidth = (sorted_data[attr][len(sorted_data) - 1] - sorted_data[attr][0]) * 1.0 / self.bin_info_array[
                    attrIndex]
                if (self.nominal_array[attrIndex]) | (binwidth == 0):
                    binwidth = 1
                while last < len(sorted_data):
                    is_last_bin = count_bins == self.bin_info_array[attrIndex] - 1
                    last = self.create_static_histogram(self.histogram_list, sorted_data, last, binwidth, attrIndex,
                                                        bin_start, is_last_bin)
                    bin_start = bin_start + binwidth
                    count_bins = count_bins + 1

        # calculate score using normalized _bin width
        # _bin width is normalized to the number of datapoints
        # save maximum score for every attr(needed to normalize score)
        max_score = []

        # loop for all histograms
        for i in range(len(self.histogram_list)):
            max_score.append(0)
            histogram = self.histogram_list[i]

            # loop for all bins
            for k in range(len(histogram)):
                _bin = histogram[k]
                _bin.total_data_size = total_data_size
                _bin.calc_score(maximum_value_of_rows[i])
                if max_score[i] < _bin.score:
                    max_score[i] = _bin.score

        for i in range(len(self.histogram_list)):
            histogram = self.histogram_list[i]
            for k in range(len(histogram)):
                _bin = histogram[k]
                _bin.normalize_score(normal, max_score[i], self.log_scale)

                # if ranked

    def predict(self, data):
        score_array = []
        for i in range(len(data)):
            each_data = data.values[i]
            value = 1
            if self.log_scale | self.ranked:
                value = 0
            for attr in range(len(data.columns)):
                score = self.get_score(self.histogram_list[attr], each_data[attr])
                if self.log_scale:
                    value = value + score
                elif self.ranked:
                    value = value + score
                else:
                    value = value * score
            score_array.append(value)
        return score_array

    def fit_predict(self, data):
        self.fit(data)
        return self.predict(data)

    @staticmethod
    def get_score(histogram, value):
        for i in range(len(histogram) - 1):
            _bin = histogram[i]
            if (_bin.range_from <= value) & (value < _bin.range_to):
                return _bin.score

        _bin = histogram[-1]
        if (_bin.range_from <= value) & (value <= _bin.range_to):
            return _bin.score
        return 0

    @staticmethod
    def check_amount(sorted_data, first_occurrence, values_per_bin, attr):
        # check if there are more than values_per_bin values of a given value
        if first_occurrence + values_per_bin < len(sorted_data):
            if sorted_data[attr][first_occurrence] == sorted_data[attr][first_occurrence + values_per_bin]:
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def create_dynamic_histogram(histogram_list, sorted_data, first_index, values_per_bin, attr_index, is_nominal):
        attr = sorted_data.columns[attr_index]

        # create new _bin
        _bin = HistogramBin(sorted_data[attr][first_index], 0, 0)

        # check if an end of the data is near
        if first_index + values_per_bin < len(sorted_data):
            last_index = first_index + values_per_bin
        else:
            last_index = len(sorted_data)

        # the first value always goes to the _bin
        _bin.add_quantitiy(1)

        # for every other value
        # check if it is the same as the last value
        # if so
        #   put it into the _bin
        # if not
        #   check if there are more than values_per_bin of that value
        #   if so
        #     open new _bin
        #   if not
        #     continue putting the value into the _bin

        cursor = first_index
        for i in range(int(first_index + 1), int(last_index)):
            if sorted_data[attr][i] == sorted_data[attr][cursor]:
                _bin.add_quantitiy(1)
                cursor = cursor + 1
            else:
                if HBOS.check_amount(sorted_data, i, values_per_bin, attr):
                    break
                else:
                    _bin.add_quantitiy(1)
                    cursor = cursor + 1

        # continue to put values in the _bin until a new values arrive
        for i in range(cursor + 1, len(sorted_data)):
            if sorted_data[attr][i] == sorted_data[attr][cursor]:
                _bin.quantity = _bin.quantity + 1
                cursor = cursor + 1
            else:
                break

        # adjust range of the bins
        if cursor + 1 < len(sorted_data):
            _bin.range_to = sorted_data[attr][cursor + 1]
        else:  # last data
            if is_nominal:
                _bin.range_to = sorted_data[attr][len(sorted_data) - 1] + 1
            else:
                _bin.range_to = sorted_data[attr][len(sorted_data) - 1]

        # save _bin
        if _bin.range_to - _bin.range_from > 0:
            histogram_list[attr_index].append(_bin)
        elif len(histogram_list[attr_index]) == 0:
            _bin.range_to = _bin.range_to + 1
            histogram_list[attr_index].append(_bin)
        else:
            # if the _bin would have length of zero
            # we merge it with previous _bin
            # this can happen at the end of the histogram
            last_bin = histogram_list[attr_index][-1]
            last_bin.add_quantitiy(_bin.quantity)
            last_bin.range_to = _bin.range_to

        return cursor + 1

    @staticmethod
    def create_static_histogram(histogram_list, sorted_data, first_index, binwidth, attr_index, bin_start, last_bin):
        attr = sorted_data.columns[attr_index]
        _bin = HistogramBin(bin_start, bin_start + binwidth, 0)
        if last_bin:
            _bin = HistogramBin(bin_start, sorted_data[attr][len(sorted_data) - 1], 0)

        last = first_index - 1
        cursor = first_index

        while True:
            if cursor >= len(sorted_data):
                break
            if sorted_data[attr][cursor] > _bin.range_to:
                break
            _bin.quantity = _bin.quantity + 1
            last = cursor
            cursor = cursor + 1

        histogram_list[attr_index].append(_bin)
        return last + 1


class HistogramBin:
    def __init__(self, range_from, range_to, quantity):
        self.range_from = range_from
        self.range_to = range_to
        self.quantity = quantity
        self.score = 0
        self.total_data_size = 0

    def get_height(self):
        width = self.range_to - self.range_from
        height = self.quantity / width
        return height

    def add_quantitiy(self, anz):
        self.quantity = self.quantity + anz

    def calc_score(self, max_score):
        if max_score == 0:
            max_score = 1

        if self.quantity > 0:
            self.score = 1.0 * self.quantity / (
                (self.range_to - self.range_from) * self.total_data_size * 1.0 / abs(max_score))

    def normalize_score(self, normal, max_score, log_scale):
        self.score = self.score * normal / max_score
        if self.score == 0:
            return
        self.score = 1 / self.score
        if log_scale:
            self.score = math.log10(self.score)


def tst_impl(td):
    total_data_size = 30
    max_data_value = 100
    hbin = HistogramBin(td[0], td[1], td[2])
    hbin.total_data_size = total_data_size
    hbin.calc_score(max_data_value)
    print(hbin.score)
    assert round(hbin.score, 2) == td[3]


def test_bin():
    tst_impl([0, 10, 25, 8.33])
    tst_impl([0, 10, 35, 11.67])
    tst_impl([0, 5, 25, 16.67])
    tst_impl([10, 20, 25, 8.33])
    tst_impl([20, 100, 25, 1.04])


def test_create_dynamic_histogram():
    data = DataFrame(data=[1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10])
    histogram_list = [[]]
    first_index = 0
    result_list = []
    # hbos = HBOS()
    while first_index < len(data):
        ret = HBOS.create_dynamic_histogram(histogram_list, data, first_index, 2, 0, False)
        result_list.append(ret)
        first_index = ret

    assert result_list == [6, 9, 14, 16, 17, 23, 27, 28]

    histogram = histogram_list[0]
    result_list = []
    for i in range(len(histogram)):
        result_list.append(histogram[i].quantity)

    assert result_list == [6, 3, 5, 2, 1, 6, 5]


def test_fit_predict():
    data = DataFrame(
        data={'attr1': [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10]})

    hbos = HBOS(log_scale=False, ranked=False, bin_info_array=[10], mode_array=["dynamic binwidth"],
                nominal_array=[False])
    result = hbos.fit_predict(data)
    result = np.round(result, 1)
    assert list(result) == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.2, 1.2, 1.2, 1.2, 1.2, 9.0, 9.0, 6.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5]

    hbos = HBOS(log_scale=False, ranked=False, bin_info_array=[10], mode_array=["static binwidth"],
                nominal_array=[False])
    result = hbos.fit_predict(data)
    result = np.round(result, 1)
    assert list(result) == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.2, 1.2, 1.2, 1.2, 1.2, 6.0, 6.0, 6.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 6.0]

    hbos = HBOS(log_scale=True, ranked=False, bin_info_array=[10], mode_array=["dynamic binwidth"],
                nominal_array=[False])
    result = hbos.fit_predict(data)
    result = np.round(result, 1)
    assert list(result) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.8, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2]

    hbos = HBOS(log_scale=True, ranked=False, bin_info_array=[10], mode_array=["static binwidth"],
                nominal_array=[False])
    result = hbos.fit_predict(data)
    result = np.round(result, 1)
    assert list(result) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.8]

    # data not sorted
    data = DataFrame(
        data={'attr1': [4, 5, 7, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10]})
    hbos = HBOS(log_scale=True, ranked=False, bin_info_array=[10], mode_array=["static binwidth"],
                nominal_array=[False])
    result = hbos.fit_predict(data)
    result = np.round(result, 1)
    assert list(result) == [0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.8]

    data0 = [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10]
    data1 = [1, 2, 2, 5, 5, 5, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10]
    data = DataFrame(data={'attr1': data0, 'attr2': data1})
    hbos = HBOS(False, False, [10, 10], ["static binwidth", "static binwidth"], [False, False])
    result = hbos.fit_predict(data)
    result = np.round(result, 1)
    assert list(result) == [8.0, 4.0, 4.0, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 1.6, 1.6, 1.6, 2.4, 2.4, 12.0, 12.0, 12.0, 2.0,
                            2.0, 2.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 6.0]

    hbos = HBOS(True, False, [9, 9], ["dynamic binwidth", "dynamic binwidth"], [False, False])
    result = hbos.fit_predict(data)
    result = np.round(result, 3)
    assert list(result) == [1.204, 1.204, 1.204, 0.903, 0.903, 0.903, 0.602, 0.602, 0.602, 0.380, 0.380, 0.380, 0.556,
                            0.556, 1.380, 1.380, 0.903, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.176, 0.176, 0.176,
                            0.176, 0.176]

    print('===== OK =====')


def test_create_static_histogram():
    data = DataFrame(data=[1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10])
    histogram_list = [[]]
    current_bin_count = 0
    attr = 0
    total_bin_count = 10
    bin_width = (data[attr][len(data) - 1] - data[attr][0]) / total_bin_count
    first_index = 0
    first_value = data[attr][0]
    result_list = []

    while first_index < len(data):
        is_last_bin = current_bin_count == total_bin_count - 1
        first_index = HBOS.create_static_histogram(histogram_list, data, first_index, bin_width, attr, first_value,
                                                   is_last_bin)
        print(first_index)
        result_list.append(first_index)
        first_value = first_value + bin_width
        current_bin_count = current_bin_count + 1

    print(result_list)
