import scipy
import scipy.stats
import matplotlib.pyplot as plt
import file_processor as fp


class StatisticsToolkit:
    def __init__(self, path_1=None, path_2=None):
        self.path_1 = path_1
        self.path_2 = path_2

    @staticmethod
    def calculate_statistical_significances(values_1, values_2):
        average_1 = sum(values_1) / float(values_1.__len__())
        average_2 = sum(values_2) / float(values_2.__len__())

        wilcoxon = scipy.stats.wilcoxon(values_1, values_2)
        t_test = scipy.stats.ttest_ind(values_1, values_2)
        mann_whitney = scipy.stats.mannwhitneyu(values_1, values_2)

        return {"size_1": len(values_1), "size_2": len(values_2), "average_1": average_1, "average_2": average_2,
                "wilcoxon": wilcoxon, "t_test": t_test, 'mann whitney: ': mann_whitney}




