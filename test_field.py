from scipy.stats.stats import pearsonr
from StatisticsToolkit import StatisticsToolkit

# x_1 = [34, 39, 37, 68, 36, 65, 41]
# x_2 = [64, 68, 56, 88, 66, 83, 61]
#
# y_1 = [39, 55, 41, 69, 37, 63, 42]
# y_2 = [62, 71, 56, 92, 58, 85, 70]
#
# z_1 = [34, 68, 7, 85, 52, 90, 0]
# z_2 = [89, 93, 64, 100, 78, 95, 80]
#
# print(pearsonr(x_1, x_2))
# print(pearsonr(y_1, y_2))
# print(pearsonr(z_1, z_2))

list_1 = [95, 87, 88, 93, 91, 91, 89, 87, 87, 84,
          95, 87, 88, 93, 91, 91, 89, 87, 87, 84]

list_2 = [92, 96, 90, 94, 93, 86, 91, 92, 88, 94,
          92, 87, 93, 96, 86, 91, 94, 85, 89, 94]

print(StatisticsToolkit.calculate_statistical_significances(list_1, list_2))