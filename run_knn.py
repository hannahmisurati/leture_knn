import pandas

from sklearn.neighbors import KNeighborsRegressor
import kfold_template

dataset = pandas.read_csv("abalone.csv")
dataset = dataset.sample(frac=1)

target = dataset.iloc[:,8]
target = target + 1.5
target = target.values

data = dataset.iloc[:,0:8]
data = pandas.get_dummies(data, columns=['Sex'])
data = data.values
# print(data)

trials = []
for w in ['uniform', 'distance']:
  for k in range(1,50):
    machine = KNeighborsRegressor(n_neighbors=k, weights='uniform')
    return_values = kfold_template.run_kfold(machine, data, target, 4, True)
    average_return_value = sum(return_values)/len(return_values)
    # print(average_return_value)
    trials.append((average_return_value, k, w))
    
trials.sort(key=lambda x: x[0], reverse=True)

print(trials[:5])