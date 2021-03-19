from pickle import load
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
plt.style.use('seaborn')

filename = sys.argv[1]  # todo: получение этого параметра из консоли
# filename = 'target1'
files = os.listdir(f'score_per_step/{filename}')
count = [elem.split('.')[0] for elem in files]
result = dict()
for i in count:
    with open(f'score_per_step/{filename}/{i}.pickle', 'rb') as f:
        tmp = load(f)
    result.update(tmp)
# with open(f'ga_result.pickle', 'rb') as f:
#     result = load(f)
#
# d = {'ga_score': [value for key, value in result.items()]}

df = pd.DataFrame.from_dict(result, orient='index', columns=['ga_score'])

fig, ax = plt.subplots(figsize=(15, 7))
plt.title(filename)
plt.plot(df)
plt.xlabel('steps')
plt.ylabel('ga_score')
plt.savefig(f'graphics/training_result_{filename}.png')
plt.show()
