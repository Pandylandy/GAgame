from pickle import load
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
plt.style.use('seaborn')

filename = sys.argv[1]  # todo: получение этого параметра из консоли
# filename = 'target1'
files = os.listdir(f'tanimoto/score_per_step/{filename}')
count = [elem.split('.')[0] for elem in files]
result = dict()
for i in count:
    with open(f'tanimoto/score_per_step/{filename}/{i}.pickle', 'rb') as f:
        tmp = load(f)
    result.update(tmp)

def reward_to_tanimoto(reward):
    return (1 + 10 ** (-10)) - (10 ** (0 - reward))   # todo: надо прервать все расчеты, перенести на сервере в нужные папки

#график с reward
df = pd.DataFrame.from_dict(result, orient='index', columns=['ga_score'])

fig, ax = plt.subplots(figsize=(15, 7))
plt.title(f'Reward for {filename}')
plt.plot(df)
plt.xlabel('steps')
plt.ylabel('ga_score')
plt.savefig(f'tanimoto/graphics/reward_{filename}.png')
# plt.show()

# график с танимомто
# result_tanimoto = dict()
# for key, value in result.items():
#     result_tanimoto[key] = reward_to_tanimoto(value)
result_tanimoto = {key: reward_to_tanimoto(value) for key, value in result.items()}

df = pd.DataFrame.from_dict(result_tanimoto, orient='index', columns=['ga_score'])

fig, ax = plt.subplots(figsize=(15, 7))
plt.title(f'Tanimoto for {filename}')
plt.plot(df)
plt.xlabel('steps')
plt.ylabel('Tanimoto')
plt.savefig(f'tanimoto/graphics/tanimoto_{filename}.png')
# plt.show()
