from pickle import load
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

with open('ga_result.pickle', 'rb') as f:
    result = load(f)

d = {'ga_score': [value for key, value in result.items()]}

df = pd.DataFrame(data=d)

fig, ax = plt.subplots(figsize=(15, 7))
plt.plot(df)
plt.xlabel('steps')
plt.ylabel('ga_score')
plt.savefig('training_result.png')
plt.show()