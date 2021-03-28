import os
from CGRtools.files import SDFRead
from pickle import dump, load

os.environ['DB'] = './data/db.shelve'
os.environ['DATA'] = './data/rules'

from RNNSynthesis.environment import SimpleSynthesis

filename = "target2"
hromosoma = [253679, 247498, 378969, 141368, 66262]

target = next(SDFRead(open(f'./data/{filename}.sdf', encoding='UTF-8')))
env = SimpleSynthesis(target, steps=10 ** 6)

def reward_to_tanimoto(reward):
    return (1 + 10 ** (-10)) - (10 ** (0 - reward))

all_rewards = []
last_reward = 0
for action in hromosoma:
    state, reward, done, info = env.step(action)  # reward не есть танимото. Танимото используется для вычесления reward.
    if state:
        if len(state) - len(target) >= 20:
            env.path.pop()
            break
        last_reward = reward
        if done:
            if env.depth < 5 and not env.stop:
                print(f'синтетический путь для молекулы {target} : {[str(step) for step in env.render()]}')
                break
            else:
                print('done, но молекула не синтезирована')
all_rewards.append(last_reward)
tmp = [str(i) for i in env.render()]
print(f'3 molecule: Tanimoto = {reward_to_tanimoto(all_rewards[-1])}, synthetic path for {target}: {tmp}')

# for action in path:      # наверняка после большего реворда был брейк. Чтобы проверить нужно полностью поторить код из фит функции.
#     state, reward, done, info = env.step(action)
# tmp = [str(i) for i in env.render()]
# print(f'2 molecule: Tanimoto = {reward_to_tanimoto(tmp_reward)}, synthetic path: {tmp}')
