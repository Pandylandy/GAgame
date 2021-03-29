from genopt.optimizer import SGA
from genopt.scalers import sigma_trunc_scaling
from genopt.selectors import tournament_selection
from genopt.crossovers import two_point_crossover
from genopt.mutators import uniform_mutation
import os
from CGRtools.files import SDFRead
from pickle import dump, load
import time
import logging

os.environ['DB'] = './data/db.shelve'
os.environ['DATA'] = './data/rules'

from RNNSynthesis.environment import SimpleSynthesis
logger = logging.getLogger("exampleApp")
logger.setLevel(logging.INFO)

# create the logging file handler
fh = logging.FileHandler("GA.log", mode='w')

formatter = logging.Formatter()
fh.setFormatter(formatter)

# add handler to logger object
logger.addHandler(fh)

logger.info("Program started")

filename = "tylenol"
target = next(SDFRead(open(f'./data/{filename}.sdf', encoding='UTF-8')))
env = SimpleSynthesis(target, steps=10 ** 6)
step_per_score = dict()


def fit_func(hromosoma):
    env.reset()
    last_reward = 0
    for action in hromosoma:
        state, reward, done, info = env.step(action)
        if state:
            if len(state) - len(target) >= 20:
                break
            last_reward = reward
            if done:
                if env.depth < 5 and not env.stop:
                    logger.info(f'синтетический путь для молекулы {target} = {[step for step in env.render()]}')
                    print(f'синтетический путь для молекулы {target} = {[step for step in env.render()]}')
                    break
                else:
                    logger.info('done, но молекула не синтезирована')
                    print('done, но молекула не синтезирована')
    return last_reward

ga = SGA(task='maximize', pop_size=50, cross_prob=0.8, mut_prob=0.2, elitism=True, n_cpu=20)
ga.set_selector_type(tournament_selection)
ga.set_scaler_type(sigma_trunc_scaling)
ga.set_crossover_type(two_point_crossover)
ga.set_mutator_type(uniform_mutation)
ga.set_fitness(fit_func)

ga.initialize(space=env.action_space, steps=10)  #длина хромосомы

# если хочу продолжить с последней популяции. НЕ ПЕРЕПУТАТЬ ДЛЯ РАЗНЫХ ТАРГЕТОВ
with open(f'ga_last_population_{filename}.pickle', 'rb') as f:
    last_pop = load(f)

ga.population = last_pop
ga.step()
#

for i in range(100000):
    start = time.time()
    ga.step()
    end = time.time()
    print(end - start)
    b_individual = ga.best_individual()
    score = b_individual.score
    statistics = ga.population.calc_statistics()
    step_per_score[i] = b_individual.score

    with open('ga_result.pickle', 'wb') as f:
        dump(step_per_score, f)

    print(f'ga.score = {score}')
    logger.info(f'ga_score-{i}: {score}')

    print(f'ga_best_individual-{i}: {b_individual}')
    logger.info(f'ga_best_individual-{i}: {b_individual}')

    logger.info(f'ga_statistics-{i}: {statistics.stats}')   # будут различные параметры в виде словаря. можно добавить туда свой ключ с лучшей хромосомой на кажом шаге

    last_pop = ga.population

    with open(f'ga_last_population_{filename}.pickle', 'wb') as f:
        dump(last_pop, f)


    if b_individual.score >= 10:
        print('молекула синтезирована!!!!!!!!!!!!!!!!!!!!!!!!')
        logger.info('молекула синтезирована!!!!!!!!!!!!!!!!!!!!!!!!')
        break


logger.info('Done')


