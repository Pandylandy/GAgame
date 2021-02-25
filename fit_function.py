import os
os.environ['DB'] = '/home/ilnur/PycharmProjects/RLFS/data/db.shelve'
os.environ['DATA'] = '/home/ilnur/PycharmProjects/RLFS/data/rules/'

from RNNSynthesis.environment import SimpleSynthesis
from CGRtools.files import SDFRead


target = next(SDFRead(open('/home/ilnur/PycharmProjects/RLFS/data/tylenol.sdf', encoding='UTF-8')))
env = SimpleSynthesis(target, steps=1000, reward_at_end=True)


