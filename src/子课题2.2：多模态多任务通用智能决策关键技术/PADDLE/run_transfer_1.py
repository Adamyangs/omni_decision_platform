from core.setup import *
from collections import OrderedDict
import argparse
import json
import tensorflow as tf
from get_knowledge import get_source_policies, extract_knowledge, set_blocedboxunlockpickup, set_blockedboxplacegoal
from pprint import pprint
import wandb
wandb.init(
    project='Knowledge-Transfer',
    name='GALOIS_PPO',
    config={
      "discount": 0.8,
      "gamma_entropy_loss": 5.0,
      "update_entropy_loss_step": 50,
      "update_entropy_loss_discount": 10,
      "seed": 0
    }
)

#@ray.remote
def start_DILP(task, name, mode, seed, variation=None, size=None, guide=None, source_policies=None, similarity_table=None):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if task == "mg":
        print("*************loading minigrid************")
        print("SEED: {}".format(seed))
        tf.set_random_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        man, env = setup_blocedboxunlockpickup(variation, env_name=args.name, size=size)
        # man, env = setup_blocedboxplacegoal(variation, env_name=args.name, size=size)
        # man, env = setup_doorgoal(variation, env_name=args.name, size=size)
        # man, env = setup_ballkey(variation, env_name=args.name, size=size)
        agent = RLDILP(man, env, state_encoding="atoms", seed=seed)
        discounting = 0.95

        critic = None
        # critic = NeuralCritic([20], 2, 1.0, learning_rate=0.001, state2vector=env.state2vector,involve_steps=False)
        w2_critic = NeuralCritic([20], 3, 1.0, learning_rate=0.001, state2vector=env.w2_state2vector,involve_steps=False, scope_name='w2_critic')
        w2_critic = None
        learner = ReinforceLearner(agent, env, 0.05, critic=critic, discounting=discounting,
                                   batched=True, steps=2000, name=name, w2_critic=w2_critic, seed=seed, source_policies=source_policies, similarity_table=similarity_table)
    if mode == "train":
        return learner.train()[-1]
    elif mode == "evaluate":
        return learner.evaluate(checkpoints=variation, repeat=20)
    else:
        raise ValueError()

x, y, max, min, std = [], [], [], [], []
def check_result(task, name, algo, seed, size = None):
    variations = [i for i in range(0,2000,50)]
    # variations = [1950]
    for variation in variations:
        tf.reset_default_graph()
        checkpoint = 'round_' + str(variation)
        print("==========="+checkpoint+"==============")
        result = start_DILP(task, name, "evaluate", seed, checkpoint, size)
        pprint(result)
        x.append(variation)
        y.append(result["mean"])
        max.append(result["max"])
        min.append(result["min"])
        std.append(result["std"])
    dataframe = pd.DataFrame({'episode': x, 'performance': y, "max": max, "min": min, "std": std})
    path = "./data/" + name.split('-')[1].lower()
    if not os.path.exists(path):
        os.makedirs(path)
    dataframe.to_csv(path+"/" + name.split('-')[1].lower() + "_" + str(seed) +".csv", index=False, sep=',')
    plt.plot(x, y, label="unlockpickup_no_guide_0", color="blue")
    plt.xlabel("episode")
    plt.ylabel("performance")
    plt.legend(loc='lower right')
    plt.show()


from pprint import pprint
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--task', default='mg')
    parser.add_argument('--algo', default='DILP')
    parser.add_argument('--name', default='MiniGrid-BlockedBoxUnlockPickup-8x8-v0')
    # parser.add_argument('--name', default='MiniGrid-BlockedBoxPlaceGoal-8x8-v0')
    # parser.add_argument('--name', default='MiniGrid-DoorGoal-8x8-v0')
    # parser.add_argument('--name', default='MiniGrid-BallKey-8x8-v0')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--size', default=8, type=int)
    parser.add_argument('--need_guide', default=False, type=bool)
    args = parser.parse_args()

    if args.mode=='train':
        if args.algo == "DILP":
            starter = start_DILP

        if args.need_guide:
            source_policies, similarity_table = set_blocedboxunlockpickup()
            # source_policies, similarity_table = set_blockedboxplacegoal()
            pprint(starter(args.task, args.name, args.mode, args.seed, size=args.size, source_policies=source_policies, similarity_table=similarity_table))
        else:
            pprint(starter(args.task, args.name, args.mode, args.seed, size=args.size))
    elif args.mode=='evaluate':
        check_result(args.task, args.name, args.algo, args.seed, args.size)

