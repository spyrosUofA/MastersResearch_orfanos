from test_new import test
from model import ActorCritic
import torch
import torch.optim as optim
import gym
import sys

def train():
    # Defaults parameters:
    #    gamma = 0.99
    #    lr = 0.02
    #    betas = (0.9, 0.999)
    #    random_seed = 543
    random_seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])

    render = False
    gamma = 0.99
    lr = 0.001
    betas = (0.9, 0.999)
    
    torch.manual_seed(random_seed)
    
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)
    
    policy = ActorCritic()
    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)
    print(lr, betas)
    
    running_reward = 0.0
    for i_episode in range(0, 10000):
        state = env.reset()
        for t in range(10000):
            action = policy(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            running_reward += reward
            if render and i_episode > 1000:
                env.render()
            if done:
                break
                    
        # Updating the policy :
        optimizer.zero_grad()
        loss = policy.calculateLoss(gamma)
        loss.backward()
        optimizer.step()
        policy.clearMemory()

        if running_reward > 4400:
            torch.save(policy.state_dict(), './preTrained/LunarLander_{}.pth'.format(random_seed))
            print("########## Solved! ##########")
            print(running_reward/20)
            test(50, random_seed)
            break
        
        if i_episode % 20 == 0:
            running_reward = running_reward/20
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
            running_reward = 0

        if i_episode > 10000:
            break
            
if __name__ == '__main__':
    train()
