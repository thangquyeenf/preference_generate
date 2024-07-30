import gym  # type: ignore
import GymBoss  # type: ignore
import argparse
import numpy as np  # type: ignore
from constant import compare_coordinates

def get_label(obs_1, obs_2, ep_reward_1, ep_reward_2, step_earn_1, step_earn_2):
    if ep_reward_1 > ep_reward_2:
        return 1.0
    elif ep_reward_1 < ep_reward_2:
        return 0.0
    else:
        coord_1 = (obs_1[0], obs_1[1])
        coord_2 = (obs_2[0], obs_2[1])

        if coord_1 == coord_2 and step_earn_1 < step_earn_2:
            return 1.0
        
        if coord_1 == coord_2 and step_earn_1 > step_earn_2:
            return 0.0

        if coord_1 == coord_2 and step_earn_1 == step_earn_2:
            return 0.5
        
        if compare_coordinates(coord1=coord_1, coord2=coord_2):
            return 1.0
        else: 
            return 0.0

def run(args):
    env = gym.make("GymBoss/BossGame-v0")
    eps_obs_1 = []
    eps_obs_2 = []
    eps_act_1 = []
    eps_act_2 = []
    labels = []
    
    for _ in range(args.num_episodes):
        observations_1 = []
        actions_1 = []
        observations_2 = []
        actions_2 = []
        obs_1 = env.reset()[0]
        ep_reward_1 = 0
        ep_reward_2 = 0
        step_earn_1 = 0
        step_earn_2 = 0
        for step in range(200):
            action = env.action_space.sample()
            next_observation, reward, done, _, _ = env.step(action)
            observations_1.append(obs_1)
            actions_1.append(action)
            ep_reward_1 += reward
            obs_1 = next_observation
            if done: 
                obs_1 = env.reset()[0]
                step_earn_1 = step

        obs_2 = env.reset()[0]
        for step in range(200):
            action = env.action_space.sample()
            next_observation, reward, done, _, _ = env.step(action)
            observations_2.append(obs_2)
            actions_2.append(action)
            ep_reward_2 += reward
            obs_2 = next_observation
            if done: 
                obs_2 = env.reset()[0]
                step_earn_2 = step

        # Chuyển đổi danh sách sang numpy.ndarray
        eps_obs_1.append(np.array(observations_1))
        eps_obs_2.append(np.array(observations_2))
        eps_act_1.append(np.array(actions_1))
        eps_act_2.append(np.array(actions_2))

        label = get_label(obs_1=obs_1, obs_2=obs_2,
                          ep_reward_1=ep_reward_1, ep_reward_2=ep_reward_2,
                          step_earn_1=step_earn_1, step_earn_2=step_earn_2)
        labels.append(label)
    
    return {
        "obs_1": np.array(eps_obs_1),
        "obs_2": np.array(eps_obs_2),
        "action_1": np.array(eps_act_1).reshape(-1, 200, 1),  # Định hình lại ở đây
        "action_2": np.array(eps_act_2).reshape(-1, 200, 1),
        "label": np.array(labels)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--path", "-p", type=str, default="pref_datasets.npz")
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--num_episodes", "-ep", type=int, default=500)
    args = parser.parse_args()
    
    pref_datasets = run(args=args)

    # Lưu dữ liệu vào tệp .npz theo đường dẫn args.path
    np.savez(args.path, obs_1=pref_datasets['obs_1'], obs_2=pref_datasets['obs_2'], action_1=pref_datasets['action_1'], action_2=pref_datasets['action_2'], label=pref_datasets['label'])

    # In thông tin để kiểm tra
    print(f"Data saved to {args.path}")
    print(pref_datasets['obs_1'].shape)
    print(pref_datasets['action_1'].shape)  # Kiểm tra hình dạng mới
    print(type(pref_datasets))
    print(pref_datasets['label'])
