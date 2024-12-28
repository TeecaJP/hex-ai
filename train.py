import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from hex_environment import HexEnvironment  # 提供された環境クラスをインポート
import os  # モデルの保存・読み込みに必要なモジュール
import matplotlib.pyplot as plt  # 学習過程の可視化

# DQNのニューラルネットワーク
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.fc(x)

# ランダムプレイヤークラス
class RandomPlayer:
    def select_action(self, legal_actions):
        return random.choice(legal_actions)

# DQNエージェント
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_min=0.2, epsilon_decay=0.995, lr=0.005, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def save_model(self, filepath):
      torch.save({
          'model_state_dict': self.model.state_dict(),
          'target_model_state_dict': self.target_model.state_dict(),
          'epsilon': self.epsilon
      }, filepath)
      print(f"Model saved to {filepath}")


    def load_model(self, filepath):
        if os.path.exists(filepath):
            try:
                checkpoint = torch.load(filepath, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                print(f"Model loaded from {filepath}")
            except KeyError as e:
                print(f"Failed to load model from {filepath}. KeyError: {e}")
            except Exception as e:
                print(f"Failed to load model from {filepath}. Error: {e}")
        else:
            print(f"No model found at {filepath}, starting with a new model.")


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, legal_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(legal_actions)
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        q_values = self.model(state_tensor).detach().cpu().numpy()[0]
        legal_q_values = [q_values[action] for action in legal_actions]
        return legal_actions[np.argmax(legal_q_values)]

    # DQNエージェント内のreplayメソッドの修正
    def replay(self, batch_size, update_target_every=1000, episode=0):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)

            target = self.model(state_tensor).detach().cpu().numpy()[0]
            if done:
                target[action] = reward
            else:
                t = self.target_model(next_state_tensor).detach().cpu().max(1)[0].item()
                target[action] = reward + self.gamma * t

            target_tensor = torch.FloatTensor(target).to(self.device).unsqueeze(0)
            output = self.model(state_tensor)

            loss = nn.MSELoss()(output, target_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # ターゲットモデルの更新
        if (episode + 1) % update_target_every == 0:
            self.update_target_model()  # ここでターゲットモデルを更新


# 学習関数
def train_dqn_agent(episodes=10000, batch_size=32, model_filepath='dqn_agent.pth', resume_training=False):
    env = HexEnvironment()
    state_size = env.board.size
    action_size = env.board.size
    agent = DQNAgent(state_size, action_size, device="cpu")
    random_player = RandomPlayer()

    # モデルを再利用する場合の処理
    if resume_training:
        agent.load_model(model_filepath)

    total_reward = 0
    episode_rewards = []

    for episode in range(episodes):
        env.reset()
        state = env.board.flatten()

        while not env.is_game_over():
            legal_actions = env.get_legal_cells()

            if env.current_player == 1:
                action = agent.select_action(state, legal_actions)
            else:
                action = random_player.select_action(legal_actions)

            env.update_board(action)
            next_state = env.board.flatten()
            reward = 0

            if env.current_player == 1:
                if env.check_winner():
                    reward = 1
                elif len(env.get_legal_cells()) == 0:
                    reward = 0
                else:
                    reward = 0
                agent.remember(state, action, reward, next_state, env.is_game_over())

            state = next_state
            total_reward += reward
            agent.replay(batch_size, episode=episode)
            env.switch_player()

        episode_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            agent.save_model(model_filepath)
            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon}")

    agent.save_model(model_filepath)

    plt.plot(episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.show()

if __name__ == "__main__":
    train_dqn_agent(episodes=10000, batch_size=32, model_filepath='_dqn_agent.pth', resume_training=True)
