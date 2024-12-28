from hex_environment import HexEnvironment
from train import DQNAgent  # DQNAgentをインポート
import os

# AIモデルをロードする関数
def load_ai_model(agent, model_filepath='dqn_agent.pth'):
    if os.path.exists(model_filepath):
        agent.load_model(model_filepath)
    else:
        print(f"No model found at {model_filepath}, starting with a new model.")
        # モデルがなければ新規モデルを初期化

# プレイ関数
def play_ai_game(model_filepath='dqn_agent.pth'):
    env = HexEnvironment()
    state_size = env.board.size  # 状態空間の次元
    action_size = env.board.size  # アクション空間の次元（盤面のセル数）

    # DQNエージェントを作成し、保存されたモデルをロード
    agent = DQNAgent(state_size, action_size, device="cpu")
    load_ai_model(agent, model_filepath)

    # ゲーム開始
    while not env.is_game_over():
        env.display_board()
        print(f"Player {env.current_player}'s turn.")

        if env.current_player == 1:  # AI（プレイヤー1）のターン
            legal_actions = env.get_legal_cells()  # 合法的なセル（アクション）を取得
            state = env.board.flatten()  # 現在の状態を1次元配列として取得
            action = agent.select_action(state, legal_actions)  # AIが選ぶアクション
            print(f"AI chose cell {action}.")
            env.update_board(action)  # 選択したアクションを実行
        else:  # ユーザー（プレイヤー2）のターン
            try:
                cell = int(input(f"Enter the cell number (0 to {env.board.size - 1}): "))
                if cell not in env.get_legal_cells():
                    print("Invalid move! Please choose a legal cell.")
                    continue
                env.update_board(cell)  # ユーザーが選んだセルを実行
            except ValueError:
                print("Invalid input! Please enter an integer.")
                continue

        # 勝者の判定
        if env.check_winner():
            env.display_board()
            print(f"Player {env.current_player} wins!")
            break

        env.switch_player()  # ターン交代

    if not env.check_winner():
        env.display_board()
        print("It's a draw!")

if __name__ == "__main__":
    play_ai_game(model_filepath='dqn_agent.pth')
