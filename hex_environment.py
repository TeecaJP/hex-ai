import numpy as np

class HexEnvironment:
    def __init__(self):
        self.current_player = 1  # プレイヤー1からスタート
        self.size = 4
        self.board = self.initialize_board()

    # 盤面の初期化
    def initialize_board(self):
        return np.zeros((self.size, self.size), dtype=int)

    # 環境のリセット
    def reset(self):
        self.board = self.initialize_board()
        self.current_player = 1  # プレイヤー1からスタートに戻す

    # 隣接セルを取得する関数
    def get_neighbors(self, cell):
        row, col = divmod(cell, self.size)  # 座標取得
        directions = [
            (-1, 0),  # 上
            (1, 0),   # 下
            (0, -1),  # 左
            (0, 1),   # 右
            (-1, 1),  # 右上
            (1, -1)   # 左下
        ]
        neighbors = []
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                neighbors.append(nr * self.size + nc)
        return neighbors

    # 配置可能なセル（値が0のセル）を取得する関数
    def get_legal_cells(self):
        return np.flatnonzero(self.board == 0).tolist()

    # 指定したセルに値を更新する関数
    def update_board(self, cell):
        if cell not in self.get_legal_cells():
            raise ValueError(f"セル {cell} は配置不可能です。")
        value = 1 if self.current_player == 1 else -1
        row, col = divmod(cell, self.size)
        self.board[row, col] = value

    # 勝利条件を判定する関数
    def check_winner(self):
        value = 1 if self.current_player == 1 else -1
        visited = set()
        stack = []

        # プレイヤー1のスタートとゴール
        start = [i for i in range(self.size) if self.board[0, i] == value]
        goal = [i for i in range(self.size * (self.size - 1), self.size * self.size) if self.board[-1, i % self.size] == value]

        # プレイヤー2のスタートとゴール
        if self.current_player == 2:
            start = [i for i in range(0, self.size * self.size, self.size) if self.board[i // self.size, 0] == value]
            goal = [i for i in range(self.size - 1, self.size * self.size, self.size) if self.board[i // self.size, -1] == value]

        stack.extend(start)

        while stack:
            current = stack.pop()
            if current in goal:
                return True

            if current not in visited:
                visited.add(current)
                neighbors = self.get_neighbors(current)
                for neighbor in neighbors:
                    row, col = divmod(neighbor, self.size)
                    if self.board[row, col] == value:
                        stack.append(neighbor)

        return False

    # 次のプレイヤーにターンを切り替える
    def switch_player(self):
        self.current_player = 1 if self.current_player == 2 else 2

    # ゲーム終了のチェック
    def is_game_over(self):
        return self.check_winner() or len(self.get_legal_cells()) == 0

    # ゲームの状態を表示（デバッグ用）
    def display_board(self):
        symbols = {0: '.', 1: '〇', -1: '×'}
        print("\n".join(" ".join(symbols[cell] for cell in row) for row in self.board))
        print()
