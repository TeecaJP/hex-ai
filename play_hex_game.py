from hex_environment import HexEnvironment

def play_game():
    env = HexEnvironment()

    while not env.is_game_over():
        env.display_board()
        print(f"Player {env.current_player}'s turn.")
        try:
            cell = int(input(f"Enter the cell number (0 to {env.board.size - 1}): "))  # env.board.sizeを使用
            env.update_board(cell)
        except ValueError as e:
            print(e)
            continue

        if env.check_winner():
            env.display_board()
            print(f"Player {env.current_player} wins!")
            break

        env.switch_player()

    if not env.check_winner():
        env.display_board()
        print("It's a draw!")

if __name__ == "__main__":
    play_game()
