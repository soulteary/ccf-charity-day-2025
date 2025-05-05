from tic_tac_toe import TicTacToeGame
from agents import RandomAgent, GreedyAgent
from data_collector import DataCollector

def main():
    game = TicTacToeGame()
    agents = [GreedyAgent(), RandomAgent()]
    collector = DataCollector()

    while not game.game_over:
        agent = agents[game.current_player - 1]
        move = agent.select_move(game)
        game.make_move(*move)
        print(game, "\n")

    print("Game Over. Winner:", "Draw" if game.winner == 0 else f"Player {game.winner}")
    collector.save_game(game)

if __name__ == "__main__":
    main()
