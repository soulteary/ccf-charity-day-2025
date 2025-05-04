import matplotlib.pyplot as plt

def visualize_board_as_pieces(board_np, save_path):
    board_size = board_np.shape[0]
    plt.figure(figsize=(6, 6))
    plt.xlim(-0.5, board_size - 0.5)
    plt.ylim(-0.5, board_size - 0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_facecolor('#DEB887')

    for i in range(board_size):
        plt.plot([-0.5, board_size - 0.5], [i - 0.5, i - 0.5], 'k', lw=0.5)
        plt.plot([i - 0.5, i - 0.5], [-0.5, board_size - 0.5], 'k', lw=0.5)

    for x in range(board_size):
        for y in range(board_size):
            if board_np[x, y] == 1:
                plt.plot(y, board_size - 1 - x, 'o', markersize=12, color='black')
            elif board_np[x, y] == 2:
                plt.plot(y, board_size - 1 - x, 'o', markersize=12, color='white', markeredgecolor='black')

    plt.xticks([])
    plt.yticks([])
    plt.title("Gomoku Game")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
