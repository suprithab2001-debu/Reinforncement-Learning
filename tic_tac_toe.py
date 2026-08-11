import random

# Q-table
q_table = {}

# Learning parameters
ALPHA = 0.8       # Learning rate
GAMMA = 0.9       # Discount factor
EPSILON = 0.2     # Exploration rate

# Winning combinations
WINNING_COMBINATIONS = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6)
]


def get_state(board):
    """Convert board into a string state."""
    return ''.join(board)


def available_moves(board):
    """Return empty positions."""
    return [i for i in range(9) if board[i] == ' ']


def check_winner(board):
    """Check whether X or O has won."""
    for a, b, c in WINNING_COMBINATIONS:
        if board[a] != ' ' and board[a] == board[b] == board[c]:
            return board[a]

    if ' ' not in board:
        return 'Draw'

    return None


def get_q_values(state):
    """Get Q-values for a state."""
    if state not in q_table:
        q_table[state] = [0.0] * 9

    return q_table[state]


def choose_action(board, training=True):
    """Choose an action using epsilon-greedy strategy."""
    moves = available_moves(board)
    state = get_state(board)
    q_values = get_q_values(state)

    # Exploration
    if training and random.random() < EPSILON:
        return random.choice(moves)

    # Exploitation
    best_value = max(q_values[i] for i in moves)
    best_moves = [i for i in moves if q_values[i] == best_value]

    return random.choice(best_moves)


def train_ai(episodes=50000):
    """Train the AI by playing against itself."""

    for episode in range(episodes):
        board = [' '] * 9
        player = 'X'
        previous_state = None
        previous_action = None

        while True:

            state = get_state(board)
            action = choose_action(board, training=True)

            board[action] = player

            result = check_winner(board)

            if result is not None:

                if result == player:
                    reward = 1
                elif result == 'Draw':
                    reward = 0.2
                else:
                    reward = -1

                q_values = get_q_values(state)

                # Q-learning update
                old_value = q_values[action]
                q_values[action] = old_value + ALPHA * (
                    reward - old_value
                )

                break

            # Next state
            next_state = get_state(board)
            next_q_values = get_q_values(next_state)

            next_moves = available_moves(board)

            max_next_value = max(
                next_q_values[i] for i in next_moves
            )

            q_values = get_q_values(state)

            old_value = q_values[action]

            q_values[action] = old_value + ALPHA * (
                GAMMA * max_next_value - old_value
            )

            # Change player
            player = 'O' if player == 'X' else 'X'

    print("AI training completed!")


def display_board(board):
    """Display the Tic-Tac-Toe board."""

    print()
    print("-------------")

    for i in range(0, 9, 3):
        print(
            "| "
            + board[i]
            + " | "
            + board[i + 1]
            + " | "
            + board[i + 2]
            + " |"
        )
        print("-------------")

    print()


def play_game():
    """Play Tic-Tac-Toe against the trained AI."""

    board = [' '] * 9

    print("\n==============================")
    print("     TIC-TAC-TOE AI GAME")
    print("==============================")
    print("You are X")
    print("AI is O")

    print("\nBoard positions:")
    print()
    print("-------------")
    print("| 1 | 2 | 3 |")
    print("-------------")
    print("| 4 | 5 | 6 |")
    print("-------------")
    print("| 7 | 8 | 9 |")
    print("-------------")

    while True:

        # Human move
        display_board(board)

        while True:
            try:
                position = int(input("Enter your position (1-9): ")) - 1

                if position in available_moves(board):
                    break

                print("Position already occupied. Try again.")

            except ValueError:
                print("Please enter a number from 1 to 9.")

        board[position] = 'X'

        result = check_winner(board)

        if result is not None:
            display_board(board)

            if result == 'X':
                print("Congratulations! You won!")
            else:
                print("It's a draw!")

            break

        # AI move
        ai_move = choose_action(board, training=False)
        board[ai_move] = 'O'

        print("AI chose position:", ai_move + 1)

        result = check_winner(board)

        if result is not None:
            display_board(board)

            if result == 'O':
                print("AI wins! Better luck next time.")
            else:
                print("It's a draw!")

            break


# Main program
if __name__ == "__main__":

    print("Training the Reinforcement Learning AI...")
    train_ai(50000)

    play_game()