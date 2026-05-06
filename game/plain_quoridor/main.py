from game.plain_quoridor.game_logic import GridGame


def main():
    game = GridGame()
    playing = True

    while playing:
        game.print_grid()

        player = int(input("Player (1 or 2): "))

        action = input("Action: (m)ove or (w)all? ")

        # 🔹 MOVIMENTO
        if action == "m":
            moves = game.available_moves(player)

            print("Available moves:")
            for i, m in enumerate(moves):
                print(i, ":", m)

            try:
                idx = int(input("Choose move index: "))
                move = moves[idx]
            except:
                print("Input non valido.")
                continue

            game.move(player, move)

        # 🔹 MURO
        elif action == "w":
            try:
                x = int(input("Wall x: "))
                y = int(input("Wall y: "))
                orientation = input("Orientation (h/v): ")
            except:
                print("Input non valido.")
                continue

            success = game.place_wall(player, (x, y), orientation)

            if not success:
                print("Muro non valido")
                continue

        else:
            print("Azione non valida")
            continue

        # 🔹 CHECK WIN
        winner = game.check_winner()
        if winner:
            game.print_grid()
            print(f"Player {winner} wins!")
            playing = False


if __name__ == "__main__":
    main()
