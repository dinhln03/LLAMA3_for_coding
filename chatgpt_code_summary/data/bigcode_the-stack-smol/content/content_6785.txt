from MoveGetter import MoveGetter
import chess

class CommandLineMoveGetter(MoveGetter):

    def getMove(self, board):
        print("\n")
        print(board)
        self.printLegalMoves(board)
        return self.getMoveFromCLI(board)

    def printLegalMoves(self, board):
        for index, move in enumerate(board.legal_moves):
            print(str(index) + ": ", end="")
            print(board.san(move))

    def getMoveFromCLI(self, board):
        selection = -1
        while(selection < 0 or selection >= len(board.legal_moves)):
            try:
                selection = int(input("Select a move "))
            except ValueError:
                print("Invalid input")
                 

        # print(board.legal_moves)
        for index, move in enumerate(board.legal_moves):
            if index == selection:
                return move