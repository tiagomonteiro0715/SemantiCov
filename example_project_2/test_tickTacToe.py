import unittest
from main import TicTacToe

class TestTicTacToe(unittest.TestCase):

    def setUp(self):
        self.ttt = TicTacToe()
        self.players_dict = {"player-1": "x", "player-2": "o"}

    def test_place_marker(self):
        result1 = self.ttt.place_marker('x', 2, 1)
        self.assertTrue(result1)

        result2 = self.ttt.place_marker('o', 2, 1)
        self.assertFalse(result2)

    def test_game_result_row1(self):
        self.ttt.board[0][0] = 'x'
        self.ttt.board[0][1] = 'x'
        self.ttt.board[0][2] = 'x'
        self.assertTrue(self.ttt.game_result(self.players_dict))

        self.ttt.board[0][0] = 'o'
        self.assertFalse(self.ttt.game_result(self.players_dict))

    def test_game_result_row2(self):
        self.ttt.board[1][0] = 'x'
        self.ttt.board[1][1] = 'x'
        self.ttt.board[1][2] = 'x'
        self.assertTrue(self.ttt.game_result(self.players_dict))

        self.ttt.board[1][0] = 'o'
        self.assertFalse(self.ttt.game_result(self.players_dict))

    def test_game_result_row3(self):
        self.ttt.board[2][0] = 'x'
        self.ttt.board[2][1] = 'x'
        self.ttt.board[2][2] = 'x'
        self.assertTrue(self.ttt.game_result(self.players_dict))

        self.ttt.board[2][0] = 'o'
        self.assertFalse(self.ttt.game_result(self.players_dict))

    def test_game_result_column1(self):
        self.ttt.board[0][0] = 'x'
        self.ttt.board[1][0] = 'x'
        self.ttt.board[2][0] = 'x'
        self.assertTrue(self.ttt.game_result(self.players_dict))

        self.ttt.board[2][0] = 'o'
        self.assertFalse(self.ttt.game_result(self.players_dict))

    def test_game_result_column2(self):
        self.ttt.board[0][1] = 'x'
        self.ttt.board[1][1] = 'x'
        self.ttt.board[2][1] = 'x'
        self.assertTrue(self.ttt.game_result(self.players_dict))

        self.ttt.board[2][1] = 'o'
        self.assertFalse(self.ttt.game_result(self.players_dict))

    def test_game_result_column3(self):
        self.ttt.board[0][2] = 'x'
        self.ttt.board[1][2] = 'x'
        self.ttt.board[2][2] = 'x'
        self.assertTrue(self.ttt.game_result(self.players_dict))

        self.ttt.board[2][2] = 'o'
        self.assertFalse(self.ttt.game_result(self.players_dict))

    def test_game_result_diagonal1(self):
        self.ttt.board[0][0] = 'x'
        self.ttt.board[1][1] = 'x'
        self.ttt.board[2][2] = 'x'
        self.assertTrue(self.ttt.game_result(self.players_dict))

        self.ttt.board[0][0] = 'o'
        self.assertFalse(self.ttt.game_result(self.players_dict))

    def test_game_result_diagonal2(self):
        self.ttt.board[0][2] = 'x'
        self.ttt.board[1][1] = 'x'
        self.ttt.board[2][0] = 'x'
        self.assertTrue(self.ttt.game_result(self.players_dict))

        self.ttt.board[1][1] = 'o'
        self.assertFalse(self.ttt.game_result(self.players_dict))


if __name__ == "__main__":
    unittest.main()
