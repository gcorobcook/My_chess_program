from chess.start_with_input import *
import unittest

class testFENmethods(unittest.TestCase):
    
    def test_en_passant(self):
        string = 'rnbqkbnr/pppppppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1'
        FEN = interpret_FEN_string(string)
        FEN.next_move((11,6,3,4,3,''))
        self.assertEqual(FEN.en_passant,(5,3))

if __name__ == '__main__':
    unittest.main()