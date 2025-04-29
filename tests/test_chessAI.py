from chess.chessAI import *
from chess.start_with_input import *
import unittest

class testAImethods(unittest.TestCase):
    
    def test_model(self):
        string = 'rnbqkbnr/pppppppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1'
        FEN = interpret_FEN_string(string)
        position = FEN.position.reshape(1,8,8,58)
        model = get_or_create_model("best_model")
        weights,v_score = model(position, training=False)
        self.assertEqual(weights.shape,(1,8*8*73))
    
    def test_evaluate_queue(self):
        leaf_queue = []
        string = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        for i in range(8):
            leaf_queue.append(interpret_FEN_string(string))
        tree = MCTS_tree()
        model = get_or_create_model("best_model")
        tree.evaluate_queue(leaf_queue,model)
    
    def test_MCTS_build(self):
        string = 'rnbqkbnr/pppppppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1'
        FEN = interpret_FEN_string(string)

if __name__ == '__main__':
    unittest.main()