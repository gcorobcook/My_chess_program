from chess.chess import *
from chess.chessAI import *
from chess.start_with_input import *
import unittest
import copy

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
        FEN = interpret_FEN_string(string)
        for i in range(8):
            leaf_queue.append((copy.deepcopy(FEN),[]))
        tree = MCTS_tree()
        current = FEN
        children = list(current.moves())
        move_count = len(children)
        model = get_or_create_model("best_model")
        tree.evaluate_queue(leaf_queue,FEN,model)
        self.assertAlmostEqual(sum(tree[string].P),1)
    
    def test_training_loop(self):
        num_games = 2
        epochs = 5
        batch_size = 32
        games_each = 1
        
        best_model = get_or_create_model("best_model")
        add_to_buffer(best_model,num_games)
        
        current_model = get_or_create_model("current_model")
        train_model(current_model,epochs,batch_size)
        current_model.save("current_model.keras",include_optimizer=True)
        
        results = compare_models(current_model,best_model,games_each)
        print("Results: ",results)
        if results['model1'] > results['model2']:
            current_model.save("best_model.keras",include_optimizer=True)

if __name__ == '__main__':
    unittest.main()