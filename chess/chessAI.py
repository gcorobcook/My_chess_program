# at some point, take constants like alpha, MCTS tree size, c_puct, temperature,...
# outside together to if name==main, or somewhere.
# Number of layers in nn, learning rate, batch size, epochs, number of games,...
# my understanding, based on e.g.
# https://stats.stackexchange.com/questions/322831/purpose-of-dirichlet-noise-in-the-alphazero-paper,
# is that alpha should be independent of MCTS tree size.

# Batch process requests to the model when using tree.build(), for speed. This is the
# main place I'm currently losing time.

from . import chess

import numpy as np
import random
import copy

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import categorical_crossentropy, MSE
from tensorflow.keras.utils import register_keras_serializable

# I will generally attempt alpha zero's strategy. I will use their
# position/move embeddings, convolutional (residual?) layers, etc.
# Then train with self-play.

# 1. Get the MCTS working. Take as input a position and a vector of probabilities
# (to be produced by the current best model), and output a new vector of
# probabilities + win probability. (Position as in the input to the neural network.)
# Also take an optional argument with a partial MCTS tree constructed earlier

class tree_node:
    def __init__(self,children=[],N=[],W=[],Q=[],P=[],visits=1):
        # the weights weight the child edges - same shape array. (Probably a list.)
        # data is the same shape array, carrying data about each child.
        # move is the move that gets you from last to self.
        self.children = children
        self.N=N
        self.W=W
        self.Q=Q
        self.P=P
        self.visits = visits

class MCTS_tree(dict):
    # Essentially a dictionary of positions,
    # with some methods respecting the tree structure
    # For each key, we store values for that position
    
    rng = np.random.default_rng()
    
    def add(self,name,children=[],N=[],W=[],Q=[],P=[],visits=1):
        self[name] = tree_node(children,N,W,Q,P,visits) # can use a hash of the name
    
    def evaluate_queue(self,leaf_queue,FEN,model):
        positions = np.stack([i[0].position for i in leaf_queue])
        weights,v_scores = model(positions,training=False)
        weights = weights.numpy().reshape(-1,8,8,73)
        
        # update data
        for i in range(8):
            current,current_path = leaf_queue[i]
            weight = weights[i]
            
            # add node with probabilities
            name = current.FEN_string()
            if name not in self:
                children = list(current.moves())
                move_count = len(children)
                probs = [weight[FEN.encode_move(move)] for move in children]
                probs = [prob/sum(probs) for prob in probs]
                self.add(name,children,[0]*move_count,[0]*move_count,[0]*move_count,
                probs,1)
            
            # update path with v_scores
            for name,move in current_path:
                self[name].W[move] += v_scores[i]
                self[name].Q[move] = self[name].W[move]/self[name].N[move]
        # Note: after reaching a checkmate/stalemate, we still get a v_score
        # and update scores for the path to it.

    def build(self,FEN,model,training=True):
        # Add Dirichlet noise inside MCTS.
        # Data is (N,W,Q,P) for each child,
        # for number of visits, total value, mean value, and prior.
        # Each time we add a node after calculating weights.
        # The tree stores FEN_positions, so it includes fullmove
        # and en passant (convenience for calculating .moves()), but only
        # FEN_position.position will be passed to the neural network, which
        # doesn't need fullmove and will have to learn en passant itself from
        # the included history.
        to_consider = 200 # may need tuning
        c_puct = 1 # work out a reasonable value
        
        root = FEN.FEN_string()
        
        if root not in self: # should only happen at the beginning of a game
            children = list(FEN.moves())
            move_count = len(children)
            
            weights,v_score = model(FEN.position.reshape(1,8,8,58), training=False)
            weights = weights.numpy().reshape(8,8,73)
            
            # calculate probs with model, + Dirichlet noise when training
            probs = [weights[FEN.encode_move(move)] for move in children]
            probs = [prob/sum(probs) for prob in probs]
            if training:
                noise = self.rng.dirichlet([0.3]*move_count)
                probs = [0.75*probs[i] + 0.25*noise[i] for i in range(move_count)]
            self.add(root,children,[0]*move_count,[0]*move_count,[0]*move_count,
            probs,1)
        else: # add Dirichlet noise to the new root node
            probs = self[root].P
            if training:
                move_count = len(probs)
                noise = self.rng.dirichlet([0.3]*move_count)
                probs = [0.75*probs[i] + 0.25*noise[i] for i in range(move_count)]
            self[root].P = probs
        
        # add nodes in mini-batches of 8
        leaf_queue = []
        while self[root].visits < to_consider:
            # select a node to add a new edge to
            current = copy.deepcopy(FEN)
            current_path = []
            best_move = None
            name = current.FEN_string()
            while name in self:
                if len(self[name].children)==0: # checkmate/stalemate - deal with below
                    break
                best_score = float('-inf')
                best_move = None
                children = self[name].children
                for i in range(len(children)):
                    score = (self[name].Q[i] + c_puct*self[name].P[i]*
                    self[name].visits**0.5/(1+self[name].N[i]))
                    if score > best_score:
                        best_score = score
                        best_move = i
                current_path.append((name,best_move))
                current.next_move(children[best_move])
                name = current.FEN_string()
            
            # add leaf to queue
            leaf_queue.append((current,current_path))
            
            # add to N on path as we go so this leaf isn't chosen again in this mini-batch
            # (unless few choices are available)
            for name,move in current_path:
                self[name].visits += 1
                self[name].N[move] += 1
            
            # evaluation and update MCTS, 8 nodes at a time
            if len(leaf_queue) == 8:
                self.evaluate_queue(leaf_queue,FEN,model)
                leaf_queue = []
    
    def subtree(self,FEN):
        # build the subtree of self starting at FEN and return
        new_tree = MCTS_tree()
        root = FEN.FEN_string()
        if root not in self:
            return new_tree
        new_tree[root] = self[root]
        children = self[root].children
        for i in range(len(children)):
            if self[root].N[i] > 0:
                move = children[i]
                current = copy.deepcopy(FEN)
                current.next_move(move)
                new_tree.update(self.subtree(current))
                # recursion
        return new_tree

# 2. Self-play. Play whole games using MCTS search to select moves. Return a
# set of positions from the game with post-MCTS probabilities and final result,
# to be sampling in training
def self_play(model):
    # at the end, when labelling pairs for the training buffer, I want to
    # implement the colour change by saying v=1 if the colour whose move it is
    # wins; -1 if they lose. So giving a 'current score' for games will require
    # changing the sign of v when black is active.
    temperature = 1 # may need tuning
    training_buffer = []
    
    new_game=chess.game()
    tree = MCTS_tree()
    for j in range(600): # i.e. that many halfmoves allowed
        if new_game.FEN.is_checkmate():
            if new_game.FEN.colour(): # black wins
                new_game.history.append('0-1')
                print(new_game.history[-1])
                for i in range(len(training_buffer)):
                    if i%2 == 0: # white moves
                        training_buffer[i][2] = -1
                    else: # black moves
                        training_buffer[i][2] = 1
            else: # white wins
                new_game.history.append('1-0')
                print(new_game.history[-2]) # print white's last move too
                print(new_game.history[-1])
                for i in range(len(training_buffer)):
                    if i%2 == 0: # white moves
                        training_buffer[i][2] = 1
                    else: # black moves
                        training_buffer[i][2] = -1
            break
        elif new_game.is_draw():
            new_game.history.append(('1/2-1/2 ',new_game.is_draw()))
            if not new_game.FEN.colour():
                print(new_game.history[-2]) # print white's last move too
            print(new_game.history[-1])
            break
        else:
            # build a MCTS tree
            tree.build(new_game.FEN,model)
            # get probabilities from the MCTS
            root = tree[new_game.FEN.FEN_string()]
            probs = root.N
            probs = [prob**(1/temperature) for prob in probs]
            probs = [prob/sum(probs) for prob in probs]
            # encode probabilities for training
            encoded_probs = np.zeros((8,8,73))
            for i in range(len(root.children)):
                encoded_probs[new_game.FEN.encode_move(root.children[i])] = probs[i]
            training_buffer.append([new_game.FEN.position,encoded_probs,0])
            # choose move
            move = random.choices(root.children,probs)[0]
            # .next_move() and prune tree
            new_game.next_move(move)
            if new_game.FEN.colour(): # for my own amusement, while the games are slow
                print(new_game.history[-1])
            tree = tree.subtree(new_game.FEN)
    return training_buffer

# 2.5. Add games to buffer

def init_worker(model_name):
    global MODEL
    MODEL = get_or_create_model(model_name,opts=False)

def self_play_wrapper():
    return self_play(MODEL)

def add_to_buffer(model_name,num_games=64):
    if os.path.exists("training_data.pkl"):
        with open("training_data.pkl", "rb") as f:
            big_training_buffer = pickle.load(f)
    else:
        big_training_buffer = []
    
    num_processes = 4
    
    with ProcessPoolExecutor(max_workers=num_processes,
        initializer=init_worker,
        initargs=(model_name,)) as executor:
        
        # Submit a task for each process to run self_play num_games//num_processes times.
        pending = []
        for _ in range(num_games):
            if len(pending) >= num_processes:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    training_buffer = future.result()
                    print("New positions to add: ",len(training_buffer))
                    big_training_buffer += training_buffer
                    print("Positions in buffer: ",len(big_training_buffer))
                    # Save the aggregated data to a pickle file.
                    with open("training_data.pkl", "wb") as f:
                        pickle.dump(big_training_buffer, f)
            pending.append(executor.submit(self_play_wrapper))
        for future in as_completed(pending):
            training_buffer = future.result()
            print("New positions to add: ",len(training_buffer))
            big_training_buffer += training_buffer
            print("Positions in buffer: ",len(big_training_buffer))
            # Save the aggregated data to a pickle file.
            with open("training_data.pkl", "wb") as f:
                pickle.dump(big_training_buffer, f)
    
    # Only keep the most recent 100,000 moves (about 200 games)
    buffer_size = len(big_training_buffer)
    if buffer_size > 100000:
        big_training_buffer = big_training_buffer[buffer_size-100000:]
    with open("training_data.pkl", "wb") as f:
        pickle.dump(big_training_buffer, f)
    
    print("Self-play completed!")
    print("Positions in buffer: ",len(big_training_buffer))

# 3. Neural network training.

# Model to output predicted rewards (Q-values) of taking an action

def conv_block(x):
    x = Conv2D(filters=256,kernel_size=3,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def residual_block(x):
    y = Conv2D(filters=256,kernel_size=3,padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters=256,kernel_size=3,padding='same')(y)
    y = BatchNormalization()(y)
    y += x
    y = Activation('relu')(y)
    
    return y

def base_layers(x):
    x = conv_block(x)
    
    num_blocks = 5
    for _ in range(num_blocks):
        x = residual_block(x)
    
    return x

def moves_head(x):
    x = Conv2D(filters=73,kernel_size=1,padding='same')(x)
    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    
    # for softmax, flatten then reshape
    x = Flatten()(x)
    x = Activation('softmax')(x)
    return x

def v_head(x):
    x = Conv2D(filters=1,kernel_size=1,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(256,activation='relu')(x)
    x = Dense(1,activation='tanh')(x)
    return x

# @register_keras_serializable()
# def my_loss(y_true,y_pred):
    # # custom loss function, combining the errors from moves_head and v_head
    # # y_pred=(p,v) is the move probs and v_score from the network;
    # # y_true=(pi,z) is the MCTS frequencies and final game result
    # pi = y_true[0]
    # z = y_true[1]
    # p = y_pred[0]
    # v = y_pred[1]
    # return categorical_crossentropy(pi, p) + MSE(z, v)

def build_model(name,opts=True):
    input_layer = Input(shape=(8,8,58))
    x = base_layers(input_layer)
    p = moves_head(x)
    v = v_head(x)
    
    model = Model(inputs=input_layer, outputs=[p,v], name=name)
    if opts:
        model.compile(optimizer=AdamW(learning_rate=0.002),
        loss=[categorical_crossentropy,MSE])
    return model

def get_training_data(batch_size):
    with open("training_data.pkl", "rb") as f:
        big_training_buffer = pickle.load(f)
    data = random.choices(big_training_buffer,k=batch_size)
    X_train = np.zeros((batch_size,8,8,58))
    pi_train = np.zeros((batch_size,8*8*73))
    z_train = np.zeros(batch_size)
    for i in range(len(data)):
        X_train[i,:,:,:] = data[i][0]
        pi_train[i,:] = data[i][1].reshape(1,8*8*73)
        z_train[i] = data[i][2]
    return X_train, pi_train, z_train

# For training, I will want to either build_model() or, if training a model I
# already have, load the model and train that.
# I will compare models every 100 epochs

def train_model(model_name,epochs=100,batch_size=256):
    model = get_or_create_model(model_name,opts=True)
    for epoch in range(epochs):
        X_train, pi_train, z_train = get_training_data(batch_size)
        model.fit(X_train, [pi_train,z_train])
        if epoch+1%10 == 0:
            print("Epoch ",epoch+1," done!")
    model.save(model_name+".keras",include_optimizer=True)

# 4. Compare models

# Takes two models (can be the same) and play them against each other

def play_2models(model1,model2):
    # So that, during training, I can check if the new model beats the old one
    # model1 plays white, model2 black.
    temperature = 1 # may need tuning
    new_game=chess.game()
    # Each model has its own tree search
    tree1 = MCTS_tree()
    tree2 = MCTS_tree()
    for j in range(600): # i.e. that many halfmoves allowed
        if new_game.FEN.is_checkmate():
            if new_game.FEN.colour(): # black wins
                new_game.history.append('0-1')
                print(new_game.history[-1])
                return model1.name
            else: # white wins
                new_game.history.append('1-0')
                print(new_game.history[-2:]) # to include the last white move
                return model2.name
        elif new_game.is_draw():
            new_game.history.append(('1/2-1/2 ',new_game.is_draw()))
            if not new_game.FEN.colour():
                print(new_game.history[-2])
            print(new_game.history[-1])
            return new_game.is_draw()
        else:
            if new_game.FEN.colour(): # white's move
                # build a MCTS tree
                tree1.build(new_game.FEN,model1,False)
                # get probabilities from the MCTS
                root = tree1[new_game.FEN.FEN_string()]
            else: # black's move
                tree2.build(new_game.FEN,model2,False)
                root = tree2[new_game.FEN.FEN_string()]
            probs = root.N
            move_count = len(probs)
            N_max = float('-inf')
            arg_max = None
            for i in range(move_count):
                if probs[i] > N_max:
                    N_max = probs[i]
                    arg_max = i
            # choose move
            move = root.children[arg_max]
            # .next_move() and prune trees
            new_game.next_move(move)
            tree1 = tree1.subtree(new_game.FEN)
            tree2 = tree2.subtree(new_game.FEN)
            if new_game.FEN.colour(): #test
                print(new_game.history[-1])
    return 'ran out of moves'

def compare_models_init_worker(model1_name,model2_name):
    global MODEL1,MODEL2
    MODEL1 = get_or_create_model(model1_name,opts=True)
    MODEL2 = get_or_create_model(model2_name,opts=False)

def play_2models_wrapper():
    return play_2models(MODEL1,MODEL2),play_2models(MODEL2,MODEL1)

def compare_models(model1_name,model2_name,games_each=10):
    results = {model1_name:0,model2_name:0}
    
    num_processes = 4
    with ProcessPoolExecutor(max_workers=num_processes,
    initializer=compare_models_init_worker,
    initargs=(model1_name,model2_name)) as executor:
        pending = []
        for _ in range(games_each):
            if len(pending) >= num_processes:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result1,result2 = future.result()
                    results[result1] = results.get(result1,0)+1
                    results[result2] = results.get(result2,0)+1
            pending.append(executor.submit(play_2models_wrapper))
        for future in as_completed(pending):
            result1,result2 = future.result()
            results[result1] = results.get(result1,0)+1
            results[result2] = results.get(result2,0)+1
    return results

def play_2models_no_trees(model1,model2):
    # Moves are selected based only on the raw probabilities output by the models.
    # model1 plays white, model2 black.
    temperature = 1
    new_game=chess.game()
    for j in range(600): # i.e. that many halfmoves allowed
        if new_game.FEN.is_checkmate():
            if new_game.FEN.colour(): # black wins
                new_game.history.append('0-1')
                return model1.name
            else: # white wins
                new_game.history.append('1-0')
                return model2.name
        elif new_game.is_draw():
            new_game.history.append('1/2-1/2')
            # print(new_game.history) #test
            return new_game.is_draw()
        else:
            if new_game.FEN.colour():
                weights,v_score = model1(new_game.FEN.position.reshape(1,8,8,58), training=False)
                # weights = np.ones((8,8,73)) # to allow testing until the model is working
                # v_score = 0 # same
            else:
                weights,v_score = model2(new_game.FEN.position.reshape(1,8,8,58), training=False)
                # weights = np.ones((8,8,73)) # to allow testing until the model is working
                # v_score = 0 # same
            weights = weights.numpy().reshape(8,8,73)
            children = list(new_game.FEN.moves())
            probs = [weights[new_game.FEN.encode_move(move)] for move in children]
            probs = [prob**(1/temperature) for prob in probs]
            probs = [prob/sum(probs) for prob in probs]
            move = random.choices(children,probs)[0]
            # .next_move() and prune tree
            new_game.next_move(move)
    return 'ran out of moves'

def compare_models_no_trees(model1,model2):
    results = {model1.name:0,model2.name:0}
    games_each = 100
    for i in range(games_each):
        result = play_2models_no_trees(model1,model2)
        results[result] = results.get(result,0)+1
    for i in range(games_each):
        result = play_2models_no_trees(model2,model1)
        results[result] = results.get(result,0)+1
    return results

def get_or_create_model(name,opts=True):
    if os.path.exists(name+".keras"):
        model = load_model(name+".keras",compile=opts)
        model.name = name
    else:
        model = build_model(name,opts)
        model.save(name+".keras",include_optimizer=False)
    return model

def main_training_loop():
    # Go through self-play, training, comparing the new model with the best one,
    # and saving training buffer and latest models
    # I could also do learning rate scheduling, tuning hyperparameters, etc...
    
    num_games = 32
    epochs = 50
    batch_size = 128
    games_each = 8
    
    while True:
        add_to_buffer("best_model",num_games)
        
        train_model("current_model",epochs,batch_size)
        
        results = compare_models("current_model","best_model",games_each)
        print("Results: ",results)
        if results["current_model"] > results["best_model"]:
            current_model.save("best_model.keras",include_optimizer=False)

if __name__ == "__main__":
    main_training_loop()