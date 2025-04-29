from .chessGUI import GUIgame
from .chess import FEN_position
import numpy as np
import sys

def interpret_FEN_string(string):
    """Take a FEN_string and return the corresponding FEN_position."""
    file_numbers = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7}
    piece_types = 'KQRBNPkqrbnp'
    
    string = string.split()
    rows = string[0].split('/')
    position = np.zeros((8,8,13*4+6))
    rank = 7
    for string_row in rows:
        file = 0
        for x in string_row:
            if x.isalpha():
                position[rank,file,piece_types.index(x)] = 1
                file += 1
            else:
                file += int(x)
        rank -= 1
    
    if string[1] == 'w': # active
        position[:,:,52] = np.ones((8,8))
    corners = {'K':0,'Q':1,'k':2,'q':3}
    if string[2] != '-': # castling
        for i in string[2]:
            position[:,:,53+corners[i]] = np.ones((8,8))
        
    
    en_passant = string[3]
    if en_passant != '-':
        en_passant = self.FEN.s2t(en_passant)
    else:
        en_passant = None
    
    position[:,:,57] += int(string[4])/100 # halfmove
    fullmove = int(string[5])
    return FEN_position(position,en_passant,fullmove)

def interpret_history(history):
    history = [[move for move in fullmove.split()[1:]] for fullmove in history.split('\n')]
    if history[0][0] == '...': # starting with black move;
        # assume in this case that we are passed a start_string with colour 'b'
        history[0] = history[0][1]
    return history

def start_game(FEN_string = None,history = None):
    """Given a FEN_string, starts a game from that position.
    Given a history, starts from the standard position and plays
    through the history to start a game at the current position.
    Given both, assumes FEN_string is the start position before
    playing through history."""
    if FEN_string == None:
        FEN_string = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    if history == None:
        GUIgame(interpret_FEN_string(FEN_string)).play()
    else:
        my_game = GUIgame(interpret_FEN_string(FEN_string))
        history = interpret_history(history) # turn a string into a list of moves
        for fullmove in history:
            for move in fullmove:
                check_move = my_game.next_move(move)
                if check_move != None:
                    break
            if check_move != None:
                break
        my_game.play()

def start_with_position_or_history():
    # accept 5 possibilities: no parameters, just FEN, just history, FEN then history
    # or history then FEN (start from FEN and apply history)
    parameters = sys.argv[1:]
    if len(parameters) == 0:
        start_game()
    elif len(parameters) == 1:
        if '.' in parameters[0]:
            start_game(history = parameters[0][1:-1])
            # for reasons I don't understand, history is otherwise passed
            # with quotation marks on either side(??).
            # FEN, on a single line, doesn't do that.
            # Sort this out.
        else:
            start_game(FEN_string = parameters[0])
    else:
        if '.' in parameters[0]: # history then FEN
            start_game(parameters[1],parameters[0][1:-1])
        else:
            start_game(parameters[0],parameters[1][1:-1])

if __name__ == '__main__':
    start_with_position_or_history()