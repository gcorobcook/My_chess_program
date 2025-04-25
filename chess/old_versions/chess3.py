# I'd like to have: a picture of the board, the game history, draw offer/
# acceptance and resignation buttons.
# Chess clocks. Increment?
# Chess engine.

#------------------------------------------------------------------------------

import numpy as np
import sys
import hashlib

class FEN_position:
    """More or less a position in FEN."""
    # Should contain information equivalent to FEN, for compatibility.

    def __init__(self,position,colour=True,
        castling=[True,True,True,True],en_passant='-',halfmove=0,fullmove=1):
        self.position = position
        # NB: Creating weird positions, e.g. more/less kings, pawns on final rank,
        # etc., may produce weird behaviour. So that's your own fault.
        self.colour = colour
        self.castling = castling
        self.en_passant = en_passant
        self.halfmove = halfmove
        self.fullmove = fullmove
        #True is white, False is black.
        #Castling rights doesn't pay attention to whether you can actually
        #castle; just whether your king/rook have moved yet. Like in FEN.
        #The Trues refer to white kingside, white queenside,... castling.
        
        # Check whether there are
        # kings/rooks on their starting squares in the start position.
        if self.position[0,4] != 'K':
            self.castling[0:1] = [False,False]
        if self.position[0,0] != 'R':
            self.castling[1] = False
        if self.position[0,7] != 'R':
            self.castling[0] = False
        if self.position[7,4] != 'k':
            self.castling[2:3] = [False,False]
        if self.position[7,0] != 'r':
            self.castling[3] = False
        if self.position[7,7] != 'r':
            self.castling[2] = False
    
    file_numbers = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7}
    file_letters = 'abcdefgh'
    
    def t2s(self,square_tuple):
        '''From a pair of numbers in range(7) to a square name.'''
        rank,file = square_tuple
        file = self.file_letters[file]
        return file+str(rank)
        
    
    def s2t(self,square_name):
        '''From a square name to a pair of numbers in range(7).'''
        file,rank = square_name
        file = self.file_numbers[file]
        return (rank,file)
    
    def in_board(self,square):
        '''Returns True if 0 leq file,rank leq 7, False otherwise.'''
        return (0 <= square[0] <= 7 and 0 <= square[1] <= 7)
    
    def slider_moves(self,rank,file,piece_name,options,protection=False):
        #These next few are subsidiary to piece_moves.
        piece_moves = set()
        colour = self.position[rank,file].isupper()
        for option in options:
            new_square = (rank,file)
            for i in range(7):
                new_square = (new_square[0]+option[0],new_square[1]+option[1])
                if not self.in_board(new_square):
                    break
                target = self.position[new_square]
                if target == '0':
                    piece_moves.add((piece_name,rank,file,new_square[0],new_square[1],''))
                elif (target.isupper() != colour) or protection:
                    piece_moves.add((piece_name,rank,file,new_square[0],new_square[1],'x'))
                    break
                else:
                    break
        return piece_moves
    
    sliders = {'r':[(1,0),(0,1),(-1,0),(0,-1)],
    'b': [(i,j) for j in [-1,1] for i in [-1,1]],
    'q':[(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]}
    
    def hopper_moves(self,rank,file,piece_name,options,protection=False):
        piece_moves = set()
        colour = self.position[rank,file].isupper()
        for option in options:
            new_square = (rank+option[0],file+option[1])
            if not self.in_board(new_square):
                continue
            target = self.position[new_square]
            if target == '0':
                piece_moves.add((piece_name,rank,file,new_square[0],new_square[1],''))
            elif (target.isupper() != colour) or protection:
                piece_moves.add((piece_name,rank,file,new_square[0],new_square[1],'x'))
        return piece_moves
    
    hoppers = {'n':[(1,2),(2,1),(-1,2),(2,-1),(-2,1),(1,-2),(-1,-2),(-2,-1)],
    'k':[(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]}
    
    def pawn_moves(self,rank,file,protection=False):
        piece_moves = set()
        colour = self.position[(rank,file)].isupper() # True for white, False for black
        piece_names = 'qrbn'
        directions = [-1,1]
        start_rank = [6,1]
        to_promote_rank = [1,6]
        new_square = (rank+directions[colour],file)
        target = self.position[new_square]
        if (target == '0') and not protection:
            if rank != to_promote_rank[colour]:
                piece_moves.add(('p',rank,file,new_square[0],new_square[1],''))
            else:
                for piece_name in piece_names:
                    piece_moves.add(('p',rank,file,new_square[0],new_square[1],piece_name))
            if rank == start_rank[colour]:
                new_square = (rank+2*directions[colour],file)
                target = self.position[new_square]
                if target == '0':
                    piece_moves.add(('p',rank,file,new_square[0],new_square[1],''))
                    
        options = [-1,1] # for capturing
        for option in options:
            new_square = (rank+directions[colour],file+option)
            if not self.in_board(new_square):
                continue
            target = self.position[new_square]
            if (new_square == self.en_passant) or protection:
                piece_moves.add(('p',rank,file,new_square[0],new_square[1],'e'))
            elif target == '0':
                continue
            if target.isupper() != colour:
                if rank != to_promote_rank[colour]:
                    piece_moves.add(('p',rank,file,new_square[0],new_square[1],'x'))
                else:
                    for piece_name in piece_names.upper():
                        piece_moves.add(('p',rank,file,new_square[0],new_square[1],piece_name))
        return piece_moves
    

    def piece_moves(self,rank,file):
        """Gives a list of valid moves for the piece in that square,
        not worrying (for now) about whether the move results in check."""
        #A move is stored as: piece, starting rank and file, ending rank and
        #file, and a string '' or 'x' for captures. En passant has 'e' for the
        #string; castling lists the king squares with string 'K','J','k' or 'j'.
        #Prawn promotions have string 'q','r','b' or 'n'; capitalised for
        #promotions with captures.
        piece = self.position[rank,file].lower()
        if piece == 'p':
            return self.pawn_moves(rank,file)
        if piece in self.sliders:
            return self.slider_moves(rank,file,piece,self.sliders[piece])
        if piece in self.hoppers:
            return self.hopper_moves(rank,file,piece,self.hoppers[piece])

    def protected_squares(self,rank,file):
        """Squares protected by the piece in (rank,file).
        That is, breaks after adding a square with a piece of the same colour,
        not before. Does not include forward pawn moves/castles, only attacks."""
        piece = self.position[rank,file].lower()
        if piece == 'p':
            moves = self.pawn_moves(rank,file,True)
        if piece in self.sliders:
            moves = self.slider_moves(rank,file,piece,self.sliders[piece],True)
        if piece in self.hoppers:
            moves = self.hopper_moves(rank,file,piece,self.hoppers[piece],True)
        if piece == '0': # as a back-stop; don't use this
            moves = set()
        return set(tuple(move[3:5]) for move in moves)
    
    def all_protected_squares(self,colour):
        # Squares protected by colour, i.e. where not-colour's king can't go.
        squares = set()
        for rank in range(8):
            for file in range(8):
                piece = self.position[rank,file]
                if piece != '0' and piece.isupper() == colour:
                    squares.update(self.protected_squares(rank,file))
        return squares

    def square_attacked(self,square,by_who):
        # Unused
        #I.e. will a move leave us in check, and can we castle
        #Here this means, is square attacked by by_who
        squares = self.all_protected_squares(by_who)
        if square in squares:
            return True
        return False
        
    def can_castle(self,corner):
        # The corners are 0,1,2,3 corresponding to white king/queenside, black king/queenside.
        if not self.castling[corner]:
            return False
        # otherwise, we know king and rook are on the right squares.
        by_who_colour = [False,False,True,True]
        king_ranks = [0,0,7,7]
        to_check = [[(0,5),(0,6)],[(0,2),(0,3)],[(7,5),(7,6)],[(7,2),(7,3)]]
        squares = self.all_protected_squares(by_who_colour[corner])
        if (king_ranks[corner],4) in squares:
            return False
        for square in to_check[corner]:
            if (self.position[square] != '0'
            or square in squares):
                return False
        to_files = [6,2,6,2]
        castle_str = 'KJkj'
        return ('k',king_ranks[corner],4,king_ranks[corner],to_files[corner],castle_str[corner])
    
    def king_axis(self,rank,file,piece_name,options,colour):
        # The squares a slider could reach if going through opponent's pieces
        # Used to check for pins
        # Return the axis (if it exists) that leads to the king
        the_axis = None
        for option in options:
            new_square = (rank,file)
            axis = [new_square]
            for i in range(7):
                new_square = (new_square[0]+option[0],new_square[1]+option[1])
                if not self.in_board(new_square):
                    break
                target = self.position[new_square]
                if target == '0':
                    # empty square
                    axis.append(new_square)
                elif target.isupper() == colour:
                    # own colour piece
                    break
                elif target.lower() == 'k':
                    # opponent's king
                    the_axis = axis
                    break
                else:
                    # other opponent's piece
                    axis.append(new_square)
            if the_axis != None:
                break
        return the_axis
    
    def non_empty_count(self,axis):
        count = 0
        for square in axis:
            if self.position[square] != '0':
                count += 1
        return count-1 # to not include the slider at the start of the axis
    
    def axes_and_attacks(self,colour):
        # returns the squares of any hoppers/pawns attacking the king,
        # and the king axes of any sliders attacking it.
        # For king axes with only one blocking piece, returns these with
        # the square of the piece.
        # This should constitute a complete list of the conditions valid
        # (non-castling) moves must satisfy to avoid check.
        hopper_checks = []
        none_king_axes = []
        one_king_axes =[]
        for rank in range(8):
            for file in range(8):
                piece = self.position[rank,file]
                if piece.lower() in 'qrb' and piece.isupper() == colour:
                    king_axis = self.king_axis(rank,file,piece,self.sliders[piece.lower()],colour)
                    if king_axis == None:
                        continue
                    count = self.non_empty_count(king_axis)
                    if count > 1:
                        continue
                    elif count == 1:
                        one_king_axes.append(king_axis)
                    else:
                        none_king_axes.append(king_axis)
                elif piece.lower() in 'np' and piece.isupper() == colour:
                    squares = self.protected_squares(rank,file)
                    for square in squares:
                        target = self.position[square]
                        if target.lower() == 'k' and target.isupper() != colour:
                            hopper_checks.append((rank,file))
        
        # at most one hopper_check or none_king_axis can be solved by a
        # non-king move
        if len(hopper_checks)+len(none_king_axes) > 1:
            return None,None
        elif len(hopper_checks) == 1:
            return hopper_checks, one_king_axes # i.e. a list of squares and pinned pieces
        elif len(none_king_axes) == 1:
            return none_king_axes[0], one_king_axes
        else: # i.e. not check, colinear moves by pinned pieces allowed
            return None, one_king_axes

    def moves(self):
        # this now needs to check validity without using consider_move,
        # which now uses moves() to check for en passant moves.
        """Gives a list of valid moves."""
        
        # I consider two cases. In either case, the king can move to precisely
        # the non-attacked squares. If not check, calculate pinned pieces and
        # and stop them moving. (Except colinear.)
        # If check, look for attacking pieces and allow block
        # or take. (Blocking only for sliders.)
        # In either case, check all sliders for king_axes and knights/pawns for
        # king attacks.
        
        # squares the king must avoid
        attacked_squares = self.all_protected_squares(not self.colour)
        
        # checks and pinned pieces
        checker_squares, one_king_axes = self.axes_and_attacks(not self.colour)
        
        move_set = set()
        for rank in range(8):
            for file in range(8):
                piece = self.position[rank,file]
                if piece.lower() in 'qrbnp' and piece.isupper() == self.colour:
                    moves = self.piece_moves(rank,file)
                    if checker_squares == None and one_king_axes == None:
                        continue # double check
                    if checker_squares == None: # not check
                        for axis in one_king_axes:
                            if (rank,file) in axis:
                                for move in moves:
                                    if tuple(move[3:5]) in axis: # colinear
                                        move_set.add(move)
                                break
                        else:
                            move_set.update(moves)
                    else: # check
                        for axis in one_king_axes:
                            if (rank,file) in axis: # pinned while in check
                                break
                        else: # block check or take - add takes to king_axes
                            for move in moves:
                                if tuple(move[3:5]) in checker_squares:
                                    move_set.add(move)
                if piece.lower() == 'k' and piece.isupper() == self.colour:
                    # allowed king moves
                    for move in self.piece_moves(rank,file):
                        if tuple(move[3:5]) not in attacked_squares:
                            move_set.add(move)
        
        #add castling moves
        corners_to_check = [[2,3],[0,1]]
        for corner in corners_to_check[self.colour]:
            result = self.can_castle(corner)
            if result != False:
                move_set.add(result)
                    
        return move_set
    
    def next_move(self,move):
        # this is no longer to be used to check moves for validity.
        # instead to create a whole new FEN position.
        # So I can modify self in-place.
        # This essentially replaces the old next_move.
        """Returns what the FEN position will be after a move."""
        self.position[move[1],move[2]] = '0'
        piece = move[0]
        if move[5].lower() in ['q','r','b','n']: # promotion
            piece = move[5]
        if self.colour:
            piece = piece.upper()
        self.position[move[3],move[4]] = piece
        if move[5] == 'e':
            self.position[move[1],move[4]] = '0'
        if move[5] == 'J':
            self.position[0,0] = '0'
            self.position[0,3] = 'R'
        if move[5] == 'K':
            self.position[0,7] = '0'
            self.position[0,5] = 'R'
        if move[5] == 'j':
            self.position[7,0] = '0'
            self.position[7,3] = 'r'
        if move[5] == 'k':
            self.position[7,7] = '0'
            self.position[7,5] = 'r'
        
        # castling
        # King or rook moving:
        if self.colour:
            if move[0] == 'k':
                self.castling[0] = False
                self.castling[1] = False
            if move[0] == 'r':
                if move[1:2] == (0,0):
                    self.castling[1] = False
                elif move[1:2] == (0,7):
                    self.castling[0] = False
        else:
            if move[0] == 'k':
                self.castling[2] = False
                self.castling[3] = False
            if move[0] == 'r':
                if move[1:2] == (7,0):
                    self.castling[3] = False
                elif move[1:2] == (7,7):
                    self.castling[2] = False
        # rook getting taken (or not there in original position)
        if move[3] == 0:
            if move[4] == 0:
                self.FEN.castling[1] = False
            elif move[4] == 7:
                self.FEN.castling[0] = False
        elif move[3] == 7:
            if move[4] == 0:
                self.FEN.castling[3] = False
            elif move[4] == 7:
                self.FEN.castling[2] = False
        
        # en passant
        if self.colour:
            if move[0] == 'p' and move[1] == 1 and move[3] == 3:
                self.en_passant = (2,move[2])
            else:
                self.en_passant = '-'
        else:
            if move[0] == 'p' and move[1] == 6 and move[3] == 4:
                self.en_passant = (5,move[2])
            else:
                self.en_passant = '-'
        
        # halfmove
        if move[5] == 'x' or move[0] == 'p':
            self.halfmove = 0
        else:
            self.halfmove += 1
        
        # fullmove
        if not self.colour:
            self.fullmove += 1
        
        # change colour
        self.colour = not self.colour
        
        # don't check if there are valid moves using the en passant square
        # I don't want to repeat the calculation of self.moves(),
        # and this is acceptable for the FEN position
        # if self.en_passant != '-':
            # for move in self.moves():
                # if move[5] == 'e':
                    # break
            # else:
                # self.en_passant = '-'

    def insufficient(self):
        pieces = [0,0,0]
        # for number of knights, light-square bishops, dark-square bishops.
        # If pieces[0] > 1, or more than one is non-zero, no draw.
        # See https://www.reddit.com/r/chess/comments/se89db/a_writeup_on_definitions_of_insufficient_material/
        # this is not exhaustive, but close enough for most purposes.
        # In a later implementation, calculate insufficient material separately
        # for each colour; this is needed for draw by timeout vs insufficient material.
        for i in range(8):
            for j in range(8):
                piece = self.position[i,j].lower()
                if piece in 'qrp':
                    return False
                if piece == 'n':
                    pieces[0] += 1
                if piece == 'b':
                    if i%2 == j%2:
                        pieces[1] += 1
                    else:
                        pieces[2] += 1
        if pieces[0] > 1:
            return False
        elif pieces[0] == 1:
            if pieces[1] == 0 and pieces[2] == 0:
                return True
            else:
                return False
        else:
            if pieces[1] == 0 or pieces[2] == 0:
                return True
            else:
                return False
    
    def is_check(self):
        # that is, if self.colour is in check
        squares = self.all_protected_squares(not self.colour)
        for square in squares:
            if (self.position[square].lower() == 'k' and
            self.position[square].isupper() == self.colour):
                return True
        return False

    def is_checkmate(self):
        if self.is_check() and len(self.moves()) == 0:
            return True
        else:
            return False

    def is_draw(self):
        if len(self.moves()) == 0 and not self.is_check():
            return 'stalemate'
        elif self.insufficient():
            return 'insufficient'
        elif self.halfmove == 100:
            return '50 move'
        else:
            return ''
        # come back to draw offers. Add an option to claim a draw after 3/50,
        # and do it automatically after 5/75.
    
    def print_board(self):
        print(self.position[::-1,:])
    
    def rep_string(self):
        # to check for repetitions
        string = ''
        
        # position
        for i in range(7,-1,-1):
            row = self.position[i,:]
            empty_count = 0
            for j in range(8):
                if row[j] == '0': # count empties
                    empty_count += 1
                else:
                    if empty_count > 0:
                        string += str(empty_count)
                        empty_count = 0
                    string += row[j]
            if empty_count > 0:
                string += str(empty_count)
            if i>0:
                string += '/'
            else:
                string += ' '
        
        string += {True:'w',False:'b'}[self.colour] + ' ' # active
        if sum(self.castling) == 0: # castling
            string += '- '
        else:
            string += ''.join('KQkq'[i] for i in range(4) if self.castling[i]) + ' '
        if self.en_passant == '-': # en passant
            string += '-'
        else:
            string += self.t2s(self.en_passant)
        
        return string
    
    def FEN_string(self):
        string = self.rep_string()
        string += ' ' + str(self.halfmove) + ' ' + str(self.fullmove)
        return string
    
    def print_FEN(self):
        self.print_board()
        print("Active colour: ", {True:'w',False:'b'}[self.colour])
        print("Castling rights: ", ''.join('KQkq'[i] for i in range(4) if self.castling[i]))
        if self.en_passant != '-':
            print("En passant target square: ", self.t2s(self.en_passant))
        print("Halfmove clock: ", self.halfmove)
        print("Fullmove number: ", self.fullmove)

# print(FEN_position().moves())

class game:
    
    start_pos = np.array([
    ['R','N','B','Q','K','B','N','R'],
    ['P','P','P','P','P','P','P','P'],
    ['0','0','0','0','0','0','0','0'],
    ['0','0','0','0','0','0','0','0'],
    ['0','0','0','0','0','0','0','0'],
    ['0','0','0','0','0','0','0','0'],
    ['p','p','p','p','p','p','p','p'],
    ['r','n','b','q','k','b','n','r']])

    def __init__(self,FEN=FEN_position(start_pos),history=[],pos_history={}):
        self.FEN = FEN
        self.history = history
        self.pos_history = pos_history
        # expansions: build a game from a FEN string, or from a
        # history (or from history plus starting FEN string)
        
        # the number of times the current position has been repeated
        rep_string = self.FEN.rep_string()
        rep_hash = hashlib.sha256(rep_string.encode('utf-8')).hexdigest()
        if rep_hash in pos_history:
            self.repeats = self.pos_history[rep_hash]
        else:
            self.pos_history[rep_hash] = 1
            self.repeats = 1

    def alg2mine(self,move):
        """Splits a standard chess notation move into my format."""
        # Check validity afterwards.

        files = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7}
        
        def process(move):
            short_move = []
            for i in move:
                if i in 'RNBQKabcdefghx':
                    short_move.append(i)
                if i.isdigit():
                    short_move.append(int(i)-1) # shift to range(7)
            return short_move # cuts off any +s, !s, ?s
        
        def promotion_move(move):
            '''Pass processed_move to this if str(move[-1]) in 'RNBQ'.'''
            nice_move = [None,None,None,None,None,None]
            nice_move[0] = 'p'
            if self.FEN.colour:
                nice_move[1] = 6
            else:
                nice_move[1] = 1
            nice_move[2] = files[move[0]]
            if move[1] == 'x':
                nice_move[5] = move[-1]
                nice_move[3] = move[3]
                nice_move[4] = files[move[2]]
            else:
                nice_move[5] = move[-1].lower()
                nice_move[3] = move[1]
                nice_move[4] = files[move[0]]
            return tuple(nice_move)
        
        def piece_move(move):
            '''Pass processed_move to this if move[0] in 'RNBQK'.'''
            nice_move = [None,None,None,None,None,None]
            nice_move[3] = move[-1]
            nice_move[4] = files[move[-2]]
            nice_move[0] = move[0].lower()
            if 'x' in move:
                nice_move[5] = 'x'
                move = move[1:-3]
            else:
                nice_move[5] = ''
                move = move[1:-2]
            if len(move) == 2:
                nice_move[1] = move[1]
                nice_move[2] = files[move[0]]
            elif len(move) ==1:
                if type(move[0]) == str:
                    nice_move[2] = files[move[0]]
                else:
                    nice_move[1] = move[0]
            #search for valid moves...
            intention = [] # list of partial matches
            valid_moves = self.FEN.moves()
            for valid_move in valid_moves:
                for i in range(6): # find partial matches
                    if nice_move[i] == None:
                        continue
                    if nice_move[i] != valid_move[i]:
                        break
                else:
                    intention.append(valid_move)
            if len(intention) > 1:
                return 'ambiguous'
            elif len(intention) == 0:
                return 'none'
            else:
                return intention[0]
        
        def pawn_move(move):
            '''Pass processed_move to this if move[0] not in 'RNBQK'.'''
            nice_move = [None,None,None,None,None,None]
            nice_move[0] = 'p'
            nice_move[2] = files[move[0]]
            nice_move[3] = move[-1]
            nice_move[4] = files[move[-2]]
            if 'x' in move:
                if self.FEN.colour:
                    nice_move[1] = move[-1]-1
                else:
                    nice_move[1] = move[-1]+1
                if self.FEN.position[nice_move[3],nice_move[4]] == '0':
                    nice_move[5] = 'e' # en passant
                else:
                    nice_move[5] = 'x'
            else:
                nice_move[5] = ''
                # differentiate between 1-square and 2-square moves
                if self.FEN.colour:
                    if self.FEN.position[nice_move[3]-1,nice_move[4]] == '0':
                        nice_move[1] = move[-1]-2
                    else:
                        nice_move[1] = move[-1]-1
                else:
                    if self.FEN.position[nice_move[3]+1,nice_move[4]] == '0':
                        nice_move[1] = move[-1]+2
                    else:
                        nice_move[1] = move[-1]+1
            return tuple(nice_move)

        # castling
        if move == '0-0' or move == 'O-O':
            if self.FEN.colour:
                return ('k',0,4,0,6,'K')
            else:
                return ('k',7,4,7,6,'k')
        if move == '0-0-0' or move == 'O-O-O':
            if self.FEN.colour:
                return ('k',0,4,0,2,'J')
            else:
                return ('k',7,4,7,2,'j')
        
        move = process(move)
        try:
            if str(move[-1]) in 'RNBQ': # promotion
                return promotion_move(move)

            # not promotion
            if move[0] in 'RNBQK': # piece move
                return piece_move(move)

            else: # pawn move
                return pawn_move(move)
        except:
            return 'none'

    def mine2alg(self,move):
        """Write one of the moves from my format in algebraic notation."""
        # Use to construct move history.
        # Assumes my move is valid.

        files = 'abcdefgh'
        
        def promotion_move(move):
            '''Pass to this if move[5].lower() in 'qrbn' and move[5] != ''.'''
            if move[5].islower():
                return files[move[2]] + str(move[3]+1) + '=' + move[5].upper()
            else:
                return files[move[2]] + 'x' + files[move[4]] + str(move[3]+1) + '=' + move[5]
        
        def pawn_move(move):
            '''Pass to this otherwise if move[0]='p'.'''
            if move[5] != '': #capture
                return files[move[2]] + 'x' + files[move[4]] + str(move[3]+1)
            else:
                return files[move[4]] + str(move[3]+1)
        
        def piece_move(move):
            '''Pass to this if move[0] != 'p' (and not castling).'''
            moves = self.FEN.moves()
            candidates = [[move[0],None,None] + list(move[3:]),
                [move[0],None,move[2]] + list(move[3:]),
                [move[0],move[1],None] + list(move[3:])]
            for candidate in candidates:
                possibles = [] # list of partial matches
                for possible in moves:
                    for i in range(6): # find partial matches
                        if candidate[i] == None:
                            continue
                        if candidate[i] != possible[i]:
                            break
                    else:
                        possibles.append(possible)
                if len(possibles) == 1:
                    break
            else:
                candidate = move[:3]
            alg_move = move[0].upper()
            if candidate[2] != None:
                alg_move += files[move[2]]
            if candidate[1] != None:
                alg_move += str(move[1]+1)
            alg_move += move[5] + files[move[4]] + str(move[3]+1)
            return alg_move

        # castling
        if move[5].lower() == 'j':
            alg_move = 'O-O-O'
        elif move[5].lower() == 'k':
            alg_move = 'O-O'

        elif move[0] == 'p':
            if move[5].lower() in 'qrbn' and move[5] != '': # promotion
                alg_move = promotion_move(move)

            else: # pawn move
                alg_move = pawn_move(move)

        else: # piece move
            alg_move = piece_move(move)
        
        return alg_move

    def add_to_history(self,move):
        """Adds a move onto the end of self.history."""
        move = self.mine2alg(move)
        
        # add to history, for printing
        if self.FEN.colour:
            self.history.append(str(self.FEN.fullmove) + '. ' + move)
        else:
            if self.history == []:
                self.history = [str(self.FEN.fullmove) + '. ... ' + move]
            else:
                whole_move = self.history[-1] + ' ' + move
                self.history[-1] = whole_move
    
    def adjust_history(self):
        """Check for checks and checkmates, after making the move,
        to add +s and #s to the history."""
        # avoids the need for consider_move in an efficient way
        if self.FEN.is_checkmate():
            self.history[-1] +='#'
        elif self.FEN.is_check():
            self.history[-1] += '+'
    
    def add_to_pos_history(self):
        """Adds a move onto self.pos_history."""
        # add to pos_history, for repetition checking (replace this with a hash)
        rep_string = self.FEN.rep_string()
        rep_hash = hashlib.sha256(rep_string.encode('utf-8')).hexdigest()
        if rep_hash in self.pos_history:
            self.pos_history[rep_hash] += 1
            self.repeats = self.pos_history[rep_hash]
        else:
            self.pos_history[rep_hash] = 1
            self.repeats = 1

    def next_move(self,move):
        """Takes input and moves a piece accordingly..."""
        if type(move) == str:
            move = self.alg2mine(move)
        if move == 'ambiguous':
            print('This could refer to multiple moves. Please try again.')
            return 'ambiguous'
        else:
            moves = self.FEN.moves()
            if move not in moves:
                print('Not a valid move. Please try again.')
                return 'invalid'
            else:
                self.add_to_history(move)
                self.FEN.next_move(move)
                self.adjust_history()
                self.add_to_pos_history()

    def is_draw(self):
        draw = self.FEN.is_draw()
        if draw != '' and draw != '50 move':
            return draw
        elif self.repeats >= 3:
            return '3 rep'
        elif draw != '':
            return draw
        else:
            return ''

    def play(self):
        start_string = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        if self.FEN.FEN_string() == start_string:
            self.FEN.print_board()
        else:
            self.FEN.print_FEN()
        while True:
            if self.FEN.is_checkmate():
                if self.FEN.colour:
                    print("0 - 1")
                    print("Black wins!")
                else:
                    print("1 - 0")
                    print("White wins!")
                break
            draw = self.is_draw()
            if draw != '':
                print("1/2 - 1/2")
                if draw == 'stalemate':
                    print("Stalemate")
                if draw == '3 rep':
                    print("Draw by repetition")
                if draw == '50 move':
                    print("Draw by the fifty-move rule")
                if draw == 'insufficient':
                    print("Draw by insufficient material")
                break
            move = input()
            if move == 'quit':
                break
            if move == '':
                continue
            if move == 'history':
                for i in self.history:
                    print(i)
                continue
            if move == 'FEN':
                self.FEN.print_FEN()
                continue
            if move == 'FEN string':
                print(self.FEN.FEN_string())
                continue
            self.next_move(move)
            self.FEN.print_board()
        print("End position: ", self.FEN.FEN_string())
        print("History: ")
        for i in self.history:
            print(i)

def interpret_FEN_string(string):
    """Take a FEN_string and return the corresponding FEN_position."""
    file_numbers = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7}
    
    string = string.split()
    rows = string[0].split('/')
    position = []
    for string_row in rows[::-1]:
        # process
        row = []
        for x in string_row:
            if x.isalpha():
                row.append(x)
            else:
                row += ['0']*int(x)
        position.append(row)
    position = np.array(position)
    
    colour = {'w':True,'b':False}[string[1]]
    castling_string = string[2]
    corners = {'K':0,'Q':1,'k':2,'q':3}
    castling = [False,False,False,False]
    if castling_string != '-':
        for letter in castling_string:
            castling[corners[letter]] = True
    en_passant = string[3]
    if en_passant != '-':
        file,rank = en_passant
        file = self.file_numbers[file]
        rank = int(rank)-1
        en_passant = (rank,file)
    halfmove = int(string[4])
    fullmove = int(string[5])
    return FEN_position(position,colour,castling,en_passant,halfmove,fullmove)

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
        game(interpret_FEN_string(FEN_string)).play()
    else:
        my_game = game(interpret_FEN_string(FEN_string))
        history = interpret_history(history) # turn a string into a list of moves
        for fullmove in history:
            for move in fullmove:
                check_move = my_game.next_move(move)
                if check_move != None:
                    break
            if check_move != None:
                break
        my_game.play()

def start_with_position_or_history(parameters):
    # accept 5 possibilities: no parameters, just FEN, just history, FEN then history
    # or history then FEN (start from FEN and apply history)
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

start_with_position_or_history(sys.argv[1:])
# to start a game, python chess3.py FEN_string,
# or python chess.py history (not yet tested),
# or similar