# Idea: I'd like to train an AI to track the 'importance' of a position and
# use that to adapt thinking time per move given constrained time for the whole game.
# Probably this is a well-undertstood problem in AI.

#------------------------------------------------------------------------------

import numpy as np
import hashlib

class FEN_position:
    """Stores a position tensor and calculates valid moves."""
    # As in alpha zero implementation: to train neural network, use 4-step history.
    # I don't see why it helps to use 2 repetition planes per position.
    # If it's in history, it's happened once; add a True layer for twice.
    # I guess their point is that memory is cheap here, and it helps
    # the network to learn about the 2nd repetition's importance(?).
    # Also I don't see the need for a fullmove layer.

    def __init__(self,position,en_passant=None,fullmove=None,little_history=None):
        if fullmove == None:
            fullmove=1
        if little_history == None:
            little_history=[]
        
        self.position = position
        self.en_passant = en_passant
        self.fullmove=fullmove
        self.little_history = little_history
        self.move_set = None
        reps = self.rep_string()
        if reps not in self.little_history:
            self.little_history.append(reps)
        
        # Check whether there are
        # kings/rooks on their starting squares in the start position.
        if self.position[0,4,0] != 1:
            self.position[:,:,53:55] = 0
        if self.position[0,0,2] != 1:
            self.position[:,:,54] = 0
        if self.position[0,7,2] != 1:
            self.position[:,:,53] = 0
        if self.position[7,4,6] != 1:
            self.position[:,:,55:57] = 0
        if self.position[7,0,8] != 1:
            self.position[:,:,56] = 0
        if self.position[7,7,8] != 1:
            self.position[:,:,55] = 0
    
    file_numbers = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7}
    file_letters = 'abcdefgh'
    
    def t2s(self,square_tuple):
        '''From a pair of numbers in range(7) to a square name.'''
        rank,file = square_tuple
        file = self.file_letters[file]
        return file+str(rank+1)
        
    
    def s2t(self,square_name):
        '''From a square name to a pair of numbers in range(7).'''
        file,rank = square_name
        file = self.file_numbers[file]
        return (int(rank)-1,file)
    
    def colour(self):
        return int(self.position[0,0,52])
    
    def in_board(self,square):
        '''Returns True if 0 leq file,rank leq 7, False otherwise.'''
        return (0 <= square[0] <= 7 and 0 <= square[1] <= 7)
    
    def slider_moves(self,piece_dim,rank,file,options,protection=False):
        #These next few are subsidiary to piece_moves.
        piece_moves = set()
        colour = (piece_dim<6) # True for white, False for black
        for option in options:
            new_square = (rank,file)
            for i in range(7):
                new_square = (new_square[0]+option[0],new_square[1]+option[1])
                if not self.in_board(new_square):
                    break
                target = self.position[new_square[0],new_square[1],:12]
                target = np.nonzero(target)[0]
                if target.size == 0:
                    piece_moves.add((piece_dim,rank,file,new_square[0],new_square[1],''))
                elif ((target[0]<6) != colour) or protection:
                    piece_moves.add((piece_dim,rank,file,new_square[0],new_square[1],'x'))
                    break
                else:
                    break
        return piece_moves
    
    sliders = {'r':[(1,0),(0,1),(-1,0),(0,-1)],
    'b': [(i,j) for j in [-1,1] for i in [-1,1]],
    'q':[(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]}
    
    def hopper_moves(self,piece_dim,rank,file,options,protection=False):
        piece_moves = set()
        colour = (piece_dim<6)
        for option in options:
            new_square = (rank+option[0],file+option[1])
            if not self.in_board(new_square):
                continue
            target = self.position[new_square[0],new_square[1],:12]
            target = np.nonzero(target)[0]
            if target.size == 0:
                piece_moves.add((piece_dim,rank,file,new_square[0],new_square[1],''))
            elif ((target[0]<6) != colour) or protection:
                piece_moves.add((piece_dim,rank,file,new_square[0],new_square[1],'x'))
        return piece_moves
    
    hoppers = {'n':[(1,2),(2,1),(-1,2),(2,-1),(-2,1),(1,-2),(-1,-2),(-2,-1)],
    'k':[(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]}
    
    pieces = 'kqrbnpkqrbnp'
    
    def pawn_moves(self,piece_dim,rank,file,protection=False):
        piece_moves = set()
        colour = int(piece_dim<6)
        piece_names = 'qrbn'
        directions = [-1,1]
        start_rank = [6,1]
        to_promote_rank = [1,6]
        new_square = (rank+directions[colour],file)
        target = self.position[new_square[0],new_square[1],:12]
        if (target == 0).all() and not protection:
            if rank != to_promote_rank[colour]:
                piece_moves.add((piece_dim,rank,file,new_square[0],new_square[1],''))
            else:
                for piece_name in piece_names:
                    piece_moves.add((piece_dim,rank,file,new_square[0],new_square[1],piece_name))
            if rank == start_rank[colour]:
                new_square = (rank+2*directions[colour],file)
                target = self.position[new_square[0],new_square[1],:12]
                if (target == 0).all():
                    piece_moves.add((piece_dim,rank,file,new_square[0],new_square[1],''))
                    
        options = [-1,1] # for capturing
        for option in options:
            new_square = (rank+directions[colour],file+option)
            if not self.in_board(new_square):
                continue
            target = self.position[new_square[0],new_square[1],:12]
            target = np.nonzero(target)[0]
            if (new_square == self.en_passant) or protection:
                piece_moves.add((piece_dim,rank,file,new_square[0],new_square[1],'e'))
            elif target.size==0:
                continue
            elif (target[0]<6) != colour:
                if rank != to_promote_rank[colour]:
                    piece_moves.add((piece_dim,rank,file,new_square[0],new_square[1],'x'))
                else:
                    for piece_name in piece_names.upper():
                        piece_moves.add((piece_dim,rank,file,new_square[0],new_square[1],piece_name))
        return piece_moves
    

    def piece_moves(self,piece_dim,rank,file):
        """Gives a list of valid moves for the piece in that square,
        not worrying (for now) about whether the move results in check."""
        #A move is stored as: piece, starting rank and file, ending rank and
        #file, and a string '' or 'x' for captures. En passant has 'e' for the
        #string; castling lists the king squares with string 'K','J','k' or 'j'.
        #Prawn promotions have string 'q','r','b' or 'n'; capitalised for
        #promotions with captures.
        if piece_dim in [5,11]: # pawn
            return self.pawn_moves(piece_dim,rank,file)
        if piece_dim in [1,2,3,7,8,9]:
            return self.slider_moves(piece_dim,rank,file,self.sliders[self.pieces[piece_dim]])
        if piece_dim in [0,4,6,10]:
            return self.hopper_moves(piece_dim,rank,file,self.hoppers[self.pieces[piece_dim]])

    def protected_squares(self,piece_dim,rank,file):
        """Squares protected by the piece in (rank,file).
        That is, breaks after adding a square with a piece of the same colour,
        not before. Does not include forward pawn moves/castles, only attacks."""
        if piece_dim in [5,11]:
            moves = self.pawn_moves(piece_dim,rank,file,True)
        if piece_dim in [1,2,3,7,8,9]:
            moves = self.slider_moves(piece_dim,rank,file,self.sliders[self.pieces[piece_dim]],True)
        if piece_dim in [0,4,6,10]:
            moves = self.hopper_moves(piece_dim,rank,file,self.hoppers[self.pieces[piece_dim]],True)
        return set(tuple(move[3:5]) for move in moves)
    
    def all_protected_squares(self,colour):
        # Squares protected by colour, i.e. where not-colour's king can't go.
        squares = set()
        piece_list = self.position[:,:,:12]
        piece_list = np.nonzero(piece_list)
        for i in range(len(piece_list[0])):
            if (piece_list[2][i]<6) == colour:
                squares.update(self.protected_squares(piece_list[2][i],
                piece_list[0][i],piece_list[1][i]))
        return squares
    
    def can_castle(self,corner):
        # The corners are 0,1,2,3 corresponding to white king/queenside, black king/queenside.
        if not self.position[0,0,53+corner]:
            return False
        # otherwise, we know king and rook are on the right squares.
        by_who_colour = [False,False,True,True][corner]
        king_rank = [0,0,7,7][corner]
        to_check = [[(0,5),(0,6)],[(0,2),(0,3)],[(7,5),(7,6)],[(7,2),(7,3)]][corner]
        squares = self.all_protected_squares(by_who_colour)
        if (king_rank,4) in squares:
            return False
        for square in to_check:
            if ((self.position[square[0],square[1],:12] != 0).any()
            or square in squares):
                return False
        to_file = [6,2,6,2][corner]
        piece_dim = [0,0,6,6][corner]
        castle_str = 'KJkj'[corner]
        return (piece_dim,king_rank,4,king_rank,to_file,castle_str)
    
    def king_axis(self,piece_dim,rank,file,options,colour):
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
                target = self.position[new_square[0],new_square[1],:12]
                target = np.nonzero(target)[0]
                if target.size == 0:
                    # empty square
                    axis.append(new_square)
                elif (target[0]<6) == colour:
                    # own colour piece
                    break
                elif target[0]%6 == 0:
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
        return sum((self.position[square[0],square[1],:12] != 0).any() for square in axis[1:])
        # to not include the slider at the start of the axis
    
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
        piece_list = self.position[:,:,:12]
        piece_list = np.nonzero(piece_list)
        king_dim = 0
        if colour:
            king_dim += 6
        for i in range(len(piece_list[0])):
            piece_dim = piece_list[2][i]
            if piece_dim in [1,2,3,7,8,9] and (piece_dim<6)==colour: # slider
                king_axis = self.king_axis(piece_dim,piece_list[0][i],piece_list[1][i],
                self.sliders[self.pieces[piece_dim]],colour)
                if king_axis == None:
                    continue
                count = self.non_empty_count(king_axis)
                if count > 1:
                    continue
                elif count == 1:
                    one_king_axes.append(king_axis)
                else:
                    none_king_axes.append(king_axis)
            elif piece_dim in [4,5,10,11] and (piece_dim<6)==colour: # knight/pawn
                squares = self.protected_squares(piece_dim,piece_list[0][i],piece_list[1][i])
                for square in squares:
                    if self.position[square[0],square[1],king_dim] == 1:
                        hopper_checks.append((piece_list[0][i],piece_list[1][i]))
        
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
        
        # if already calculated, I don't need to do it again.
        if self.move_set != None:
            return self.move_set
        
        # I consider two cases. In either case, the king can move to precisely
        # the non-attacked squares. If not check, calculate pinned pieces and
        # and stop them moving. (Except colinear.)
        # If check, look for attacking pieces and allow block
        # or take. (Blocking only for sliders.)
        # In either case, check all sliders for king_axes and knights/pawns for
        # king attacks.
        
        # squares the king must avoid
        attacked_squares = self.all_protected_squares(not self.colour())
        
        # checks and pinned pieces
        checker_squares, one_king_axes = self.axes_and_attacks(not self.colour())
        
        move_set = set()
        piece_list = self.position[:,:,:12]
        piece_list = np.nonzero(piece_list)
        for i in range(len(piece_list[0])):
            piece_dim = piece_list[2][i]
            rank = piece_list[0][i]
            file = piece_list[1][i]
            if piece_dim %6 != 0 and (piece_dim<6)==self.colour(): # not kings
                moves = self.piece_moves(piece_dim,rank,file)
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
            if piece_dim %6 == 0 and (piece_dim<6)==self.colour():
                # allowed king moves
                for move in self.piece_moves(piece_dim,rank,file):
                    if tuple(move[3:5]) not in attacked_squares:
                        move_set.add(move)
        
        #add castling moves
        corners_to_check = [[2,3],[0,1]]
        for corner in corners_to_check[self.colour()]:
            result = self.can_castle(corner)
            if result != False:
                move_set.add(result)
        
        self.move_set = move_set
        return move_set
    
    def next_move(self,move):
        """Changes self to what the position will be after a move."""
        # Move stack of positions back one
        self.position[:,:,13:52] = self.position[:,:,:39]
        
        # Create new position in the first 13 planes
        self.position[move[1],move[2],move[0]] = 0
        
        piece_dim = move[0]
        piece_dims = {'q':1,'r':2,'b':3,'n':4}
        special = move[5]
        if special.lower() in ['q','r','b','n']: # promotion
            piece_dim = piece_dims[special.lower()]
            if not self.colour():
                piece_dim += 6
        self.position[move[3],move[4],:12] = 0 # for captures
        self.position[move[3],move[4],piece_dim] = 1
        if special == 'e': # en passant
            self.position[move[1],move[4],(piece_dim+6)%12] = 0
        if special == 'J':
            self.position[0,0,2] = 0
            self.position[0,3,2] = 1
        if special == 'K':
            self.position[0,7,2] = 0
            self.position[0,5,2] = 1
        if special == 'j':
            self.position[7,0,8] = 0
            self.position[7,3,8] = 1
        if special == 'k':
            self.position[7,7,8] = 0
            self.position[7,5,8] = 1
        
        # castling rights
        # King or rook moving:
        if move[:3] == (0,0,4):
            self.position[:,:,53:55] = 0
        elif move[:3] == (2,0,0):
            self.position[:,:,54] = 0
        elif move[:3] == (2,0,7):
            self.position[:,:,53] = 0
        elif move[:3] == (6,7,4):
            self.position[:,:,55:57] = 0
        elif move[:3] == (8,7,0):
            self.position[:,:,56] = 0
        elif move[:3] == (8,7,7):
            self.position[:,:,55] = 0
        # rook getting taken
        if move[3:5] == (0,0):
            self.position[:,:,54] = 0
        elif move[3:5] == (0,7):
            self.position[:,:,53] = 0
        elif move[3:5] == (7,0):
            self.position[:,:,56] = 0
        elif move[3:5] == (7,7):
            self.position[:,:,55] = 0
        
        # en passant
        if move[0] == 5 and move[1] == 1 and move[3] == 3:
            self.en_passant = (2,move[2])
        elif move[0] == 11 and move[1] == 6 and move[3] == 4:
            self.en_passant = (5,move[2])
        else:
            self.en_passant = None
        
        # halfmove
        if move[5] == 'x' or move[0]%6 == 5:
            self.position[:,:,57] = np.zeros((8,8))
        else:
            self.position[:,:,57] += 1/100
        
        # fullmove
        if not self.colour():
            self.fullmove += 1
        
        # change colour
        self.position[:,:,52] = (self.position[0,0,52] +1)%2
        
        # little history, for tracking repetitions for self.position[13,:,:], etc.
        reps = self.rep_string()
        if reps in self.little_history:
            self.position[:,:,13] = 1
        self.little_history.append(reps)
        if len(self.little_history) > 4:
            self.little_history = self.little_history[1:]
        
        self.move_set = None
    
    def encode_move(self,move):
        # take a move and return its coordinates in the (8,8,73) array
        # used for the AI
        under_prom = ['n','b','r']
        direction = (move[3]-move[1],move[4]-move[2])
        if move[5] in under_prom: # under-promotions
            direction = direction[1]+1 # shift to range(3)
            move_type = direction*3+under_prom.index(move[5])+64
        elif move[0] in [4,10]: # knights
            knight_moves = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
            arrow = knight_moves.index(direction)
            move_type = arrow+56
        else: # other pieces
            if direction[1]==0 and direction[0]>0: # north
                arrow=0
                length=direction[0]
            elif direction[0]==direction[1] and direction[0]>0: # north-east, etc.
                arrow=1
                length=direction[0]
            elif direction[0]==0 and direction[1]>0:
                arrow=2
                length=direction[1]
            elif direction[0]==-direction[1] and direction[0]<0:
                arrow=3
                length=direction[1]
            elif direction[1]==0 and direction[0]<0:
                arrow=4
                length=-direction[0]
            elif direction[0]==direction[1] and direction[0]<0:
                arrow=5
                length=-direction[0]
            elif direction[0]==0 and direction[1]<0:
                arrow=6
                length=-direction[1]
            elif direction[0]==-direction[1] and direction[0]>0:
                arrow=7
                length=direction[0]
            move_type = arrow*7+length
        
        return (move[1],move[2],move_type)
    
    def decode_net_move(self,net_move):
        # take coordinates in an array of shape (8,8,73) and return the move
        move = [None,net_move[0],net_move[1],None,None,'']
        target = self.position[move[1],move[2],:12]
        move[0] = np.nonzero(target)[0][0]
        if net_move[2] < 56: # queen-type moves
            directions = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
            arrow = net_move[2]//7
            arrow = directions[arrow]
            length = net_move[2]%7
            move[3],move[4] = move[1]+length*arrow[0],move[2]+length*arrow[1]
        elif net_move[2] < 64: # knight moves
            knight_moves = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
            arrow = knight_moves[net_move[2]-56]
            move[3],move[4] = move[1]+arrow[0],move[2]+arrow[1]
        else: # under-promotions
            under_prom = ['n','b','r']
            move_type = net_move[2]-64
            direction = move_type//3-1 # shift back to -1,0,1
            prom_type = under_prom[move_type%3]
            move[3],move[4] = move[1]+1,move[2]+direction
            if direction:
                move[5] = prom_type.upper()
            else:
                move[5] = prom_type
            return move
        
        # calculate move[5] type for 'x','q','Q','e' without using self.moves()
        # also castling
        castle_moves = [(0,2),(0,-2),(6,2),(6,-2)]
        if (move[0],move[4]-move[2]) in castle_moves:
            move[5] = 'KJkj'[castle_moves.index((move[0],move[4]-move[2]))]
        elif move[0] in [5,11] and move[3] in [0,7]: # promotion
            if move[4] == move[2]: # without capture
                move[5] = 'q'
            else: # with capture
                move[5] = 'Q'
        elif (move[0] in [5,11]) and (move[3],move[4]) == self.en_passant: # en passant
            move[5] = 'e'
        else:
            target = self.position[move[3],move[4],:12]
            if target.any():
                move[5] = 'x'
        return move
        

    def insufficient(self):
        piece_types = [0,0,0]
        # for number of knights, light-square bishops, dark-square bishops.
        # If pieces[0] > 1, or more than one is non-zero, no draw.
        # See https://www.reddit.com/r/chess/comments/se89db/a_writeup_on_definitions_of_insufficient_material/
        # this is not exhaustive, but close enough for most purposes.
        # In a later implementation, calculate insufficient material separately
        # for each colour; this is needed for draw by timeout vs insufficient material.
        piece_list = self.position[:,:,:12]
        piece_list = np.nonzero(piece_list)
        for i in range(len(piece_list[0])):
            piece_dim = piece_list[2][i]
            if piece_dim in [1,2,5,7,8,11]: # queens, rooks, pawns
                return False
            if piece_dim in [4,10]: # knight
                piece_types[0] += 1
            elif piece_dim in [3,9]: # bishop
                if piece_list[0][i]%2 == piece_list[1][i]%2:
                    piece_types[1] += 1
                else:
                    piece_types[2] += 1
        
        if piece_types[0] > 1:
            return False
        elif piece_types[0] == 1:
            if piece_types[1] == 0 and piece_types[2] == 0:
                return True
            else:
                return False
        else:
            if piece_types[1] == 0 or piece_types[2] == 0:
                return True
            else:
                return False
    
    def is_check(self):
        # that is, if self.colour is in check
        squares = self.all_protected_squares(not self.colour())
        king_dim = [6,0][self.colour()]
        for square in squares:
            if self.position[square[0],square[1],king_dim] == 1:
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
        elif self.position[0,0,57] >= 1:
            return '50 move'
        else:
            return ''
    
    def print_board(self):
        piece_types = 'KQRBNPkqrbnp'
        for rank in range(7,-1,-1):
            row = ['0']*8
            for file in range(8):
                target = self.position[rank,file,:12]
                target = np.nonzero(target)[0]
                if target.size != 0:
                    row[file] = piece_types[target[0]]
            print(row)
    
    def rep_string(self):
        # to check for repetitions
        piece_types = 'KQRBNPkqrbnp'
        string = ''
        
        # position
        for i in range(7,-1,-1):
            row = self.position[i,:,:12]
            empty_count = 0
            for j in range(8):
                target = np.nonzero(row[j,:])[0]
                if target.size == 0: # count empties
                    empty_count += 1
                else:
                    if empty_count > 0:
                        string += str(empty_count)
                        empty_count = 0
                    string += piece_types[target[0]]
            if empty_count > 0:
                string += str(empty_count)
            if i>0:
                string += '/'
            else:
                string += ' '
        
        string += {True:'w',False:'b'}[self.position[0,0,52]] + ' ' # active
        if sum(self.position[0,0,53:57]) == 0: # castling
            string += '- '
        else:
            string += ''.join('KQkq'[i] for i in range(4) if self.position[0,0,53+i]) + ' '
        if self.en_passant == None: # en passant
            string += '-'
        else:
            string += self.t2s(self.en_passant)
        
        return string
    
    def FEN_string(self):
        string = self.rep_string()
        string += ' ' + str(int(100*self.position[0,0,57])) + ' ' + str(self.fullmove)
        return string
    
    def print_FEN(self):
        self.print_board()
        print("Active colour: ", {True:'w',False:'b'}[self.position[0,0,52]])
        print("Castling rights: ", ''.join('KQkq'[i] for i in range(4) if self.position[0,0,53+i]))
        if self.en_passant != None:
            print("En passant target square: ", self.t2s(self.en_passant))
        print("Halfmove clock: ", int(100*self.position[0,0,57]))
        print("Fullmove number: ", self.fullmove)

# print(FEN_position().moves())

class game:
    
    def __init__(self,FEN=None,history=None,pos_history=None):
        if history == None:
            history = []
        if pos_history == None:
            pos_history = {}
        if FEN == None:
            start_pos = np.zeros((8,8,13*4+6)) # the first 12 planes are for: KQRBNPkqrbnp
            piece_pos = [(0,4,0),(0,3,1),(0,0,2),(0,7,2),(0,2,3),(0,5,3),(0,1,4),(0,6,4),
            (7,4,6),(7,3,7),(7,0,8),(7,7,8),(7,2,9),(7,5,9),(7,1,10),(7,6,10)]
            for pos in piece_pos:
                start_pos[pos] = 1
            start_pos[1,:,5] += 1
            start_pos[6,:,11] += 1
            start_pos[:,:,52:57] += 1 # colour, castling*4, halfmove
            FEN = FEN_position(start_pos)
        
        self.FEN = FEN
        self.history = history
        self.pos_history = pos_history
        self.result = None
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
            colour = self.FEN.colour()*6
            nice_move[0] = 11-colour
            if colour:
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
            colour = self.FEN.colour()*6
            piece_dim = 'KQRBN'.index(move[0])+6-colour
            nice_move[0] = piece_dim
            nice_move[3] = move[-1]
            nice_move[4] = files[move[-2]]
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
            nice_move = [None]*6
            colour = self.FEN.colour()*6
            nice_move[0] = 11-colour
            nice_move[2] = files[move[0]]
            nice_move[3] = move[-1]
            nice_move[4] = files[move[-2]]
            if 'x' in move:
                if colour:
                    nice_move[1] = move[-1]-1
                else:
                    nice_move[1] = move[-1]+1
                if (self.FEN.position[nice_move[3],nice_move[4],:12] == 0).all():
                    nice_move[5] = 'e' # en passant
                else:
                    nice_move[5] = 'x'
            else:
                nice_move[5] = ''
                # differentiate between 1-square and 2-square moves
                if colour:
                    if self.FEN.position[nice_move[3]-1,nice_move[4],nice_move[0]] == 0:
                        nice_move[1] = move[-1]-2
                    else:
                        nice_move[1] = move[-1]-1
                else:
                    if self.FEN.position[nice_move[3]+1,nice_move[4],nice_move[0]] == 0:
                        nice_move[1] = move[-1]+2
                    else:
                        nice_move[1] = move[-1]+1
            return tuple(nice_move)

        # castling
        if move == '0-0' or move == 'O-O':
            if self.FEN.colour():
                return (0,0,4,0,6,'K')
            else:
                return (6,7,4,7,6,'k')
        if move == '0-0-0' or move == 'O-O-O':
            if self.FEN.colour():
                return (0,0,4,0,2,'J')
            else:
                return (6,7,4,7,2,'j')
        
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
        pieces = 'KQRBNPKQRBNP'
        
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
            alg_move = pieces[move[0]]
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

        elif move[0]%6 == 5:
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
        if self.FEN.colour():
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
                if self.repeats == 2: # give repeat info to the neural network
                    self.FEN.position[:,:,12] = 1

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
                if self.FEN.colour():
                    self.result = '0-1'
                    print(self.result)
                    print("Black wins!")
                else:
                    self.result = '1-0'
                    print(self.result)
                    print("White wins!")
                break
            draw = self.is_draw()
            if draw != '':
                self.result = '1/2-1/2'
                print(self.result)
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
            # print(list(self.FEN.moves()))
            # print(list(self.mine2alg(move) for move in self.FEN.moves()))
        print("End position: ", self.FEN.FEN_string())
        print("History: ")
        for i in self.history:
            print(i)

if __name__ == '__main__':
    game().play()