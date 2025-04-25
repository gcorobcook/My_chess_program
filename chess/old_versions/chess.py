# Work out how to do a graphical representation and make moves by dragging.
# I'd like to have: a picture of the board, the game history, draw offer/
# acceptance and resignation buttons.
# Chess clocks. Increment?
# Very basic chess engine?

#------------------------------------------------------------------------------

import copy

class FEN_position(object):
	"""More or less a position in FEN."""

	start_pos = [
		['r','n','b','q','k','b','n','r'],
		['p','p','p','p','p','p','p','p'],
		['0','0','0','0','0','0','0','0'],
		['0','0','0','0','0','0','0','0'],
		['0','0','0','0','0','0','0','0'],
		['0','0','0','0','0','0','0','0'],
		['P','P','P','P','P','P','P','P'],
		['R','N','B','Q','K','B','N','R']]

	def __init__(self,position=start_pos,colour=True,
		castling=[True,True,True,True],en_passant='-',halfmove=0,fullmove=1):
		self.position = position
		# self.new_position = copy.deepcopy(position)
		self.colour = colour
		self.castling = castling
		self.en_passant = en_passant
		self.halfmove = halfmove
		self.fullmove = fullmove
		#True is white, False is black, for convenience.
		#Castling rights doesn't pay attention to whether you can actually
		#castle; just whether your king/rook have moved yet. Like in FEN.
		#The Trues refer to white kingside, white queenside,... castling.

	def rook_moves(self,file,rank):
		#These next few are subsidiary to piece_moves.
		piece_moves = set()
		options = [(1,0),(0,1),(-1,0),(0,-1)]
		for option in options:
			new_square = (file,rank)
			for i in range(7):
				new_square = (new_square[0]+option[0],new_square[1]+option[1])
				if not (1 <= new_square[0] <= 8 and 1 <= new_square[1] <= 8):
					break
				target = self.position[8-new_square[1]][new_square[0]-1]
				if target == '0':
					piece_moves.add(('r',file,rank,new_square[0],new_square[1],''))
				elif target.isupper() == self.colour:
					break
				else:
					piece_moves.add(('r',file,rank,new_square[0],new_square[1],'x'))
					break
		return piece_moves

	def knight_moves(self,file,rank):
		piece_moves = set()
		options = [(1,2),(2,1),(-1,2),(2,-1),(-2,1),(1,-2),(-1,-2),(-2,-1)]
		for option in options:
			new_square = (file+option[0],rank+option[1])
			if not (1 <= new_square[0] <= 8 and 1 <= new_square[1] <= 8):
				continue
			target = self.position[8-new_square[1]][new_square[0]-1]
			if target == '0':
				piece_moves.add(('n',file,rank,new_square[0],new_square[1],''))
			elif target.isupper() != self.colour:
				piece_moves.add(('n',file,rank,new_square[0],new_square[1],'x'))
		return piece_moves

	def bishop_moves(self,file,rank):
		piece_moves = set()
		options = [(1,1),(-1,1),(1,-1),(-1,-1)]
		for option in options:
			new_square = (file,rank)
			for i in range(7):
				new_square = (new_square[0]+option[0],new_square[1]+option[1])
				if not (1 <= new_square[0] <= 8 and 1 <= new_square[1] <= 8):
					break
				target = self.position[8-new_square[1]][new_square[0]-1]
				if target == '0':
					piece_moves.add(('b',file,rank,new_square[0],new_square[1],''))
				elif target.isupper() == self.colour:
					break
				else:
					piece_moves.add(('b',file,rank,new_square[0],new_square[1],'x'))
					break
		return piece_moves

	def queen_moves(self,file,rank):
		piece_moves = set()
		options = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]
		for option in options:
			new_square = (file,rank)
			for i in range(7):
				new_square = (new_square[0]+option[0],new_square[1]+option[1])
				if not (1 <= new_square[0] <= 8 and 1 <= new_square[1] <= 8):
					break
				target = self.position[8-new_square[1]][new_square[0]-1]
				if target == '0':
					piece_moves.add(('q',file,rank,new_square[0],new_square[1],''))
				elif target.isupper() == self.colour:
					break
				else:
					piece_moves.add(('q',file,rank,new_square[0],new_square[1],'x'))
					break
		return piece_moves

	def king_moves(self,file,rank):
		piece_moves = set()
		options = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
		for option in options:
			new_square = (file+option[0],rank+option[1])
			if not (1 <= new_square[0] <= 8 and 1 <= new_square[1] <= 8):
				continue
			target = self.position[8-new_square[1]][new_square[0]-1]
			if target == '0':
				piece_moves.add(('k',file,rank,new_square[0],new_square[1],''))
			elif target.isupper() != self.colour:
				piece_moves.add(('k',file,rank,new_square[0],new_square[1],'x'))
		return piece_moves

	def white_pawn_moves(self,file,rank):
		piece_moves = set()
		if rank == 7:
			options = [(0,1,'q'),(0,1,'r'),(0,1,'b'),(0,1,'n')]
		else:
			options = [(0,1)]
		for option in options:
			new_square = (file+option[0],rank+option[1])
			target = self.position[8-new_square[1]][new_square[0]-1]
			if target == '0' and rank != 7:
				piece_moves.add(('p',file,rank,new_square[0],new_square[1],''))
				if rank == 2:
					target2 = self.position[4][new_square[0]-1]
					if target2 == '0':
						piece_moves.add(('p',file,rank,new_square[0],4,''))
			elif target == '0':
				piece_moves.add(('p',file,rank,new_square[0],new_square[1],option[2]))
		if rank == 7:
			options2 = [(1,1,'q'),(1,1,'r'),(1,1,'b'),(1,1,'n'),
				(-1,1,'q'),(-1,1,'r'),(-1,1,'b'),(-1,1,'n')]
		else:
			options2 = [(1,1),(-1,1)]
		for option in options2:
			new_square = (file+option[0],rank+option[1])
			if not (1 <= new_square[0] <= 8):
				continue
			target = self.position[8-new_square[1]][new_square[0]-1]
			if new_square == self.en_passant:
				piece_moves.add(('p',file,rank,new_square[0],new_square[1],'e'))
			if target == '0':
				continue
			if target.isupper() != self.colour and rank != 7:
				piece_moves.add(('p',file,rank,new_square[0],new_square[1],'x'))
			elif target.isupper() != self.colour:
				piece_moves.add(('p',file,rank,new_square[0],new_square[1],option[2].upper()))
		return piece_moves

	def black_pawn_moves(self,file,rank):
		piece_moves = set()
		if rank == 2:
			options = [(0,-1,'q'),(0,-1,'r'),(0,-1,'b'),(0,-1,'n')]
		else:
			options = [(0,-1)]
		for option in options:
			new_square = (file+option[0],rank+option[1])
			target = self.position[8-new_square[1]][new_square[0]-1]
			if target == '0' and rank != 2:
				piece_moves.add(('p',file,rank,new_square[0],new_square[1],''))
				if rank == 7:
					target2 = self.position[3][new_square[0]-1]
					if target2 == '0':
						piece_moves.add(('p',file,rank,new_square[0],5,''))
			elif target == '0':
				piece_moves.add(('p',file,rank,new_square[0],new_square[1],option[2]))
		if rank == 2:
			options2 = [(1,-1,'q'),(1,-1,'r'),(1,-1,'b'),(1,-1,'n'),
				(-1,-1,'q'),(-1,-1,'r'),(-1,-1,'b'),(-1,-1,'n')]
		else:
			options2 = [(1,-1),(-1,-1)]
		for option in options2:
			new_square = (file+option[0],rank+option[1])
			if not (1 <= new_square[0] <= 8):
				continue
			target = self.position[8-new_square[1]][new_square[0]-1]
			if new_square == self.en_passant:
				piece_moves.add(('p',file,rank,new_square[0],new_square[1],'e'))
			if target == '0':
				continue
			if target.isupper() != self.colour and rank != 2:
				piece_moves.add(('p',file,rank,new_square[0],new_square[1],'x'))
			elif target.isupper() != self.colour:
				piece_moves.add(('p',file,rank,new_square[0],new_square[1],option[2].upper()))
		return piece_moves

	def piece_moves(self,file,rank):
		"""Gives a list of valid moves for the piece in that square,
		not worrying (for now) about whether the move results in check."""
		#A move is stored as: piece, starting file and rank, ending file and
		#rank, and a string '' or 'x' for captures. En passant has 'e' for the
		#string; castling lists the king squares with string 'K','J','k' or 'j'.
		#Prawn promotions have string 'q','r','b' or 'n'; capitalised for
		#promotions with captures.
		#Files are listed as 1 to 8.
		piece = self.position[8-rank][file-1]
		piece_moves = set()
		if piece.lower() == 'r':
			piece_moves.update(self.rook_moves(file,rank))
		elif piece.lower() == 'n':
			piece_moves.update(self.knight_moves(file,rank))
		elif piece.lower() == 'b':
			piece_moves.update(self.bishop_moves(file,rank))
		elif piece.lower() == 'q':
			piece_moves.update(self.queen_moves(file,rank))
		elif piece.lower() == 'k':
			piece_moves.update(self.king_moves(file,rank))
		elif piece == 'P':
			piece_moves.update(self.white_pawn_moves(file,rank))
		elif piece == 'p':
			piece_moves.update(self.black_pawn_moves(file,rank))
		return piece_moves

	def consider_move(self,move):
		"""Returns what the position will be after a move."""
		new_position = copy.deepcopy(self.position)
		new_position[8-move[2]][move[1]-1] = '0'
		piece = move[0]
		if move[5].lower() in 'qrbn' and move[5] != '':
			piece = move[5]
		if self.colour:
			new_position[8-move[4]][move[3]-1] = piece.upper()
		else:
			new_position[8-move[4]][move[3]-1] = piece.lower()
		if move[5] == 'e':
			if self.colour:
				new_position[3][move[3]-1] = '0'
			else:
				new_position[4][move[3]-1] = '0'
		if move[5] == 'J':
			new_position[7][0] = '0'
			new_position[7][3] = 'R'
		if move[5] == 'K':
			new_position[7][7] = '0'
			new_position[7][5] = 'R'
		if move[5] == 'j':
			new_position[0][0] = '0'
			new_position[0][3] = 'r'
		if move[5] == 'k':
			new_position[0][7] = '0'
			new_position[0][5] = 'r'
		return new_position

	def potensh_moves(self):
		move_set = set()
		for rank in range(1,9):
			for file in range(1,9):
				square = self.position[8-rank][file-1]
				if square.isalpha() and square.isupper() == self.colour:
					move_set.update(self.piece_moves(file,rank))
		return move_set

	def square_attacked(self,square,by_who):
		#I.e. will a move leave us in check, and can we castle
		#Here this means, is square attacked by by_who
		new_position = FEN_position(self.position,by_who)
		move_set = new_position.potensh_moves()
		for move in move_set:
			if (move[3],move[4]) == square:
				return True
		return False

	def is_check(self,by_who):
		#I.e. is the other player in check by by_who.
		found = False
		for rank in range(1,9):
			for file in range(1,9):
				square = self.position[8-rank][file-1]
				if square.lower() == 'k' and not (square.isupper() == by_who):
					king_square = (file,rank)
					found = True
					break
			if found == True:
				break
		if self.square_attacked(king_square,by_who):
			return True
		else:
			return False

	def moves(self):
		"""Gives a list of valid moves."""
		move_set = self.potensh_moves()
		if self.colour:
			if self.castling[0]:
				if self.position[7][5] == '0' and not self.square_attacked((5,1),False):
					if self.position[7][6] == '0' and not self.square_attacked((6,1),False):
						move_set.add(('k',5,1,7,1,'K'))
			if self.castling[1]:
				if self.position[7][3] == '0' and not self.square_attacked((5,1),False):
					if self.position[7][2] == '0' and not self.square_attacked((4,1),False):
						if self.position[7][1] == '0':
							move_set.add(('k',5,1,3,1,'J'))
		else:
			if self.castling[2]:
				if self.position[0][5] == '0' and not self.square_attacked((5,8),True):
					if self.position[0][6] == '0' and not self.square_attacked((6,8),True):
						move_set.add(('k',5,8,7,8,'k'))
			if self.castling[3]:
				if self.position[0][3] == '0' and not self.square_attacked((5,8),True):
					if self.position[0][2] == '0' and not self.square_attacked((4,8),True):
						if self.position[0][1] == '0':
							move_set.add(('k',5,8,3,8,'j'))
		#Check for moves that leave us in check.
		valid = set()
		for move in move_set:
			new_position = self.consider_move(move)
			new_FEN = FEN_position(new_position)
			if not new_FEN.is_check(not self.colour):
				valid.add(move)
		return valid

	def insufficient(self):
		pieces = [0,0,0]
		# for number of knights, light-square bishops, dark-square bishops
		# (mod 2). If pieces[0] > 1, or more than one is non-zero, no draw.
		# See http://www.e4ec.org/immr.html: this is not exhaustive, but
		# close enough for most purposes.
		for i in range(8):
			for j in range(8):
				piece = self.position[i][j].lower()
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

	def is_checkmate(self):
		if self.is_check(not self.colour) and len(self.moves()) == 0:
			return True
		else:
			return False

	def is_draw(self):
		if len(self.moves()) == 0 and not self.is_check(not self.colour):
			return 'stalemate'
		elif self.insufficient():
			return 'insufficient'
		elif self.halfmove == 100:
			return '50 move'
		else:
			return ''
		# come back to draw offers. Add an option to claim a draw after 3/50,
		# and do it automatically after 5/75.

# print(FEN_position().moves())

class game(object):

	def __init__(self,FEN=FEN_position(),history=[],pos_history=None):
		self.FEN = FEN
		self.history = history
		if pos_history == None:
			self.pos_history = [[[self.FEN.position,self.FEN.colour,
			self.FEN.castling,False],1]]
		else:
			self.pos_history = pos_history
		# the False is extra en passant moves: if there is a self.FEN.en_passant
		# pawn, this records that there are pawns there to take advantage.
		self.repeats = self.pos_history[-1][1]
		# the number of times the current position has been repeated
		self.result = '*'

	def alg2mine(self,move):
		"""Splits a standard chess notation move into my format."""
		# Check validity afterwards.

		files = {'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8}
		
		def process(move):
			short_move = []
			for i in move:
				if i in 'RNBQKabcdefghx':
					short_move.append(i)
				if i.isdigit():
					short_move.append(int(i))
			return short_move

		if move == '0-0' or move == 'O-O':
			if self.FEN.colour:
				return ('k',5,1,7,1,'K')
			else:
				return ('k',5,8,7,8,'k')
		if move == '0-0-0' or move == 'O-O-O':
			if self.FEN.colour:
				return ('k',5,1,3,1,'J')
			else:
				return ('k',5,8,3,8,'j')
		else:
			move = process(move)
			nice_move = [None,None,None,None,None,None]
			if str(move[-1]) in 'RNBQ' and move[-2] != '=':
				move = move[:-1] + ['=',move[-1]]
			if move[-2] == '=': # promotion
				nice_move[0] = 'p'
				nice_move[1] = files[move[0]]
				if self.FEN.colour:
					nice_move[2] = 7
				else:
					nice_move[2] = 2
				if move[1] == 'x':
					nice_move[5] = move[-1]
					nice_move[3] = files[move[2]]
					nice_move[4] = move[3]
				else:
					nice_move[5] = move[-1].lower()
					nice_move[3] = files[move[0]]
					nice_move[4] = move[1]
				return tuple(nice_move)

			nice_move[3] = files[move[-2]]
			nice_move[4] = move[-1]

			if move[0] in 'RNBQK': # piece move
				nice_move[0] = move[0].lower()
				if 'x' in move:
					nice_move[5] = 'x'
					move = move[1:-3]
				else:
					nice_move[5] = ''
					move = move[1:-2]
				if len(move) == 2:
					nice_move[1] = files[move[0]]
					nice_move[2] = move[1]
				elif len(move) ==1:
					if move[0].isalpha():
						nice_move[1] = files[move[0]]
					else:
						nice_move[2] = move[0]
				#search for valid moves...
				intention = []
				valid_moves = self.FEN.moves()
				for valid_move in valid_moves:
					for i in range(6):
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

			else: # pawn move
				nice_move[0] = 'p'
				nice_move[1] = files[move[0]]
				if 'x' in move:
					if self.FEN.colour:
						nice_move[2] = move[-1]-1
					else:
						nice_move[2] = move[-1]+1
					if self.FEN.position[8-nice_move[4]][nice_move[3]-1] == '0':
						nice_move[5] = 'e' # en passant
					else:
						nice_move[5] = 'x'
				else:
					nice_move[5] = ''
					if self.FEN.colour:
						if self.FEN.position[9-nice_move[4]][nice_move[3]-1] == '0':
							nice_move[2] = move[-1]-2
						else:
							nice_move[2] = move[-1]-1
					else:
						if self.FEN.position[7-nice_move[4]][nice_move[3]-1] == '0':
							nice_move[2] = move[-1]+2
						else:
							nice_move[2] = move[-1]+1
				return tuple(nice_move)

	def mine2alg(self,move):
		"""Write one of the moves from my format in algebraic notation."""
		# Use to construct move history.
		# Assumes my move is valid.

		files = {1:'a',2:'b',3:'c',4:'d',5:'e',6:'f',7:'g',8:'h'}

		#castling
		if move[5].lower() == 'j':
			alg_move = 'O-O-O'
		elif move[5].lower() == 'k':
			alg_move = 'O-O'

		#promotion
		elif move[0] == 'p':
			if move[5] in 'qrbn' and move[5] != '':
				alg_move = files[move[1]] + str(move[4]) + '=' + move[5].upper()
			elif move[5] in 'QRBN' and move[5] != '':
				alg_move = files[move[1]] + 'x' + files[move[3]] + str(move[4]) + '=' + move[5]

			elif move[5] != '': #capture
				alg_move = files[move[1]] + 'x' + files[move[3]] + str(move[4])
			else:
				alg_move = files[move[3]] + str(move[4])

		elif move[0] != 'p':
			moves = self.FEN.potensh_moves()
			# the extra file/rank information is included even if another
			# potential move would be ruled out by it being check.
			candidates = [[move[0],None,None] + list(move[3:]),
				[move[0],move[1],None] + list(move[3:]),
				[move[0],None,move[2]] + list(move[3:]),]
			for candidate in candidates:
				possibles = []
				for possible in moves:
					for i in range(6):
						if candidate[i] == None:
							continue
						if candidate[i] != possible[i]:
							break
					else:
						possibles.append(possible)
				if len(possibles) == 1:
					break
			alg_move = move[0].upper()
			if candidate[1] != None:
				alg_move += files[move[1]]
			if candidate[2] != None:
				alg_move += str(move[2])
			alg_move += move[5] + files[move[3]] + str(move[4])

		new_position = self.FEN.consider_move(move)
		new_FEN = FEN_position(new_position,colour=(not self.FEN.colour))
		if new_FEN.is_checkmate():
			alg_move += '#'
		elif new_FEN.is_check(self.FEN.colour):
			alg_move += '+'
		# print(move,alg_move)
		return alg_move

	def add_to_history(self,move):
		"""Adds a move onto the end of self.history and self.pos_history."""
		move = self.mine2alg(move)
		if self.FEN.colour:
			self.history.append(str(self.FEN.fullmove) + '. ' + move)
		else:
			if self.history == []:
				self.history = [str(self.FEN.fullmove) + '. ... ' + move]
			else:
				whole_move = self.history[-1] + ' ' + move
				self.history[-1] = whole_move
		extra = False
		target = self.FEN.en_passant
		if target != '-':
			if self.FEN.colour:
				squares = [(target[0]-1,5),(target[0]+1,5)]
				for square in squares:
					if square[0]<1 or square[0]>8:
						continue
					if self.FEN.position[8-square[1]][square[0]-1] == 'P':
						extra = True
			else:
				squares = [(target[0]-1,4),(target[0]+1,4)]
				for square in squares:
					if square[0]<1 or square[0]>8:
						continue
					if self.FEN.position[8-square[1]][square[0]-1] == 'p':
						extra = True
		new_pos = [self.FEN.position,self.FEN.colour,self.FEN.castling,extra]
		for i in range(len(self.pos_history)):
			if new_pos == self.pos_history[i][0]:
				self.pos_history[i][1] += 1
				self.repeats = self.pos_history[i][1]
				break
		else:
			self.pos_history.append([new_pos,1])
			self.repeats = 1
		# print(self.repeats)

	def next_move(self,move):
		"""Takes input and moves a piece accordingly..."""
		if type(move) == str:
			move = self.alg2mine(move)
		# print(move) #test
		if move == 'ambiguous':
			print('This could refer to multiple moves. Please try again.')
		elif move in self.FEN.moves():
			self.add_to_history(move)
			#do move
			self.FEN.position = self.FEN.consider_move(move)
			if self.FEN.colour:
				if move[0] == 'k':
					self.FEN.castling[0] = False
					self.FEN.castling[1] = False
				if move[0] == 'r':
					if move[1] == 1:
						self.FEN.castling[1] = False
					elif move[1] == 8:
						self.FEN.castling[0] = False
			else:
				if move[0] == 'k':
					self.FEN.castling[2] = False
					self.FEN.castling[3] = False
				if move[0] == 'r':
					if move[1] == 1:
						self.FEN.castling[3] = False
					elif move[1] == 8:
						self.FEN.castling[2] = False
			if move[4] == 1:
				if move[3] == 1:
					self.FEN.castling[1] = False
				elif move[3] == 8:
					self.FEN.castling[0] = False
			elif move[4] == 8:
				if move[3] == 1:
					self.FEN.castling[3] = False
				elif move[3] == 8:
					self.FEN.castling[2] = False
			if self.FEN.colour:
				if move[0] == 'p' and move[2] == 2 and move[4] == 4:
					self.FEN.en_passant = (move[1],3)
				else:
					self.FEN.en_passant = '-'
			else:
				if move[0] == 'p' and move[2] == 7 and move[4] == 5:
					self.FEN.en_passant = (move[1],6)
				else:
					self.FEN.en_passant = '-'
			if move[5] == 'x' or move[5] == 'e' or move[0] == 'p':
				self.FEN.halfmove = 0
			else:
				self.FEN.halfmove += 1
			if not self.FEN.colour:
				self.FEN.fullmove += 1
			self.FEN.colour = not self.FEN.colour
		else:
			print('Not a valid move. Please try again.')

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
		for i in self.FEN.position:
			print(i)
		while True:
			move = input()
			if move == 'history':
				for i in self.history:
					print(i)
			else:
				self.next_move(move)
				for i in self.FEN.position:
					print(i)
				# print self.FEN.castling
				# print self.FEN.is_check(False)
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

# game().play()