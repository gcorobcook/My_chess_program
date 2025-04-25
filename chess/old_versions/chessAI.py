import pygame
import chess
import random
from copy import deepcopy

def randomAI(game):
	"""Takes a chess game object and returns a random (my format) move."""
	moves = game.FEN.moves()
	return random.choice(list(moves))

# General idea: define a value function for positions that is somehow recursive:
# minimise the max value of the other player's moves or something.

# Two stages: basic evaluation by hand (currently piecevalue, can be improved)
# and iteration a few moves forward (currently judgepos, can do more pruning).

def unst_max(iter,key=(lambda x:x)):
	"""Takes an iterable and returns the element that maximises lambda of it.
	Deliberately unstable: if several elements give the same value, it
	picks a random one to avoid loops."""
	# (But why can't my AI see when it's a draw in one and avoid it?)
	maximum = key(iter[0])
	answers = []
	for x in iter:
		if key(x) > maximum:
			answers = [x]
			maximum = key(x)
		elif key(x) == maximum:
			answers.append(x)
	return random.choice(answers)

def unst_min(iter,key=(lambda x:x)):
	minimum = key(iter[0])
	answers = []
	for x in iter:
		if key(x) < minimum:
			answers = [x]
			minimum = key(x)
		elif key(x) == minimum:
			answers.append(x)
	return random.choice(answers)

def piecevalue(game):
	"""Total points value of my pieces - opponents'."""
	# give wins high value and draws 0.
	values = {'p':1,'n':3,'b':3,'r':5,'q':9,'k':0}
	mult = {True:1,False:-1}
	total = 0
	if game.FEN.is_checkmate(): # if I write a few value functions, move these
	# bits to a bigger 'total value' function.
		return -1000*mult[game.FEN.colour]
	if game.is_draw() != '':
		return 0
	for i in range(8):
		for j in range(8):
			piece = game.FEN.position[i][j]
			if piece == '0':
				continue
			total += values[piece.lower()]*mult[piece.isupper()]
	return total

def judgepos(game,depth=0):
	# recursive step
	moves = game.FEN.moves()
	values = []
	max_depth = 0
	for move in moves:
		new_game = deepcopy(game)
		new_game.next_move(move)
		if depth < max_depth:
			new_value = judgepos(new_game,depth+1)
		else:
			new_value = [move,piecevalue(new_game)]
		values.append(new_value)
	if game.FEN.colour:
		return unst_max(values,key=lambda x:x[1])
	else:
		return unst_min(values,key=lambda x:x[1])

def chessAI(game):
	return judgepos(game)[0]