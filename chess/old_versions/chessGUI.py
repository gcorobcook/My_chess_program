#still to do: resign, offer draw buttons, button to save pgns in archive.
#make it so player1 and clock1 are always white.
#work out dirty rects for clocks.
#make AI.

import pygame
import os
import chess
import chessAI
import datetime
from tkinter import Tk

# cs = 64

# board_x = 40
# board_y = 40
# cell_colour = (255,255,255)
# cell_x = board_x
# cell_y = board_y
# for i in range(8):
	# for j in range(8):
		# pygame.draw.rect(background,cell_colour,(cell_x,cell_y,cs,cs))
		# cell_colour = tuple(255-cell_colour[0] for k in range(3))
		# cell_x += cs
	# cell_colour = tuple(255-cell_colour[0] for k in range(3))
	# cell_y += cs
	# cell_x = board_x

class GUIgame(chess.game):
	def __init__(self):
		#set player=False to play from black's perspective
		pygame.init()
		chess.game.__init__(self,chess.FEN_position())
		self.start_time = 300 #both players start with 5 minutes by default
		self.time1 = 300
		self.time2 = 300
		self.time_choices = [1,2,3,5,10,15,30,60,"No limit"]
		self.increment = 0
		self.increment_choices = [0,1,2,3,5,10,15,30]
		self.cs = 72
		self.player1 = "Player 1"
		self.player2 = "Player 2"
		self.player = True # from white perspective: play_w or twopl
		self.play_option = 'twopl'

		self.screen = pygame.display.set_mode((14*self.cs,9*self.cs))

	def write_text(self,text,size):
		colour = (0,120,0)
		if size == 'tiny':
			return pygame.font.Font(None,self.cs//4).render(text,True,colour)
		if size == 'small':
			return pygame.font.Font(None,self.cs//2).render(text,True,colour)
		if size == 'big':
			return pygame.font.Font(None,4*self.cs//5).render(text,True,colour)

	def pos2pic(self):
		"""Takes a position and creates a picture with all the pieces in place.
		Later the pieces should be sprites, but I haven't got to that yet."""
		board = pygame.image.load(os.path.join("Images","board.jpg"))
		board = pygame.transform.scale(board,(8*self.cs,8*self.cs))
		board.convert()
		self.screen.blit(board,(self.board_x,self.board_y))
		self.pieces = {'P':'pawn_white',
			'N':'knight_white',
			'B':'bishop_white',
			'R':'rook_white',
			'Q':'queen_white',
			'K':'king_white',
			'p':'pawn_black',
			'n':'knight_black',
			'b':'bishop_black',
			'r':'rook_black',
			'q':'queen_black',
			'k':'king_black'}
		cell_x = self.board_x
		cell_y = self.board_y
		my_range = range(8)
		if not self.player:
			my_range = range(7,-1,-1)
		for i in my_range:
			for j in my_range:
				piece = self.FEN.position[i][j]
				if piece != '0':
					piece = pygame.image.load(os.path.join("Images",self.pieces[piece]+".png"))
					piece = pygame.transform.scale(piece,(self.cs,self.cs))
					if self.play_option == 'twopl' and self.FEN.position[i][j].islower():
						piece = pygame.transform.rotate(piece,180)
					piece.convert()
					self.screen.blit(piece,(cell_x,cell_y))
				cell_x += self.cs
			cell_y += self.cs
			cell_x = self.board_x

	def draw_screen(self):
		background = pygame.Surface(self.screen.get_size())
		background.fill((150,200,255))
		background.convert()
		self.background = background
		self.screen.blit(self.background,(0,0))

		self.board_x = self.cs//2
		self.board_y = self.cs//2

		self.pos2pic()
		pygame.display.flip()

	def draw_start_screen(self):

		class button(pygame.Surface):
			def __init__(self,text,bw=None,bh=None,border=True,bordwidth=2,
				clicked=False,backcolour=(255,220,178),clbackcolour=(255,190,112)):
				#obviously I could have more options, like border colour.
				self.text = text
				# this takes text that has already been rendered
				# so it could equally well take a picture
				self.tw,self.th = self.text.get_size()
				if bw == None:
					bw = int(self.tw)
				if bh == None:
					bh = int(self.th)
				self.bw = bw
				self.bh = bh
				pygame.Surface.__init__(self,(self.bw,self.bh))
				self.border = border
				self.bordwidth = bordwidth
				self.clicked = clicked
				self.backcolour = backcolour
				self.clbackcolour = clbackcolour
				self.fill(self.backcolour)
				if self.clicked:
					self.fill(self.clbackcolour)
				self.convert()
				if self.border:
					pygame.draw.rect(self,(0,0,0),(0,0,self.bw,self.bh),self.bordwidth)
				self.blit(self.text,(self.bw/2-self.tw/2,self.bh/2-self.th/2))

		#buttons: play as white,play as black,2 player,ok
		#various time controls
		start_box = pygame.Surface((self.screen.get_width()-2*self.cs,self.screen.get_height()-2*self.cs))
		start_box.fill((255,200,133))
		start_box.convert()
		start_game = self.write_text("Start Game",'big')
		start_game = button(start_game,border=False,backcolour=(255,200,133))
		sgx = start_box.get_width()/2-start_game.get_width()/2
		sgy = self.cs//2
		start_box.blit(start_game,(sgx,sgy))
		game_time = self.write_text("Time (minutes)",'small')
		game_time = button(game_time,border=False,backcolour=(255,200,133))
		start_box.blit(game_time,(self.cs/2,5*self.cs/2))
		game_incr = self.write_text("Increment (seconds)",'small')
		game_incr = button(game_incr,border=False,backcolour=(255,200,133))
		start_box.blit(game_incr,(self.cs/2,4*self.cs))
		play_w = self.write_text("Play as White",'small')
		cl = (self.play_option == 'play_w')
		play_w = button(play_w,5*self.cs//2,self.cs//2,clicked=cl)
		play_b = self.write_text("Play as Black",'small')
		cl = (self.play_option == 'play_b')
		play_b = button(play_b,5*self.cs//2,self.cs//2,clicked=cl)
		twopl = self.write_text("Two Players",'small')
		cl = (self.play_option == 'twopl')
		twopl = button(twopl,5*self.cs//2,self.cs//2,clicked=cl)
		ok = self.write_text("OK",'small')
		ok = button(ok,self.cs,self.cs//2)
		sbw,sbh = start_box.get_size()
		start_box.blit(play_w,(self.cs/2,3*self.cs/2))
		start_box.blit(play_b,(3*self.cs,3*self.cs/2))
		start_box.blit(twopl,(11*self.cs/2,3*self.cs/2))
		xpos = self.cs/2
		for i in self.time_choices[:-1]:
			cl = False
			if self.start_time != None:
				if i == self.start_time//60:
					cl = True
			butt = self.write_text(str(i),'small')
			butt = button(butt,self.cs//2,self.cs//2,clicked=cl)
			start_box.blit(butt,(xpos,3*self.cs))
			xpos += self.cs/2
		butt = self.write_text(str(self.time_choices[-1]),'small')
		cl = False
		if self.start_time == None:
			cl = True
		butt = button(butt,3*self.cs//2,self.cs//2,clicked=cl)
		start_box.blit(butt,(xpos,3*self.cs))
		xpos = self.cs/2
		for i in self.increment_choices:
			cl = False
			if i == self.increment:
				cl = True
			butt = self.write_text(str(i),'small')
			butt = button(butt,self.cs//2,self.cs//2,clicked=cl)
			start_box.blit(butt,(xpos,9*self.cs/2))
			xpos += self.cs/2
		start_box.blit(ok,(sbw-3*self.cs/2,sbh-self.cs))
		self.screen.blit(start_box,(self.cs,self.cs))
		pygame.display.flip()

	def starting(self):
		"""Start screen."""
		self.draw_screen()
		self.draw_start_screen()
		startloop = True
		while startloop:
			# there must be better ways to check for rects being clicked :/
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					return 'quit'
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_ESCAPE:
						return 'quit'
					elif event.key == pygame.K_RETURN:
						startloop = False
				if event.type == pygame.MOUSEBUTTONDOWN:
					x,y = event.pos
					x /= self.cs
					y /= self.cs
					if 5/2 < y < 3: #player
						x -= 3/2
						x /= 5/2
						x = int(x)
						# print(x)
						options = ['play_w','play_b','twopl']
						for i in range(3):
							if i == x:
								self.play_option = options[i]
						if x == 0:
							self.player = True
							self.player1 = "Human"
							self.player2 = "Computer"
						elif x == 1:
							self.player = False
							self.player1 = "Human"
							self.player2 = "Computer"
						else:
							self.player = True
							self.player1 = "Player 1"
							self.player2 = "Player 2"
					elif 4 < y < 9/2: #time
						x = int(2*x-3)
						for i in range(len(self.time_choices)+2):
							if i == x:
								if i < len(self.time_choices)-1:
									self.start_time = 60*self.time_choices[i]
								else:
									self.start_time = None
								self.time1 = self.start_time
								self.time2 = self.start_time
					elif 11/2 < y < 6: #increment
						x = int(2*x-3)
						for i in range(len(self.increment_choices)):
							if i == x:
								self.increment = self.increment_choices[i]
					elif 7 < y < 15/2: #ok
						if 23/2 < x < 25/2:
							startloop = False
					self.draw_start_screen()

	def draw_details(self):
		# player names, any other buttons
		people = pygame.Surface((4*self.cs,self.cs//2))
		people.fill((255,255,255))
		people.convert()
		pygame.draw.rect(people,(0,0,0),(0,0,people.get_width(),people.get_height()),2)
		self.people1 = people
		self.people2 = people.copy()
		self.people_x = self.board_x+8*self.cs+self.cs//2
		self.people2_y = self.board_y
		self.people_y = self.board_y + 15*self.cs//2
		player1_text = self.write_text(self.player1,'small')
		player2_text = self.write_text(self.player2,'small')
		self.people1.blit(player1_text,(self.cs//8,self.cs//4-player1_text.get_height()//2))
		if self.play_option == 'twopl':
			player2_text = pygame.transform.rotate(player2_text,180)
			self.people2.blit(player2_text,(31*self.cs//8-player2_text.get_width(),self.cs//4-player2_text.get_height()//2))
		else:
			self.people2.blit(player2_text,(self.cs//8,self.cs//4-player2_text.get_height()//2))
		self.screen.blit(self.background,(0,0))
		self.screen.blit(self.people1,(self.people_x,self.people_y))
		self.screen.blit(self.people2,(self.people_x,self.people2_y))
		pygame.display.flip()

	def draw_clocks(self):
		"""Reblits the clocks to keep them counting. Later only refresh
		dirty rects of these."""
		
		def pretty_time(time):
			if time <= 0:
				return '0.00'
			timer = int(time)
			secs = str(timer%60)
			if len(secs) == 1:
				secs = '0'+secs
			mins = timer//60
			hrs = 0
			if mins > 60:
				hrs = mins//60
				mins = mins%60
			mins = str(mins)
			if hrs != 0:
				if len(mins) == 1:
					mins = '0'+mins
				return str(hrs)+'.'+str(mins)+'.'+str(secs)
			else:
				return str(mins)+'.'+str(secs)

		clocks = pygame.Surface((4*self.cs,self.cs//2))
		clocks.fill((255,255,255))
		clocks.convert()
		pygame.draw.rect(clocks,(0,0,0),(0,0,4*self.cs,self.cs//2),2)
		clocks1 = clocks
		clocks2 = clocks.copy()

		time1r = pretty_time(self.time1)
		time1r = self.write_text(time1r,'small')
		time2r = pretty_time(self.time2)
		time2r = self.write_text(time2r,'small')
		clocks1.blit(time1r,(3*self.cs,self.cs//4-time1r.get_height()//2))
		clocks2.blit(time2r,(3*self.cs,self.cs//4-time2r.get_height()//2))
		if self.player:
			clocks2_y = self.board_y+self.cs//2
			clocks_y = self.board_y + 7*self.cs
		else:
			clocks_y = self.board_y+self.cs//2
			clocks2_y = self.board_y + 7*self.cs
		# clocks1 is white's time and clocks2 black's, not player1 and player2
		# this might change in the future
		if self.play_option == 'twopl':
			clocks2 = pygame.transform.rotate(clocks2,180)
		self.screen.blit(clocks1,(self.people_x,clocks_y))
		self.screen.blit(clocks2,(self.people_x,clocks2_y))

	def draw_history(self,topline):
		# Redo after each move to add the new move
		# The idea is to scroll by hand, by having the up and down arrows
		# change the topline.
		history_box = pygame.Surface((2*self.cs,41*self.cs/8))
		history_box.fill((255,255,255))
		history_box.convert()
		pygame.draw.rect(history_box,(0,0,0),(0,0,2*self.cs,41*self.cs/8),2)
		for i in range(topline,len(self.history)):
			move = self.history[i].split(' ')
			move = [self.write_text(part,'tiny') for part in move]
			y = (i-topline)*self.cs//4+self.cs//8
			history_box.blit(move[0],(3*self.cs//8-move[0].get_width(),y))
			history_box.blit(move[1],(self.cs//2,y))
			if len(move) == 3:
				history_box.blit(move[2],(9*self.cs//8,y))
		self.screen.blit(history_box,(11*self.cs,31*self.cs//16))

	def draw_resign(self):
		"""Offer draw and resign buttons."""
		#For one player mode, only resign button. For two player, both buttons
		#for both players. Positions change when there are no clocks.
		#Require confirmation.
		butts = pygame.Surface((self.cs//2,self.cs//2))
		butts.fill((255,255,255))
		butts.convert()
		pygame.draw.rect(butts,(0,0,0),(0,0,self.cs//2,self.cs//2),2)
		res_butts = butts
		draw_butts = butts.copy()
		r = self.write_text('0','small')
		d = self.write_text('½','small')
		res_butts.blit(r,(self.cs//4-r.get_width()//2,self.cs//4-r.get_height()//2))
		res_butts2 = pygame.transform.rotate(res_butts,180)
		draw_butts.blit(d,(self.cs//4-d.get_width()//2,self.cs//4-d.get_height()//2))
		draw_butts2 = pygame.transform.rotate(draw_butts,180)
		if self.start_time != None:
			butts_y = self.people_y - self.cs
			butts2_y = self.people2_y + self.cs
		else:
			butts_y = self.people_y - self.cs//2
			butts2_y = self.people2_y + self.cs//2
		if self.play_option != 'twopl':
			self.screen.blit(res_butts,(self.people_x,butts_y))
		else:
			self.screen.blit(res_butts,(self.people_x,butts_y))
			self.screen.blit(draw_butts,(self.people_x+self.cs//2,butts_y))
			self.screen.blit(res_butts2,(self.people_x,butts2_y))
			self.screen.blit(draw_butts2,(self.people_x+self.cs//2,butts2_y))

	def pixel2square(self,pos):
		file,rank = pos
		file -= self.board_x
		file //= self.cs
		file += 1
		rank -= self.board_y
		rank //= self.cs
		rank = 8-rank
		if not self.player:
			file = 9-file
			rank = 9-rank
		if 1 <= file and file <= 8 and 1 <= rank and rank <= 8:
			return [file,rank]
		else:
			return None

	def pixel2littlesquare(self,pos):
		file,rank = pos
		file -= self.board_x
		file %= self.cs
		file //= self.cs//2
		rank -= self.board_y
		rank %= self.cs
		rank //= self.cs//2
		#squares = [(0,0),(1,0),(0,1),(1,1)]
		pieces = 'qrbn'
		if self.play_option == 'twopl' and not self.FEN.colour:
			pieces = pieces[::-1]
		return pieces[file + 2*rank]

	def clicks2move(self,event,move):
		"""Interpret a series of mouse clicks/drags as a pair of squares
		representing the move. Check moves for validity at this stage;
		only accept moves that are. Return in my 6 character format."""
		click = self.pixel2square(event.pos)
		# deal with pawn promotion case first
		if move[4] != False:
			if event.type == pygame.MOUSEBUTTONDOWN:
				if click != move[3:5]:
					self.pos2pic()
					pygame.display.flip()
					return [False for i in range(6)]
				else:
					new_piece = self.pixel2littlesquare(event.pos)
					if self.FEN.position[8-click[1]][click[0]-1] != '0':
						new_piece = new_piece.upper()
					move[5] = new_piece
					return tuple(move)
			else:
				return move
		# get 2 squares which could be a move
		if event.type == pygame.MOUSEBUTTONDOWN:
			if click == None:
				return [False for i in range(6)]
			if move[0] == False:
				piece = self.FEN.position[8-click[1]][click[0]-1]
				if piece.isalpha() and piece.isupper() == self.FEN.colour:
					move[0] = piece.lower()
					move[1],move[2] = click
					return move
				else:
					return [False for i in range(6)]
			else:
				move[3],move[4] = click
		if event.type == pygame.MOUSEBUTTONUP:
			if click == None or move[0] == False:
				return [False for i in range(6)]
			if click == move[1:3]:
				return move
			else:
				move[3],move[4] = click
		# check whether this is a move
		# for valid in self.FEN.moves():
			# if move[:5] == list(valid)[:5]:
				# print(valid)
		for valid in self.FEN.moves():
			if move[:5] == list(valid)[:5]:
				if valid[5] == '':
					return valid
				elif valid[5].lower() in 'qrbn':
					# this is a pawn promotion.
					return move
				else:
					return valid
		# if not, for down clicks, take click as new 1st square.
		# for up clicks, throw all away.
		if event.type == pygame.MOUSEBUTTONDOWN:
			return self.clicks2move(event,[False for i in range(6)])
		else:
			return [False for i in range(6)]

	def timevsinsufficient(self):
		# I will only give draws for time-out vs insufficient material
		# when the other person has a king and *nothing else*. USCF disagrees
		# but this is a close simple approximation to FIDE.
		for i in range(8):
			for j in range(8):
				if self.FEN.position[i][j].lower() in 'qrbnp':
					if self.FEN.position[i][j].islower() == self.FEN.colour:
						return False
		return True

	def get_move(self):
		"""Loops until we get a move."""
		move = [False for i in range(6)]
		mainloop = True
		while mainloop:
			# check for input, and send it to the functions
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					return 'quit'
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_ESCAPE:
						return 'quit'
					if event.key == pygame.K_UP:
						if self.topline > 0:
							self.topline -= 1
							self.draw_history(self.topline)
					if event.key == pygame.K_DOWN:
						if self.topline < len(self.history)-20:
							self.topline += 1
							self.draw_history(self.topline)
				elif event.type in [pygame.MOUSEBUTTONDOWN,pygame.MOUSEBUTTONUP]:
					move = self.clicks2move(event,move)
					if not False in move:
						return move
					elif move[4] != False: #pawn promotions
						#show clickable piece pictures
						pieces = ['queen','rook','bishop','knight']
						if self.player:
							cell_x = self.board_x + (move[3]-1)*self.cs
							cell_y = self.board_y + (8-move[4])*self.cs
						else:
							cell_x = self.board_x + (8-move[3])*self.cs
							cell_y = self.board_y + (move[4]-1)*self.cs
						for i in range(4):
							if self.play_option == 'twopl' and not self.FEN.colour:
								piece = pieces[3-i]
							else:
								piece = pieces[i]
							if self.FEN.colour:
								piece += '_white.png'
							else:
								piece += '_black.png'
							piece = pygame.image.load(os.path.join("Images",piece))
							piece = pygame.transform.scale(piece,(self.cs//2,self.cs//2))
							if self.play_option == 'twopl' and not self.FEN.colour:
								piece = pygame.transform.rotate(piece,180)
							piece.convert()
							little_x = cell_x + (i%2)*self.cs//2
							little_y = cell_y + (i//2)*self.cs//2
							self.screen.blit(piece,(little_x,little_y))
							pygame.display.flip()
			if self.start_time != None:
				milliseconds = self.clock.tick(self.fps)
				if self.FEN.colour:
					self.time1 -= milliseconds/1000
				else:
					self.time2 -= milliseconds/1000
				self.draw_clocks()
				if self.time1 <= 0 or self.time2 <= 0:
					if self.timevsinsufficient():
						self.result = '1/2-1/2'
						return 'timevsinsuf'
					if self.time1 <= 0:
						self.result = '0-1'
						return 'winbtime'
					else:
						self.result = '1-0'
						return 'winwtime'
				pygame.display.flip()

	def playing(self):
		self.draw_screen()
		self.draw_details()
		self.pos2pic()
		self.draw_history(0)
		if self.start_time != None:
			self.draw_clocks()
			self.clock = pygame.time.Clock()
			self.fps = 30
		self.draw_resign()
		pygame.display.flip()
		move = [False for i in range(6)]
		self.topline = 0
		mainloop = True
		while mainloop:
			if self.play_option == 'play_w':
				if self.FEN.colour:
					move = self.get_move()
				else:
					game = chess.game(self.FEN,self.history,self.pos_history)
					move = chessAI.chessAI(game) # computer move
					if self.start_time != None:
						milliseconds = self.clock.tick(self.fps)
						self.time2 -= milliseconds/1000
			if self.play_option == 'play_b':
				if self.FEN.colour:
					game = chess.game(self.FEN,self.history,self.pos_history)
					move = chessAI.chessAI(game) # computer move
					if self.start_time != None:
						milliseconds = self.clock.tick(self.fps)
						self.time1 -= milliseconds/1000
				else:
					move = self.get_move()
			if self.play_option == 'twopl':
				move = self.get_move()
			if move in ['quit','winbtime','winwtime','timevsinsuf']:
				return move
			else:
				if self.start_time != None:
					if self.FEN.colour:
						self.time1 += self.increment
					else:
						self.time2 += self.increment
				self.next_move(move)
				if self.FEN.colour:
					print(self.history[-1])
				self.pos2pic()
				topline = max(0,len(self.history)-20)
				self.draw_history(topline)
				pygame.display.flip()
				if self.FEN.is_checkmate() or self.is_draw() != '':
					if self.FEN.is_checkmate():
						if self.FEN.colour:
							self.result = '0-1'
							return 'winb'
						else:
							self.result = '1-0'
							return 'winw'
					else:
						self.result = '1/2-1/2'
						return self.is_draw()

	def print_results(self,text1,text2):
			text1 = self.write_text(text1,'big')
			offset1_x = self.board_x + 4*self.cs - text1.get_width()//2
			offset1_y = self.board_y + 7*self.cs//2 - text1.get_height()//2
			text2 = self.write_text(text2,'big')
			offset2_x = self.board_x + 4*self.cs - text2.get_width()//2
			offset2_y = self.board_y + 9*self.cs//2 - text2.get_height()//2
			self.screen.blit(text1,(offset1_x,offset1_y))
			self.screen.blit(text2,(offset2_x,offset2_y))

	def make_FEN(self):
		final_FEN = ''
		spaces = 0
		for i in range(8):
			for j in range(8):
				if self.FEN.position[i][j] == '0':
					spaces += 1
				else:
					if spaces != 0:
						final_FEN += str(spaces)
						spaces = 0
					final_FEN += self.FEN.position[i][j]
			if spaces != 0:
				final_FEN += str(spaces)
				spaces = 0
			if i < 7:
				final_FEN += '/'
		if self.FEN.colour:
			final_FEN += ' w '
		else:
			final_FEN += ' b '
		castles = 'KQkq'
		nocastle = '- '
		for i in range(4):
			if self.FEN.castling[i]:
				final_FEN += castles[i]
				nocastle = ' '
		final_FEN += nocastle
		files = {1:'a',2:'b',3:'c',4:'d',5:'e',6:'f',7:'g',8:'h'}
		if self.FEN.en_passant != '-':
			ep = files[self.FEN.en_passant[0]] + str(self.FEN.en_passant[1]) + ' '
		else:
			ep = '- '
		final_FEN = final_FEN + ep + str(self.FEN.halfmove) + ' ' + str(self.FEN.fullmove)
		return final_FEN

	def make_PGN(self,next=''): ###
		final_PGN = '[Event "Casual Game"]\n[Site "Ged\'s Computer"]\n'
		date = '[Date "' + str(datetime.date.today()) + '"]\n'
		round = '[Round "-"]\n'
		if self.play_option != 'play_b':
			w = self.player1
			b = self.player2
		else:
			w = self.player2
			b = self.player1
		w = '[White "' + w + '"]\n'
		b = '[Black "' + b + '"]\n'
		res = '[Result "' + self.result + '"]\n'
		if self.start_time != None:
			tc1 = 60*self.start_time
			tc2 = self.increment
			tc = '[TimeControl "' + str(tc2) + '+' + str(tc2) + '"]\n'
		else:
			tc = '[TimeControl "-"]\n'
		if next == '':
			end = ''
		else:
			if next in ['winb','winw','stalemate','3 rep','50 move','insufficient']:
				term = 'normal'
			elif next in ['winbtime','winwtime']:
				term = 'time forfeit'
			elif next == 'timevsinsuf':
				term = 'time vs insufficient material'
			end = '[Termination "' + term + '"]\n'
		final_PGN = final_PGN + date + round + w + b + res + tc + end + '\n'
		for i in self.history:
			# PGN standard represents castling and promotion differently from
			# FIDE standard. May need to reprocess.
			# Add new lines?
			final_PGN += i
			final_PGN += ' '
		if self.result != '*':
			final_PGN += self.result
		else:
			final_PGN = final_PGN[:-1]
		return final_PGN

	def add_copy_buttons(self):
		"""Draw buttons to copy FEN and PGN to the clipboard."""
		copy_button = pygame.Surface((2*self.cs,self.cs//2))
		copy_button.fill((255,255,255))
		copy_button.convert()
		pygame.draw.rect(copy_button,(0,0,0),(0,0,copy_button.get_width(),copy_button.get_height()),2)
		self.copy_button1 = copy_button
		self.copy_button2 = copy_button.copy()
		self.copy_button_x = self.board_x + 17*self.cs//2
		self.copy_button2_y = self.board_y + 4*self.cs
		self.copy_button1_y = self.board_y + 7*self.cs//2
		self.copy_text1 = self.write_text('Copy FEN','small')
		self.copy_text2 = self.write_text('Copy PGN','small')
		self.copy_button1.blit(self.copy_text1,(self.cs//8,self.cs//4-self.copy_text1.get_height()//2))
		self.copy_button2.blit(self.copy_text2,(self.cs//8,self.cs//4-self.copy_text2.get_height()//2))
		self.screen.blit(self.copy_button1,(self.copy_button_x,self.copy_button1_y))
		self.screen.blit(self.copy_button2,(self.copy_button_x,self.copy_button2_y))

	def ending(self,next):
		if next == 'winb':
			text1="0 - 1"
			text2="Black wins!"
		elif next == 'winw':
			text1="1 - 0"
			text2="White wins!"
		elif next == 'winbtime':
			text1="0 - 1"
			text2="Black wins on time!"
		elif next == 'winwtime':
			text1="1 - 0"
			text2="White wins on time!"
		else:
			text1="1/2 - 1/2"
			if next == 'stalemate':
				text2="Stalemate"
			if next == '3 rep':
				text2="Draw by repetition"
			if next == '50 move':
				text2="Draw by the fifty-move rule"
			if next == 'insufficient':
				text2="Draw by insufficient material"
			if next == 'timevsinsuf':
				text2="Time-out vs insufficient material"
		self.print_results(text1,text2)
		self.add_copy_buttons()
		pygame.display.flip()
		endloop = True
		while endloop:
			event = pygame.event.wait()
			if event.type == pygame.QUIT:
				endloop = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					endloop = False
				if event.key == pygame.K_UP:
					if self.topline > 0:
						self.topline -= 1
						self.draw_history(self.topline)
				if event.key == pygame.K_DOWN:
					if self.topline < len(self.history)-20:
						self.topline += 1
						self.draw_history(self.topline)
				pygame.display.flip()
			elif event.type == pygame.MOUSEBUTTONDOWN:
				x,y = event.pos
				if self.copy_button_x < x and x < self.copy_button_x + 2*self.cs:
					if self.copy_button1_y < y and y < self.copy_button2_y:
						r = Tk()
						r.withdraw()
						r.clipboard_clear()
						r.clipboard_append(self.make_FEN())
						r.update()
					elif self.copy_button2_y < y and y < self.copy_button2_y + self.cs//2:
						r = Tk()
						r.withdraw()
						r.clipboard_clear()
						r.clipboard_append(self.make_PGN(next))
						r.update()

	def play(self):
		"""To play the game."""
		next = self.starting()
		if next != 'quit':
			next = self.playing()
		if next != 'quit':
			self.ending(next)

GUIgame().play()
pygame.quit()