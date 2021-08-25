import tkinter as tk
from tkinter import ttk
from tkinter import *
from random import randint
import random
import math
import tensorflow as tf
import numpy as np

modelList = ['CNN','NN']

#Sizes
size_of_board = 600
rows = 28
cols = 28
boxLengthX = size_of_board / cols
boxLengthY = size_of_board / rows

#Colors
backgroundColor = 'gray75'
borderColor = 'black'


class drawBoard:
	def __init__(self):

		self.window = tk.Tk()
		self.window.title('ML Number Guesser')
		self.window.geometry(str(size_of_board+25)+'x'+str((size_of_board+125)))

		self.window.bind("<B1-Motion>", self.draw_border)
		self.window.bind("<B3-Motion>", self.remove_border)

		self.window.bind("<Button-1>", self.draw_border)
		self.window.bind("<Button-3>", self.remove_border)

		self.buttonFrame = tk.Frame(self.window, width=size_of_board)
		self.buttonFrame.pack(side=TOP)

		self.startButton = tk.Button(self.buttonFrame,command=self.start_button_click, text='Start')
		self.startButton.pack(side = LEFT)

		self.clearButton = tk.Button(self.buttonFrame,command=self.clear_button_click, text='Clear')
		self.clearButton.pack(side = LEFT)

		self.modelChoice = tk.StringVar(self.window)
		self.modelChoice.set(modelList[0])

		self.modelOptMenu = ttk.OptionMenu(self.window, self.modelChoice, modelList[1], *modelList)
		self.modelOptMenu.config(width=50)
		self.modelOptMenu.pack()

		self.canvas = tk.Canvas(self.window,width=size_of_board, height=size_of_board, background=backgroundColor)
		self.canvas.pack()

		self.predictionLabel = tk.Label(self.window, text='No Prediction')
		self.predictionLabel.pack(side = BOTTOM)

		self.init_board()



	def init_board(self):
		self.board = {}

		for i in range(rows):
			for j in range(cols):
				self.board[i,j] = 0

		for i in range(rows):
			self.canvas.create_line(
				i * size_of_board / rows, 0, i * size_of_board / rows, size_of_board,
			)

		for i in range(cols):
			self.canvas.create_line(
				0, i * size_of_board / cols, size_of_board, i * size_of_board / cols,
			)

	def mainloop(self):
		while True:
			self.window.update()

	def find_neighbors(self, point):
		if point in self.board:
			x,y = point
			neighbors = []
			if y < cols-1 and self.board[x,y+1] != 1:
				neighbors.append((x,y+1))
			if y > 0 and self.board[x,y-1] != 1:
				neighbors.append((x,y-1))
			if x < rows-1 and self.board[x+1,y] != 1:
				neighbors.append((x+1,y))
			if x > 0 and self.board[x-1,y] != 1:
				neighbors.append((x-1,y))
			return neighbors
		return

	def draw_border(self, event):
		widget = self.window.winfo_containing(event.x_root, event.y_root)
		if str(widget) == '.!canvas':
			#boxLengthX = size_of_board / cols
			#boxLengthY = size_of_board / rows

			i1 = event.x // boxLengthX
			i2 = event.y // boxLengthY

			intensityRadius = 1
			thickness = size_of_board / cols

			colorNum = random.randint(0,25)
			gradient = hex(colorNum)
			if len(gradient) == 3:
				grd_list = list(gradient)
				grd_list.insert(2,'0')
				gradient = ''.join(grd_list)
			color = '#'+gradient[2:]*3
			self.canvas.create_rectangle(i1*boxLengthX, i2*boxLengthY, (i1+1)*boxLengthX, (i2+1)*boxLengthY, fill=color)
			self.board[i1,i2] = 1
			point =(i1,i2)

			for neighbor in self.find_neighbors(point):
				x,y = neighbor
				colorNum = random.randint(50,175)
				gradient = hex(colorNum)
				color = '#'+gradient[2:]*3
				self.canvas.create_rectangle(x*boxLengthX, y*boxLengthY, (x+1)*boxLengthX, (y+1)*boxLengthY, fill=color)
				self.board[x,y] = colorNum / 255.0

	def remove_border(self, event):
		widget = self.window.winfo_containing(event.x_root, event.y_root)
		if str(widget) == '.!canvas':
			i1 = event.x // boxLengthX
			i2 = event.y // boxLengthY
			if self.board[i1,i2] != 0:
				self.canvas.create_rectangle(i1*boxLengthX, i2*boxLengthY, (i1+1)*boxLengthX, (i2+1)*boxLengthY, fill=backgroundColor)
				self.board[i1,i2] = 0
				point =(i1,i2)

				for neighbor in self.find_neighbors(point):
					x,y = neighbor
					self.canvas.create_rectangle(x*boxLengthX, y*boxLengthY, (x+1)*boxLengthX, (y+1)*boxLengthY, fill=backgroundColor)
					self.board[x,y] = 0

	def start_button_click(self):
		# TODO
		if self.modelChoice.get() == 'CNN':
			model = tf.keras.models.load_model('cnn.h5')
		elif self.modelChoice.get() == 'NN':
			model = tf.keras.models.load_model('nn.h5')

		matrix = []

		for i in range(rows):
			vector = []
			for j in range(cols):
				vector.append(self.board[j,i])
				value = self.board[j,i]
			matrix.append(vector)

		matrix = np.array(matrix)
		if self.modelChoice.get() == 'CNN':
			matrix = np.expand_dims(matrix,axis=2)
			matrix = np.expand_dims(matrix,axis=0)
		elif self.modelChoice.get() == 'NN':
			matrix = np.expand_dims(matrix,axis=0)
		result = model.predict(matrix)
		prob = max(result[0])
		prediction = list(result[0]).index(prob)
		print('Its a ', prediction)
		self.predictionLabel['text'] = 'Your number is a'+str(prediction)


	def clear_button_click(self):
		self.canvas.delete("all")
		self.init_board()


newBoard = drawBoard()
newBoard.mainloop()