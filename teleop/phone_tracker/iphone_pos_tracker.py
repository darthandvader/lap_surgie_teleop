import numpy as np
import socket
import json
from threading import Thread

HOST = "0.0.0.0"
PORT = 20000

class iPhonePosTracker:
	
	def __init__(self):
		self.init_tf = np.eye(4)
		self.curr_tf = np.eye(4)

		# to convert from iPhone coordinate system
		self.axis_conv = np.array([
			[ 0, -1,  0,  0],
			[ 0,  0,  1,  0],
			[-1,  0,  0,  0],
			[ 0,  0,  0,  1]
		])
		self.start_server()
		self.tf_fetch_thread = Thread(target=self.thread_fetch_tf)
		self.tf_fetch_thread.start()

	def start_server(self):
		self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.server_socket.bind((HOST, PORT))
		print(f"iPhone Pos Tracking Server listening on {HOST}:{PORT}")

	def thread_fetch_tf(self):
		print("Waiting for iPhone data...")
		data, addr = self.server_socket.recvfrom(65535)
		print("Recieved data from", addr)
		pos_data = json.loads(data.decode())
		print("init_tf:")
		self.init_tf = np.array(pos_data["transform"])
		print(self.init_tf)
		while True:
			data, addr = self.server_socket.recvfrom(65535)
			pos_data = json.loads(data.decode())
			self.curr_tf = np.array(pos_data["transform"])

	def get_tf(self):
		return (self.axis_conv @ np.linalg.inv(self.init_tf) @ self.curr_tf @ np.linalg.inv(self.axis_conv))

