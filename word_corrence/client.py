#!/usr/bin/python3

import socket
import sys

host = '127.0.0.1'
port = 34772


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host , port))

if len(sys.argv) > 1:
	msg = sys.argv[1].strip()
	if msg:
		sock.sendall(msg.encode())
		reply = sock.recv(4096)
		print(reply.decode())

sock.close()
