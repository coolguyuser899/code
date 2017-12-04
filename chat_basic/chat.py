"""
Simple chat program from below
https://www.youtube.com/watch?v=DIPZoZheMTo

To run:
server
python3 chat.py
client
telnet localhost 10000
"""

import socket
import threading   #allow multiple connection at any one time

#AF_INET = IP4, SOCK_STREAM = TCP, SOCK_DGRAM = UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#sockk to bind an address and port, python tuple, 0.0.0.0 = available to any ip address
sock.bind(('0.0.0.0', 10000))

#allow one connection
sock.listen(1)

connections = []

def handler(c, a):
    global connections     #allow to access connections
    while True:
        data = c.recv(1024)    #receive data from connection c
        for connection in connections:
            connection.send(bytes(data))    #send back data, convert string to bytes
        if not data:
            connections.remove()
            c.close()
            break

while True:
    c, a = sock.accept()   #connection and client address
    cThread = threading.Thread(target=handler, args=(c,a))  #threading library Thread method, method is handler, pass args c,a
    cThread.daemon = True   #allow program exit regardless any thread is running
    cThread.start()
    connections.append(c)   #when new connection append
    print(connections)