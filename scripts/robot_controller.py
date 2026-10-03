      
import socket
import struct
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import signal
import functools
import time
import select
import errno

import threading

_DEFAULT_ROBOT_CONFIG = {
    0: {'host':'0.0.0.0', 'action_port':10000, 'keyboard_port':10001},
}

class RobotRegistry:
    def __init__(self):
        self.registry = _DEFAULT_ROBOT_CONFIG
    
    def register_robot(self, robot_id, host, action_port, keyboard_port):
        if robot_id in self.registry:
            print(f"Error: Robot ID {robot_id} already registered.")
            return False
        else:
            self.registry[robot_id] = {'host': host, 'action_port': action_port, 'keyboard_port':keyboard_port}
            print(f"Robot ID {robot_id} registered successfully with host {host}, \
                        action_port {action_port} and keyboard_port {keyboard_port}.")
            return True

    def get_robot_info(self, robot_id):
        if robot_id in self.registry:
            return self.registry[robot_id]
        else:
            print(f"Error: Robot ID {robot_id} not found.")
            return None
        
    def exist(self, robot_id):
        return (robot_id in self.registry)

class RobotCommunication:
    def __init__(self, robot_id, host=None, action_port=None, keyboard_port=None):
        self.robot_register = RobotRegistry()
        if self.robot_register.exist(robot_id):
            self.robot_info = self.robot_register.get_robot_info(robot_id)
        else:
            self.robot_register.register_robot(robot_id, host, action_port, keyboard_port)
            self.robot_info = self.robot_register.get_robot_info(robot_id)

        self.action_sock = None
        self.action_conn = None

        self.keyboard_sock = None
        self.keyboard_conn = None

        self.client_socks = []

    def connect(self):
        self.action_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.action_sock.setblocking(True)
        self.action_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.action_sock.bind((self.robot_info['host'], self.robot_info['action_port']))
        self.action_sock.listen(1)
        print(f"Listening on action port {self.action_sock.getsockname()[1]}")

        action_thread = threading.Thread(target=self.handle_action_client)
        action_thread.start()
        action_thread.join()    

    def handle_action_client(self):
        self.action_conn, addr = self.action_sock.accept()
        print(f"Accepted connection from {addr}")

    def recv_image(self, index):
        image_size = struct.unpack('<L', self.action_conn.recv(4))[0]
        image = self.recvall(self.action_conn, image_size)
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def recv_action_data(self):
        data_size = struct.unpack('<L', self.action_conn.recv(4))[0]
        data = self.recvall(self.action_conn, data_size)
        action_data = json.loads(data.decode('utf8'))
        return action_data
    
    def accept_connections(self):
        try:
            client_sock, addr = self.keyboard_sock.accept()
            client_sock.setblocking(0)  # Set non-blocking mode for the client socket
            self.client_socks.append(client_sock)
        except BlockingIOError:
            # The socket is in non-blocking mode and there are no pending connections
            pass
    

    def send_dict(self, dict_data):
        data_str = json.dumps(dict_data)
        data_bytes = data_str.encode('utf-8')
        self.action_conn.sendall(struct.pack('<L', len(data_bytes)))
        self.action_conn.sendall(data_bytes)

    def close(self):
        self.action_sock.close()
        for sock in self.client_socks:
            sock.close()
        # self.keyboard_sock.close()

    @staticmethod
    def recvall(sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf
    

class RobotController:
    def __init__(self,
             robot_id:int,
             host:str=None,
             port:str=None,
             max_time_step:int=10000
        ):
        self.robot_comm = RobotCommunication(robot_id, host, port)
        self.max_time_step = max_time_step
        self.global_step = 0

    def connect(self):
        self.robot_comm.connect()

    def close(self):
        self.robot_comm.close()

    def prediction(self, views, actions) -> dict:
        raise NotImplementedError
    
    def recv_image(self, cam_names:list = ['camera_left', 'camera_front', 'camera_right']) -> dict:
        views = {}
        for i,name in enumerate(cam_names):
            image = np.array(self.robot_comm.recv_image(i)) # left
            views[name] = image[None,:]
        return views
    
    def recv_action(self):
        return self.robot_comm.recv_action_data()
        
    def reset(self):
        self.global_step = 0

    
    def set_zero(self):
        record_signal = {
            'cmd':'INIT_ZERO',
            'gripper':[0.0, 0.0]
        }
        self.robot_comm.send_dict(record_signal)
        self.reset()
    
    def reset_to_zero(self):
        self.record_stop()
        self.set_zero()
        
    def run(self,
            record_mode=False,
            cam_names:list=['camera_left', 'camera_front', 'camera_right']):
        
        while self.global_step <= self.max_time_step:            

            self.global_step += 1

            action = self.recv_action()
            view = self.recv_image(cam_names)
            
            pred = self.prediction(view, action)
            self.robot_comm.send_dict(pred)


    