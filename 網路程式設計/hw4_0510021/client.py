import sys
import socket
import json
import os
import stomp
import time
conn = [0]*1000
conn_name = []
class MyListener(stomp.ConnectionListener):
    def on_error(self,headers,message):
        print('receive an error "%s"' % message)
    def on_message(self,headers,message):
        print('%s' % message)
        


class Client(object):
    def __init__(self, ip, port):
        try:
            socket.inet_aton(ip)
            if 0 < int(port) < 65535:
                self.ip = ip
                self.port = int(port)
            else:
                raise Exception('Port value should between 1~65535')
            self.cookie = {}
        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    def run(self):
        while True:
            cmd = sys.stdin.readline()
            if cmd == 'exit' + '\n':
                return
            if cmd != os.linesep:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect((self.ip, self.port))
                        cmd = self.__attach_token(cmd)
                        s.send(cmd.encode())
                        resp = s.recv(4096).decode()
                        self.__show_result(json.loads(resp), cmd)
                        
                except Exception as e:
                    print(e, file=sys.stderr)
    
    def __show_result(self, resp, cmd=None):
        if 'message' in resp:
            print(resp['message'])

        if 'invite' in resp:
            if len(resp['invite']) > 0:
                for l in resp['invite']:
                    print(l)
            else:
                print('No invitations')

        if 'friend' in resp:
            if len(resp['friend']) > 0:
                for l in resp['friend']:
                    print(l)
            else:
                print('No friends')

        if 'post' in resp:
            if len(resp['post']) > 0:
                for p in resp['post']:
                    print('{}: {}'.format(p['id'], p['message']))
            else:
                print('No posts')
                
        if 'groups' in resp:
            if len(resp['groups']) > 0:
                for g in resp['groups']:
                    print(g)
            else:
                print('No groups')

        if cmd:
            command = cmd.split()
            #print(command[0])
            if resp['status'] == 0 and command[0] == 'login':
                self.cookie[command[1]] = resp['token']
                if not resp['token'] in  conn_name:
                    conn_name.append(resp['token'])
                    conn[conn_name.index(resp['token'])] = stomp.Connection10()
                    conn[conn_name.index(resp['token'])].set_listener('',MyListener())
                    conn[conn_name.index(resp['token'])].start()
                    conn[conn_name.index(resp['token'])].connect('admin','password',wait=True)
                    for g in resp['group']:
                        print('join group: '+g)
                        conn[conn_name.index(resp['token'])].subscribe('/topic/'+g)
                    conn[conn_name.index(resp['token'])].subscribe('/queue/'+command[1])
            if resp['status'] == 0 and command[0] == 'logout':  
                conn[conn_name.index(command[1])].disconnect()
                del conn[conn_name.index(command[1])]
                conn_name.remove(command[1])
                
            if resp['status'] == 0 and command[0] == 'delete':
                conn[conn_name.index(command[1])].disconnect()
                del conn[conn_name.index(command[1])]
                conn_name.remove(command[1])
                
            if resp['status'] == 0 and command[0] == 'create-group':
                #print('/topic/'+resp['group'])
                conn[conn_name.index(command[1])].subscribe('/topic/'+resp['group'])
                
            if resp['status'] == 0 and command[0] == 'join-group':
                #print('/topic/'+resp['group'])
                conn[conn_name.index(command[1])].subscribe('/topic/'+resp['group'])
                
           
                

    def __attach_token(self, cmd=None):
        if cmd:
            command = cmd.split()
            if len(command) > 1:
                if command[0] != 'register' and command[0] != 'login':
                    if command[1] in self.cookie:
                        command[1] = self.cookie[command[1]]
                    else:
                        command.pop(1)
            return ' '.join(command)
        else:
            return cmd


def launch_client(ip, port):
    c = Client(ip, port)
    c.run()

if __name__ == '__main__':
    
    if len(sys.argv) == 3:
        launch_client(sys.argv[1], sys.argv[2])
    else:
        print('Usage: python3 {} IP PORT'.format(sys.argv[0]))