from collections import deque
import numpy as np
import threading
import colorsys
import copy
from heapq import *




class Point(object):

    def __init__(self, x=0, y=0):
        self.x = x;
        self.y = y;

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    #좌표 + 해줄때 쓸거.

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y 
    # 좌표 그냥 =로 대입할 때 쓸거.


    def __gt__(self, other):
        return self if self.x > other.x else other



dir = [Point(0, -1), Point(0, 1), Point(1, 0), Point(-1, 0)]
class Maze():
    m = 0
    n = 0
    matrix = []
    time = 0
    length = 0

    def __init__(self, _m, _n, _matrix):
        self.m = _m;
        self.n = _n;
        
        self.matrix = copy.deepcopy(_matrix);
        # 배열은 deepcopy해야 값이 다 복사 됨.



def bfs(start, goal ,_maze):

    #큐를 이용함.
    q = deque()

    #방문했던 노드는 1로 표시할거기 때문에, vis라는 방문 체크 배열을 만들어서 다 0으로 초기화
    vis = [[0 for j in range(_maze.n)] for i in range(_maze.m)]
    #찾았는지 확인하기 위한 bool 변수
    found = False

    # 최단경로를 알아내기 위해 각 노드의 부모 노드가 뭔지 저장하는 배열
    parent = [[Point() for j in range(_maze.n)] for i in range(_maze.m)]
    q.append(start)
    vis[start.y][start.x] = 1;

    cnt = 1 # time을 구하기 위한 변수. 시작점부터 무조건 탐색하니까 1부터 시작
    while len(q) > 0 :
        now = q.popleft()
        
 

        # 상하좌우로 탐색
        for i in dir : 
            # 다음 좌표
            next = now + i

            # 좌표가 범위를 벗어나면 continue로 넘어간다
            if next.x < 0 or next.x >= _maze.n or next.y < 0 or next.y >= _maze.m :
                continue

            # goal 찾았으니까 탈출! 그리고 goal의 부모도 표시하고
            if next == goal: 
                found = True
                q.clear()
                parent[next.y][next.x] = now
                break

            # 아직 방문 안했는데 통로면 queue에 넣는다    
            if _maze.matrix[next.y][next.x] !=1 and vis[next.y][next.x] == 0 : 
                vis[next.y][next.x] = 1
                q.append(next)
                cnt += 1
                parent[next.y][next.x] = now # trace를 위한 부분. 부모가 뭐였는지 기록

    # 최단경로를 저장할 리스트
    path = []

    # 만약 찾으면?
    if found : 
        # goal부터 시작해서 parent를 저장해둔 배열을 이용해서 trace를 시작한다.
        p = goal
        while p != start :
            # 시작점과 도착점은 5로 바꾸지 않는다.
            if p!=start and _maze.matrix[p.y][p.x] !=4 :
                _maze.matrix[p.y][p.x] = 5
            path.append(p)
            p = parent[p.y][p.x]
        path.append(p)
        path.reverse() # 최단 경로 생성
        _maze.length += len(path)-1
        _maze.time += cnt
    else : 
        print("Can't find path...")


def heuristic(start, goal) : 
    return abs(start.x - goal.x) + abs(start.y - goal.y)


def greedy_best(start, goal, _maze):
    #힙큐를 이용함.
    pq = []

    #방문했던 노드는 1로 표시할거기 때문에, vis라는 방문 체크 배열을 만들어서 다 0으로 초기화
    vis = [[0 for j in range(_maze.n)] for i in range(_maze.m)]
    #찾았는지 확인하기 위한 bool 변수
    found = False

    # 최단경로를 알아내기 위해 각 노드의 부모 노드가 뭔지 저장하는 배열
    parent = [[Point() for j in range(_maze.n)] for i in range(_maze.m)]
    
    #현재까지 실제 cost 저장할 배열
    cost = [[0 for j in range(_maze.n)] for i in range(_maze.m)]


    heappush(pq, (0,start))
    vis[start.y][start.x] = 1;
    
    cnt = 1 # time을 구하기 위한 변수. 시작점부터 무조건 탐색하니까 1부터 시작

    while len(pq) > 0 :

        now = heappop(pq)[1]
 
        cost[now.y][now.x] = cnt


        # 상하좌우로 탐색
        for i in dir : 
            # 다음 좌표
            next = now + i

            # 좌표가 범위를 벗어나면 continue로 넘어간다
            if next.x < 0 or next.x >= _maze.n or next.y < 0 or next.y >= _maze.m :
                continue

            # goal 찾았으니까 탈출! 그리고 goal의 부모도 표시하고
            if next == goal: 
                found = True
                pq.clear()
                parent[next.y][next.x] = now
                break

            # 아직 방문 안했는데 통로면 pq에 넣는다.    
            if _maze.matrix[next.y][next.x] !=1 and vis[next.y][next.x] == 0 : 
   
                vis[next.y][next.x] = 1
                priority = heuristic(next, goal)
                heappush(pq, (priority, next))
                cnt += 1
                parent[next.y][next.x] = now # trace를 위한 부분. 부모가 뭐였는지 기록

    # 최단경로를 저장할 리스트
    path = []
    
    # 만약 찾으면?
    if found : 
        # goal부터 시작해서 parent를 저장해둔 배열을 이용해서 trace를 시작한다.
        p = goal
        while p != start :
            # 시작점과 도착점은 5로 바꾸지 않는다.
            if p!=start and _maze.matrix[p.y][p.x] !=4 :
                _maze.matrix[p.y][p.x] = 5
            path.append(p)
            p = parent[p.y][p.x]
        path.append(p)
        path.reverse() # 최단 경로 생성
        _maze.length += len(path)-1
        _maze.time += cnt
    else : 
        print("Can't find path...")



def a_star(start, goal ,_maze):

    #힙큐를 이용함.
    pq = []

    #방문했던 노드는 1로 표시할거기 때문에, vis라는 방문 체크 배열을 만들어서 다 0으로 초기화
    vis = [[0 for j in range(_maze.n)] for i in range(_maze.m)]
    #찾았는지 확인하기 위한 bool 변수
    found = False

    # 최단경로를 알아내기 위해 각 노드의 부모 노드가 뭔지 저장하는 배열
    parent = [[Point() for j in range(_maze.n)] for i in range(_maze.m)]
    
    #현재까지 실제 cost 저장할 배열
    cost = [[0 for j in range(_maze.n)] for i in range(_maze.m)]


    heappush(pq, (0,start))
    vis[start.y][start.x] = 1;
    
    cnt = 1 # time을 구하기 위한 변수. 시작점부터 무조건 탐색하니까 1부터 시작

    while len(pq) > 0 :

        now = heappop(pq)[1]

        cost[now.y][now.x] = cnt


        # 상하좌우로 탐색
        for i in dir : 
            # 다음 좌표
            next = now + i

            # 좌표가 범위를 벗어나면 continue로 넘어간다
            if next.x < 0 or next.x >= _maze.n or next.y < 0 or next.y >= _maze.m :
                continue

            # goal 찾았으니까 탈출! 그리고 goal의 부모도 표시하고
            if next == goal: 
                found = True
                pq.clear()
                parent[next.y][next.x] = now
                break

            # 아직 방문 안했는데 통로면 pq에 넣는다.    
            if _maze.matrix[next.y][next.x] !=1 and vis[next.y][next.x] == 0 : 
   
                vis[next.y][next.x] = 1
                priority = cost[now.y][now.x] + heuristic(next, goal)
                heappush(pq, (priority, next))
                cnt += 1
                parent[next.y][next.x] = now # trace를 위한 부분. 부모가 뭐였는지 기록

    # 최단경로를 저장할 리스트
    path = []
    
    # 만약 찾으면?
    if found : 
        # goal부터 시작해서 parent를 저장해둔 배열을 이용해서 trace를 시작한다.
        p = goal
        while p != start :
            # 시작점과 도착점은 5로 바꾸지 않는다.
            if p!=start and _maze.matrix[p.y][p.x] !=4 :
                _maze.matrix[p.y][p.x] = 5
            path.append(p)
            p = parent[p.y][p.x]
        path.append(p)
        path.reverse() # 최단 경로 생성
        _maze.length += len(path)-1
        _maze.time += cnt
    else : 
        print("Can't find path...")



def dfs(start, goal ,_maze):
    
    #리스트를 그냥 스택처럼 사용
    s = []
    #방문했던 노드는 1로 표시할거기 때문에, vis라는 방문 체크 배열을 만들어서 다 0으로 초기화
    vis = [[0 for j in range(_maze.n)] for i in range(_maze.m)]
    
    #찾았는지 확인하기 위한 변수
    found = False

    # 최단경로를 알아내기 위해 각 노드의 부모 노드가 뭔지 저장하는 배열
    parent = [[Point() for j in range(_maze.n)] for i in range(_maze.m)]
    s.append(start)
    vis[start.y][start.x] = 1;

    #print(goal.y,end='')
    cnt = 1
    while len(s) > 0 :

        now = s.pop()

        #도착지 찾았으니까 반복문을 빠져나간다.
        

        # 상하좌우로 탐색
        for i in dir : 
            # 다음 좌표
            next = now + i
            # 좌표가 범위를 벗어나면 continue로 넘어간다
            if next.x < 0 or next.x >= _maze.n or next.y < 0 or next.y >= _maze.m :
                continue
            # 아직 방문 안했는데 통로면 queue에 넣는다    
            
            if next == goal : 
                found = True
                s.clear()
                parent[next.y][next.x] = now
                break

            if _maze.matrix[next.y][next.x] !=1 and vis[next.y][next.x] == 0 : 
                vis[next.y][next.x] = 1
                s.append(next)
                cnt += 1
                parent[next.y][next.x] = now # trace를 위한 부분. 부모가 뭐였는지 기록

    path = []

    if found : 
        p = goal
        while p != start :
            if p!=start and _maze.matrix[p.y][p.x] !=4 :
                _maze.matrix[p.y][p.x] = 5
            path.append(p)
            p = parent[p.y][p.x]
        path.append(p)
        path.reverse() # 최단 경로 생성
        _maze.length += len(path)-1
        _maze.time += cnt
    else : 
        print("Can't find path...")







def first_floor(start, goal1, goal2, _maze):
    
    #bfs(start, goal1, _maze)
    #bfs(goal1, goal2, _maze)

    dfs(start, goal1, _maze)
    dfs(goal1, goal2, _maze)

    #a_star(start, goal1, _maze)
    #a_star(goal1, goal2, _maze)
    
    #greedy_best(start, goal1, _maze)
    #greedy_best(goal1, goal2, _maze)

def second_floor(start, goal1, goal2, _maze):

    #bfs(start, goal1, _maze)
    #bfs(goal1, goal2, _maze)

    #dfs(start, goal1, _maze)
    #dfs(goal1, goal2, _maze)

    #a_star(start, goal1, _maze)
    #a_star(goal1, goal2, _maze)
    
    greedy_best(start, goal1, _maze)
    greedy_best(goal1, goal2, _maze)

def third_floor(start, goal1, goal2, _maze):

    #bfs(start, goal1, _maze)
    #bfs(goal1, goal2, _maze)

    #dfs(start, goal1, _maze)
    #dfs(goal1, goal2, _maze)

    #a_star(start, goal1, _maze)
    #a_star(goal1, goal2, _maze)
    
    greedy_best(start, goal1, _maze)
    greedy_best(goal1, goal2, _maze)

def fourth_floor(start, goal1, goal2, _maze):

    #bfs(start, goal1, _maze)
    #bfs(goal1, goal2, _maze)

    #dfs(start, goal1, _maze)
    #dfs(goal1, goal2, _maze)

    #a_star(start, goal1, _maze)
    #a_star(goal1, goal2, _maze)
    
    greedy_best(start, goal1, _maze)
    greedy_best(goal1, goal2, _maze)

def fifth_floor(start, goal1, goal2, _maze):

    #bfs(start, goal1, _maze)
    #bfs(goal1, goal2, _maze)

    #dfs(start, goal1, _maze)
    #dfs(goal1, goal2, _maze)

    #a_star(start, goal1, _maze)
    #a_star(goal1, goal2, _maze)
    
    greedy_best(start, goal1, _maze)
    greedy_best(goal1, goal2, _maze)


def main():
    naming = [" ", "first", "second", "third", "fourth", "fifth"]
    matrix = []
    funcdict = {
        1: first_floor,
        2: second_floor,
        3: third_floor,
        4: fourth_floor,
        5: fifth_floor
    }

    m = 0
    n = 0
    try:
        floor = int(input("Choose the floor want to escape: "))
        temp = floor
        f = open(naming[floor] + "_floor_input.txt", "r")
        floor, m, n = map(int, f.readline().split())
        #맨 첫줄 읽어서 몇 층인지랑 map크기 받아옴.
        
        #키의 위치, 시작점, 도착점을 저장할 튜플
        key = [-1, 0]
        st = [-1, 0]
        ed = [-1, 0]


        while True:
            line = f.readline()#한 줄 식 입력 받음.
            
            if(key[1] == 0): key[0] += 1

            if(st[1] == 0): st[0] += 1

            if(ed[1] == 0): ed[0] += 1

            if not line: break
            temp_list = list(map(int, line.split()))
            #입력 받은 한 줄을 int map 형태로 바꿈.
            if(6 in temp_list): key[1] = temp_list.index(6)
            #입력 받으면서 키 위치를 찾는다.
            
            if(3 in temp_list): st[1] = temp_list.index(3)
            #입력 받으면서 시작점을 찾는다.

            if(4 in temp_list): ed[1] = temp_list.index(4)
            #입력 받으면서 도착점을 찾는다.
            matrix.append(temp_list)
        f.close()

        key = tuple(key)
        st = tuple(st)
        ed = tuple(ed)



        #시작 좌표, key 좌표, goal 좌표를 Point 형태로 만든다
        start = Point(st[1], st[0])
        goal1 = Point(key[1], key[0])
        goal2 = Point(ed[1], ed[0])

        
        _maze = Maze(m, n, matrix)

        funcdict[floor](start, goal1, goal2, _maze)

        #미로 탐색 결과를 파일에 출력
        f = open(naming[floor] + "_floor_output.txt","w")
        for i in range(_maze.m) :
            line = map(str, _maze.matrix[i])
            f.write(" ".join(line))
            f.write("\n")
        f.write("---\n")
        f.write("length=%d\n"%(_maze.length))
        f.write("time=%d\n"%(_maze.time))
        f.close()
        
    except:
        return -1
    if(temp <= 0 or temp > 6):
        return -1
    return 0

if __name__ == '__main__':
    val = main()
    if(val < 0):
        print("Please check input.")
