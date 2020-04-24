"""Strategy 4"""
import sys, traceback 
from heapq import heappush, heappop
import p4_utils as p4  
import deceptor as d

class Agent(object):
    """Simple dpp strategies return path that maximises LDP"""
    def __init__(self, **kwargs):
        pass
            
    def reset(self, **kwargs):
        self.stepgen = self.step_gen()
        
    def setGoals(self, poss_goals):
        #Called by controller to pass in poss goals
        self.poss_goals = poss_goals
        
    def getNext(self, mapref, current, goal, timeremaining):
        #print "received request"
        self.start = current 
        self.real_goal = goal
        self.mapref = mapref
        return next(self.stepgen)
        
    def step_gen(self):
        #print "running generator"
        all_goal_coords = [self.real_goal] + self.poss_goals
        goal_obs = d.generateGoalObs(self.mapref, self.start, all_goal_coords)
        rmp, argmin = d.rmp(self.mapref, self.start, goal_obs)

        heatmap = d.HeatMap(self.mapref, goal_obs)
        target = d.findTarget(self.mapref, rmp, argmin, self.real_goal, heatmap)
        try:
            path1 = self.customAstar(target, heatmap)
        except:
            print traceback.format_exc()
        path2 = self.mapref.optPath(target, self.real_goal)
        path = path1[1:] + path2[1:]
        self.path2 = path2
        
        for step in path:
            yield step     
            
    def getWorkings(self):
        print "getting workings"
        return ((self.path2, "yellow"),)
        
        
    def customAstar(self, target, heatmap):   
        start = self.start
        model = self.mapref
        if start == target:
            return []     # 0 steps, empty self.path

        path = []          # path as list of coordinates
        closedlist = {}    # dictionary of expanded nodes - key=coord, data = node
        openlist = []      # heap as pqueue on f_val
        pruned = []
        
        # initialise openlist with a
        # node = (f, g, coord, parent)  and a has g=0 and f = h and no parent
        heappush(openlist, (model.getH(start, target), 0, start, None))
        
        while openlist:
            # pop best node from open list (lowest f)
            node = heappop(openlist)
            current_f, current_g, current, parent = node
            if current in closedlist or current_g == float('inf'):
                continue # node has to be ignored: blocked or already visited

            # add node to closelist
            closedlist[current] = node

            # goal reached?
            if current == target:
                path = [current]
                while not current == start:
                    current = closedlist[current][p4.P_POS]
                    path.insert(0, current)
                return path
                
            # expand current node by getting all successors and adding them to open list
            adjacents = (model.getAdjacents(current))
            for adj in adjacents:
                if adj in closedlist or adj in pruned:   
                    #print adj, "closed"
                    continue   
                #print "considering truthfulness..."
                if model.getCell(adj) == "@":
                    #print "impassable"
                    continue
                if heatmap.checkNode(adj):
                    print adj, "pruned"
                    pruned.append(adj)
                    continue
                adjg = current_g + model.getCost(adj, current)
                adjf = adjg + model.getH(adj, target)
                adjnode = (adjf, adjg, adj, current)
                heappush(openlist, adjnode)
                
        print "path not found"