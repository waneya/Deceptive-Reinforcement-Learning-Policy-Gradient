from collections import Counter
import math
import qFunction
from scipy.special import softmax
import random
import os.path
import numpy as np
DISCOUNT_FACTOR = 0.99
TERM_V = 10000.0
EPSILON = 0.0
ACTIONS = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
ALPHA = 0.3
TEAMP = 0.01
BASICT = 0.9
DEBUG = True
MODE = "dissimulation"
Q_DIR = "../qfunc/"
SAVE = True


class Agent(object):
    def __init__(self, lmap, real_goal, fake_goals, map_file, start):
        self.lmap = lmap
        self.A = ALPHA
        self.real_goal = real_goal
        self.fake_goals = fake_goals
        real_q_file = Q_DIR + map_file + ".{:d}.{:d}.q".format(real_goal[0], real_goal[1])
        self.real_q = qFunction.QFunction(lmap.width, lmap.height)
        if os.path.isfile(real_q_file):
            if DEBUG:
                print "loading q function for", real_goal
            self.real_q.load(real_q_file)
        else:
            if DEBUG:
                print "training q function for", real_goal
            qFunction.train(self.real_q, lmap, real_goal, TERM_V, DISCOUNT_FACTOR)
            self.real_q.save(real_q_file)
        self.fake_q = []
        for i, fg in enumerate(fake_goals):
            fake_q_file = Q_DIR + map_file + ".{:d}.{:d}.q".format(fg[0], fg[1])
            fq = qFunction.QFunction(lmap.width, lmap.height)
            if os.path.isfile(fake_q_file):
                if DEBUG:
                    print "loading q function for", fg
                fq.load(fake_q_file)
            else:
                if DEBUG:
                    print "training q function for", fg
                qFunction.train(fq, lmap, fg, TERM_V, DISCOUNT_FACTOR)
                fq.save(fake_q_file)
            self.fake_q.append(fq)
        self.matrix = lmap.matrix
        self.newMap = []
        for i in self.matrix:
            arr = []
            for j in i:
                if j == 'T':
                    arr.append(1)
                else:
                    arr.append(0)
            self.newMap.append(arr)
        self.allQtable =  [self.real_q] +  self.fake_q
        self.allGoals = [self.real_goal] + self.fake_goals
        self.start = start
        self.costForGoal = [0] * len(self.allGoals)
        self.minDist = [self.cost(self.start, goal, self.allQtable[index]) for index,  goal in enumerate(self.allGoals)]
        self.path = self.recursivePathModel()
        print(self.path)
        print(len(self.path))

    def getSQvalue(self, state):
        list = self.getQValueForActions(state)
        return list


    def getQValueForActions(self, state):
        x, y = state
        q_value_list = []
        for i, a in enumerate(ACTIONS):
            v = self.real_q.qValue(state, i)
            if v == float('-inf'):
                v = -100
            q_value_list.append(v)
        return q_value_list

    def getAllactions(self, state):
        x, y = state
        actionList = []
        for a in ACTIONS:
            state_p = (x + a[0], y + a[1])
            actionList.append(state_p)
        return actionList

    def temp(self, pathLength):
        return 1 + (2 * pathLength)

    def reset(self, **kwargs):
        self.count = 0


    def countState(self, path, currentState):
        count = 0
        for s in path:
            if currentState == s:
                count = count + 1
        return count

    def getAction(self, state):
        nextStateList = self.getAllactions(state)
        stateList = []
        for index, next_state in enumerate(nextStateList):
            if next_state == self.real_goal or self.lmap.isPassable(next_state):
                stateList.append(next_state)
        return stateList

    def recursivePathModel(self):
        currentState = self.start
        pathList = []
        w = 8000
        while currentState != self.real_goal:
            transtionList = []
            weightedList = []
            qActionList = self.getSQvalue(currentState)
            pathList.append(currentState)
            actionList = self.getAction(currentState)
            actionListAll = self.getAllactions(currentState)
            for index, actionState in enumerate(actionListAll):
                i, actionRM = self.rationalMeasureQvalue(actionState)
                if (actionState in actionList) and (actionState not in pathList):
                    diff = - actionRM
                elif (actionState in actionList) and (actionState in pathList):
                    n = self.countState(pathList, actionState)
                    diff = - actionRM - (n)
                else:
                    diff = 0
                transtionList.append(diff)
            transtionList = self.soft_max(transtionList)
            temp = self.temp(len(pathList))
            for index, t in enumerate(transtionList):
               # print(transtionList[index] * w * temp)
               # print(qActionList[index])
                weightedList.append(
                    float(float(ALPHA) * float(transtionList[index] * w * temp) + float((1 - ALPHA)) * float(
                        (qActionList[index]/80))))
            weightedList = self.soft_max(weightedList)
            actions = []
            weights = []
            for index, act in enumerate(actionListAll):
                if act in actionList:
                    actions.append(act)
                    weights.append(weightedList[index])
            m = max(weights)
            possibleMax = [i for i, j in enumerate(weights) if j == m]
            n = random.choice(possibleMax)
            past =  currentState
            currentState = actions[n]
            a_index = self.findActionbyState(currentState, past)
            self.getCostforAll(past, currentState, a_index)
            print(currentState)
        return pathList

    def soft_max(self, x):
        x = np.array(x)
        return softmax(x)

    def findActionbyState(self, current, next):
        for i, a in enumerate(ACTIONS):
            state_p = (current[0] + a[0], current[1] + a[1])
            if state_p == next:
                return i
        return -1


    def cost(self, currentState, goal, qtable):
        cost = qtable.value(goal)- qtable.value(currentState)
        return cost

    def rationalMeasureQvalue(self, currentState):
        rationalMeasurementList = []
        for index, goal in enumerate(self.allGoals):
            p = self.cost(currentState, goal, self.allQtable[index])
            rationalmeasurementScore = float(self.minDist[index]) / float((self.costForGoal[index] + p))
            rationalMeasurementList.append(rationalmeasurementScore)
        m = max(rationalMeasurementList)
        i = rationalMeasurementList.index(m)
        return i, m

    def getCostforAll(self, current, next,action):
        for index, cost in enumerate(self.costForGoal):
            self.costForGoal[index] = cost + (self.allQtable[index].value(next) - self.allQtable[index].qValue(current, action))

    def getNext(self, mapref, current, goal, timeremaining):
        self.count = self.count + 1
        return self.path[self.count]













    def vkProbabilityModel(self, path, goalset, goal):
        o = 0;
        count = 0
        s0 = path[0]
        path = path[1:]
        for currentState in path:
            a = len(self.bfs(np.array(self.newMap), s0, (goal[0], goal[1])))
            b = len(self.bfs(np.array(self.newMap), currentState, (goal[0], goal[1])))
            goalValue = (float(a) / float(o + b))
            check = False
            for g in goalset:
                c = len(self.bfs(np.array(self.newMap), s0, (g[0], g[1])))
                d = len(self.bfs(np.array(self.newMap), currentState, (g[0], g[1])))
                gValue = (float(c) / float((o + d)))
                if gValue >= goalValue:
                    check = True
            if check:
                count = count + 1
            o = o + 1
        return float(count)

    def dissimulationModelSingle(self, path):
        o = 0;
        s0 = path[0]
        path = path[1:]
        totalDissimulation = 0
        g = self.real_goal
        for currentState in path:
            dissimulation = 0
            a = len(self.bfs(np.array(self.newMap), s0, (g[0], g[1])))
            b = len(self.bfs(np.array(self.newMap), currentState, (g[0], g[1])))
            m = o + b
            gValue = float(0.8 * (float(a) / float(m)))
            dissimulation = dissimulation + float((gValue * math.log(gValue, 2)))
            o = o + 1
            totalDissimulation = totalDissimulation + float(-dissimulation)
        return totalDissimulation / float(len(path))

    def dissimulationModel(self, path):
        o = 0;
        s0 = path[0]
        # l = len(path)
        path = path[1:]
        totalDissimulation = 0
        k = math.log(len(self.allGoals), 2)
        vec = []
        for currentState in path:
            dissimulation = 0
            gList = []
            for g in self.allGoals:
                a = len(self.bfs(np.array(self.newMap), s0, (g[0], g[1])))
                b = len(self.bfs(np.array(self.newMap), currentState, (g[0], g[1])))
                m = o + b
                gValue = float(a) / float(m)
                gList.append(gValue)
            total = sum(gList)
            realg = []
            for gs in gList:
                a = float(gs) / float(total)
                realg.append(0.8 * a)
            for gValue in realg:
                dissimulation = dissimulation + float((gValue * math.log(gValue, 2)))
            o = o + 1
            vec.append(-dissimulation)
            totalDissimulation = totalDissimulation + float((-k) * dissimulation)
        return totalDissimulation, totalDissimulation / float(len(path)), vec

    def calculationAlongPath(self):
        alldis, averageDis, vec = self.dissimulationModel(self.path)
        print alldis, averageDis, vec
