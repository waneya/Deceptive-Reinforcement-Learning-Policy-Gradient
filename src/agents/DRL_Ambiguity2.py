"""strategy 1"""
from heapq import heappush, heappop, heapify
import random
from time import time
import math
import sys, traceback
import pandas as pd
import numpy as np

import deceptor as d

OCT_CONST = 0.414
INF = float('inf')
X_POS = 0
Y_POS = 1

DEBUG_DRL_EQUATIONS = False
DEBUG_PROBABILITY_CALCULATIONS = False
DEBUG_PRUNING = True
DEBUG_QVALUES = False
DEBUG_ACTIONS = False
USE_Q_TABLES = False # if false, reward function wil be used


class Agent(object):
    """Simple dpp strategies return path that maximises LDP"""

    def __init__(self, **kwargs):
        #self.observationsSoFar = []
        #self.qtables_pandas = {}
        pass


    def initializeDRLAmbiguity(self, mapref, start, realGoal, possGoals):
        self.qtables_pandas = {}
        self.observationsSoFar = []
        self.availableActions = ["up", "down", "left", "right", "up_left", "up_right", "down_left", "down_right"]

        self.start = start
        self.currentPosition = self.start
        self.realGoal = realGoal
        self.possGoals = possGoals
        self.mapref = mapref
        self.all_goal_coords = [self.realGoal] + self.possGoals
        self.residualRewardEachGoal = {}
        self.actionAdvantageEachGoal = {}
        self.prunedGoalsThisStep = []
        self.pruningParameter = 0.0

        # first action taken randomly
        #self.actualActionTaken = random.choice(self.availableActions)
        #print "\n------------First Random Action Taken As " +self.actualActionTaken+ "--------------\n"
        #self.observationsSoFar.append(self.actualActionTaken)
        self.deltaREachGoalSoFar = {} # stores the running sum of difference of QValue for each goal

        self.probabilityEachGoal = {}  # This is P(ri)


        # Load the Q_Tables in Pandas
        # Initialize the P(ri) uniformly

        for goal in self.all_goal_coords:
            self.probabilityEachGoal[goal] = (1/float(len(self.all_goal_coords)))
            self.deltaREachGoalSoFar[goal] = 0

            if USE_Q_TABLES:
                try:
                    df = pd.read_csv(str(goal) + "_Q_table.csv", header=None)
                    #df.columns = df.iloc[0]
                    #df = df[1:]
                    self.qtables_pandas[goal] = df

                except Exception:
                    print("Exception occured while loading Q-Tables")


    def reset(self, **kwargs):

        self.stepgen = self.step_gen()

    def setGoals(self, poss_goals):
        # Called by controller to pass in poss goals
        self.possGoals = poss_goals

    def getNext(self, mapref, current, goal, timeremaining):

        self.currentPosition = current
        self.realGoal = goal
        self.mapref = mapref

        #return self.stepgen

        return next(self.stepgen)


    def convertActionToCoordinates(self, mapref, action, current):

        #print"Received action is: ", action, "received position is: ", current


        if action in self.availableActions:
            x = current[0]
            y = current[1]

            new_x = x
            new_y = y

            if action == "up":
                new_y = y - 1

            elif action == "down":
                new_y = y + 1

            elif action == "left":
                new_x = x - 1

            elif action == "right":
                new_x = x + 1

            elif action == "up_left":
                new_x = x - 1
                new_y = y - 1

            elif action == "up_right":
                new_x = x + 1
                new_y = y - 1

            elif action == "down_left":
                new_x = x - 1
                new_y = y + 1

            elif action == "down_right":
                new_x = x + 1
                new_y = y + 1


            nextState = (new_x, new_y)
            status = "step"


            # To check the reasonable of action
            if new_x < 0 or new_x > mapref.info['width'] - 1:
                if DEBUG_ACTIONS:
                    print "position: ", nextState, " for action: ", action, " is out of boundry"

                nextState = current
                status = 'no_step'
            if new_y < 0 or new_y > mapref.info['height'] - 1:
                if DEBUG_ACTIONS:
                    print "position: ", nextState, " for action: ", action, " is out of boundry"

                nextState = current
                status = 'no_step'

            if mapref.matrix[new_x][new_y] != ".":
                if DEBUG_ACTIONS:
                    print "obstruction at position: ", nextState, " for action: ", action
                nextState = current

                status = 'obstacle'

        else:
            if DEBUG_ACTIONS:

                print("Action not found while converting")
            nextState = current
            status = 'no_step'

        #print "Next State is: ", nextState

        return nextState,  status

    def getLigitimateActions(self, position):

        ligitimateActions = []

        for action in self.availableActions:
            #print action
            coord, status = self.convertActionToCoordinates(self.mapref, action, position)
            #if DEBUG_DRL_EQUATIONS:
            #print "Status of Action ", action, " is " , status
            if status == "step":
                ligitimateActions.append((coord,action))

        return ligitimateActions

    def getRewardForAction(self, cordinatesAfterAction, goal, action):


        atGoalReward  = 1000
        distanceNormalizer = float(self.mapref.info['width'] * self.mapref.info['height'])
        distance = math.sqrt(((cordinatesAfterAction[0] - goal[0]) ** 2) + ((cordinatesAfterAction[1] - goal[1]) ** 2))
        normalizedDistance = distance/distanceNormalizer
        closenessToGoal = 1 - normalizedDistance

        if DEBUG_QVALUES:
            print "--------For Action: ",action," corresponding to position: ", cordinatesAfterAction, "--------"
            print " Squared Distance: ", distance
            print " Distance Normalizer (Area of map): ", distanceNormalizer
            print " Normalized Distance: ", normalizedDistance
            print " Closeness To Goal: ", closenessToGoal

        reward = closenessToGoal # add others as we progress

        isGoal = cordinatesAfterAction == goal

        if isGoal:
            reward +=  atGoalReward

        return reward


    def getBestQValueForAGoal(self, goal, currentPosition):

        if USE_Q_TABLES:

            if DEBUG_QVALUES:
                print"\n=================Best Q Value for current position and current goal using Q-Tables==============================="
                print "\nBestQ:------------Current Goal is :", goal , "Position is: ", currentPosition , "---------\n"

            df = self.qtables_pandas[goal]

            current_position = df.loc[df[0] == str(currentPosition)]
            del current_position[0]
            if DEBUG_DRL_EQUATIONS:
                print "Q-Table Values based on above are: ",current_position
            best_q_value = float((current_position.astype(float).max(axis=1).values[0]))

            if DEBUG_QVALUES:
                print "Best Q-Value is :", best_q_value
            best_action_index = int(current_position.astype(float).idxmax(axis=1).values[0])

            #best_action = self.availableActions[best_action_index - 1]
            #print("Best Action is: ", best_action, type(best_action))

            return best_q_value

        else:
            valueOfActions = {}

            ligitimateActions = self.getLigitimateActions(currentPosition)
            if DEBUG_ACTIONS or DEBUG_QVALUES:
                print"\n=================Best Q Value for current position and current goal using Reward Function===================="
                print "================Legitimate Actions are: ====================="
                print len(ligitimateActions), ": ", ligitimateActions

            for items in ligitimateActions:
                coord = items[0]
                action = items[1]
                # print action

                valueOfActions[action] = self.getRewardForAction(coord, goal, action)

            # Get argmin(entropyOfActions)
            if DEBUG_QVALUES:
                print "\n\nAll Qvalues Are: ", valueOfActions

            bestAction = max(valueOfActions, key=valueOfActions.get)
            best_q_value = max(valueOfActions.values())
            if DEBUG_QVALUES or DEBUG_ACTIONS:
                print"Action Taken is: ", bestAction

            return best_q_value


    def getQValueOfAction(self, goal, legitimateAction, currentPosition):

        if USE_Q_TABLES:

            if DEBUG_QVALUES:
                print"\n================Q-Value of Action Using Q-Tables ============================"
                #print "\n----Q-Value Of Action---------"

            df = self.qtables_pandas[goal]

            #Selected Row For Current Position
            current_position = df.loc[df[0] == str(currentPosition)]
            del current_position[0]
            indexOfAction = self.availableActions.index(legitimateAction)

            # current position has one extra column as it is pandas
            qValueOfAction = float(current_position[indexOfAction + 1].astype(float).values[0])

            if DEBUG_QVALUES:
                print "Q-Value of Action is :", qValueOfAction

            return qValueOfAction

        else:
            if DEBUG_QVALUES:
                print"\n==================Q-value of Action Using Reward Function ============================"
            cordinatesAfterAction = self.convertActionToCoordinates(self.mapref,legitimateAction,currentPosition)[0]
            qValueOfAction = self.getRewardForAction(cordinatesAfterAction, goal, legitimateAction)
            if DEBUG_QVALUES:
                print "Q-Value of Action is :", qValueOfAction
            return qValueOfAction


    def getQdifference(self, goal, position, legitimateAction):

        bestQ = self.getBestQValueForAGoal(goal, position)
        qOfAction = self.getQValueOfAction(goal, legitimateAction, position)
        qDifferenceForGivenAction = (qOfAction - bestQ) / 1000

        if DEBUG_DRL_EQUATIONS:
            print " For goal: ", goal, " and  action : ", legitimateAction , " Best Q is: ", bestQ, " qOfAction is :", qOfAction


        return qDifferenceForGivenAction


    # This function is used to calculate the
    # Entropy based on Selected Action
    # and observations so far (o.a)
    def getEntropyOfAction(self, currentPosition, action, listOtherThenPruned):


        if DEBUG_PROBABILITY_CALCULATIONS:
            print "\n-------------Calculating Entropy for action ", action,"---------------------\n"



        probabilityRgivenODotAction = {}
        tempDeltaREachGoal = self.deltaREachGoalSoFar #Running sum so far based on observations



        if DEBUG_PROBABILITY_CALCULATIONS:
            print "\n ---------Getting Delta-R and Sum of Delta-R for new action--------"
        for goal in listOtherThenPruned:
            if DEBUG_PROBABILITY_CALCULATIONS:
                print "-------Checking Goal: ",goal,"----------\n",
                print " Delta R before Action" , tempDeltaREachGoal[goal]




            # Running sum per goal for all observarins
            # so far plus this action being evaluated
            qDifferenceForGivenAction =  self.getQdifference(goal, currentPosition, action)
            tempDeltaREachGoal[goal] = tempDeltaREachGoal[goal] + qDifferenceForGivenAction

            if DEBUG_PROBABILITY_CALCULATIONS:
                print " Q-Difference: " , qDifferenceForGivenAction
                print " Updated Delta R for goal: ",goal, " is: " ,tempDeltaREachGoal[goal]

        tempExpSumOfDeltaRs = 0
        if DEBUG_PROBABILITY_CALCULATIONS:
            print "\n----Now Calculating Sum Of Exponents of Delta R's of Eacg Goal------ "
        for goal in listOtherThenPruned:
            # Sum of deltaRs of each goal
            tempExpSumOfDeltaRs  = tempExpSumOfDeltaRs + math.exp(tempDeltaREachGoal[goal])
            if DEBUG_PROBABILITY_CALCULATIONS:
                print " Updated exponenets sum of delta R all goals: ", tempExpSumOfDeltaRs

        if DEBUG_PROBABILITY_CALCULATIONS:
            print "\n ---------Getting Probability R given o.a for new action--------"

        entropy = 0
        for goal in listOtherThenPruned:

            exponentOfDeltaRofGoal = math.exp(tempDeltaREachGoal[goal])
            probabilityRgivenODotAction[goal] = (exponentOfDeltaRofGoal/tempExpSumOfDeltaRs) * self.probabilityEachGoal[goal]
            logOfProbability = math.log(probabilityRgivenODotAction[goal], 2)
            entropyThisGoal = (-1 * probabilityRgivenODotAction[goal] * logOfProbability)
            entropy = entropy + entropyThisGoal

            if DEBUG_PROBABILITY_CALCULATIONS:
                print "----goal: ", goal, "------"
                print "  Delta - R is ", tempDeltaREachGoal[goal]
                print "  Sum of exponents of delta-r is: ", tempExpSumOfDeltaRs
                print "  Probability R given o.a for goal is ", probabilityRgivenODotAction[goal]
                print "  Entropy for this goal is: ",entropyThisGoal
                print "  Sum of Entropy of goals so far is: ", entropy
                #print "  Probability R given o.a is: ", probabilityRgivenODotAction


        return entropy


    # Based on Equation 4 of the
    # DRL papaer
    def getEntropyMaximizingAction(self):

        entropyOfActions = {}
        # Below for loop is computing deltaR for each goal
        # Calculation below is based on all the actions
        # that have been taken so far

        listOtherThenPruned = [item for item in self.all_goal_coords if item not in self.prunedGoalsThisStep]
        if DEBUG_PRUNING:
            print "Total Goals are: ", self.all_goal_coords
            print "Pruned Goals This Step are: ", self.prunedGoalsThisStep
            print "Goals being considered are: ", listOtherThenPruned

        # Get each action get the entropy
        # after adding this action to observations
        # so far
        position = self.currentPosition
        legitimateActions = self.getLigitimateActions(position)
        for item in legitimateActions:
                action = item[1]
                entropyOfActions[action] = self.getEntropyOfAction(self.currentPosition, action,listOtherThenPruned)
                #print "Entropy of Action from Calling MEthod: ", entropyOfActions

        # Get argmin(entropyOfActions)
        if DEBUG_DRL_EQUATIONS:
            print "\n\nAll Entropies Are: ", entropyOfActions

        bestAction = min(entropyOfActions, key=entropyOfActions.get)



        if DEBUG_DRL_EQUATIONS:
            print"Action Taken is: ", bestAction

        return bestAction

    def updateResidualRewardAndActionAdvantage(self):

        if DEBUG_PRUNING:
            print "\n--------------Updating Residual Reward and Getting Action Advantage --------------"

        if len(self.observationsSoFar) >= 2: #at-least two actions have been taken

            if len(self.all_goal_coords) > 1: # Atleast one bogus goal remaining



                for goal in self.possGoals:
                    qDashPosition = self.observationsSoFar[-1][0] # the last action in observation history
                    #print "qDashPosition", qDashPosition
                    qDashActionTaken = self.observationsSoFar[-1][1]
                    qPosition = self.observationsSoFar[-2][0]
                    #print "qPosition: ", qPosition
                    qActionTaken = self.observationsSoFar[-2][1]
                    qDashValue = self.getQValueOfAction(goal, qDashActionTaken, qDashPosition)
                    qValue = self.getQValueOfAction(goal, qActionTaken, qPosition)
                    totalResidualRewardThisGoal = qDashValue - qValue # @TODO Verify From Professor

                    #print "Total Residual Reward This Pair: ", totalResidualRewardThisGoal

                    if goal not in self.residualRewardEachGoal:
                        self.residualRewardEachGoal[goal] = totalResidualRewardThisGoal
                    else:
                        self.residualRewardEachGoal[goal] = self.residualRewardEachGoal[goal] + totalResidualRewardThisGoal

                    #print "Residual Reward Updated: ", self.residualRewardEachGoal[goal]

                    actionAdvantage = qDashValue - self.residualRewardEachGoal[goal]
                    #print "Check"
                    self.actionAdvantageEachGoal[goal] = actionAdvantage
                    #print "Action Advantage This Goal: ", self.actionAdvantageEachGoal[goal]

                    if DEBUG_PRUNING:
                        print "The goal is: ", goal
                        print "   Q-Dash position and Action is: ", qDashPosition, " ", qDashActionTaken
                        print "   Q-Dash value is: ",qDashValue
                        print "   Q position and Action is: ", qPosition, " ", qActionTaken
                        print "   Q value is: ", qValue
                        print "   Residual Reward This pair is: ", totalResidualRewardThisGoal
                        print "   Total Residual Reward is: ", self.residualRewardEachGoal[goal]
                        print "   Action advantage (based on Q-dash (last observation): ", self.actionAdvantageEachGoal[goal]


    def pruneGoalsThisStep(self):
        self.prunedGoalsThisStep = []
        for goal in self.possGoals:

            if self.actionAdvantageEachGoal[goal] < self.pruningParameter:
                if goal not in self.prunedGoalsThisStep:
                    self.prunedGoalsThisStep.append(goal)
                    if DEBUG_PRUNING:
                        print "    Goal ", goal," has been pruned for next step"



    def drlAmbiguityPath(self, currentPos, finalUndetected ):

        drlAmbiguityPath = [currentPos]
        if self.start == finalUndetected:

            if DEBUG_DRL_EQUATIONS == True:
                print "self.start = finalUndetected"
            return drlAmbiguityPath



        while currentPos!=finalUndetected:

            #print "\n======================Getting Entropy Maximizing Action=========================="

            #if len(self.prunedGoalsThisStep) == len(self.possGoals):
            if False:
                valueOfActions = {}

                ligitimateActions = self.getLigitimateActions(self.currentPosition)
                if DEBUG_ACTIONS:
                    print "================Legitimate Actions are: ====================="
                    print len(ligitimateActions), ": ", ligitimateActions

                for items in ligitimateActions:
                    coord = items[0]
                    action = items[1]
                    # print action

                    valueOfActions[action] = self.getRewardForAction(coord, self.realGoal, action)

                # Get argmin(entropyOfActions)
                if DEBUG_DRL_EQUATIONS:
                    print "\n\nAll Qvalues Are: ", valueOfActions

                bestAction = max(valueOfActions, key=valueOfActions.get)

            else:
                bestAction = self.getEntropyMaximizingAction()
                #print "\n=====================Entropy Maximizing Action Is: ", bestAction, "========================"
                self.observationsSoFar.append((self.currentPosition, bestAction))
                if DEBUG_PRUNING:
                    print "Observations so Far: ", self.observationsSoFar
                #print "\n========================Updating Variables=================================="


                if len(self.observationsSoFar) >= 2:
                    self.updateResidualRewardAndActionAdvantage()
                    self.pruneGoalsThisStep()

                # Update DeltaR For each goal
                # after the bestAction has been selected
                for goal in self.all_goal_coords:
                    self.deltaREachGoalSoFar[goal] = self.deltaREachGoalSoFar[goal] + self.getQdifference(goal, self.currentPosition, bestAction)




            # updating the action taken.
            # Corresponding values will be
            # updated during next iteration
            # of step_gen
            self.actualActionTaken = bestAction
            coordinatesAfterAction = self.convertActionToCoordinates(self.mapref, bestAction, currentPos)[0]

            print ("\n\n ====Best(Entropy Maximizing) Action Is: ",bestAction, " Coordinates After Action is: ", coordinatesAfterAction, " Final Position: ", finalUndetected,"====" )
            drlAmbiguityPath.append(coordinatesAfterAction)
            currentPos = coordinatesAfterAction
            self.currentPosition = currentPos
            #print "\n========================Variables Updated=================================="


        return drlAmbiguityPath



    def step_gen(self):


        print "running generator"
        all_goal_coords = [self.realGoal] + self.possGoals
        goal_obs = d.generateGoalObs(self.mapref, self.currentPosition, all_goal_coords)
        rmp, argmin = d.rmp(self.mapref, self.currentPosition, goal_obs)

        path1 = self.drlAmbiguityPath(self.currentPosition, self.realGoal)
        #path2 = self.mapref.optPath(argmin.coord, self.realGoal)
        #path = path1[1:] + path2[1:]
        #self.path2 = path2

        for step in path1:
            yield step






    def getFullPath(self, mapref, start, goal, poss_goals, heatmap):
        # returns cost and path
        all_goal_coords = [goal] + poss_goals
        goal_obs = d.generateGoalObs(mapref, start, all_goal_coords)
        rmp, argmin = d.rmp(mapref, start, goal_obs)
        cost1, path1 = mapref.optPath(start, argmin.coord, 2)
        cost2, path2 = mapref.optPath(argmin.coord, goal, 2)

        return cost1 + cost2, path1[1:] + path2[1:]

