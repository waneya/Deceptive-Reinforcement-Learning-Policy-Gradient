import os.path
import math
import policy_gradient
from policy_gradient import P4Environemnt
import numpy as np
from collections import Counter




EPSILON = 0.00
ACTIONS = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
ROOT = "G:\\Semester 1 2020\\COMP90055\\DRL-Policy-Gradient"
PP_DIR = os.path.join(ROOT,'drl','PP')

BETA = 1
DECEPTIVE = True
USE_SINGLE_POLICY = True

SIMPLE_SMOOTH = True
PRUNE = False
DEBUG = True

USE_PRE_SET_PARAMETERS = True



class Agent(object):
    def __init__(self, lmap, real_goal, fake_goals, map_file, start):
        print("Loading Agent")
        self.lmap = lmap
        self.real_goal = real_goal
        self.fake_goals = fake_goals
        self.startPosition = start # used for policy training
        self.simple_prune_increment_param = float(0.0)
        self.simple_prune_decrement_param = float(0.0)
        self.allGoals = [self.real_goal] + self.fake_goals

        if DEBUG:
            print self.fake_goals


        # Section of code below will either
        # load policy parameters for real
        # and fake goals, or train them
        # and save the results.
        def loadParamOrTrainPolicy(agent, goal, allGoals):

            if USE_SINGLE_POLICY:
                goalParaFile = PP_DIR + map_file + "param_singlepolicy_" + str(len(policy_gradient.INITIAL_WEIGHTS_SINGLE_POLICY)) + "_goal({:d}.{:d}).npy".format(goal[0], goal[1])
            else:
                goalParaFile = PP_DIR + map_file + "param_multipolicy_" + str(len(policy_gradient.INITIAL_WEIGHTS_SINGLE_POLICY)) + "_goal({:d}.{:d}).npy".format(goal[0], goal[1])

            # a separte environment file is created
            # so that associated functions can be
            # added to it
            env = P4Environemnt(agent.lmap, agent.startPosition, goal, allGoals)

            # In policy variable below:
            # Can also set alpha and gamma value
            policy = policy_gradient.LinearPolicy(env)
            #policy.parameters = policy_gradient.INITIAL_WEIGHTS
            #notHasParameter = #len(policy.parameters) < 1

            fileFound = os.path.isfile(goalParaFile)
            if USE_PRE_SET_PARAMETERS:

                print "Using pre-set parameters for goal: ", goal
                if USE_SINGLE_POLICY:
                    policy.parameters = policy_gradient.INITIAL_WEIGHTS_SINGLE_POLICY
                else:
                    policy.parameters = policy_gradient.INITIAL_WEIGHTS_MULTIPLE_POLICIES

            elif fileFound :

                policy.loadParameters(goalParaFile)
                print "loading parameters from filefor goal: ", goal

            else:

                print "training parameters for goal: ", goal
                policy_gradient.trainPolicy(policy) # policy and training parameters defined in policy_gradient
                policy.saveParameters(goalParaFile)

            print policy.parameters
            return policy

        self.realGoalPolicy = loadParamOrTrainPolicy(self, self.real_goal,self.allGoals)

        if not USE_SINGLE_POLICY:
            self.fakeGoalsPolicy = []
            for i, fg in enumerate(fake_goals):

                fakeGoalPolicy = loadParamOrTrainPolicy(self, fg,self.allGoals)
                self.fakeGoalsPolicy.append(fakeGoalPolicy)

        # @TODO after this point first work on the TODO of policy_gradient and then work after wards


        self.sum_q_diff = [0.0] * (len(fake_goals) + 1)
        self.d_set = set(range(len(fake_goals)))
        self.passed = set()
        self.closest = [0.0] * len(fake_goals)
        self.history = set()
        # show all q tables for debugging
        # self.real_q.printQtbl()
        # for fq in self.fake_q:
        #     fq.printQtbl()



    def entropyMaximizingAction(self, current):
        # This part is currently under development
        # It requires Debugging
        # 6th July 2020

        if not USE_SINGLE_POLICY:

            #only possible with multiple policies

            # here we consider fake as best action for real goal
            # and real as the entropy maximizing action

            allPolicies = np.array([self.realGoalPolicy] + self.fakeGoalsPolicy)

            allGoals = np.array([self.real_goal] + self.fake_goals)
            number_of_goals = 2 # 1 real and 1 least entropy
            number_of_fake_goals = number_of_goals - 1

            realPolicy = self.realGoalPolicy

            envReal = realPolicy.env

            alpha = 0.15

            # below three values are one time values
            # that will bge added and subtracted
            # from probabilities
            simple_prune = alpha / number_of_goals
            simple_prune_entropy = simple_prune  # this value will be decreased from max_entropy

            simple_prune_best = number_of_fake_goals * simple_prune  # this value will be added to real goal
            entropy_action_prob = 1.0
            best_actionprob = 0.0
            stageTwoProbs = [entropy_action_prob ,best_actionprob]

            stageTwoProbs[0] += self.simple_prune_decrement_param

            stageTwoProbs[1] += self.simple_prune_increment_param  # add because 2nd term is -ve

            #assert alpha < stageTwoProbs[1]

            reChoose = True  # do-while

            while reChoose:  # unless a legimite action is obtained, keep trying

                # A 2-D array with actions in columns
                # and policy wise probs in rows
                policy_wise_actions_probs = []
                for policy in allPolicies:
                     actions_probs = policy.getAllActionsProbabilities()
                     policy_wise_actions_probs.append(actions_probs)

                # A 2-D numpy array with action in rows
                # and policy wise probs in columns
                actions_wise_policy_probs = np.array(policy_wise_actions_probs).T

                action_entropies = []
                for action_probs in actions_wise_policy_probs:
                    information_gain = 0.0
                    for prob in action_probs:
                        information_gain += prob * math.log(prob, 2)
                    entropy = -1.0 * information_gain
                    action_entropies.append(entropy)

                # A 1-D numpy array having entropies
                # of all actions
                action_entropies = np.array(action_entropies)

                max_entropy_index = np.argmax(action_entropies)
                #real_best_index = np.argpartition(action_entropies, -2)[-2] # 2nd largest entropy
                real_best_index = realPolicy.getHighestProbabilityActionIndex()


                firstStageActions = [max_entropy_index, real_best_index]

                #actionTakenIndex = np.argmax(action_entropies)

                actionTakenIndex = np.random.choice(firstStageActions, 1, p=stageTwoProbs)[0]

                # all policies have same action space
                # so no matter whose actions is chosen
                action = envReal.actions[actionTakenIndex]

                next = (current[0] + action[0], current[1] + action[1])
                status,cos = envReal.getNewStateStatus(next)

                # Code below will rechoose stage one stochastic
                # action if returned action is not legitimate
                reChoose = not status == 'step'

                # If confusion of the agents end
                # (i.e. it does not keep
                # moving around observed states)
                # make it deceptive by decresing real
                # goal prob to initials and fake goal
                # to initials
                confused = next in self.history
                if not reChoose and \
                        not confused and \
                        (self.simple_prune_increment_param - simple_prune_best) > 0:  # make sure probability does not increase above 1
                    self.simple_prune_increment_param -= simple_prune_best
                    self.simple_prune_decrement_param += simple_prune_entropy

            if SIMPLE_SMOOTH:
                # @TODO also capture behaviour without this
                confused = next in self.history
                total_fake_goal_prob = np.sum(stageTwoProbs[1:])
                total_fake_goal_pruning = number_of_fake_goals * simple_prune_entropy
                if confused and \
                        (self.simple_prune_increment_param + simple_prune_best) < 1: #(total_fake_goal_prob - total_fake_goal_pruning):
                    self.simple_prune_increment_param += float(simple_prune_best)
                    self.simple_prune_decrement_param -= float(simple_prune_entropy)

            # update requires policy and env variables
            # this is because getHighest, getStochastic and
            # all real_best_index policy methods work on there variables
            # @TODO Update others if needed
            for policy in allPolicies:
                env = policy.getPolicyEnvironment()
                env.current = next

            # terminal, newState = self.realGoalPolicy.env.takeActions(bestActionIndex)
            self.history.add(next)
            return next

    def getBestAction(self,policy,current):

        env = policy.getPolicyEnvironment()
        state = current
        closeness = []
        for act_index in range(len(env.actions)):
            close = env.getClosenessToGoalFeature(state, act_index)
            closeness.append(close)
        closeness = np.array(closeness)
        bestActionIndex = np.argmax(closeness)

        return bestActionIndex

    def honest(self, current):


        policy = self.realGoalPolicy
        env = policy.getPolicyEnvironment()
        env.current = current

        if USE_SINGLE_POLICY:

            bestActionIndex = self.getBestAction(policy, current)

        else:
            bestActionIndex = policy.getHighestProbabilityActionIndex(current)


        actionTakenIndex = bestActionIndex

        action = env.actions[actionTakenIndex]

        next = (current[0] + action[0], current[1] + action[1])

        # update requires policy and env variables
        # this is because getHighest, getStochastic and
        # all other policy methods work on there variables
        #@TODO Update others if needed
        #env.current = next # highest probability method calculates on env.current
        env.takeAction(actionTakenIndex) #updates current as well


        #terminal, newState = self.realGoalPolicy.env.takeActions(bestActionIndex)
        self.history.add(next)

        return next



    def irrationalAgent(self, current):

        # idea is to choose best action from both real
        # and fake policy. As agent become confused
        # probabilty of besr action is increased at the
        # expense of fake stochastic. As the confusion reduces
        # probability of fake stochastic fake is increased.
        # A limit is imposed on increase in probability
        # of fake stochastic after confusion in secing if
        allGoals = np.array([self.real_goal] + self.fake_goals)
        if USE_SINGLE_POLICY:
            allPolicies = np.array([self.realGoalPolicy])
            number_of_choices = 2 # move towards real or diverge

        else:
            number_of_choices = len(allGoals)
            allPolicies = np.array([self.realGoalPolicy] + self.fakeGoalsPolicy)


            #number_of_goals = number_of_choices
        assert number_of_choices > 1
        choices_other_real = number_of_choices - 1 # number of fake goals
        realPolicy = self.realGoalPolicy
        envReal = realPolicy.env

        if USE_SINGLE_POLICY:
            default_prob_real = 0.05
            default_prob_others = 0.95
            delta_prob = 0.4* default_prob_others
            maintain_at = 0.55
            retantion = maintain_at - default_prob_real
        else:

            default_prob_real = 1.0/number_of_choices
            default_prob_others = (1.0/number_of_choices) * choices_other_real
            delta_prob= 0.15 * default_prob_real
            maintain_at =1.25 * default_prob_real
            retantion  = maintain_at - default_prob_real



        # below three values are one time values
        # that will bge added and subtracted
        # from probabilities of real and fake goals
        simple_prune = delta_prob/number_of_choices
        simple_prune_fake = simple_prune # this value will be decreased from each fake goal
        simple_prune_real = choices_other_real * simple_prune # this value will be added to real goal

        stageTwoProbs = np.zeros(number_of_choices) # use for pruning

        #update the probs as per environment situation

        if USE_SINGLE_POLICY:
            stageTwoProbs[0] = default_prob_real + self.simple_prune_increment_param
            stageTwoProbs[1] = default_prob_others + self.simple_prune_decrement_param
        else:
            for prob in range(0, number_of_choices):
                    stageTwoProbs[prob] = default_prob_real
                    if prob == 0:  # Real Goal
                        stageTwoProbs[prob] += self.simple_prune_increment_param
                    else:  # for fake goals
                        stageTwoProbs[prob] += self.simple_prune_decrement_param  # add because 2nd term is -ve



        reChoose = True # do-while

        while reChoose: #unless a legimite action is obtained, keep trying

            firstStageActions = []

            if USE_SINGLE_POLICY:
                closest_index = self.getBestAction(realPolicy, current)
                realPolicy.env.current = current
                policy_index = realPolicy.getHighestProbabilityActionIndex(current)
                firstStageActions = [closest_index, policy_index]
            else:
                for policy in allPolicies:
                    policy.env.current = current
                    bestAction = policy.getHighestProbabilityActionIndex(current)
                    firstStageActions.append(bestAction)


            actionTakenIndex = np.random.choice(firstStageActions, 1, p=stageTwoProbs)[0]


            # all policies have same action space
            # so no matter whose actions is chosen
            action = envReal.actions[actionTakenIndex]

            next = (current[0] + action[0], current[1] + action[1])
            status,cos = envReal.getNewStateStatus(next)

            # Code below will rechoose stage one stochastic
            # action if returned action is not legitimate
            reChoose = not status == 'step'

                # If confusion of the agents end
                # (i.e. it does not keep
                # moving around observed states)
                # make it deceptive by decresing real
                # goal prob to initials and fake goal
                # to initials
            confused = next in self.history #@TODO try to optimize this. it causes excessive movement
            margin = 2* simple_prune_real
            #probability_not_decrease_zero = (self.simple_prune_increment_param - simple_prune_real) > (margin + default_prob_real)
            probability_not_decrease_zero = (stageTwoProbs[0] - simple_prune_real) >= (default_prob_real) + retantion
            if not reChoose and \
                    not confused and \
                    probability_not_decrease_zero: #make sure probability does not decrease below 0
                    self.simple_prune_increment_param -= simple_prune_real
                    self.simple_prune_decrement_param += simple_prune_fake



        if SIMPLE_SMOOTH:


                #probability_not_exceed_one = (self.simple_prune_increment_param + default_prob_real + simple_prune_real) < (1 - margin)
                probability_not_exceed_one = (stageTwoProbs[0] + simple_prune_real) <= (1 - margin)
                total_fake_goal_prob = np.sum(stageTwoProbs[1:])
                total_fake_goal_pruning = choices_other_real * simple_prune_fake
                if confused and probability_not_exceed_one:
                    self.simple_prune_increment_param += simple_prune_real
                    self.simple_prune_decrement_param -= simple_prune_fake


        # update requires policy and env variables
        # this is because getHighest, getStochastic and
        # all other policy methods work on there variables
        #@TODO Update others if needed
        for policy in allPolicies:
            env = policy.getPolicyEnvironment()
            env.takeAction(actionTakenIndex)


        #terminal, newState = self.realGoalPolicy.env.takeActions(bestActionIndex)
        self.history.add(next)
        return next


    def getNext(self, mapref, current, goal, timeremaining=100):
        if DECEPTIVE:
            move = self.irrationalAgent(current)
            #move = self.entropyMaximizingAction(current) # this agent is not working
            #move = self.obsEvl(current)
            print move
        else:
            move = self.honest(current)
            print move
            # honest is honest with Multiple policies
            # with single policy, honest include divergence feature


        return move

    def getPath(self, mapref, start, goal):
        path = [start]
        current = start
        while current != goal:
            move = self.getNext(mapref, current, goal)
            path.append(move)
            current = move
        return path

    def reset(self, **kwargs):
        self.sum_q_diff = [0.0] * (len(self.fake_goals) + 1)
        self.d_set = set(range(len(self.fake_goals)))
        self.passed = set()
        self.closest = [0.0] * len(self.fake_goals)
        self.history = set()
        self.simple_prune_increment_param = 0
        self.simple_prune_decrement_param = 0

        if USE_SINGLE_POLICY:
            allPolicies = [self.realGoalPolicy]

        else:
            allPolicies = [self.realGoalPolicy] + self.fakeGoalsPolicy

        for goalPolicy in allPolicies:
            goalPolicy.env.current = self.startPosition
            goalPolicy.env.useOptimumCost = False
            goalPolicy.env.history = []
            goalPolicy.env.getClosenessToAllGoalsValue = np.full( (len(goalPolicy.env.allGoals),len(goalPolicy.env.actions) ),np.inf)

