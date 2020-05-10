import os.path
import math
import policy_gradient
from policy_gradient import P4Environemnt
import numpy as np
from collections import Counter




EPSILON = 0.00
ACTIONS = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
PP_DIR = "../drl/PP/"
BETA = 1
DECEPTIVE = False
SIMPLE_SMOOTH = True
SINGLE_POLICY = True
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
        self.simple_prune_real = float(0.0)
        self.simple_prune_fake = float(0.0)
        self.allGoals = [self.real_goal] + self.fake_goals

        if DEBUG:
            print self.fake_goals


        # Section of code below will either
        # load policy parameters for real
        # and fake goals, or train them
        # and save the results.
        def loadParamOrTrainPolicy(agent, goal, allGoals):
            goalParaFile = PP_DIR + map_file + "param_"+str(policy_gradient.NUMBER_PARAMETERS)+"_goal({:d}.{:d}).npy".format(goal[0], goal[1])

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

                print "loading pre-set parameters for goal: ", goal
                policy.parameters = policy_gradient.INITIAL_WEIGHTS

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

        if not SINGLE_POLICY:
            self.fakeGoalsPolicy = []
            for i, fg in enumerate(fake_goals):

                fakeGoalPolicy = loadParamOrTrainPolicy(self, fg, self.allGoals)
                self.fakeGoalsPolicy.append(fakeGoalPolicy)




        self.sum_q_diff = [0.0] * (len(fake_goals) + 1)
        self.d_set = set(range(len(fake_goals)))
        self.passed = set()
        self.closest = [0.0] * len(fake_goals)
        self.history = set()
        # show all q tables for debugging
        # self.real_q.printQtbl()
        # for fq in self.fake_q:
        #     fq.printQtbl()

    def fakeGoalElimination(self, current, m_idx):
        eli_set = set()
        for fg in self.d_set:
            fq = self.fakeGoalsPolicy[fg]
            nq = fq.qValue(current, m_idx)
            if DEBUG:
                print "check elimination:", fg, "nq:", nq
            if nq == policy_gradient.EPISODES:
                if DEBUG:
                    print "pass fake goal"
                self.passed.add(fg)
                eli_set.add(fg)
            elif self.diverge(fg, current, m_idx):
                eli_set.add(fg)
        self.d_set -= eli_set
        if DEBUG:
            print "remove:", eli_set, "remain:", self.d_set, "\n"

    def fakeGoalReconsideration(self, current, m_idx):
        for fg, fq in enumerate(self.fakeGoalsPolicy):
            if fg in self.d_set:
                continue
            if DEBUG:
                print "reconsidering", fg
            if fg in self.passed:
                if DEBUG:
                    print "\tpassed before"
                continue
            if fq.value(current) < self.closest[fg]:
                if DEBUG:
                    print "\tcloser before"
                continue
            if not self.diverge(fg, current, m_idx):
                if DEBUG:
                    print "add", fg, "back"
                self.d_set.add(fg)

    def diverge(self, fg, current, m_idx):
        act = ACTIONS[m_idx]
        state_p = (current[0] + act[0], current[1] + act[1])
        fq = self.fakeGoalsPolicy[fg]
        q = fq.value(current)
        nq = fq.value(state_p)
        if DEBUG:
            print "divergence test", fg, "action:", m_idx, "q:", q, "nq:", nq
        return nq < q

    def obsEvl(self, current):
        if DEBUG:
            print "\ncurrent: ", current
        x, y = current
        candidates = list()
        rqs = self.realGoalPolicy.value(current)
        for i, a in enumerate(ACTIONS):
            state_p = (x + a[0], y + a[1])
            if DEBUG:
                print "\n", current, "->", state_p, "action", i
            if not self.lmap.isPassable(state_p, current) or state_p in self.history:
                continue
            rnq = self.realGoalPolicy.value(state_p)
            rq = self.realGoalPolicy.qValue(current, i)
            if DEBUG:
                print "next+e: ", rnq * (1 + EPSILON), " qs: ", rqs
            if rnq * (1 + EPSILON) >= rqs:
                if DEBUG:
                    print "realg: q*: {:.3f}, q: {:.3f}, q_diff: {:.3f}, nq: {:.3f}, qd_ratio: {:.3f}" \
                        .format(rqs, rq, rqs - rq, rnq, (rqs - rq) / rqs)
                    for fg in self.d_set:
                        fq = self.fakeGoalsPolicy[fg]
                        qs = fq.value(current)
                        nq = fq.value(state_p)
                        q = fq.qValue(current, i)
                        print "fake{:d}: q*: {:.3f}, q: {:.3f}, q_diff: {:.3f}, nq: {:.3f}, qd_ratio: {:.3f}" \
                            .format(fg, qs, q, qs - q, nq, (qs - q) / qs)
                if len(self.d_set) == 0:
                    candidates.append((0, rq, i))
                else:
                    q_diffs = []
                    tmp_d_set = set()
                    for fg in self.d_set:
                        if PRUNE:
                            if self.diverge(fg, current, i):
                                continue
                        tmp_d_set.add(fg)
                        fq = self.fakeGoalsPolicy[fg]
                        qs = fq.value(current)
                        q = fq.qValue(current, i)
                        q_diffs.append(self.sum_q_diff[fg] + qs - q)
                    if DEBUG:
                        print "possible deceptive goals:", tmp_d_set
                    q_diffs.append(self.sum_q_diff[-1] + rqs - rq)
                    if DEBUG:
                        print "q_diffs:", q_diffs
                    sum_q_diffs = sum(q_diffs)
                    if sum_q_diffs > 0:
                        q_diffs = [qd / sum_q_diffs for qd in q_diffs]
                    if DEBUG:
                        print "norm q_diffs:", q_diffs
                    probs = [math.exp(-qd) for qd in q_diffs]
                    sum_probs = sum(probs)
                    if sum_probs > 0:
                        probs = [p / sum_probs for p in probs]
                    entropy = 0.0
                    for p in probs:
                        entropy += p * math.log(p, 2)
                    candidates.append((-entropy, rq, i))
                    if DEBUG:
                        print "probs: {}\nentropy: {:.5f}\n".format(str(probs), -entropy)
            else:
                if DEBUG:
                    print "action diverges from real goal"
        candidates = sorted(candidates, reverse=True)
        if DEBUG:
            print str(candidates), "\n"
        a_idx = candidates[0][2]
        if DEBUG:
            print "action selected:", a_idx
        if PRUNE:
            # eliminate fake goal
            self.fakeGoalElimination(current, a_idx)
            # and reconsider
            self.fakeGoalReconsideration(current, a_idx)
        # update sum q_diff
        rq = self.realGoalPolicy.qValue(current, a_idx)
        self.sum_q_diff[-1] += rqs - rq
        for fg, fq in enumerate(self.fakeGoalsPolicy):
            qs = fq.value(current)
            q = fq.qValue(current, a_idx)
            self.sum_q_diff[fg] += qs - q
        if DEBUG:
            print "sum q diff:", self.sum_q_diff, "\n"
        move = (x + ACTIONS[a_idx][0], y + ACTIONS[a_idx][1])
        # update closest point
        for fg, fq in enumerate(self.fakeGoalsPolicy):
            nq = fq.value(move)
            self.closest[fg] = max(self.closest[fg], nq)
        # update history
        self.history.add(move)
        return move

    def entropyMaximizingAction(self, current):


        allPolicies = np.array([self.realGoalPolicy] + self.fakeGoalsPolicy)

        allGoals = np.array([self.real_goal] + self.fake_goals)
        number_of_goals = 2 # 1 real and 1 least entropy
        number_of_fake_goals = number_of_goals - 1

        realPolicy = self.realGoalPolicy

        envReal = realPolicy.env

        alpha = 0.15

        # below three values are one time values
        # that will bge added and subtracted
        # from probabilities of real and fake goals
        simple_prune = alpha / number_of_goals
        simple_prune_fake = simple_prune  # this value will be decreased from each fake goal

        simple_prune_real = number_of_fake_goals * simple_prune  # this value will be added to real goal
        entropy_action_prob = 1.0
        best_actionprob = 0.0
        stageTwoProbs = [best_actionprob ,entropy_action_prob ]




        stageTwoProbs[0] += self.simple_prune_real

        stageTwoProbs[1] += self.simple_prune_fake   # add because 2nd term is -ve

        #assert alpha < stageTwoProbs[1]

        reChoose = True  # do-while

        while reChoose:  # unless a legimite action is obtained, keep trying

            # A 2-D array with actions in columns
            # and policy wise probs in rows
            policy_wise_actions_probs = []
            for policy in allPolicies:
                 actions_probs = policy.getAllActionsProbabilities(current)
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

            max_entropy = np.argmax(action_entropies)
            #other = np.argpartition(action_entropies, -2)[-2] # 2nd largest entropy
            other = realPolicy.getHighestProbabilityActionIndex(current)


            firstStageActions = [other, max_entropy]

            #actionTakenIndex = np.argmax(action_entropies)

            actionTakenIndex = np.random.choice(firstStageActions, 1, p=stageTwoProbs)[0]

            # all policies have same action space
            # so no matter whose actions is chosen
            action = envReal.actions[actionTakenIndex]

            next = (current[0] + action[0], current[1] + action[1])
            status = envReal.getNewStateStatus(next)

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
                    (self.simple_prune_real - simple_prune_real) > 0:  # make sure probability does not increase above 1
                self.simple_prune_real -= simple_prune_real
                self.simple_prune_fake += simple_prune_fake

        if SIMPLE_SMOOTH:
            # @TODO also capture behaviour without this
            confused = next in self.history
            total_fake_goal_prob = np.sum(stageTwoProbs[1:])
            total_fake_goal_pruning = number_of_fake_goals * simple_prune_fake
            if confused and \
                    (self.simple_prune_real + simple_prune_real) < 1: #(total_fake_goal_prob - total_fake_goal_pruning):
                self.simple_prune_real += float(simple_prune_real)
                self.simple_prune_fake -= float(simple_prune_fake)

        # update requires policy and env variables
        # this is because getHighest, getStochastic and
        # all other policy methods work on there variables
        # @TODO Update others if needed
        for policy in allPolicies:
            env = policy.getPolicyEnvironment()
            env.current = next

        # terminal, newState = self.realGoalPolicy.env.takeActions(bestActionIndex)
        self.history.add(next)
        return next

    def honest(self, current):



        policy = self.realGoalPolicy
        env = policy.getPolicyEnvironment()

        bestActionIndex = policy.getHighestProbabilityActionIndex(current)
        #bestActionIndex = policy.getStochasticActionIndex(current)

        actionTakenIndex = bestActionIndex

        action = env.actions[actionTakenIndex]

        next = (current[0] + action[0], current[1] + action[1])

        # update requires policy and env variables
        # this is because getHighest, getStochastic and
        # all other policy methods work on there variables
        #@TODO Update others if needed
        #env.current = next # highest probability method calculates on env.current
        env.takeAction(actionTakenIndex)


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
        allPolicies = np.array([self.realGoalPolicy] + self.fakeGoalsPolicy)

        allGoals = np.array([self.real_goal] + self.fake_goals)
        number_of_goals = len(allGoals)
        number_of_fake_goals = number_of_goals - 1


        realPolicy = self.realGoalPolicy
        fakePolicy = self.fakeGoalsPolicy[0]
        envReal = realPolicy.env


        alpha= 0.1

        
        # below three values are one time values
        # that will bge added and subtracted
        # from probabilities of real and fake goals
        simple_prune = alpha/number_of_goals
        simple_prune_fake = simple_prune # this value will be decreased from each fake goal

        simple_prune_real = number_of_fake_goals * simple_prune # this value will be added to real goal

        stageTwoProbs = np.zeros(number_of_goals)



        for prob in range(0, number_of_goals):
            stageTwoProbs[prob] = 1.0 / number_of_goals
            if prob == 0:  # Real Goal
                stageTwoProbs[prob] += self.simple_prune_real
            else:  # for fake goals
                stageTwoProbs[prob] += self.simple_prune_fake  # add because 2nd term is -ve



        reChoose = True # do-while

        while reChoose: #unless a legimite action is obtained, keep trying

            firstStageActions = []
            for policy in allPolicies:
                bestAction = policy.getHighestProbabilityActionIndex(current)
                firstStageActions.append(bestAction)


            actionTakenIndex = np.random.choice(firstStageActions, 1, p=stageTwoProbs)[0]


            # all policies have same action space
            # so no matter whose actions is chosen
            action = envReal.actions[actionTakenIndex]

            next = (current[0] + action[0], current[1] + action[1])
            status = envReal.getNewStateStatus(next)
            
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
                (self.simple_prune_real - simple_prune_real) > simple_prune_real: #make sure probability does not decrease below 0
                self.simple_prune_real -= simple_prune_real
                self.simple_prune_fake += simple_prune_fake



        if SIMPLE_SMOOTH:
            # @TODO also capture behaviour without this
            confused = next in self.history
            total_fake_goal_prob = np.sum(stageTwoProbs[1:])
            total_fake_goal_pruning = number_of_fake_goals * simple_prune_fake
            if confused and (self.simple_prune_real + simple_prune_real) < (1 - simple_prune_real):
                self.simple_prune_real += simple_prune_real
                self.simple_prune_fake -= simple_prune_fake


        # update requires policy and env variables
        # this is because getHighest, getStochastic and
        # all other policy methods work on there variables
        #@TODO Update others if needed
        for policy in allPolicies:
            env = policy.getPolicyEnvironment()
            env.current = next



        #terminal, newState = self.realGoalPolicy.env.takeActions(bestActionIndex)
        self.history.add(next)
        return next


    def getNext(self, mapref, current, goal, timeremaining=100):
        if DECEPTIVE and not SINGLE_POLICY:
            #move = self.stochastic2(current)
            #move = self.entropyMaximizingAction(current)
            move = self.obsEvl(current)
        else:
            move = self.honest(current)
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
        self.simple_prune_real = 0
        self.simple_prune_fake = 0
        self.realGoalPolicy.env.current = self.startPosition

        if not SINGLE_POLICY:
            for goalPolicy in self.fakeGoalsPolicy:
                goalPolicy.env.current = self.startPosition

