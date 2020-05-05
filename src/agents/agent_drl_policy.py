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
DECEPTIVE = True
SIMPLE_SMOOTH = True
PRUNE = True
DEBUG = True


class Agent(object):
    def __init__(self, lmap, real_goal, fake_goals, map_file, start):
        print("Loading Agent")
        self.lmap = lmap
        self.real_goal = real_goal
        self.fake_goals = fake_goals
        self.startPosition = start # used for policy training
        self.simple_prune = 0

        if DEBUG:
            print self.fake_goals


        # Section of code below will either
        # load policy parameters for real
        # and fake goals, or train them
        # and save the results.
        def loadParamOrTrainPolicy(agent, goal):
            goalParaFile = PP_DIR + map_file + "_goal({:d}.{:d}).npy".format(goal[0], goal[1])

            # a separte environment file is created
            # so that associated functions can be
            # added to it
            env = P4Environemnt(agent.lmap, agent.startPosition, goal)

            # In policy variable below:
            # Can also set alpha and gamma value
            policy = policy_gradient.LinearPolicy(env)

            fileFound = os.path.isfile(goalParaFile)
            if fileFound:
                if DEBUG:
                    print "loading parameters for goal: ", goal
                policy.loadParameters(goalParaFile)
            else:
                if DEBUG:
                    print "training parameters for goal: ", goal

                policy_gradient.trainPolicy(policy) # policy and training parameters defined in policy_gradient
                policy.saveParameters(goalParaFile)

            return policy

        self.realGoalPolicy = loadParamOrTrainPolicy(self, self.real_goal)
        self.fakeGoalsPolicy = []
        for i, fg in enumerate(fake_goals):

            fakeGoalPolicy = loadParamOrTrainPolicy(self, fg)
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
        rqs = self.policy.value(current)
        for i, a in enumerate(ACTIONS):
            state_p = (x + a[0], y + a[1])
            if DEBUG:
                print "\n", current, "->", state_p, "action", i
            if not self.lmap.isPassable(state_p, current) or state_p in self.history:
                continue
            rnq = self.policy.value(state_p)
            rq = self.policy.qValue(current, i)
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
        rq = self.policy.qValue(current, a_idx)
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

    def honest(self, current):

        policy = self.realGoalPolicy
        env = policy.getPolicyEnvironment()

        bestActionIndex = policy.getHighestProbabilityActionIndex()

        actionTakenIndex = bestActionIndex

        action = env.actions[actionTakenIndex]

        next = (current[0] + action[0], current[1] + action[1])

        # update requires policy and env variables
        # this is because getHighest, getStochastic and
        # all other policy methods work on there variables
        #@TODO Update others if needed
        env.current = next # highesy probability method calculates on env.current


        #terminal, newState = self.realGoalPolicy.env.takeActions(bestActionIndex)
        self.history.add(next)

        return next

    def stochastic1(self, current):
        # idea is to choose stochastic action from both real
        # and fake policy. As agent become confused
        # probabilty of besr action is increased at the
        # expense of fake stochastic. As the confusion reduces
        # probability of fake stochastic fake is increased.
        # A limit is imposed on increase in probability
        # of fake stochastic after confusion in secing if

        # illiegitimate steps can be
        # returned based on policy
        # probability

        realPolicy = self.realGoalPolicy
        fakePolicy = self.fakeGoalsPolicy[0]
        envReal = realPolicy.env
        envFake = fakePolicy.env
        realStoProb = 0.5
        realBestProb = 0.0
        fakeStoProb = 0.5
        #next = current

        simple_prune_increment = 0.02

        perform = True # do-while

        while perform: #unless a legimite action is obtained, keep trying
            realBestIndex = realPolicy.getHighestProbabilityActionIndex()
            realStochasticIndex =  realPolicy.getStochasticActionIndex()
            fakeStochasticIndex =  fakePolicy.getStochasticActionIndex()

            possibleActions = [realStochasticIndex, fakeStochasticIndex, realBestIndex]
            #selectionProbability = [realStoProb + simple_prune, fakeStoProb - simple_prune]
            selectionProbability = np.array([realStoProb , fakeStoProb - self.simple_prune, realBestProb + self.simple_prune])
            sumOfProbability = np.sum(selectionProbability)

            actionTakenIndex = np.random.choice(possibleActions, 1, p=selectionProbability)[0]
            #actionTakenIndex = stochasticActionIndex

            # all policies have same action space
            # so no matter whose actions is chosen
            action = envReal.actions[actionTakenIndex]

            next = (current[0] + action[0], current[1] + action[1])
            status = envReal.getStateStatus(next)
            perform = not status == 'step' #reslect stochastic action if next state is not legitimate
            if perform == False and \
                self.simple_prune > 4*simple_prune_increment and \
                next not in self.history:
                self.simple_prune -= simple_prune_increment


        if SIMPLE_SMOOTH:
            # @TODO also capture behaviour without this
            if next in self.history and self.simple_prune < (fakeStoProb - simple_prune_increment):
                self.simple_prune += simple_prune_increment


        # update requires policy and env variables
        # this is because getHighest, getStochastic and
        # all other policy methods work on there variables
        #@TODO Update others if needed
        envReal.current = next
        envFake.current = next


        #terminal, newState = self.realGoalPolicy.env.takeActions(bestActionIndex)
        self.history.add(next)
        return next

    def stochastic2(self, current):

        # all comments of stochastic 1 True,
        # except that best action is selected rather
        # than stochastic

        realPolicy = self.realGoalPolicy
        fakePolicy = self.fakeGoalsPolicy[0]
        envReal = realPolicy.env
        envFake = fakePolicy.env
        realBestProb = 0.5
        realStoProb = 0.0
        fakeBestProb = 0.5
        #next = current

        simple_prune_increment = 0.02

        perform = True # do-while

        while perform: #unless a legimite action is obtained, keep trying
            realStoIndex = realPolicy.getStochasticActionIndex()
            realBestIndex =  realPolicy.getHighestProbabilityActionIndex()
            fakeBestIndex =  fakePolicy.getHighestProbabilityActionIndex()

            possibleActions = [realBestIndex, fakeBestIndex, realStoIndex]

            selectionProbability = np.array([realBestProb , fakeBestProb - self.simple_prune, realStoProb + self.simple_prune])


            actionTakenIndex = np.random.choice(possibleActions, 1, p=selectionProbability)[0]


            # all policies have same action space
            # so no matter whose actions is chosen
            action = envReal.actions[actionTakenIndex]

            next = (current[0] + action[0], current[1] + action[1])
            status = envReal.getStateStatus(next)
            perform = not status == 'step' #reslect stochastic action if next state is not legitimate
            if perform == False and \
                self.simple_prune > 2*simple_prune_increment and \
                next not in self.history:
                self.simple_prune -= simple_prune_increment


        if SIMPLE_SMOOTH:
            # @TODO also capture behaviour without this
            if next in self.history and self.simple_prune < (fakeBestProb - simple_prune_increment):
                self.simple_prune += simple_prune_increment


        # update requires policy and env variables
        # this is because getHighest, getStochastic and
        # all other policy methods work on there variables
        #@TODO Update others if needed
        envReal.current = next
        envFake.current = next


        #terminal, newState = self.realGoalPolicy.env.takeActions(bestActionIndex)
        self.history.add(next)
        return next


    def getNext(self, mapref, current, goal, timeremaining=100):
        if DECEPTIVE:
            move = self.stochastic2(current)
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
        self.simple_prune = 0
        self.realGoalPolicy.env.current = self.startPosition
        for goalPolicy in self.fakeGoalsPolicy:
            goalPolicy.env.current = self.startPosition

