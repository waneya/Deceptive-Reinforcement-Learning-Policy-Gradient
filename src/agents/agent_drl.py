import os.path
import math
import qFunction
from collections import Counter


ALPHA = 0.2
GAMMA = 1

TERM_V = 10000.0
EPSILON = 0.00
ACTIONS = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
Q_DIR = "../drl/qfunc/"
BETA = 1
DECEPTIVE = False
PRUNE = True
DEBUG = True


class Agent(object):
    def __init__(self, lmap, real_goal, fake_goals, map_file):
        self.lmap = lmap
        self.real_goal = real_goal
        self.fake_goals = fake_goals
        if DEBUG:
            print self.fake_goals
        real_q_file = Q_DIR + map_file + ".{:d}.{:d}.q".format(real_goal[0], real_goal[1])
        self.real_q = qFunction.QFunction(lmap.width, lmap.height)
        if os.path.isfile(real_q_file):
            if DEBUG:
                print "loading q function for", real_goal
            self.real_q.load(real_q_file)
        else:
            if DEBUG:
                print "training q function for", real_goal
            qFunction.train(self.real_q, lmap, real_goal, TERM_V, GAMMA)
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
                qFunction.train(fq, lmap, fg, TERM_V, GAMMA)
                fq.save(fake_q_file)
            self.fake_q.append(fq)
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
            fq = self.fake_q[fg]
            nq = fq.qValue(current, m_idx)
            if DEBUG:
                print "check elimination:", fg, "nq:", nq
            if nq == TERM_V:
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
        for fg, fq in enumerate(self.fake_q):
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
        fq = self.fake_q[fg]
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
        rqs = self.real_q.value(current)
        for i, a in enumerate(ACTIONS):
            state_p = (x + a[0], y + a[1])
            if DEBUG:
                print "\n", current, "->", state_p, "action", i
            if not self.lmap.isPassable(state_p, current) or state_p in self.history:
                continue
            rnq = self.real_q.value(state_p)
            rq = self.real_q.qValue(current, i)
            if DEBUG:
                print "next+e: ", rnq * (1 + EPSILON), " qs: ", rqs
            if rnq * (1 + EPSILON) >= rqs:
                if DEBUG:
                    print "realg: q*: {:.3f}, q: {:.3f}, q_diff: {:.3f}, nq: {:.3f}, qd_ratio: {:.3f}" \
                        .format(rqs, rq, rqs - rq, rnq, (rqs - rq) / rqs)
                    for fg in self.d_set:
                        fq = self.fake_q[fg]
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
                        fq = self.fake_q[fg]
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
        rq = self.real_q.qValue(current, a_idx)
        self.sum_q_diff[-1] += rqs - rq
        for fg, fq in enumerate(self.fake_q):
            qs = fq.value(current)
            q = fq.qValue(current, a_idx)
            self.sum_q_diff[fg] += qs - q
        if DEBUG:
            print "sum q diff:", self.sum_q_diff, "\n"
        move = (x + ACTIONS[a_idx][0], y + ACTIONS[a_idx][1])
        # update closest point
        for fg, fq in enumerate(self.fake_q):
            nq = fq.value(move)
            self.closest[fg] = max(self.closest[fg], nq)
        # update history
        self.history.add(move)
        return move

    def honest(self, current):
        if DEBUG:
            print "\ncurrent: ", current
        x, y = current
        candidates = Counter()
        for i, a in enumerate(ACTIONS):
            state_p = (x + a[0], y + a[1])
            if DEBUG:
                print "\n", current, "->", state_p, "action", i
            if not self.lmap.isPassable(state_p, current) or state_p in self.history:
                continue
            q = self.real_q.qValue(current, i)
            if DEBUG:
                print "Q value:", q
            candidates[state_p] = q
        move = candidates.most_common()[0][0]
        if DEBUG:
            print "candidates:", candidates, "\nmove:", move
        # update history
        self.history.add(move)
        return move

    def getNext(self, mapref, current, goal, timeremaining=100):
        if DECEPTIVE:
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

