# coding=utf-8
# Part of the code has been
# referenced from https://github.com/jklaise/personal_website/blob/master/notebooks/rl_policy_gradients.ipynb
# and modified to fit the LinearRegression
# and multiple actions requirement
# of p4

import numpy as np
from scipy.special import softmax
import math
ACTIONS = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
NUMBER_PARAMETERS = 1
ALPHA = 0.1
GAMMA = 0.9
GLOBAL_SEED = 0
EPISODES = 50
SUMMED_PARAMETER_UPDATE = False
P4_BASED_LOSS = True


class P4Environemnt:

    def __init__(self, lmap, start, goal):

        self.lmap = lmap
        self.start =start
        self.goal = goal
        self.current = start
        self.actions = ACTIONS

    def getStateStatus(self, state):


        if P4_BASED_LOSS:
            # @TODO Based on cost calculations in P4
            current = self.current
            cost = self.lmap.getCost(state, current)

            if cost == float('inf'):
                status = 'nostep'
            else:
                status = 'step'

            return status

        else:

            new_x = state[0]
            new_y = state[1]
            mapref = self.lmap


            # To check the reasonable of action
            if new_x < 0 or new_x > mapref.info['width'] - 1:
                status = 'OOB'

            elif new_y < 0 or new_y > mapref.info['height'] - 1:

                status = 'OOB'

            elif mapref.matrix[new_x][new_y] != ".":
                status = 'obstacle'

            else:
                status = "step"

            return status

    def reinitiateEnvironment(self):
        self.current = self.start



    def getClosenessToGoalFeature(self, action):


            newState = (self.current[0] + action[0], self.current[1] + action[1])
            status = self.getStateStatus(newState)

            if status == "step":
                distanceNormalizer = float(self.lmap.info['width'] * self.lmap.info['height'])
                distance = math.sqrt(
                    ((newState[0] - self.goal[0]) ** 2) + ((newState[1] - self.goal[1]) ** 2))
                normalizedDistance = distance / distanceNormalizer
                closenessToGoal = 1 - normalizedDistance
                featureValueForAction = closenessToGoal


            else:
                featureValueForAction = -10

            return featureValueForAction

    def takeAction(self, actionIndex):

        action = self.actions[actionIndex]

        newState = (self.current[0] + action[0], self.current[1] + action[1])
        done = newState == self.goal
        self.current = newState

        return done, newState


    def getFeatures(self, actions):
        #avtionWiseFeatures = {}
        actionWiseFeaturecVector = []
        features =[]

        for action in actions:
            featureOneForEachAction = self.getClosenessToGoalFeature(action)
            #featureTwo
            #featureThree
            #.....

            #avtionWiseFeatures[action] = [featureOneForEachAction]     #More features can be added in this list
            features = [featureOneForEachAction] #More features can be added in this list
            actionWiseFeaturecVector.append(features)     #More features can be added in this list


        #More features can be added
        assert len(features) == NUMBER_PARAMETERS
        return actionWiseFeaturecVector



class LinearPolicy:

    def __init__(self, env, alpha = ALPHA, gamma = GAMMA):
        self.alpha = alpha
        self.gamma = gamma
        self.env = env
        # parameters are loaded or trained
        self.parameters = None

    def linearRegression(self):
        # use self.agent to get the state-action based h(s,a)
        # not sure if it is needed now

        pass


    def getHighestProbabilityActionIndex(self):

        stepActionWiseFeatures = self.env.getFeatures(self.env.actions)  # returned value is not numpy array
        stepActionsProb, stepActionsRewards = \
            self.getActionsRewardsAndProbabilities(self.env.actions,
                                                     stepActionWiseFeatures)  # returned values are numpy array



        # Action with highest prpbability as per policy
        bestActionIndex = np.argmax(stepActionsProb)

        bestAction = self.env.actions[bestActionIndex]
        next = (self.env.current[0] + bestAction[0], self.env.current[1] + bestAction[1])
        #bestActionReward = stepActionsRewards[bestActionIndex]
        #bestActionProb = stepActionsProb[bestActionIndex]

        return bestActionIndex

    def getStochasticActionIndex(self):
        # Beware stochastic actions
        # can lead to OOB or obstacles
        # with very low probability
        stepActionWiseFeatures = self.env.getFeatures(self.env.actions)  # returned value is not numpy array
        # below two returned variables are numpy array
        stepActionsProb, stepActionsRewards = \
            self.getActionsRewardsAndProbabilities(self.env.actions,
                                                     stepActionWiseFeatures)
        stepActionsIndex = np.array(range(len(self.env.actions))) #Python 2.x list index generation

        stochasticActionIndex = np.random.choice(stepActionsIndex, 1, p=stepActionsProb)[0]# return array


        return stochasticActionIndex


    def getActionsRewardsAndProbabilities(self, actions, actionWiseFeatures):
        # sample an action in proportion to probabilities
        '''

        :param actions: The available action for this state
        :param actionWiseFeatures: a 2-D array containng feature-vector of all actions
        :return:
                probs: a 1-D numpy array carrying probabilities of all actions as per softmax of h(s,a,theta)
                rewards: a 1-D numpy array carrying dot product of h(s,a,theta) and parameter vector
        '''

        valueGivenParam = []
        #featureVector = []

        #actionsIndex = np.array([0,1,2,3,4,5,6,7])

        for actionFeatures in actionWiseFeatures:
            features = np.array(actionFeatures)
            parameters = self.parameters
            assert len(features) == len(parameters)
            reward = np.dot(features, parameters)
            valueGivenParam.append(reward)
            #featureVector.append(features)

        rewards = np.array(valueGivenParam)
        probs = softmax(rewards) #softmax function from SciPi
        #actionWiseFeatureVector = np.array(featureVector)

        return probs,rewards  #,actionsIndex# , actionWiseFeatureVector




    def grad_log_p(self,obs, featuresPerAction, policyProbs, actionsIndex, rewards):

        '''

        :param rewards: 1-D numpy array specifying rewards obsrved at each step of episode
        :param obs: 1-D numpy array specifying coordinates of position at each step of episode
        :param actionsIndex: 1-D numpy array specifying indexes of action at each step of episode
        :param policyProbs: 1-D numpy array specifying policy(s,a,theta) at each step of episode
        :param featuresPerAction: 2-D numpy array. For each action available at state it stores
                                    feature vector as 1-D numpy array for each step of episode
       :return: a 1-D numpy array having gradient_of_log_of_policy for each step in episode
       '''

        gradientEachStep = []

        if SUMMED_PARAMETER_UPDATE:
            # Sum of features implementaion
            for step in range(0,len(obs)):
                actionTaken = actionsIndex[step]

                stepFeaturesAllActions = featuresPerAction[step]
                stepProbsAllActions = policyProbs[step]

                phi_s_a = stepFeaturesAllActions[actionTaken]
                firstTerm = np.sum(phi_s_a)

                secondTerm = 0
                for act_index in range (0 , len(stepProbsAllActions)): # feature and prob values of all actions at this step
                    phi_s_actIndex = stepFeaturesAllActions[act_index]
                    prob = stepProbsAllActions[act_index]
                    secondTerm += np.sum(phi_s_actIndex) * prob



                gradient = firstTerm - secondTerm
                gradientEachStep.append(gradient)

        else:

            for step in range(0,len(obs)):
                actionTaken = actionsIndex[step]

                stepFeaturesAllActions = featuresPerAction[step]
                stepProbsAllActions = policyProbs[step]

                phi_s_a = stepFeaturesAllActions[actionTaken]

                numberOfFeatures = len(phi_s_a)
                expected = np.zeros(numberOfFeatures)
                for act_index in range (0 , len(stepProbsAllActions)): # feature and prob values of all actions at this step
                    phi_s_actIndex = stepFeaturesAllActions[act_index]
                    prob = stepProbsAllActions[act_index]
                    expected += phi_s_actIndex * prob



                gradient = phi_s_a - expected
                gradientEachStep.append(gradient)

        return np.array(gradientEachStep)


    def discount_rewards(self, rewards):


        '''

        :param self:
        :param rewards:
        :return: 1-D numpy array representing the cumulative discount reward
                 from the step t to terminal step
        '''
        # calculate temporally adjusted, discounted rewards

        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        for i in reversed(range(0, len(rewards))):
            cumulative_rewards = cumulative_rewards * self.gamma + rewards[i]
            discounted_rewards[i] = cumulative_rewards

        return discounted_rewards

    def update(self, rewards, obs, actionsIndex, policyProbs, featuresPerAction): #all arguments are episode arguments
        '''
        :param rewards: 1-D numpy array specifying rewards obsrved at each step of episode
        :param obs: 1-D numpy array specifying coordinates of position at each step of episode
        :param actionsIndex: 1-D numpy array specifying indexes of action at each step of episode
        :param policyProbs: 1-D numpy array specifying policy(s,a,theta) at each step of episode
        :param featuresPerAction: 2-D numpy array. For each action available at state it stores
                                    feature vector as 1-D numpy array for each step of episode
        :return:
        '''

        if SUMMED_PARAMETER_UPDATE:
            # calculate gradients for each action over all observations
            grad_log_p = self.grad_log_p(obs, featuresPerAction, policyProbs, actionsIndex, rewards)
                #np.array([self.grad_log_p(ob)[action] for ob,action in zip(obs,actions)])

            # calculate temporaly adjusted, discounted rewards
            discounted_rewards = self.discount_rewards(rewards)

            # gradients times rewards at each step
            # updateValueAllSteps = (np.multiply(grad_log_p, discounted_rewards)) * self.alpha

            # @ToDO Check if the assumptions below are correct

            # gradients times rewards at each step multiplied by alpha
            updateValueAllSteps = (np.multiply(grad_log_p, discounted_rewards)) * self.alpha

            # Assumption1: adding updated value
            # one by one is same as adding once
            # while updating parameters
            # Assumption 2:
            # all parameters will receive same update
            totalUpdate = np.sum(updateValueAllSteps)

            # gradient ascent on parameters by adding whole episode update
            self.parameters +=totalUpdate

        else:
            # calculate gradients for each action over all observations
            grad_log_p = self.grad_log_p(obs, featuresPerAction, policyProbs, actionsIndex, rewards)
                #np.array([self.grad_log_p(ob)[action] for ob,action in zip(obs,actions)])

            # calculate temporaly adjusted, discounted rewards
            discounted_rewards = self.discount_rewards(rewards)


            # gradients times rewards at each step
            # updateValueAllSteps = (np.multiply(grad_log_p, discounted_rewards)) * self.alpha

            # @ToDO Check if the assumptions below are correct

            # gradients times rewards at each step multiplied by alpha
            updateValueAllSteps = (grad_log_p* discounted_rewards[:,None]) * self.alpha
            #a * b[:, None]

            # Assumption1: adding updated value
            # one by one is same as adding once
            # while updating parameters
            # Assumption 2:
            # all parameters will receive same update
            totalUpdate = np.sum(updateValueAllSteps, axis = 0)

            # gradient ascent on parameters by adding whole episode update
            self.parameters = np.add(self.parameters ,totalUpdate)


    def getPolicyEnvironment(self):
        return self.env

    def loadParameters(self, file_name):

        self.parameters = np.load(file_name)


    def saveParameters(self, file_name):
        np.save(file_name, self.parameters)

    def qValue(self, state, action):
        x, y = state
        return self.q_tbl[x][y][action]

    def value(self, state):
        x, y = state
        return max(self.q_tbl[x][y])

    def valueIteration(self, lmap, goal, discount):
        converge = True
        for x in range(self.width):
            for y in range(self.height):
                state = (x, y)
                if state == goal:
                    continue
                for z, a in enumerate(ACTIONS):
                    old_q = self.q_tbl[x][y][z]
                    state_p = (x + a[0], y + a[1])
                    cost = lmap.getCost(state_p, previous=state)
                    if cost < float('inf'):
                        q = discount * self.value(state_p) - cost
                        if q > old_q:
                            converge = False
                            self.q_tbl[x][y][z] = q

        return converge

def run_episode(env, policy,seed):

        '''Returns:
        episodetotalreward
        episodeRewards
        episodeObservations
        epsidoeActions, np.array(episodeProbs), np.array(stepActionWiseFeatures)

        '''

        episodeObservations = []
        episodetotalreward = 0
        epsidoeActions = []
        episodeRewards = []
        episodeProbs = []
        episodeActionjWiseFeatures = []

        env.reinitiateEnvironment()
        terminal = env.current == env.goal
        step =0
        while not terminal:

            stepActionWiseFeatures = env.getFeatures(env.actions) # returned value is not numpy array
            stepActionsProb, stepActionsRewards = \
                policy.getActionsRewardsAndProbabilities(env.actions, stepActionWiseFeatures) #returned values are numpy array

            stepActionsIndex = np.array(range(len(env.actions)))  # Python 2.x list index generation
            stochasticActionIndex = np.random.choice(stepActionsIndex, 1, p=stepActionsProb)[0]  # return array
            #step = step+1
            #print step

            # Action with highest prpbability as per policy
            #bestActionIndex = np.argmax(stepActionsProb)
            #bestActionReward = stepActionsRewards[bestActionIndex]

            stochasticActionReward = stepActionsRewards[stochasticActionIndex]

            episodeObservations.append(env.current)

            terminal, newState = env.takeAction(stochasticActionIndex) #also updates env.current to next step

            stepReward = stochasticActionReward
            stepAction = stochasticActionIndex #action taken
            stepProbs = stepActionsProb # all probabilities

            episodetotalreward += stepReward
            episodeRewards.append(stepReward)
            epsidoeActions.append(stepAction)
            episodeProbs.append(stepProbs)
            episodeActionjWiseFeatures.append(np.array(stepActionWiseFeatures))

        return episodetotalreward, np.array(episodeRewards), np.array(episodeObservations), \
               np.array(epsidoeActions), np.array(episodeProbs), np.array(episodeActionjWiseFeatures)


def train(env,policy, MAX_EPISODES=1000, seed=None, evaluate=False):

        episode_rewards = []

        # train until MAX_EPISODES
        for i in range(MAX_EPISODES):
            print 'Training Episode : ',i, " out of: ", MAX_EPISODES
            # run a single episode
            total_reward, rewards, observations, actionsIndex, policyProbs,featuresPerAction = run_episode(env, policy,seed)

            # keep track of episode rewards
            episode_rewards.append(total_reward)

            # update policy
            policy.update(rewards, observations, actionsIndex, policyProbs, featuresPerAction)
            #print"EP: " + str(i) + " Score: " + str(total_reward) + " ", end="\r", flush=False

            # evaluation call after training is finished - evaluate last trained policy on 100 episodes
        if evaluate:
            pass

        return episode_rewards, policy

def trainPolicy(policy):


    env = policy.env
    np.random.seed(GLOBAL_SEED)
    policy.alpha = ALPHA
    policy.gamma = GAMMA
    policy.parameters = np.random.rand(NUMBER_PARAMETERS)

    episode_rewards, policytrained = train(
                                    env,
                                    policy=policy,
                                    MAX_EPISODES=EPISODES,
                                    seed=GLOBAL_SEED,
                                    evaluate=False,
                                    )