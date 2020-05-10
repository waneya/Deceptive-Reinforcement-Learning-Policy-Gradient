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
INITIAL_WEIGHTS = np.array([55 #closeness
                            ,60 #divergence +ve value means penalize divergence and favour action closer to all goals
                            ,-10 # -ve means penalize
                            #,500 # reachedGoal
                            ])
NUMBER_PARAMETERS = len(INITIAL_WEIGHTS)
ALPHA = 0.001
GAMMA = 0.95
GLOBAL_SEED = 0
EPISODES = 10
SUMMED_PARAMETER_UPDATE = False
P4_BASED_LOSS = True
STEPS_EACH_EPISODE = 2000



class P4Environemnt:

    def __init__(self, lmap, start, goal, allGoals):

        self.lmap = lmap
        self.start =start
        self.goal = goal
        self.current = start
        self.actions = ACTIONS
        self.allGoals = allGoals
        self.history = []

    def getNewStateStatus(self, state):


        if P4_BASED_LOSS:
            # @TODO Based on cost calculations in P4

            previous = None
            if len(self.history) > 1:
                #previous = self.history[-1] # bad idea
                previous = self.current
            cost = self.lmap.getCost(state, previous)

            #current = None
            #cost = self.lmap.getCost(state, current)


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
            if new_x < 1 or new_x >= mapref.info['width'] - 1:
                status = 'OOB'

            elif new_y < 1 or new_y >= mapref.info['height'] - 1:

                status = 'OOB'

            elif mapref.matrix[new_x][new_y] != ".":
                status = 'obstacle'

            else:
                status = "step"

            return status

    def reinitiateEnvironment(self):
        self.current = self.start

    def normalized_euclidean_distance(self, state1, state2):

        distanceNormalizer = float(self.lmap.info['width'] * self.lmap.info['height'])
        distance = math.sqrt(
            ((state1[0] - state2[0]) ** 2) + ((state1[1] - state2[1]) ** 2))
        normalizedDistance = distance / distanceNormalizer

        return normalizedDistance

    def getStateFeaturesAllActions(self, state, actions):
        '''

        :param state:
        :param actions:
        :return: a 2-D numpy  array containng feature-vector of all actions
        '''



        actionWiseFeaturecVector = []
        features =[]
        FIRST_FEATURE = 0

        for act_index in range(len(actions)):

            features = self.getStateActionFeatures(state, act_index)
            actionWiseFeaturecVector.append(features)


        assert len(features) == NUMBER_PARAMETERS
        actionWiseFeaturecVector = np.array(actionWiseFeaturecVector)

        # below lines further normalize closeness by subtracting lowest columns
        #feature_one_normalized = actionWiseFeaturecVector[:,FIRST_FEATURE] - np.min(actionWiseFeaturecVector[:, FIRST_FEATURE])
        #actionWiseFeaturecVector[:, FIRST_FEATURE] = feature_one_normalized.T

        return actionWiseFeaturecVector


    def getStateActionFeatures(self,state,act_index):


        featureOne = self.getClosenessToGoalFeature(state, act_index)
        featureTwo = self.getDivergenceFromAllGoalsFeature(state, act_index)
        featureThree = self.stepFeature(state, act_index)
        featureFour = self.goalReachedFeature(state, act_index)

        features = [featureOne
                    ,featureTwo
                    ,featureThree
                    #,featureFour
                    ]

        return features


    def getStateActionReward(self, state, act_index):
        # This function gives the reward
        # for chosen action during the training
        # phase. Currently we are not learning
        # the weights of the  rewards
        rewards_weights = INITIAL_WEIGHTS
        features = self.getStateActionFeatures(state,act_index)
        reward = np.dot(rewards_weights, features)
        return reward


    def getClosenessToGoalFeature(self, state, act_index):
        # This feature ensures least number of steps are taken


            action = self.actions[act_index]

            newState = (state[0] + action[0], state[1] + action[1])
            status = self.getNewStateStatus(newState)

            if status == "step":


                normalizedDistance = self.normalized_euclidean_distance(newState,self.goal)
                closenessToGoal = 1 - normalizedDistance
                featureValueForAction = closenessToGoal  # further normalizing


            else:
                featureValueForAction = 0

            return featureValueForAction



    def getDivergenceFromAllGoalsFeature(self, state, act_index):
        '''

        :param state:
        :param actions:
        :param act_index:
        :return: Total divergence from all goals for act_index.
                 Higher the divergence, more irrational the action
                 because it, overall, diverges from all goals.
        '''


        actions = self.actions

        action = actions[act_index]
        newState = (state[0] + action[0], state[1] + action[1])
        status = self.getNewStateStatus(newState)


        if status == "step":

            goal_wise_action_divergence = []
            for goal in self.allGoals:

                actions_closeness_goal = []
                for action in actions:
                    newState = (state[0] + action[0], state[1] + action[1])
                    status = self.getNewStateStatus(newState)
                    if status == "step":
                        normlaizedDistance  = self.normalized_euclidean_distance(newState, goal)
                        closeness = 1 -normlaizedDistance
                    else:
                        closeness = 0
                    actions_closeness_goal.append(closeness)
                actions_closeness_goal = np.array(actions_closeness_goal)
                actions_closeness_goal -= np.min(actions_closeness_goal)
                best_action_index = np.argmax(actions_closeness_goal)
                best_action_closeness = np.max(actions_closeness_goal)
                given_action_closeness = actions_closeness_goal[act_index]
                action_divergence_from_optimal_this_goal =   given_action_closeness - best_action_closeness
                goal_wise_action_divergence.append(action_divergence_from_optimal_this_goal)
            goal_wise_action_divergence = np.array(goal_wise_action_divergence)
            total_divergence = np.sum(goal_wise_action_divergence)

            featureValue =  total_divergence

        else:
            featureValue = 0

        return featureValue

    def stepFeature(self, state, act_index):


        action = self.actions[act_index]
        newState = (state[0] + action[0], state[1] + action[1])



        status = self.getNewStateStatus(newState)

        infinity = float('inf')
        if status == 'step':
            cost = 1
        else:
            cost = infinity



        if cost == infinity:
            featureValue = 100.0
        else:
            featureValue = cost/100.0 #

        if len(self.history)>0:
            #nearHistory = self.history[-10:]
            nearHistory = self.history
            if newState in nearHistory:
                featureValue = 100
                #pass

        if newState == self.goal:
            featureValue = -25.0
        if newState in self.allGoals[1:]: # i.e. fake goals
            featureValue = 100.0

        return featureValue

    def goalReachedFeature(self, state, act_index):


        actions = self.actions
        action = actions[act_index]
        newState = (state[0] + action[0], state[1] + action[1])

        if newState == self.goal:
            featureValue = 5
        else:
            featureValue = 0

        return featureValue




    def getClosenessToFakeGoalsFeature(self, state, act_index):
        action = self.actions[act_index]
        newState = (state[0] + action[0], state[1] + action[1])
        status = self.getNewStateStatus(newState)
        number_fake_goals = len(self.allGoals) - 1
        if status == "step":

            fg_closness = np.zeros(number_fake_goals)
            for fake_goal_ind in range(1, number_fake_goals):  # index 0 is real goal
                fake_goal = self.allGoals[fake_goal_ind]

                normalizedDistance = self.normalized_euclidean_distance(fake_goal, newState)
                closenessToGoal = 1 - normalizedDistance
                fg_closness[fake_goal_ind - 1] = closenessToGoal


        else:
            featureValueForAction = -10

        return featureValueForAction



    def takeAction(self, act_index):

        action = self.actions[act_index]

        newState = (self.current[0] + action[0], self.current[1] + action[1])
        done = newState == self.goal
        self.history.append(self.current)
        self.current = newState

        return done, newState







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

    def getPolicyBasedActionProbs(self, state, actions):

        '''

        :param actions: The available action for this state
        :param state
        :return:
                probs: a 1-D numpy array carrying probabilities of all actions as per softmax of h(s,a,theta)
                rewards: a 1-D numpy array carrying dot product of h(s,a,theta) and parameter vector
        '''

        stepActionWiseFeatures = self.env.getStateFeaturesAllActions(state,actions)

        valueGivenParam = []


        for actionFeatures in stepActionWiseFeatures:
            features = actionFeatures
            parameters = self.parameters
            assert len(features) == len(parameters)
            h_s_a_theta = np.dot(features, parameters)
            valueGivenParam.append(h_s_a_theta)
            #featureVector.append(features)

        allAction_h_s_a_theta = np.array(valueGivenParam)
        probs = softmax(allAction_h_s_a_theta ) #softmax function from SciPi


        return probs


    def getHighestProbabilityActionIndex(self, state ):

        stepActionsProb= self.getPolicyBasedActionProbs(state, self.env.actions)

        # Action with highest prpbability as per policy
        bestActionIndex = np.argmax(stepActionsProb)

        bestAction = self.env.actions[bestActionIndex]
        next = (self.env.current[0] + bestAction[0], self.env.current[1] + bestAction[1])
        #bestActionReward = stepActionsRewards[bestActionIndex]
        #bestActionProb = stepActionsProb[bestActionIndex]

        return bestActionIndex

    def getAllActionsProbabilities(self, state):

        stepActionsProb = self.getPolicyBasedActionProbs(state, self.env.actions)

        return stepActionsProb

    def getStochasticActionIndex(self, state):
        # Beware stochastic actions
        # can lead to OOB or obstacles
        # with very low probability
        stepActionsProb = self.getPolicyBasedActionProbs(state, self.env.actions)
        stepActionsIndex = np.array(range(len(self.env.actions))) #Python 2.x list index generation

        stochasticActionIndex = np.random.choice(stepActionsIndex, 1, p=stepActionsProb)[0]# return array


        return stochasticActionIndex


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
                step_action = actionsIndex[step]

                step_features_AllStateActions = featuresPerAction[step]
                step_probs_alStateActions = policyProbs[step]

                phi_s_a = step_features_AllStateActions[step_action]
                firstTerm = np.sum(phi_s_a)

                secondTerm = 0
                for act_index in range (0 , len(step_probs_alStateActions)): # feature and prob values of all actions at this step
                    phi_s_actIndex = step_features_AllStateActions[act_index]
                    prob = step_probs_alStateActions[act_index]
                    secondTerm += np.sum(phi_s_actIndex) * prob



                gradient = firstTerm - secondTerm
                gradientEachStep.append(gradient)

        else:

            for step in range(0,len(obs)):
                step_action = actionsIndex[step]

                step_features_AllStateActions = featuresPerAction[step]
                step_probs_alStateActions = policyProbs[step]

                phi_s_a = step_features_AllStateActions[step_action]

                numberOfFeatures = len(phi_s_a)
                expected = np.zeros(numberOfFeatures)
                for act_index in range (0 , len(step_probs_alStateActions)): # feature and prob values of all actions at this step
                    phi_s_actIndex = step_features_AllStateActions[act_index]
                    prob = step_probs_alStateActions[act_index]
                    expectation = phi_s_actIndex * prob
                    expected += expectation



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

        # this update function make discounted updates
        # from step t to terminal step at the end of the
        # episode.
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

        stepActionsProb, stepActionsRewards = self.getPolicyBasedActionProbs(state, self.env.actions)

        return stepActionsRewards[action]

    def value(self, state):
        stepActionsProb, stepActionsRewards = self.getPolicyBasedActionProbs(state, self.env.actions)
        return np.max(stepActionsRewards)

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
        while not terminal and step < STEPS_EACH_EPISODE:
            stepActionWiseFeatures = env.getStateFeaturesAllActions(env.current, env.actions)
            stepActionsProb = policy.getPolicyBasedActionProbs(env.current, env.actions)

            stepActionsIndex = np.array(range(len(env.actions)))  # Python 2.x list index generation
            stochasticActionIndex = np.random.choice(stepActionsIndex, 1, p=stepActionsProb)[0]  # return array
            #bestActionIndex = np.argmax(stepActionsProb)

            # Pass the action and choose another one
            # if it takes us out of boundry or obstacle
            stochasticAction = env.actions[stochasticActionIndex]
            newState = (env.current[0] + stochasticAction[0], env.current[1] + stochasticAction[1])
            status = env.getNewStateStatus(newState)
            if status !="step":
                #print "continued"
                step+=1
                continue

            # get the reward for all state
            # so that it can be normalized
            # by lowest reward
            all_rewards =[]
            for act_index in range(len(env.actions)):
                reward = env.getStateActionReward(env.current,act_index)
                all_rewards.append(reward)
            all_rewards = np.array(all_rewards)
            all_rewards -= np.min(all_rewards)
            all_rewards *= 10

            stochasticActionReward = all_rewards[stochasticActionIndex]



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
            step += 1
            #print newState
            #print step

        return episodetotalreward, np.array(episodeRewards), np.array(episodeObservations), \
               np.array(epsidoeActions), np.array(episodeProbs), np.array(episodeActionjWiseFeatures), terminal


def train(env,policy, MAX_EPISODES=1000, seed=None, evaluate=False):

        episode_rewards = []

        # train until MAX_EPISODES
        for i in range(MAX_EPISODES):
            print 'Training Episode : ',i, " out of: ", MAX_EPISODES
            # run a single episode
            total_reward, rewards, observations, actionsIndex, policyProbs,featuresPerAction, terminal = run_episode(env, policy,seed)

            # keep track of episode rewards
            episode_rewards.append(total_reward)

            # update policy
            policy.update(rewards, observations, actionsIndex, policyProbs, featuresPerAction)
            print "Parameters after Episode: ", policy.parameters, " Reached Goal: ", terminal
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
    #policy.parameters = np.random.rand(NUMBER_PARAMETERS)
    policy.parameters = INITIAL_WEIGHTS

    episode_rewards, policytrained = train(
                                    env,
                                    policy=policy,
                                    MAX_EPISODES=EPISODES,
                                    seed=GLOBAL_SEED,
                                    evaluate=False,
                                    )