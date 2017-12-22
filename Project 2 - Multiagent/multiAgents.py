# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
import random, util
from game import Agent
#

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        food = newFood.asList()
        number_of_food = len(food)

        closestDistance = 999999
        #Get the minimum distance to the closet food. Give the number of food importance by multiplying it by 1000.
        for coordinate in food:
            closestDistance = min(manhattanDistance(coordinate, newPos) + number_of_food*1000, closestDistance)

        #Pacman will get stuck at the last food. This will tell him that just go get the closest food and finish the game.
        if number_of_food == 0:
            closestDistance = 0

        #Note: As features, try the reciprocal of important values (such as distance to food) rather than just the values themselves.
        total = -closestDistance

        distance_to_ghost = 999999
        #Get the minimum distance to the closet ghost.
        for coordinate in newGhostStates:
            distance_to_ghost = min(manhattanDistance(coordinate.getPosition(), newPos), distance_to_ghost)

        #The ghost is near, run pacman run!
        if distance_to_ghost <= 1:
            total = total - 999999

        return total
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        final_score, final_direction = self.maxfunction(gameState, self.depth)
        return final_direction

    def maxfunction(self, gameState, depth):
        # You have reached the end.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), " "

        # Pacman index id is zero
        pacman_actions = gameState.getLegalActions(0)
        # This is like -ve infinity for us
        best_score = -999999
        # There is no best action right now.
        best_action = None
        # For every action of pacman, generate the state and send it to min function for evaluation.
        for action in pacman_actions:
            # Get the score for the depth for pacman. We only have on pacman.
            max_value = self.minfunction(gameState.generateSuccessor(0, action), depth, 1)[0]
            # If the max_value is bigger than the best score. Then the max value becomes the best score.
            if max_value > best_score:
                best_score = max_value
                # the action you did got you the best score. Save it.
                best_action = action
        # Return both the best_score and the best action that gets you that score.
        return best_score, best_action

    def minfunction(self, gameState, depth, ghost_number):
        # You have reached the end.
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), " "

        # Ghost index id is one and above.
        ghost_actions = gameState.getLegalActions(ghost_number)
        # This is like +ve infinity for us
        best_score = 999999
        # There is no best action right now.
        best_action = None
        # We need to get the minimum from all the ghost.
        if (ghost_number != gameState.getNumAgents() - 1):
            # For every action of ghost, generate the state and send it to min function for evaluation.
            for action in ghost_actions:
                # Remember, you are sending both the best_score and the best action. Hence the [0] in the end.
                # Get the min_value from all the ghost. Keep calling for all the ghost till you get the minimum.
                min_value = self.minfunction(gameState.generateSuccessor(ghost_number, action), depth, ghost_number + 1)[0]
                if min_value < best_score:
                    best_score = min_value
                    # the action you did got you the best score. Save it.
                    best_action = action
        # We need to get the maximum for the pacman now.
        else:
            # For every action of ghost, generate the state and send it to max function for evaluation.
            for action in ghost_actions:
                #Remember, you are sending both the best_score and the best action. Hence the [0] in the end.
                #Follow the Algorithm.
                min_value = self.maxfunction(gameState.generateSuccessor(ghost_number, action), depth - 1)[0]
                if min_value < best_score:
                    best_score = min_value
                    # the action you did got you the best score. Save it.
                    best_action = action
        # Return both the best_score and the best action that gets you that score.
        return best_score, best_action

        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Alpha is intially is -ve infinite
        alpha = -999999
        # Beta is initally is +ve infi
        beta = 999999
        temp, ans = self.maxfunction(gameState, self.depth, alpha, beta)
        return ans

    def maxfunction(self, gameState, depth, alpha, beta):
        # You have reached the end.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), " "

        # Pacman index id is zero
        pacman_actions = gameState.getLegalActions(0)
        # This is like -ve infinity for us
        best_score = -999999
        # There is no best action right now.
        best_action = None
        # For every action of pacman, generate the state and send it to min function for evaluation.
        for action in pacman_actions:
            # Get the score for the depth for pacman. We only have on pacman.
            max_value = self.minfunction(gameState.generateSuccessor(0, action), depth, 1, alpha, beta)[0]
            # If the max_value is bigger than the best score. Then the max value becomes the best score.
            if max_value > best_score:
                best_score = max_value
                # the action you did got you the best score. Save it.
                best_action = action
            # We will now see the max between alpha and the bes score.
            alpha = max(alpha, best_score)
            # If we find that best_score is bigger than beta, then we do purning
            if best_score > beta:
                # Return the best score.
                return best_score, ' '
        # Return both the best_score and the best action that gets you that score.
        return best_score, best_action

    def minfunction(self, gameState, depth, ghost_number, alpha, beta):
        # You have reached the end.
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), " "

        # Ghost index id is one and above.
        ghost_actions = gameState.getLegalActions(ghost_number)
        # This is like +ve infinity for us
        best_score = 999999
        # There is no best action right now.
        best_action = None
        # We need to get the minimum from all the ghost.
        if (ghost_number != gameState.getNumAgents() - 1):
            # For every action of ghost, generate the state and send it to min function for evaluation.
            for action in ghost_actions:
                # Remember, you are sending both the best_score and the best action. Hence the [0] in the end.
                # Get the min_value from all the ghost. Keep calling for all the ghost till you get the minimum.
                min_value = self.minfunction(gameState.generateSuccessor(ghost_number, action), depth, ghost_number + 1, alpha, beta)[0]
                if min_value < best_score:
                    best_score = min_value
                    # the action you did got you the best score. Save it.
                    best_action = action
                    # We will now see the min between beta and best_score
                    beta = min(beta, best_score)
                    # If we find that best_score is less than alpha, then we do purning
                    if best_score < alpha:
                        return best_score, ' '
        # We need to get the maximum for the pacman now.
        else:
            # For every action of ghost, generate the state and send it to max function for evaluation.
            for action in ghost_actions:
                # Remember, you are sending both the best_score and the best action. Hence the [0] in the end.
                # Follow the Algorithm.
                min_value = self.maxfunction(gameState.generateSuccessor(ghost_number, action), depth - 1, alpha, beta)[0]
                if min_value < best_score:
                    best_score = min_value
                    # the action you did got you the best score. Save it.
                    best_action = action

                    beta = min(min, best_score)
                    if best_score < alpha:
                        return best_score, ' '
        # Return both the best_score and the best action that gets you that score.
        return best_score, best_action

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        final_score, final_direction = self.maxfunction(gameState, self.depth)
        return final_direction

    def maxfunction(self, gameState, depth):
        # You have reached the end.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), " "

        # Pacman index id is zero
        pacman_actions = gameState.getLegalActions(0)
        # This is like -ve infinity for us
        best_score = -999999
        # There is no best action right now.
        best_action = None
        # For every action of pacman, generate the state and send it to min function for evaluation.
        for action in pacman_actions:
            # Get the score for the depth for pacman. We only have on pacman.
            max_value = self.minfunction(gameState.generateSuccessor(0, action), depth, 1)[0]
            #Increase the count
            # If the max_value is bigger than the best score. Then the max value becomes the best score.
            if max_value > best_score:
                best_score = max_value
                # the action you did got you the best score. Save it.
                best_action = action
        # Return both the best_score and the best action that gets you that score.
        return best_score, best_action

    def minfunction(self, gameState, depth, ghost_number):
        # You have reached the end.
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), " "

        # Ghost index id is one and above.
        ghost_actions = gameState.getLegalActions(ghost_number)
        # This is like +ve infinity for us
        best_score = 999999
        # There is no best action right now.
        best_action = None
        # We need to get the minimum from all the ghost.
        if (ghost_number != gameState.getNumAgents() - 1):
            # For every action of ghost, generate the state and send it to min function for evaluation.
            min_value = 0
            count = 0
            for action in ghost_actions:
                # Remember, you are sending both the best_score and the best action. Hence the [0] in the end.
                # Get the min_value from all the ghost. Keep calling for all the ghost till you get the minimum.
                # Even thought the variable name is min. It is just a add up of values.
                min_value = min_value + self.minfunction(gameState.generateSuccessor(ghost_number, action), depth, ghost_number + 1)[0]
                # Keep the count of number of values.
                count = count + 1
                if min_value < best_score:
                    best_score = min_value
                    # the action you did got you the best score. Save it.
                    best_action = action
        # We need to get the maximum for the pacman now.
        else:
            # For every action of ghost, generate the state and send it to max function for evaluation.
            min_value = 0
            count = 0
            for action in ghost_actions:
                # Remember, you are sending both the best_score and the best action. Hence the [0] in the end.
                # Follow the Algorithm.
                # Even thought the variable name is min. It is just a add up of values.
                min_value = min_value + self.maxfunction(gameState.generateSuccessor(ghost_number, action), depth - 1)[0]
                # Keep the count of number of values.
                count = count + 1
                if min_value < best_score:
                    best_score = min_value
                    # the action you did got you the best score. Save it.
                    best_action = action
        # Return both the best_score and the best action that gets you that score.
        # Return the average. So min_value (which hold all the value) divide by number of values in count.
        return min_value/count, best_action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Pretty similar to the project 2 part 1.
    # The only difference is the addition of currentScore
    pacman_position = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    numberOfFoodsLeft = len(foodList)
    ghostList = currentGameState.getGhostStates()
    currentScore = currentGameState.getScore()
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghostList]
    min_food_distance = 999999
    for coordinate in foodList:
        min_food_distance = min(manhattanDistance(pacman_position, coordinate) + numberOfFoodsLeft * 1000, min_food_distance)

    if numberOfFoodsLeft == 0:
        min_food_distance = 0

    total = -min_food_distance

    min_ghost_distance = 999999
    for coordinate in ghostList:
        min_ghost_distance = min(manhattanDistance(pacman_position, coordinate.getPosition()), min_ghost_distance)

    if min_ghost_distance <= 1:
        total = total - 999999

    #Just a random combination.
    total = total + currentScore - numberOfFoodsLeft + sum(newScaredTimes)
    return total


    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

