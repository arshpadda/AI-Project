# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Stack
from util import Queue
from util import PriorityQueue
from util import manhattanDistance

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    #Stack to hold the node that have been visited along with the path taken from the start node to reach that node.
    stack = Stack()
    #Set to hold the node explored.
    explorednode = set()
    #Get the start node.
    startnode = problem.getStartState()
    #Push the starting node on the Stack along with an empty set to know the direction in order to reach the node.
    stack.push((startnode,[]))
    #Loop till the stack is empty
    while stack.isEmpty() is not True:
        #Pop the currentnode and the direction from the stack
        currentnode, direction = stack.pop()
        #We will now add the node to set of explored node.
        explorednode.add(currentnode)
        #If the node is the goal. We made it!!
        if problem.isGoalState(currentnode):
            #The direction holds the way to reach till the goal from the start node.
            return direction
        #Loop for each successor(child) of the current node.
        for (successor, action, stepCost) in problem.getSuccessors(currentnode):
            #If the successor(child) is not explored
            if successor not in explorednode:
                #Add the successor to the stack along with the path to reach it.
                stack.push((successor, direction + [action]))

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #Queue to hold the node along with the path taken from the start node to reach that node
    queue = Queue()
    #Set to hold the node explored.
    explorednode = set()
    # Get the start node.
    startnode = problem.getStartState()
    # Push the starting node on the Queue along with an empty set to know the direction in order to reach the node.
    queue.push((startnode,[]))

    # Loop till the queue is empty
    while queue.isEmpty() is not True:
        # Pop the currentnode and the direction from the queue
        currentnode, direction = queue.pop()
        # Check if the currentnode is not in explorednode.
        if currentnode not in explorednode:
           # We will now add the node to set of explored node.
            explorednode.add(currentnode)
            # If the node is the goal. We made it!!
            if problem.isGoalState(currentnode):
                # The direction holds the way to reach till the goal from the start node.
                return direction
            # Loop for each successor(child) of the current node.
            for (successor, action, stepCost) in problem.getSuccessors(currentnode):
                # If the successor(child) is not explored
                if successor not in explorednode:
                    # Add the successor to the queue along with the path to reach it.
                    queue.push((successor, direction + [action]))

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Priority Queue to hold the node along with the path taken from the start node to reach that node
    pqueue = PriorityQueue()
    #Set to hold the node explored.
    explorednode = set()
    # Get the start node.
    startnode = problem.getStartState()
    # Push the starting node on the Queue along with an empty set to know the direction in order to reach the node.
    pqueue.push((startnode,[]),0)

    # Loop till the priority queue is empty
    while pqueue.isEmpty() is not True:
        # Pop the currentnode and the direction from the priority queue
        (currentnode,direction)  = pqueue.pop()
        # Check if the currentnode is not in the explored node.
        if currentnode not in explorednode:
            # We will now add the node to set of explored node.
            explorednode.add(currentnode)
            # If the node is the goal. We made it!!
            if problem.isGoalState(currentnode):
                # The direction holds the way to reach till the goal from the start node.
                return direction
            # Loop for each successor(child) of the current node.
            for (successor, action, stepCost) in problem.getSuccessors(currentnode):
                # Add the successor to the queue along with the path to reach it.
                if successor not in explorednode:
                    # Add the successor to the queue along with the path to reach it.
                    pqueue.push((successor, direction + [action]), problem.getCostOfActions(direction + [action]))

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Priority Queue to hold the node along with the path taken from the start node to reach that node
    pqueue = PriorityQueue()
    #Set to hold the node explored.
    explorednode = set()
    # Get the start node.
    startnode = problem.getStartState()
    # Push the starting node on the Queue along with an empty set to know the direction in order to reach the node.
    pqueue.push((startnode, []), 0)

    # Loop till the priority queue is empty
    while pqueue.isEmpty() is not True:
        # Pop the currentnode and the direction from the priority queue
        (currentnode, direction) = pqueue.pop()
        # Check if the currentnode is not in the explored node.
        if currentnode not in explorednode:
            # We will now add the node to set of explored node.
            explorednode.add(currentnode)
            # If the node is the goal. We made it!!
            if problem.isGoalState(currentnode):
                # The direction holds the way to reach till the goal from the start node.
                return direction
            # Loop for each successor(child) of the current node.
            for (successor, action, stepCost) in problem.getSuccessors(currentnode):
                # Add the successor to the queue along with the path to reach it.
                if successor not in explorednode:
                    # Add the successor to the queue along with the path to reach it.
                    pqueue.push((successor, direction + [action]), problem.getCostOfActions(direction + [action]) + heuristic(successor, problem))

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
