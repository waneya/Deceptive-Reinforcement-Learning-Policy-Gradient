
AGENT_FILE = "agent_rm"         # agent filename - must be in src/agents/
MAP_FILE = "drl/49_low.map"  	# map filename - must be in maps (sister dir to src)
START = (17, 17)                # coordinates of start location in (col,row) format
GOAL = (40, 38)                 # coordinates of goal location in (col,row) format
POSS_GOALS = [(13, 28), (35, 21),(32, 45), (20, 46)]

GUI = True                      # True = show GUI, False = run on command line
SPEED = 0.0                     # delay between displayed moves in seconds
DEADLINE = 100                  # Number of seconds to reach goal
HEURISTIC = 'octile'            # may be 'euclid' or 'manhattan' or 'octile' (default = 'euclid')
DIAGONAL = True                 # Only allows 4-way movement when False (default = True)
FREE_TIME = 0.000               # Step times > FREE_TIME are timed iff REALTIME = True
DYNAMIC = False                 # Implements runtime changes found in script.py when True
STRICT = True                   # Allows traversal of impassable cells when False (default = True)
PREPROCESS = False              # Gives agent opportunity to preprocess map (default = False)
#COST_MODEL = 'mixed_real'      # May be 'mixed' (default), 'mixed_real', 'mixed_opt1' or 'mixed_opt2'
COST_FILE = "../costs/G1-W5-S10.cost"
