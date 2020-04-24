# Deceptive Path Planning (DPP) System for p4

This code allows **p4** to be used to generate and display _deceptive_ paths. It has been developed for and used in the following paper:

* Peta Masters and Sebastian Sardina. [Deceptive path-planning](https://bitbucket.org/ssardina-research/p4-simulator-gr/src/master/src/DPP). In _Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)_, pages 4368--4375, Melbourne, Australia, 2017. 



## USAGE NOTES

Deceptive path-planners (e.g. `agent_ds1.py`, `agent_ds2.py`, etc) are able to generate deceptive paths.

File `deceptor.py` contains helper routines.

Pass extra possible goals in config file or as array of coordinate tuples in `POSS_GOALS` command line parameter.

Call agent in from command line or menu and run in the usual way.


* For automated testing, generate problems sets exactly as for goal recognition (see `/src/GR` folder). 
Locate `dpp.py` in `src/` folder as calling script: inputs problems from nominated GR file, runs strategies (i.e. deceptive path-planning agents), and outputs results to csv. 
Currently, you need to hard-code the agent names in the dpp script.