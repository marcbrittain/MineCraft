# Episode 1

This is episode 1 of the minecraft video series. In this episode, our AI Ralph is attempting to navigate to the lapis lazuli block.
Ralph quickly learns that if he takes the wrong action, he will end up in the lava and that will end the episode. We used standard Q-learning to solve this problem.

The reward structure for this problem is set up as follows:

* -100 for landing on a lava block
* +100 for landing on a lapis lazuli block
* -1 for every action taken

The reason for including the -1 reward for every action is to provide incentive for arriving at the lapis lazuli block as fast as possible. 
This helps Ralph converge to the optimal route faster and allows him to safetly make it to the lapis lazuli block :)
