Elijah Pelofske
IBM challenge

Higher energy states could be used as an additional computational basis for transmon qubit quantum computers. 
The problem is figuring out a way to effectively cntrol and use these higher energy states. 
In particular, it is interesting to consider if one try to use these higher energy sttes as 
additional computational basis, or try to restrict these higher enrgy states and remain in
the [0, 1] basis. 

In this project we are interested in learning more about these higher energy states. 

There were several problems that were encountered though;

1. Importing qiskit Pulse using qiskit==0.25 failed for an unkown reason (import related error). 
Therefore, we reverted to qiskit==0.23 and then went through the tutorial found at 
https://qiskit.org/textbook/ch-quantum-hardware/accessing_higher_energy_states.html 
in order to replicate the experiments to discrimate between |0>, |1> and |2> states. 

2. The next problem was that the queue time for these calibration experiments 
was quite long (e.g. 30 minutes). Therefore it was difficult to get results
in a timely manner. 

3. We also encountered some problems with matplotlib not displaying the generated plots.
It is unkown if this is due to the python environment we are using or some other
compatability problem. 

The summary is that most of the issues we encountered were software related in the process of
setting up and understanding the problem at hand. 
