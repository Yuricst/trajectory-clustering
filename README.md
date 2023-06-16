# trajectory-clustering
Python library for trajectory clustering. 


### Notes
- The standard way for trajectory clustering is to use DBSCAN with Dynamic Time Warping (DTW).
- DTW is a similarity measure for time series data.
    - It is non-metric as it does not satisfy the triangular inequality. 
    - The order of the samples in the trajectory has to satisfy monotonicity, but the sampling time does not have huge effect because DTW adjusts the pairs to be compared between trajectories to minimize the total distance, under the monotonicity constraints. 
- To account for the phase w.r.t. the Sun's rotational period, this information has to be added to the state space.
    - This is because DTW only check the order and doesn't take absolute nor relative time into consideration.
    - Inside the DTW algorithm, we need to adjust the distance function between the two states, as to account for the periodicity of the phase. 
        - We can use Euclidian distance, but only for the phase dimenison we can take $|\theta| \% \pi$. 