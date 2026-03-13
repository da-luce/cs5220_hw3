# HW3: Distributed Memory Particle Simulation

Name: `Dalton Luce` \
CornellID: `5354540` \
Perlmutter Username: `dcl252`

## Strong Scaling Plot

![](./strong_scaling_analysis.png)

<!--A strong scaling plot in log-log scale that shows the performance of your code on 1e6 and 2e6 particles using 64 processes on 1 nodes, 128 processes across 2 nodes with 64 processes per node, and 256 processes across 2 nodes with 128 processes per node. Include a line for ideal scaling as well.-->

<!-- The plot goes below the ideal scaling at some points -->

This plot shows the parallel scaling behavior of the particle simulation for 1e6 and 2e6 particles. The plot follows a downward trend for both simulation loads, demonstrating that my implementation is benefiting from additional parallel resources. At some points, the curve falls below the ideal scaling line, but levels out towards higher levels of parallelism.

My hypothesis for why this is, is that my 1D decomposition of the domain is limiting my parallel efficiency. As the number of processor grows, the ratio of actual computation to ghost particle exchange decreases. With a 2D decomposition, the "surface area" of ghost regions would be lower, limiting communication more. However, I found it difficult to implement 2D decomposition without relying on AI for parts of the implementation, so I did not submit this as my own work. Instead, I chose to submit only my 1D decomposition implementation, which, while guided by AI in some portions (see below), reflects my own implementation and understanding.

## AI Disclosure

I did not collaborate with other students on this assignments. This report was written entirely by me, except for the python script to parse run output and generate the scaling plot. Google Gemini was used to create this script.

Citations are present in code comments where papers or generative AI were used/referenced. The major driving architecture influence from AI was:

- The use of vectors. Google Gemini highlighted from the get-go that my buffers should be CPP vectors, as OpenMPI requires contiguous memory for communication. - Confirmed what buffers I needed for correct communication with my decomposition
- Google Gemini also suggested the ghost exchange should be made async, meaning ghosts are exchanged while bins within each slice are calculated. I implemented this and it improved my parallel efficiency.
- Google Gemini also helped with my understanding of the OpenMPI API, specifically understanding how to use request tags to wait on multiple directions of ghost particle exchanges at the same time.
