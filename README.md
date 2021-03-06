# TMM_1d_waveguide
Transfer Matrix Method for calculating Natural Frequency, Dispersion Relation, and Forced Response of general 1d waveguide. 

![state vector equation](https://user-images.githubusercontent.com/69582527/90011980-81b18780-dcdd-11ea-81e5-a34f8ab93516.gif)

![matrix](https://user-images.githubusercontent.com/69582527/90011992-84ac7800-dcdd-11ea-895f-13fb0c7053c5.gif)

![state vector](https://user-images.githubusercontent.com/69582527/90011996-85dda500-dcdd-11ea-8fbe-9019cf5024ae.gif)

Where A and B are both 6 by 6 matrices which is function of angular frequency, curvature, and torsion.
z is corresponding 12 dimensional state vector, which represents resultant displacement, rotation, internal force, internal moment at each point along the waveguide. 

Since, generally curvature and torsion is not constant throughout the waveguide, this program calculates transfer matrix by integration using Runge-Kutta 4th order method. And further uses transfer matrix to calculate natural frequency, dispersion relation, and forced response.

The code automatically calculates curvature and torsion for 1d waveguide, thus the only input is the parametric equation of the waveguide. 

We hope further usage of this program provides opportunity for everyone to calculate the mechanical characteristic of 1d waveguides. 
