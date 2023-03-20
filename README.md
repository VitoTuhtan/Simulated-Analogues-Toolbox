# Simulated-Analogues-Toolbox
# A collection of scripts used to post process HPC simulations of self gravitating plasma (RAMSES):  https://bitbucket.org/thaugboelle/ramses/wiki/Maintain
# 
# Outputs of the scripts were used to train a Deep Convolutional Neural Network that is used to infer physical properties of protostellar binary systems.
# The core of the algorithm is implemented in the script parallel_disk_characteristics_pickle.py where the data is extracted from the simulation and stored in a systematic way in order to streamline the training of the neural network. 
# 
# The abstract of the Simulated Analogues project: 
#Over the past few years, it’s become increasingly evident that star formation is a multi-scale problem, and therefore only global simulations that properly account for the connection from the large-scale gas flow to the accreting protostar can be used to understand protostellar systems. At the same time long wavelength interferometers (ALMA, NOEMA) are able to make observations with tens of AU resolution for the nearest young stellar objects (YSO).
#Using high resolution simulations and post processing methods, we aim to bridge the gap between simulations and observations of binary YSOs. Our goal is to create synthetic observations and perform a down-selection from large datasets of synthetic images to a handful of matching candidates in a semi-automatic way. From synthetic observations we infer the underlying physics that drives the creation and evolution of YSO binaries.
#We simulate the creation and evolution of YSO binaries with RAMSES, a 3D MHD adaptive mesh refinement code. By post-processing the simulation data with the radiative transfer code RADMC-3D, we produce synthetic observations. We deploy Deep Convolutional Neural Networks (DCNN) which analyzes an observation and the synthetic images database to perform the down-selection and predict several system parameters for the observed system.
#We apply our method on the observations of systems IRAS-2A to find their simulated analogues. We describe the chosen simulated analogues and analyse their respective pre-collapse environment, evolutionary stage, accreting disk properties.
