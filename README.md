# LTRO-Project
Project that I worked on during my masters. Involves Time series forecasting and Vehicle routing

Mount the entire folder to Visual Studio code/Any IDE before executing any program to avoid file not found errors. Make sure Jupyter Notebook works on the said IDE. 

Explanation for notebooks and Folders inside 'src'  and 'notebooks' folder

Main tools for Forecasting and clustering
The 'LTRO_Tool.ipynb' is the tool that would be used to completely forcast a particular point, cluster predictions and to execute the VRP problem. Instructions are provided inside the Jupyter notebook

The 'Multiple Forecast.ipynb' is used to forecast for multiple points and different waste types. Instructions are provided inside the Jupyter notebook. When this notebook is run, it gives 4 excel sheets as the output. 
So, in the future if this is run and clustering needs to be done for the new predictions, make sure to make a new excel sheet 'Predictions.xlsx' with all the output sheet combined in the order of BX,Verre,OM and Carton. 

The scripts inside 'src' folder were used to test the codes and add the codes in the notebooks.

The Following python scripts were built to use in the above tools
Forecasting.py - To forecast the fill rate for a particular point and particular wastes
Clustering.py - To cluster the predicted fill rate of points based on the prediction values and distance from the depot
VRP.py - To solve the VRP of the cluster obtained from above script

'Results' folder contains Predictions obtained by running 'Multiple Forecast.ipynb' file
