# Prediction-of-overdose-Mortaliry-in-the-USA

## Outline:

  Motivation

  Project

a. clean data

b. exploratory data analysis (EDA)

c. modeling

  Future Work

## Motivation

The opioid epidemic has turned into one of the major public health catastrophes for this generation of Americans. Similar to what tobacco or HIV were to earlier generations, the opioid epidemic appears to be this era’s defining public health crisis. I want to see if it was possible to build a model to predict opioid-related mortality by State  since this type of model might give insights into where and how to target interventions.

## Project

Therefore, I decided to build a model using time series to predict the future of overdose mortality in the USA. I used a public dataset from https://data.worldbank.org/. The data contains eight features and 817 entries.

  a.First, I had to clean the data. I replaced the word ‘Suppressed’ for a zero value. Then, I convert the Deaths feature from an object into a numeric value. Finally, I subset my dataset by State.

  b.After, we performed exploratory data analysis (EDA) on my data by separating the data into States. I also made maps to see which states had the higher death by overdose in different years.
  
  ![](/Screen%20Shot%202019-11-02%20at%205.31.33%20PM.png)
  ![](/Screen%20Shot%202019-11-03%20at%2011.50.28%20AM.png)
  
  c.I used Arima model to predict future outcomes of overdose mortality by each State. I also implement time series using a packed made by Facebook called Prophet. With this packed, I could implement a more detailed analysis.
  ![](/Screen%20Shot%202019-11-03%20at%2011.51.29%20AM.png)
  ![](/Screen%20Shot%202019-11-03%20at%2011.51.50%20AM.png)
  ![](/Screen%20Shot%202019-11-03%20at%2011.52.09%20AM.png)

## Future Work

In order to further improve my model, I would like to incorporate additional features such as county, or even cities will enhance my model. I would like to find more recent data (available was 2014). Finally, I would like to implement time series forecasting using neural networks.

