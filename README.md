
# Introduction

**This is an on-going project that is not finished. We are actively working on it and anticipate finishing the project in early 2022.**
  
We created a sample of financial time-series images (candlestick charts and line graphs) and developed a machine learning image classification algorithm to identify previously labeled technical indicators (e.g., RSI). This project is an open-source, approximate replication of

Naftali Cohen, Tucker Balch, & Manuela Veloso. [Trading via Image Classification](https://arxiv.org/abs/1907.10046). *arXiv preprint arXiv:1907.10046*, 2019.

## Prerequisites

-  [Alpaca account](https://app.alpaca.markets/signup)

-  [vscode](https://code.visualstudio.com/)

-  [docker](https://www.docker.com/)

To run this project, you will need to have an Alpaca account. Alpaca has eligibility requirements that are detailed [here](https://alpaca.markets/support/requirements-alpaca-brokerage-account/).

We developed the project using VS code and docker. These requirements ensure that you can replicate our work. However, the code can be easily modified to run on any set-up.

## Steps

1. We retrieved the daily open, high, low, and close data on 505 firms listed in the S&P500 (see list_of_SP_500.csv in the data folder). The data ranges from 01 January 2011 to 01 January 2019. The data was retrieved from [Alpaca](https://alpaca.markets).

2. We calculated three technical indicators: [moving average convergence/divergence](https://en.wikipedia.org/wiki/MACD) (MACD), [relative strength index](https://en.wikipedia.org/wiki/Relative_strength_index) (RSI), and [Bollinger Bands](https://en.wikipedia.org/wiki/Bollinger_Bands) (BB). Each technical indicator resulted in a buy or no-buy signal for each trading day. However, we had too few buy signals for the relative strength index, so we did not consider it further.

3. For each technical indicator, we randomly sampled 20 buy and 20 no-buy signals for each ticker in the S&P 500.  

4. We created a line plot and candle stick plot for each of the randomly sampled signals. Here are some example plots (note: these examples were re-sized for illustrative purposes). 

<img src="/workspaces/image-classification-for-technical-indicators/sample_candle_plot.jpg" alt="drawing" width="300"/>
<img src="/workspaces/image-classification-for-technical-indicators/sample_line_plot.jpg" alt="drawing" width="300"/>

# Reproducing Work

Our work can be reproduced by running the notebooks. Because of the amount of data, you can run a smaller version of the project in step 1. Both notebooks are linked below.

1a. [Upload Plots](notebooks/00_upload_plots.ipynb)

1b. [Upload Plots (minimal example)](notebooks/00_upload_plots_minimal_example.ipynb)

2. Coming Soon!