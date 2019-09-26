# awesome-deep-trading
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

List of code, papers, and resources for AI/deep learning/machine learning/neural networks applied to algorithmic trading.

Open access: all rights granted for use and re-use of any kind, by anyone, at no cost, under your choice of either the free MIT License or Creative Commons CC-BY International Public License.

© 2019 Craig Bailes ([@cbailes](https://github.com/cbailes) | [Patreon](https://www.patreon.com/craigbailes) | [contact@craigbailes.com](mailto:contact@craigbailes.com))

# Contents
- [Papers](#papers)
  * [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
  * [Long Short-Term Memory (LSTMs)](#long-short-term-memory-lstms)
  * [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
  * [Social Processing](#social-processing)
    + [Sentiment Analysis](#sentiment-analysis)
- [Repositories](#repositories)
  * [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans-1)
  * [Cryptocurrency](#cryptocurrency) 
  * [Datasets](#datasets)
- [Resources](#resources)
  * [Presentations](#presentations)
  * [Courses](#courses)
  * [Guides & Further Reading](#guides--further-reading)

# Papers

* [Classification-based Financial Markets Prediction using Deep Neural Networks](https://arxiv.org/pdf/1603.08604) - Matthew Dixon, Diego Klabjan, Jin Hoon Bang (2016)
* [Deep Learning for Limit Order Books](https://arxiv.org/pdf/1601.01987) - Justin Sirignano (2016)
* [High-Frequency Trading Strategy Based on Deep Neural Networks](https://link.springer.com/chapter/10.1007%2F978-3-319-42297-8_40) - Andrés Arévalo, Jaime Niño, German Hernández, Javier Sandoval (2016)
* [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/pdf/1706.10059) - Zhengyao Jiang, Dixing Xu, Jinjun Liang (2017)
* [Agent Inspired Trading Using Recurrent Reinforcement Learning and LSTM Neural Networks](https://arxiv.org/pdf/1707.07338.pdf) - David W. Lu (2017)
* [Algorithmic Trading Using Deep Neural Networks on High Frequency Data](https://link.springer.com/chapter/10.1007/978-3-319-66963-2_14) - Andrés Arévalo, Jaime Niño, German Hernandez, Javier Sandoval, Diego León, Arbey Aragón (2017)
* [Deep Hedging](https://arxiv.org/pdf/1802.03042) - Hans Bühler, Lukas Gonon, Josef Teichmann, Ben Wood (2018)
* [Deep Neural Networks in High Frequency Trading](https://arxiv.org/pdf/1809.01506) - Prakhar Ganesh, Puneet Rakheja (2018)
* [Stock Trading Bot Using Deep Reinforcement Learning](https://link.springer.com/chapter/10.1007/978-981-10-8201-6_5) - Akhil Raj Azhikodan, Anvitha G. K. Bhat, Mamatha V. Jadhav (2018)
* [Financial Trading as a Game: A Deep Reinforcement Learning Approach](https://arxiv.org/pdf/1807.02787) - Chien Yi Huang (2018)
* [Practical Deep Reinforcement Learning Approach for Stock Trading](https://arxiv.org/pdf/1811.07522) - Zhuoran Xiong, Xiao-Yang Liu, Shan Zhong, Hongyang Yang, Anwar Walid (2018)
* [Algorithmic Trading and Machine Learning Based on GPU](http://ceur-ws.org/Vol-2147/p09.pdf) - Mantas Vaitonis, Saulius Masteika, Konstantinas Korovkinas (2018)
* [A quantitative trading method using deep convolution neural network ](https://iopscience.iop.org/article/10.1088/1757-899X/490/4/042018/pdf) - HaiBo Chen, DaoLei Liang, LL Zhao (2019)
* [Deep learning in exchange markets](https://www.sciencedirect.com/science/article/pii/S0167624518300702) - Rui Gonçalves, Vitor Miguel Ribeiro, Fernando Lobo Pereira, Ana Paula Rocha (2019)
* [Financial Trading Model with Stock Bar Chart Image Time Series with Deep Convolutional Neural Networks](https://arxiv.org/abs/1903.04610) - Omer Berat Sezer, Ahmet Murat Ozbayoglu (2019)
* [Deep Reinforcement Learning for Financial Trading Using Price Trailing](https://ieeexplore.ieee.org/document/8683161) -  Konstantinos Saitas Zarkias, Nikolaos Passalis, Avraam Tsantekidis, Anastasios Tefas (2019)
* [Cooperative Multi-Agent Reinforcement Learning Framework for Scalping Trading](https://arxiv.org/abs/1904.00441) - Uk Jo, Taehyun Jo, Wanjun Kim, Iljoo Yoon, Dongseok Lee, Seungho Lee (2019)
* [Improving financial trading decisions using deep Q-learning: Predicting the number of shares, action strategies, and transfer learning](https://www.sciencedirect.com/science/article/pii/S0957417418306134) - Gyeeun Jeong, Ha Young Kim (2019)
* [Deep Execution - Value and Policy Based Reinforcement Learning for Trading and Beating Market Benchmarks](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3374766) - Kevin Dabérius, Elvin Granat, Patrik Karlsson (2019)
* [An Empirical Study of Machine Learning Algorithms for Stock Daily Trading Strategy](https://www.hindawi.com/journals/mpe/2019/7816154/ref/) - Dongdong Lv, Shuhan Yuan, Meizi Li, Yang Xiang (2019)
* [Enhancing Time Series Momentum Strategies Using Deep Neural Networks](https://arxiv.org/abs/1904.04912) - Bryan Lim, Stefan Zohren, Stephen Roberts (2019)
* [Multi-Agent Deep Reinforcement Learning for Liquidation Strategy Analysis](https://arxiv.org/abs/1906.11046) - Wenhang Bao, Xiao-yang Liu (2019)
* [Deep learning-based feature engineering for stock price movement prediction](https://www.sciencedirect.com/science/article/abs/pii/S0950705118305264) - Wen Long, Zhichen Lu, Lingxiao Cui (2019)
* [Deep Hierarchical Strategy Model for Multi-Source Driven Quantitative Investment](https://ieeexplore.ieee.org/abstract/document/8743385) - Chunming Tang, Wenyan Zhu, Xiang Yu (2019)

## Convolutional Neural Networks (CNNs)
* [A deep learning based stock trading model with 2-D CNN trend detection](https://www.researchgate.net/publication/323131323_A_deep_learning_based_stock_trading_model_with_2-D_CNN_trend_detection) - Ugur Gudelek, S. Arda Boluk, Murat Ozbayoglu, Murat Ozbayoglu (2017)
* [Algorithmic Financial Trading with Deep Convolutional Neural Networks: Time Series to Image Conversion Approach](https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach) (2018)
* [DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](https://ieeexplore.ieee.org/abstract/document/8673598) - Zihao Zhang, Stefan Zohren, Stephen Roberts (2019)

## Long Short-Term Memory (LSTMs)
* [Application of Deep Learning to Algorithmic Trading - Stanford CS229](http://cs229.stanford.edu/proj2017/final-reports/5241098.pdf) (2017)

## Generative Adversarial Networks (GANs)
* [Generating Realistic Stock Market Order Streams](https://openreview.net/pdf?id=rke41hC5Km) - Anonymous Authors (2018)
* [Stock Market Prediction on High-Frequency Data Using Generative Adversarial Nets](https://www.researchgate.net/publication/324541558_Stock_Market_Prediction_on_High-Frequency_Data_Using_Generative_Adversarial_Nets) - Xingyu Zhou, Zhisong Pan, Guyu Hu, Siqi Tang (2018)
* [Generative Adversarial Networks for Financial Trading Strategies Fine-Tuning and Combination](https://deepai.org/publication/generative-adversarial-networks-for-financial-trading-strategies-fine-tuning-and-combination) - Adriano Koshiyama (2019)
* [Stock Market Prediction Based on Generative Adversarial Network](https://doi.org/10.1016/j.procs.2019.01.256) - Kang Zhang, Guoqiang Zhong, Junyu Dong, Shengke Wang, Yong Wang (2019)
* 

## Social Processing
### Sentiment Analysis
* [Improving Decision Analytics with Deep Learning: The Case of Financial Disclosures](https://arxiv.org/pdf/1508.01993) - Stefan Feuerriegel, Ralph Fehrer (2015)
* [Using Machine Learning to Predict Stock Prices](https://medium.com/analytics-vidhya/using-machine-learning-to-predict-stock-prices-c4d0b23b029a) - Vivek Palaniappan (2018)
* [Stock Prediction Using Twitter](https://towardsdatascience.com/stock-prediction-using-twitter-e432b35e14bd) - Khan Saad Bin Hasan (2019)

# Repositories
* [Yvictor/TradingGym](https://github.com/Yvictor/TradingGym) - Trading and Backtesting environment for training reinforcement learning agent or simple rule base algo
* [Rachnog/Deep-Trading](https://github.com/Rachnog/Deep-Trading) - Experimental time series forecasting
* [jobvisser03/deep-trading-advisor](https://github.com/jobvisser03/deep-trading-advisor) - Deep Trading Advisor uses MLP, CNN, and RNN+LSTM with Keras, zipline, Dash and Plotly
* [rosdyana/CNN-Financial-Data](https://github.com/rosdyana/CNN-Financial-Data) - Deep Trading using a Convolutional Neural Network
* [iamSTone/Deep-trader-CNN-kospi200futures](https://github.com/iamSTone/Deep-trader-CNN-kospi200futures) - Kospi200 index futures Prediction using CNN
* [ha2emnomer/Deep-Trading](https://github.com/ha2emnomer/Deep-Trading) - Keras-based LSTM RNN 
* [gujiuxiang/Deep_Trader.pytorch](https://github.com/gujiuxiang/Deep_Trader.pytorch) - This project uses Reinforcement learning on stock market and agent tries to learn trading. PyTorch based.
* [ZhengyaoJiang/PGPortfolio](https://github.com/ZhengyaoJiang/PGPortfolio) - PGPortfolio: Policy Gradient Portfolio, the source code of "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"
* [yuriak/RLQuant](https://github.com/yuriak/RLQuant) - Applying Reinforcement Learning in Quantitative Trading (Policy Gradient, Direct RL)
* [ucaiado/QLearning_Trading](https://github.com/ucaiado/QLearning_Trading) - Trading Using Q-Learning
* [laikasinjason/deep-q-learning-trading-system-on-hk-stocks-market](https://github.com/laikasinjason/deep-q-learning-trading-system-on-hk-stocks-market) - Deep Q learning implementation on the Hong Kong Stock Exchange
* [golsun/deep-RL-trading](https://github.com/golsun/deep-RL-trading) - Codebase for paper "Deep reinforcement learning for time series: playing idealized trading games" by Xiang Gao
* [huseinzol05/Stock-Prediction-Models](https://github.com/huseinzol05/Stock-Prediction-Models) - Gathers machine learning and deep learning models for Stock forecasting, included trading bots and simulations
* [jiewwantan/StarTrader](https://github.com/jiewwantan/StarTrader) - Trains an agent to trade like a human using a deep reinforcement learning algorithm: deep deterministic policy gradient (DDPG) learning algorithm
* [notadamking/RLTrader](https://github.com/notadamking/RLTrader) - A cryptocurrency trading environment using deep reinforcement learning and OpenAI's gym

## Generative Adversarial Networks (GANs)
* [borisbanushev/stockpredictionai](https://github.com/borisbanushev/stockpredictionai) - A notebook for stock price movement prediction using an LSTM generator and CNN discriminator
* [kah-ve/MarketGAN](https://github.com/kah-ve/MarketGAN) - Implementing a Generative Adversarial Network on the Stock Market

## Cryptocurrency
* [samre12/deep-trading-agent](https://github.com/samre12/deep-trading-agent) - Deep Reinforcement Learning-based trading agent for Bitcoin using DeepSense Network for Q function approximation.
* [ThirstyScholar/trading-bitcoin-with-reinforcement-learning](https://github.com/ThirstyScholar/trading-bitcoin-with-reinforcement-learning) - Trading Bitcoin with Reinforcement Learning
* [lefnire/tforce_btc_trader](https://github.com/lefnire/tforce_btc_trader) - A TensorForce-based Bitcoin trading bot (algo-trader). Uses deep reinforcement learning to automatically buy/sell/hold BTC based on price history.

## Datasets
* [kaggle/Huge Stock Market Dataset](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) - Historical daily prices and volumes of all U.S. stocks and ETFs
* [Alpha Vantage](https://www.alphavantage.co/) - Free APIs in JSON and CSV formats, realtime and historical stock data, FX and cryptocurrency feeds, 50+ technical indicators  
* [Quandl](https://quandl.com/)

# Resources
## Presentations
* [BigDataFinance Neural Networks Intro](http://bigdatafinance.eu/wp/wp-content/uploads/2016/06/Tefas_BigDataFinanceNeuralNetworks_Intro_Web.pdf) - Anastasios Tefas, Assistant Professor at Aristotle University of Thessaloniki (2016)
* [Trading Using Deep Learning: Motivation, Challenges, Solutions](http://on-demand.gputechconf.com/gtc-il/2017/presentation/sil7121-yam-peleg-deep-learning-for-high-frequency-trading%20(2).pdf) - Yam Peleg, GPU Technology Conference (2017)
* [FinTech, AI, Machine Learning in Finance](https://www.slideshare.net/sanjivdas/fintech-ai-machine-learning-in-finance) - Sanjiv Das (2018)
* [Deep Residual Learning for Portfolio Optimization:With Attention and Switching Modules](https://engineering.nyu.edu/sites/default/files/2019-03/NYU%20FRE%20Seminar-Jifei%20Wang%20%28slides%29.pdf) - Jeff Wang, Ph.D., NYU

## Courses
* [Artificial Intelligence for Trading (ND880) nanodegree at Udacity](https://www.udacity.com/course/ai-for-trading--nd880) (+[GitHub code repo](https://github.com/udacity/artificial-intelligence-for-trading))
* [Neural Networks in Trading course by Dr. Ernest P. Chan at Quantra](https://quantra.quantinsti.com/course/neural-networks-deep-learning-trading-ernest-chan) 

## Meetups
* [Artificial Intelligence in Finance & Algorithmic Trading on Meetup](https://www.meetup.com/Artificial-Intelligence-in-Finance-Algorithmic-Trading/) (New York City)

## Guides & Further Reading
* [Neural networks for algorithmic trading. Simple time series forecasting](https://medium.com/@alexrachnog/neural-networks-for-algorithmic-trading-part-one-simple-time-series-forecasting-f992daa1045a) - Alex Rachnog (2016)
* [Predicting Cryptocurrency Prices With Deep Learning](https://dashee87.github.io/deep%20learning/python/predicting-cryptocurrency-prices-with-deep-learning/) - David Sheehan (2017)
* [Introduction to Learning to Trade with Reinforcement Learning](http://www.wildml.com/2018/02/introduction-to-learning-to-trade-with-reinforcement-learning/) - Denny Britz (2018)
* [Webinar: How to Forecast Stock Prices Using Deep Neural Networks](https://www.youtube.com/watch?v=RMh8AUTQWQ8) - Erez Katz, Lucena Research (2018)
* [Creating Bitcoin trading bots don’t lose money](https://towardsdatascience.com/creating-bitcoin-trading-bots-that-dont-lose-money-2e7165fb0b29) - Adam King (2019)
* [Why Deep Reinforcement Learning Can Help Improve Trading Efficiency](https://medium.com/@viktortachev/why-deep-reinforcement-learning-can-help-improve-trading-efficiency-5af57e8faf9d) - Viktor Tachev (2019)
* [Optimizing deep learning trading bots using state-of-the-art techniques](https://towardsdatascience.com/using-reinforcement-learning-to-trade-bitcoin-for-massive-profit-b69d0e8f583b) - Adam King (2019)
* [Using the latest advancements in deep learning to predict stock price movements](https://towardsdatascience.com/aifortrading-2edd6fac689d) - Boris Banushev (2019)
* [Introduction to Deep Learning Trading in Hedge Funds](https://www.toptal.com/deep-learning/deep-learning-trading-hedge-funds) - Neven Pičuljan
