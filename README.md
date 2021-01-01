# awesome-deep-trading
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

List of code, papers, and resources for AI/deep learning/machine learning/neural networks applied to algorithmic trading.

Open access: all rights granted for use and re-use of any kind, by anyone, at no cost, under your choice of either the free MIT License or Creative Commons CC-BY International Public License.

© 2021 Craig Bailes ([@cbailes](https://github.com/cbailes) | [Patreon](https://www.patreon.com/craigbailes) | [contact@craigbailes.com](mailto:contact@craigbailes.com))

# Contents
- [Papers](#papers)
  * [Meta Analyses & Systematic Reviews](#meta-analyses--systematic-reviews)
  * [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
  * [Long Short-Term Memory (LSTMs)](#long-short-term-memory-lstms)
  * [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
  * [High Frequency](#high-frequency)
  * [Portfolio](#portfolio)
  * [Reinforcement Learning](#reinforcement-learning)
  * [Vulnerabilities](#vulnerabilities)
  * [Cryptocurrency](#cryptocurrency)
  * [Social Processing](#social-processing)
    + [Behavioral Analysis](#behavioral-analysis)
    + [Sentiment Analysis](#sentiment-analysis)
- [Repositories](#repositories)
  * [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans-1)
  * [Guides](#guides)
  * [Cryptocurrency](#cryptocurrency-1) 
  * [Datasets](#datasets)
    + [Simulation](#simulation)
- [Resources](#resources)
  * [Presentations](#presentations)
  * [Courses](#courses)
  * [Further Reading](#further-reading)

# Papers

* [Classification-based Financial Markets Prediction using Deep Neural Networks](https://arxiv.org/pdf/1603.08604) - Matthew Dixon, Diego Klabjan, Jin Hoon Bang (2016)
* [Deep Learning for Limit Order Books](https://arxiv.org/pdf/1601.01987) - Justin Sirignano (2016)
* [High-Frequency Trading Strategy Based on Deep Neural Networks](https://link.springer.com/chapter/10.1007%2F978-3-319-42297-8_40) - Andrés Arévalo, Jaime Niño, German Hernández, Javier Sandoval (2016)
* [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/pdf/1706.10059) - Zhengyao Jiang, Dixing Xu, Jinjun Liang (2017)
* [Agent Inspired Trading Using Recurrent Reinforcement Learning and LSTM Neural Networks](https://arxiv.org/pdf/1707.07338.pdf) - David W. Lu (2017)
* [Deep Hedging](https://arxiv.org/pdf/1802.03042) - Hans Bühler, Lukas Gonon, Josef Teichmann, Ben Wood (2018)
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
* [Recipe for Quantitative Trading with Machine Learning](http://dx.doi.org/10.2139/ssrn.3232143) - Daniel Alexandre Bloch (2019)
* [Exploring Possible Improvements to Momentum Strategies with Deep Learning](http://hdl.handle.net/2105/49940) - Adam Takács, X. Xiao (2019)
* [Enhancing Time Series Momentum Strategies Using Deep Neural Networks](https://arxiv.org/abs/1904.04912) - Bryan Lim, Stefan Zohren, Stephen Roberts (2019)
* [Multi-Agent Deep Reinforcement Learning for Liquidation Strategy Analysis](https://arxiv.org/abs/1906.11046) - Wenhang Bao, Xiao-yang Liu (2019)
* [Deep learning-based feature engineering for stock price movement prediction](https://www.sciencedirect.com/science/article/abs/pii/S0950705118305264) - Wen Long, Zhichen Lu, Lingxiao Cui (2019)
* [Review on Stock Market Forecasting & Analysis](https://www.researchgate.net/publication/340583328_Review_on_Stock_Market_Forecasting_Analysis-LSTM_Long-Short_Term_Memory_Holt's_Seasonal_MethodANN_Artificial_Neural_Network_ARIMA_Auto_Regressive_Integrated_Minimum_Average_PCA_MLP_Multi_Layers_Percep) - Anirban Bal, Debayan Ganguly, Kingshuk Chatterjee (2019)
* [Neural Networks as a Forecasting Tool in the Context of the Russian Financial Market Digitalization](https://www.researchgate.net/publication/340474330_Neural_Networks_as_a_Forecasting_Tool_in_the_Context_of_the_Russian_Financial_Market_Digitalization) - Valery Aleshin, Oleg Sviridov, Inna Nekrasova, Dmitry Shevchenko (2020)
* [Deep Hierarchical Strategy Model for Multi-Source Driven Quantitative Investment](https://ieeexplore.ieee.org/abstract/document/8743385) - Chunming Tang, Wenyan Zhu, Xiang Yu (2019)
* [Finding Efficient Stocks in BSE100: Implementation of Buffet Approach INTRODUCTION](https://www.researchgate.net/publication/340501895_Asian_Journal_of_Management_Finding_Efficient_Stocks_in_BSE100_Implementation_of_Buffet_Approach_INTRODUCTION) - Sherin Varghese, Sandeep Thakur, Medha Dhingra (2020)
* [Deep Learning in Asset Pricing](https://arxiv.org/abs/1904.00745) - Luyang Chen, Markus Pelger, Jason Zhu (2020)

## Meta Analyses & Systematic Reviews
* [Application of machine learning in stock trading: a review](http://dx.doi.org/10.14419/ijet.v7i2.33.15479) - Kok Sheng Tan, Rajasvaran Logeswaran (2018)
* [Evaluating the Performance of Machine Learning Algorithms in Financial Market Forecasting: A Comprehensive Survey](https://arxiv.org/abs/1906.07786) - Lukas Ryll, Sebastian Seidens (2019)
* [Reinforcement Learning in Financial Markets](https://www.mdpi.com/2306-5729/4/3/110/pdf) - Terry Lingze Meng, Matloob Khushi (2019)
* [Financial Time Series Forecasting with Deep Learning: A Systematic Literature Review: 2005-2019](https://arxiv.org/abs/1911.13288) - Omer Berat Sezer, Mehmet Ugur Gudelek, Ahmet Murat Ozbayoglu (2019)
* [A systematic review of fundamental and technical analysis of stock market predictions](https://www.researchgate.net/publication/335274959_A_systematic_review_of_fundamental_and_technical_analysis_of_stock_market_predictions) - Isaac kofi Nti, Adebayo Adekoya, Benjamin Asubam Weyori (2019)

## Convolutional Neural Networks (CNNs)
* [A deep learning based stock trading model with 2-D CNN trend detection](https://www.researchgate.net/publication/323131323_A_deep_learning_based_stock_trading_model_with_2-D_CNN_trend_detection) - Ugur Gudelek, S. Arda Boluk, Murat Ozbayoglu, Murat Ozbayoglu (2017)
* [Algorithmic Financial Trading with Deep Convolutional Neural Networks: Time Series to Image Conversion Approach](https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach) - Omer Berat Sezar, Murat Ozbayoglu (2018)
* [DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](https://ieeexplore.ieee.org/abstract/document/8673598) - Zihao Zhang, Stefan Zohren, Stephen Roberts (2019)

## Long Short-Term Memory (LSTMs)
* [Application of Deep Learning to Algorithmic Trading, Stanford CS229](http://cs229.stanford.edu/proj2017/final-reports/5241098.pdf) - Guanting Chen, Yatong Chen, Takahiro Fushimi (2017)
* [Stock Prices Prediction using Deep Learning Models](https://arxiv.org/abs/1909.12227) - Jialin Liu, Fei Chao, Yu-Chen Lin, Chih-Min Lin (2019)
* [Deep Learning for Stock Market Trading: A Superior Trading Strategy?](https://doi.org/10.14311/NNW.2019.29.011) - D. Fister, J. C. Mun, V. Jagrič, T. Jagrič, (2019)
* [Performance Evaluation of Recurrent Neural Networks for Short-Term Investment Decision in Stock Market](https://www.researchgate.net/publication/339751012_Performance_Evaluation_of_Recurrent_Neural_Networks_for_Short-Term_Investment_Decision_in_Stock_Market) - Alexandre P. da Silva, Silas S. L. Pereira, Mário W. L. Moreira, Joel J. P. C. Rodrigues, Ricardo A. L. Rabêlo, Kashif Saleem (2020)
* [Research on financial assets transaction prediction model based on LSTM neural network](https://doi.org/10.1007/s00521-020-04992-7) - Xue Yan, Wang Weihan & Miao Chang (2020)
* [Prediction Of Stock Trend For Swing Trades Using Long Short-Term Memory Neural Network Model](https://www.researchgate.net/publication/340789607_Prediction_Of_Stock_Trend_For_Swing_Trades_Using_Long_Short-Term_Memory_Neural_Network_Model) - Varun Totakura, V. Devasekhar, Madhu Sake (2020)
* [A novel Deep Learning Framework: Prediction and Analysis of Financial Time Series using CEEMD and LSTM](https://www.sciencedirect.com/science/article/abs/pii/S0957417420304334) - Yong'an Zhang, Binbin Yan, Memon Aasma (2020)
* [Deep Stock Predictions](https://arxiv.org/abs/2006.04992) - Akash Doshi, Alexander Issa, Puneet Sachdeva, Sina Rafati, Somnath Rakshit (2020)

## Generative Adversarial Networks (GANs)
* [Generative Adversarial Networks for Financial Trading Strategies Fine-Tuning and Combination](https://deepai.org/publication/generative-adversarial-networks-for-financial-trading-strategies-fine-tuning-and-combination) - Adriano Koshiyama (2019)
* [Stock Market Prediction Based on Generative Adversarial Network](https://doi.org/10.1016/j.procs.2019.01.256) - Kang Zhang, Guoqiang Zhong, Junyu Dong, Shengke Wang, Yong Wang (2019)
* [Generative Adversarial Network for Stock Market price Prediction](https://cs230.stanford.edu/projects_fall_2019/reports/26259829.pdf) - Ricardo Alberto Carrillo Romero (2019)
* [Generative Adversarial Network for Market Hourly Discrimination](https://mpra.ub.uni-muenchen.de/id/eprint/99846) - Luca Grilli, Domenico Santoro (2020)

## High Frequency
* [Algorithmic Trading Using Deep Neural Networks on High Frequency Data](https://link.springer.com/chapter/10.1007/978-3-319-66963-2_14) - Andrés Arévalo, Jaime Niño, German Hernandez, Javier Sandoval, Diego León, Arbey Aragón (2017)
* [Stock Market Prediction on High-Frequency Data Using Generative Adversarial Nets](https://doi.org/10.1155/2018/4907423) - Xingyu Zhou, Zhisong Pan, Guyu Hu, Siqi Tang, Cheng Zhao (2018)
* [Deep Neural Networks in High Frequency Trading](https://arxiv.org/pdf/1809.01506) - Prakhar Ganesh, Puneet Rakheja (2018)
* [Application of Machine Learning in High Frequency Trading of Stocks](https://www.ijser.org/researchpaper/Application-of-Machine-Learning-in-High-Frequency-Trading-of-Stocks.pdf) - Obi Bertrand Obi (2019)

## Portfolio
* [Multi Scenario Financial Planning via Deep Reinforcement Learning AI](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3516480) - Gordon Irlam (2020)
* [G-Learner and GIRL: Goal Based Wealth Management with Reinforcement Learning](https://arxiv.org/abs/2002.10990) - Matthew Dixon, Igor Halperin (2020)
* [Reinforcement-Learning based Portfolio Management with Augmented Asset Movement Prediction States](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-YeY.4483.pdf) - Yunan Ye, Hengzhi Pei, Boxin Wang, Pin-Yu Chen, Yada Zhu, Jun Xiao, Bo Li (2020)

## Reinforcement Learning
* [Reinforcement learning in financial markets - a survey](http://hdl.handle.net/10419/183139) - Thomas G. Fischer (2018)
* [AlphaStock: A Buying-Winners-and-Selling-Losers Investment Strategy using Interpretable Deep Reinforcement Attention Networks](https://arxiv.org/abs/1908.02646) - Jingyuan Wang, Yang Zhang, Ke Tang, Junjie Wu, Zhang Xiong
* [Capturing Financial markets to apply Deep Reinforcement Learning](https://arxiv.org/abs/1907.04373) - Souradeep Chakraborty (2019)
* [Reinforcement Learning for FX trading](http://stanford.edu/class/msande448/2019/Final_reports/gr2.pdf) - Yuqin Dai, Chris Wang, Iris Wang, Yilun Xu (2019)
* [An Application of Deep Reinforcement Learning to Algorithmic Trading](https://arxiv.org/abs/2004.06627) - Thibaut Théate, Damien Ernst (2020)
* [Single asset trading: a recurrent reinforcement learning approach](http://urn.kb.se/resolve?urn=urn:nbn:se:mdh:diva-47505) - Marko Nikolic (2020)
* [Beat China’s stock market by using Deep reinforcement learning](https://www.researchgate.net/profile/Huang_Gang9/publication/340438304_Beat_China's_stock_market_by_using_Deep_reinforcement_learning/links/5e88e007299bf130797c7a68/Beat-Chinas-stock-market-by-using-Deep-reinforcement-learning.pdf) - Gang Huang, Xiaohua Zhou, Qingyang Song (2020)
* [An Adaptive Financial Trading System Using Deep Reinforcement Learning With Candlestick Decomposing Features](https://doi.org/10.1109/ACCESS.2020.2982662) - Ding Fengqian, Luo Chao (2020)
* [Application of Deep Q-Network in Portfolio Management](https://arxiv.org/abs/2003.06365) - Ziming Gao, Yuan Gao, Yi Hu, Zhengyong Jiang, Jionglong Su (2020)
* [Deep Reinforcement Learning Pairs Trading with a Double Deep Q-Network](https://ieeexplore.ieee.org/abstract/document/9031159) - Andrew Brim (2020)
* [A reinforcement learning model based on reward correction for quantitative stock selection](https://doi.org/10.1088/1757-899X/768/7/072036) - Haibo Chen, Chenyu Zhang, Yunke Li (2020)
* [AAMDRL: Augmented Asset Management with Deep Reinforcement Learning](https://arxiv.org/abs/2010.08497) - Eric Benhamou, David Saltiel, Sandrine Ungari, Abhishek Mukhopadhyay, Jamal Atif (2020)

## Guides
* [Stock Price Prediction And Forecasting Using Stacked LSTM- Deep Learning](https://www.youtube.com/watch?v=H6du_pfuznE) - Krish Naik (2020) 
* [Comparing Arima Model and LSTM RNN Model in Time-Series Forecasting](https://analyticsindiamag.com/comparing-arima-model-and-lstm-rnn-model-in-time-series-forecasting/) - Vaibhav Kumar (2020)
* [LSTM to predict Dow Jones Industrial Average: A Time Series forecasting model](https://medium.com/analytics-vidhya/lstm-to-predict-dow-jones-industrial-average-time-series-647b0115f28c) - Sarit Maitra (2020)

## Vulnerabilities
* [Adversarial Attacks on Deep Algorithmic Trading Policies](https://arxiv.org/abs/2010.11388) - Yaser Faghan, Nancirose Piazza, Vahid Behzadan, Ali Fathi (2020)

## Cryptocurrency
* [Recommending Cryptocurrency Trading Points with Deep Reinforcement Learning Approach](https://doi.org/10.3390/app10041506) - Otabek Sattarov, Azamjon Muminov, Cheol Won Lee, Hyun Kyu Kang, Ryumduck Oh, Junho Ahn, Hyung Jun Oh, Heung Seok Jeon (2020)

## Social Processing
### Behavioral Analysis
* [Can Deep Learning Predict Risky Retail Investors? A Case Study in Financial Risk Behavior Forecasting](https://www.researchgate.net/publication/329734839_Can_Deep_Learning_Predict_Risky_Retail_Investors_A_Case_Study_in_Financial_Risk_Behavior_Forecasting) - Yaodong Yang, Alisa Kolesnikova, Stefan Lessmann, Tiejun Ma, Ming-Chien Sung, Johnnie E.V. Johnson (2019)
* [Investor behaviour monitoring based on deep learning](https://www.tandfonline.com/doi/full/10.1080/0144929X.2020.1717627?casa_token=heptguQeb3kAAAAA%3AB1D3L4udpW0l3nw0sJHSpZ9tvDjptW3HfDqa_3XrUS-9owFARbHnurpSdtCy54KzR05aTdNTwhbnMA) - Song Wang, Xiaoguang Wang, Fanglin Meng, Rongjun Yang, Yuanjun Zhao (2020)

### Sentiment Analysis
* [Improving Decision Analytics with Deep Learning: The Case of Financial Disclosures](https://arxiv.org/pdf/1508.01993) - Stefan Feuerriegel, Ralph Fehrer (2015)
* [Big Data: Deep Learning for financial sentiment analysis](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-017-0111-6) - Sahar Sohangir, Dingding Wang, Anna Pomeranets, Taghi M. Khoshgoftaar (2018)
* [Using Machine Learning to Predict Stock Prices](https://medium.com/analytics-vidhya/using-machine-learning-to-predict-stock-prices-c4d0b23b029a) - Vivek Palaniappan (2018)
* [Stock Prediction Using Twitter](https://towardsdatascience.com/stock-prediction-using-twitter-e432b35e14bd) - Khan Saad Bin Hasan (2019)
* [Sentiment and Knowledge Based Algorithmic Trading with Deep Reinforcement Learning](https://arxiv.org/abs/2001.09403) - Abhishek Nan, Anandh Perumal, Osmar R. Zaiane (2020)

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

### Simulation
* [Generating Realistic Stock Market Order Streams](https://openreview.net/pdf?id=rke41hC5Km) - Anonymous Authors (2018)
* [Deep Hedging: Learning to Simulate Equity Option Markets](https://arxiv.org/abs/1911.01700) - Magnus Wiese, Lianjun Bai, Ben Wood, Hans Buehler (2019)

# Resources
## Presentations
* [BigDataFinance Neural Networks Intro](http://bigdatafinance.eu/wp/wp-content/uploads/2016/06/Tefas_BigDataFinanceNeuralNetworks_Intro_Web.pdf) - Anastasios Tefas, Assistant Professor at Aristotle University of Thessaloniki (2016)
* [Trading Using Deep Learning: Motivation, Challenges, Solutions](http://on-demand.gputechconf.com/gtc-il/2017/presentation/sil7121-yam-peleg-deep-learning-for-high-frequency-trading%20(2).pdf) - Yam Peleg, GPU Technology Conference (2017)
* [FinTech, AI, Machine Learning in Finance](https://www.slideshare.net/sanjivdas/fintech-ai-machine-learning-in-finance) - Sanjiv Das (2018)
* [Deep Residual Learning for Portfolio Optimization:With Attention and Switching Modules](https://engineering.nyu.edu/sites/default/files/2019-03/NYU%20FRE%20Seminar-Jifei%20Wang%20%28slides%29.pdf) - Jeff Wang, Ph.D., NYU

## Courses
* [Artificial Intelligence for Trading (ND880) nanodegree at Udacity](https://www.udacity.com/course/ai-for-trading--nd880) (+[GitHub code repo](https://github.com/udacity/artificial-intelligence-for-trading))
* [Neural Networks in Trading course by Dr. Ernest P. Chan at Quantra](https://quantra.quantinsti.com/course/neural-networks-deep-learning-trading-ernest-chan)
* [Machine Learning and Reinforcement Learning in Finance Specialization by NYU at Coursera](https://www.coursera.org/specializations/machine-learning-reinforcement-finance)

## Meetups
* [Artificial Intelligence in Finance & Algorithmic Trading on Meetup](https://www.meetup.com/Artificial-Intelligence-in-Finance-Algorithmic-Trading/) (New York City)

## Further Reading
* [Neural networks for algorithmic trading. Simple time series forecasting](https://medium.com/@alexrachnog/neural-networks-for-algorithmic-trading-part-one-simple-time-series-forecasting-f992daa1045a) - Alex Rachnog (2016)
* [Predicting Cryptocurrency Prices With Deep Learning](https://dashee87.github.io/deep%20learning/python/predicting-cryptocurrency-prices-with-deep-learning/) - David Sheehan (2017)
* [Introduction to Learning to Trade with Reinforcement Learning](http://www.wildml.com/2018/02/introduction-to-learning-to-trade-with-reinforcement-learning/) - Denny Britz (2018)
* [Webinar: How to Forecast Stock Prices Using Deep Neural Networks](https://www.youtube.com/watch?v=RMh8AUTQWQ8) - Erez Katz, Lucena Research (2018)
* [Creating Bitcoin trading bots that don’t lose money](https://towardsdatascience.com/creating-bitcoin-trading-bots-that-dont-lose-money-2e7165fb0b29) - Adam King (2019)
* [Why Deep Reinforcement Learning Can Help Improve Trading Efficiency](https://medium.com/@viktortachev/why-deep-reinforcement-learning-can-help-improve-trading-efficiency-5af57e8faf9d) - Viktor Tachev (2019)
* [Optimizing deep learning trading bots using state-of-the-art techniques](https://towardsdatascience.com/using-reinforcement-learning-to-trade-bitcoin-for-massive-profit-b69d0e8f583b) - Adam King (2019)
* [Using the latest advancements in deep learning to predict stock price movements](https://towardsdatascience.com/aifortrading-2edd6fac689d) - Boris Banushev (2019)
* [RNN and LSTM — The Neural Networks with Memory](https://levelup.gitconnected.com/rnn-and-lstm-the-neural-networks-with-memory-24e4cb152d1b) - Nagesh Singh Chauhan (2020)
* [Introduction to Deep Learning Trading in Hedge Funds](https://www.toptal.com/deep-learning/deep-learning-trading-hedge-funds) - Neven Pičuljan
