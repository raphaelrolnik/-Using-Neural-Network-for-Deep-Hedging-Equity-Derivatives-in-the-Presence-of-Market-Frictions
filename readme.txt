------ readme text ------




-- presentation of the file --

This file is explaining in details the guidelines to follow in order to be guided through our dissertation's deep hedging algorithm. 




-- plan of the algorithm --

The algorithm is structured the same way as our dissertation's Results & Analysis part : 
I. Initial configuration (we build a neural network hedge vs a black scholes benchmark on our historical raw data for our two datasets). 
	a) Initial run of our neural network with the parameters fine-tuned in part V.
	b) Initial run of our neural network with a set of initial number of neurons and loss functions (sensitivity analysis)

II. Robustness analysis of our algorithm with generation of synthetic data with a blend of 4 models (Heston, Bates, VG-CIR, rBergomi)


III. Testing our initial algorithm (part. I.a) with additional inputs in our neural network : RSI, interest rates and market indexes


IV. Incorporating transaction costs to our algorithm


V. Computational justifications for the choices in the neural network architecture and features (based on our datasets features)
	a) Number of neurons in the LSTM NN
	b) Loss function used as basis in the LSTM NN
	c) Batch size and number of epochs used in the LSTM NN



-- data files --


Attached are also every data that we use : 

D1. Our main data for the algorithm : historical prices of MBG.DE and MSFT stocks

D2. Additional inputs for part III. : 
	- Macroeconomic indicators (interest rates) : the historical Federal Funds Effective Rate (FEDFUND) for the MSFT neural 	network and the historical Euro Main refinancing operations fixed rate for the MBG neural network
	- Market indicators (market index performance) : the S&P500 historical data index for the MSFT neural network and the 	DAX historical data index for the MBG neural network



Some additional points : 
	- the code is working as a whole, it needs to be runned step by step from I. to IV.
	- we are storing the results in new dataframes each time in order to be able to see the results in a general fashion 	(thus the v1,v2,v3...s for each part each time). 



-- instructions --

To run the code, ensure the directory structure is as follows:
Source code - 7CCSMPRJ ROLNIK Raphaël/
	README.txt
	originality.txt
	requirements.txt
	main_code.py
	Data/
		D1. Raw starting data/
			MBG.DE
        		MSFT
		D2. Additional inputs/
			Macroeconomic indicators (interest rates)/
				ECB rates
				Fed rates
       			Market indicators (market indexes)/
 				^GDAXI
				^SPX

Navigate to the dissertation_code directory and execute the Python script using:
$ python main_code.py