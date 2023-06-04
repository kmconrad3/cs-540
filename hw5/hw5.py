import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys



def load_data(filepath): #q2
    df = pd.read_csv(filepath)
    
    fig,ax = plt.subplots(figsize=(5,5))
    ax.plot(df["year"],df["days"])
    ax.set_ylabel("Number of frozen days")
    ax.set_xlabel("Year")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(np.arange(min(df["year"]), max(df["year"])+1, 1))
    plt.savefig("plot.png")
    
    return df



def calc_features(df): #q3
    rows = df.shape[0]
    ones = [1 for i in range(rows)]
    X = np.column_stack([ones,df["year"]])
    print("Q3a:")
    print(X)
    
    Y = np.array(df["days"])
    print("Q3b:")
    print(Y)
    
    return X,Y



def regression(X,Y):
    Z = np.dot(np.transpose(X),X)
    print("Q3c:")
    print(Z)
    
    I = np.linalg.inv(Z)
    print("Q3d:")
    print(I)
    
    PI = np.dot(I, np.transpose(X))
    print("Q3e:")
    print(PI)
    
    hat_beta = np.dot(PI, Y)
    print("Q3f:")
    print(hat_beta)
               
    return hat_beta



def predict(hat_beta):
    y_test = hat_beta[0] + 2021*hat_beta[1]
    print("Q4: " + str(y_test))

    
    
def interpretation(hat_beta):
    if hat_beta[1] < 0:
        print("Q5a: <")
        print("Q5b: The number of days Lake Mendota remains frozen decreases with increase in the year.")
    if hat_beta[1] == 0:
        print("Q5a: =")
        print("Q5b: The number of days Lake Mendota remains frozen is not affected by the year.")
    if hat_beta[1] > 0:
        print("Q5a: >")
        print("Q5b: The number of days Lake Mendota remains frozen increases with increase in the year.")     

        
        
def limitation(hat_beta):
    x_star = -1*hat_beta[0]/hat_beta[1]
    print("Q6a: " + str(x_star))
    print("Q6b: We can observe from the prediction a general decreasing trend in our plot, which suggests the number of days Lake Mendota remains frozen decreases as the number of years increase.")

    
    
if __name__ == "__main__":
    filepath = sys.argv[1]
    df = load_data(filepath)
    X,Y = calc_features(df)
    hat_beta = regression(X,Y)
    predict(hat_beta)
    interpretation(hat_beta)
    limitation(hat_beta)
