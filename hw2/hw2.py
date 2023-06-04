import sys
import math
import string

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)



def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict.fromkeys(string.ascii_uppercase, 0)
    with open (filename,encoding='utf-8') as f:
        for line in f:
            #print(line)
            for char in line:
                    if char.upper() in X:
                        X[char.upper()]+= 1
    #print(filename)                    
    return X



def q_two(filename, letterpos):
    #letter position all tho e1 and s1, indexing starts at 0
    P_X_given_e= get_parameter_vectors()[0][letterpos] #e
    P_X_given_s= get_parameter_vectors()[1][letterpos] #s
    #print(list(shred(filename).values()))
    #print("ok")
    letter_co= list(shred(filename).values())[letterpos] #Xi
    
    return round( letter_co * math.log(P_X_given_e), 4 ), round( letter_co * math.log(P_X_given_s), 4 )



def q_three(filename):
    #bayes rule
    P_ye=.6
    P_ys=.4
    
    sum_q2_e=0
    sum_q2_s=0
    for i in range(0,26):
        sum_q2_e+=q_two(filename, i)[0] 
        sum_q2_s+=q_two(filename, i)[1]
        
    F_ye = math.log(P_ye) + sum_q2_e
    F_ys = math.log(P_ys) + sum_q2_s
    
    return round(F_ye, 4), round(F_ys, 4)



def q_four(filename):
    # if statement for english
    # elif statement for spanish, then below is an else statement
    P_ye_X = 1/(1 + math.exp(q_three(filename)[1] - q_three(filename)[0]) )
   
    return round(P_ye_X, 4)



### proper print format
print("Q1")
for i in shred("letter.txt"):
    print("{i} {x}".format(i=i, x=shred("letter.txt").get(i))) 
    
print("Q2")
print("{e:0.4f}\n{s:0.4f}".format(e=q_two("letter.txt", 0)[0],s=q_two("letter.txt", 0)[1]) )

print("Q3")
print("{e:0.4f}\n{s:0.4f}".format(e=q_three("letter.txt")[0], s=q_three("letter.txt")[1]) )

print("Q4")
print("{e:0.4f}".format( e=q_four("letter.txt") ) )