# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 23:41:35 2018

@author: User
"""
#**First-Step**Making imports******
import pandas as pd
import numpy as np
from collections import Counter
#stopword **********************************************
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

#Read input file*****************************************
fake_input = open("clean_fake-Train.txt", "r")
real_input = open("clean_real-Train.txt","r")
test_input = pd.read_csv('test.csv', sep=',')
test_input.columns = ['id', 'category'] 

#UNIGRAM AND BIGRAM BAG OF WORDS IMPLEMENTATION*******************************************************

bow_all_u = {} #unigram all words dictionary
bow_ri_u = {}  #unigram all real input words in dictionary
bow_fi_u = {}  #unigram all fake input word in dictionary

bow_all_b = {} #bigram all words dictionary
bow_ri_b = {}  #bigram all real input words in dictionary
bow_fi_b = {}  #bigram all fake input word in dictionary

r_line=0
f_line=0

datafake =[] #fake data all line
datareal =[] #real data all line
words_counter = [] #all word

for line in fake_input.readlines():
    f_line += 1
    line = line.rstrip("\n")
    datafake.append(line)
    line = line.split(" ")
    #unigram dictioary
    for words in line:
        #if words not in stopWords:
            words_counter.append(words)
            x = words
            if x not in bow_fi_u.keys():
                bow_fi_u[x] = 1
            else:
                bow_fi_u[x] += 1
            y = words
            if y not in bow_all_u.keys():
                bow_all_u[y] = 1
            else:
                bow_all_u[y] += 1
    #bigram dictionary
    for i in range(len(line)-1): 
        
        z = line[i],line[i+1]
        if z not in bow_fi_b.keys():
            bow_fi_b[z] = 1
        else:
            bow_fi_b[z] += 1

        t = line[i],line[i+1]
        if  t not in bow_all_b.keys():
            bow_all_b[t] = 1
        else:
            bow_all_b[t] += 1
    

for line in real_input.readlines():
    r_line += 1
    line = line.rstrip("\n")
    datareal.append(line)
    line = line.split(" ")
    #unigram dictionary
    for words in line:
        #if words not in stopWords:
            words_counter.append(words)
            x = words
            if x not in bow_ri_u.keys():
                bow_ri_u[x] = 1
            else:
                bow_ri_u[x] += 1 
    
            y = words
            if y not in bow_all_u.keys():
                bow_all_u[y] = 1
            else:
                bow_all_u[y] += 1
    #bigram dictionary
    for i in range(len(line)-1): 
        z = line[i],line[i+1]
        if  z not in bow_ri_b.keys():
            bow_ri_b[z] = 1
        else:
            bow_ri_b[z] += 1
 
        t = line[i],line[i+1] 
        if  t not in bow_all_b.keys():
            bow_all_b[t] = 1
        else:
            bow_all_b[t] += 1

#print("Unigram Fake input unique word count : ",len(bow_fi_u))
#print("Unigram Real input unique word count : ", len(bow_ri_u))
#print("Unigram All unique word count : ", len(bow_all_u))

#print("#######################################################")

#print("Bigram Fake input unique word count : ",len(bow_fi_b))
#print("Bigram Real input unique word count : ", len(bow_ri_b))
#print("Bigram All unique word count : ", len(bow_all_b))

#print("#######################################################")
#highest 3 words reals unigram 
#print("highest 3 words reals unigram :")
values1=list(bow_ri_u.values())
values1.sort()
last1=values1[-3:]
for i in bow_ri_u:
    if bow_ri_u[i] in last1:
        i
        #print(i, bow_ri_u[i])
#print("#######################################################")
#print("highest 3 words fake unigram :")
values2=list(bow_fi_u.values())
values2.sort()
last2=values2[-3:]
for i in bow_fi_u:
    if bow_fi_u[i] in last2:
        i
        #print(i, bow_fi_u[i])
#print("#######################################################")
        
#Naïve Bayes Learning
#Prior Probability 
p_real =(r_line / (r_line + f_line))
p_fake =(f_line / (r_line + f_line))
#print("P(real) = ",p_real)
#print("P(fake) = ", p_fake)

#Naïve Bayes Classification
#Likelihood

total_r_uni = 0
for r in bow_ri_u.values():
    total_r_uni += r  
#print("unigram total word real : " , total_r_uni)
    
total_f_uni = 0
for f in bow_fi_u.values():
    total_f_uni += f
#print("unigram total word fake : " , total_f_uni)

#print("#######################################################")

total_r_bi = 0
for r in bow_ri_b.values():
    total_r_bi += r  
#print("bigram total word real : " , total_r_bi)
    
total_f_bi = 0
for f in bow_fi_b.values():
    total_f_bi += f
#print("bigram total word fake : " , total_f_bi)

#print("#######################################################")

#*****************************************************************************
p_real_dict_uni = {}
p_fake_dict_uni = {}

t_line = 0
uni_true_classify = 0
uni_false_classify = 0 
val1=[]

for i in range(0 , len(test_input)):
    p_r = 1
    p_f = 1
    t_line += 1
    test_line = test_input.values[i][0]
    for word in test_line.split(" "):
        #if word not in stopWords:
        a = word
        if a not in bow_ri_u.keys():
            p_real_dict_uni[a] = (0 + 1)/(total_r_uni + len(bow_all_u))
        else:
            p_real_dict_uni[a] = (bow_ri_u[a] + 1)/(total_r_uni + len(bow_all_u)) 
                
        b = word
        if b not in bow_fi_u.keys():
            p_fake_dict_uni[b] = (0 + 1)/(total_f_uni + len(bow_all_u)) 
        else:
            p_fake_dict_uni[b] = (bow_fi_u[b] + 1)/(total_f_uni + len(bow_all_u))
                
        c = word
        p_r  = (p_r * p_real_dict_uni[c])
        p_f  = (p_f * p_fake_dict_uni[c])
            
    p_r = (p_r * p_real)
    p_f = (p_f * p_fake)
    
    if p_f > p_r:
        val1.append('fake')  
    else:
        val1.append('real')
            
#********************************************************************************

#print("total test number: ",t_line)

#print("#######################################################")

num1 =0
for v in val1:
    if v == test_input.values[num1][1]:
        uni_true_classify += 1
    else:
        uni_false_classify += 1
    num1 += 1

#print("Unigram true classify : ", uni_true_classify)
#print("Unigram false classify : ", uni_false_classify)

#print("#######################################################")

accuracy_uni = 100 * (uni_true_classify / t_line)
print("unigram Accuracy : ",accuracy_uni)

#********************************************************************************
p_real_dict_bi = {}
p_fake_dict_bi = {}

val2=[]

for i in range(0 , len(test_input)):
    p_r = 1
    p_f = 1
    test_line = test_input.values[i][0]
        
    line_b = test_line.split(" ")
    
    for i in range(len(line_b)-1): 
        
        a = line_b[i],line_b[i+1]
        if a not in bow_ri_b.keys():
            p_real_dict_bi[a] = (0 + 1)/(total_r_bi + len(bow_all_b))
        else:
            p_real_dict_bi[a] = (bow_ri_b[a] + 1)/(total_r_bi + len(bow_all_b)) 
       
        b = line_b[i],line_b[i+1]
        if b not in bow_fi_b.keys():
            p_fake_dict_bi[b] = (0 + 1)/(total_f_bi + len(bow_all_b))
        else:
            p_fake_dict_bi[b] = (bow_fi_b[b] + 1)/(total_f_bi + len(bow_all_b)) 
        
        c = line_b[i],line_b[i+1]
        p_r  = (p_r * p_real_dict_bi[c])
        p_f = (p_f * p_fake_dict_bi[c])
 
    p_r = np.log10(p_r * p_real)
    p_f = np.log10(p_f * p_fake)
    
    if p_f > p_r:
        val2.append('fake')
    else:
        val2.append('real')
        
#print("#######################################################")

bi_true_classify = 0
bi_false_classify = 0
num2 =0
for v in val2:
    if v == test_input.values[num2][1]:
        bi_true_classify += 1
    else:
        bi_false_classify += 1
    num2 += 1
    
#print("bigram true classify : ", bi_true_classify)
#print("bigram false classify : ", bi_false_classify)

#print("#######################################################")

accuracy_bi = 100 * (bi_true_classify / t_line)
print("bigram Accuracy : ",accuracy_bi)

#print("################PART 3 ###################################")

bag_of_word = Counter(words_counter)
all_lines = datafake + datareal

'''
P(class) = p_real , p_fake
P(word|class) = #of occurance of words in headlines of that class / total number of headlines of that class
'''
real_prior = np.log10(p_real)
fake_prior = np.log10(p_fake)

pword_bow = {}
for w in bag_of_word.keys():
    
    hl_num_f = 1
    for l in datafake:
        if w in l:
            hl_num_f += 1
    w_fake = hl_num_f / len(datafake)
    
    hl_num_r = 1
    for l in datareal:
        if w in l:
            hl_num_r += 1
    w_real = hl_num_r / len(datareal)

    pword_bow[w] = [w_real, w_fake]

'''
#PRESENCE

P(class|word) = P(word|class) * P(class) / (P(word|fake)*P(fake) + P(word|real)*P(real))

'''
real_w = []
fake_w = []

for w in pword_bow.keys():
    val = pword_bow[w]
    real_w.append([w, val[0] * p_real / (val[0] * p_real + val[1] * p_fake)])
    fake_w.append([w, val[1] * p_fake / (val[0] * p_real + val[1] * p_fake)])

'''print("*******Presence for real***********")
for i in range(10):
    print(sorted(real_w, key=lambda tup: tup[1], reverse=True)[i])
    
print("******Presence for fake************")
for i in range(10):
    print(sorted(fake_w, key=lambda tup: tup[1], reverse=True)[i])'''

#print("##############################################################################")

'''
#ABSENCE

P(~word|class) = 1- P(word|class)
P(~word) = headlines without word / # headlines

P(class|~word) = P(~word|class)*P(class)/P(~word)

'''
real_not_w = []
fake_not_w = []

for w in pword_bow.keys():
    headline_num = 1
    for l in all_lines:
        if not w in l:
            headline_num += 1
    not_word = headline_num / len(all_lines)
    val = pword_bow[w]
    fake_not_w.append([w, (1 - val[1]) * p_fake / not_word])
    real_not_w.append([w, (1 - val[0]) * p_real / not_word])

'''print("****Absence for real******")
for i in range(10):
    print(sorted(real_not_w, key=lambda tup: tup[1], reverse=True)[i])

print("****Absence for fake******")
for i in range(10):
    print(sorted(fake_not_w, key=lambda tup: tup[1], reverse=True)[i])'''

