######################################
#####################################
import math
import numpy as np
import scipy.io as sio  #matlab
import matplotlib.pyplot as plt

def read_data():
    filename='mnist_train.mat'  
    tra_data=sio.loadmat(filename)
    tra_data=tra_data["mnist_train"]
    tra_data/=256.0 

    filename='mnist_train_labels.mat'  
    label=sio.loadmat(filename)
    label=label["mnist_train_labels"]
    return tra_data,label

def initialize_parameters(n_x, n_y,hid_num): 
    w1=0.2*np.random.random((hid_num,n_x))-0.1
    w2=0.2*np.random.random((hid_num,hid_num))-0.1
    w3=0.2*np.random.random((n_y,hid_num))-0.1
    hid_offset1 = np.zeros((hid_num,1))
    hid_offset2 = np.zeros((hid_num,1))
    out_offset = np.zeros((n_y,1))
    parameters = {"w1": w1,
                  "hid_offset1": hid_offset1,
                  "w2": w2,
                  "hid_offset2": hid_offset2,
                  "w3": w3,
                  "out_offset": out_offset}
    return parameters

# sigmoid
def get_act(x):
      act_vec = []
      for i in x:
          act_vec.append(1/(1+np.exp(-i)))
      act_vec = np.array(act_vec)
      act_vec=act_vec.reshape((len(act_vec),1))
      return act_vec

def propagation(losslist,corretinlist,inp_lrate = 0.2,hid_lrate = 0.2,err_th = 0.01):
    filename = 'mnist_test.mat'
    test = sio.loadmat(filename)
    test_s = test["mnist_test"]
    test_s /= 256.0
    filename = 'mnist_test_labels.mat'
    testlabel = sio.loadmat(filename)
    test_l = testlabel["mnist_test_labels"]
    right = np.zeros(10)  
    numbers = np.zeros(10) 


    tra_data,label=read_data()
    data_num=len(tra_data)  #60000
    n_y=10
    n_x=len(tra_data[0]) #784
    hid_num=int(math.sqrt(n_x+10)+5) 
    parameters=initialize_parameters(n_x,n_y,hid_num)
    w1 = parameters["w1"]
    hid_offset1 = parameters["hid_offset1"]
    w2 = parameters["w2"]
    hid_offset2 = parameters["hid_offset2"]
    w3 = parameters["w3"]
    out_offset = parameters["out_offset"]
    for count in range(0,data_num):
         print (count)
         t_label = np.zeros((n_y,1))
         t_label[label[count]] = 1
         
         tra=tra_data[count].reshape((len(tra_data[count]),1))
         hid_value1 = np.dot(w1,tra) + hid_offset1      
         hid_act1 = get_act(hid_value1)                
         hid_value2 = np.dot(w2,hid_act1) + hid_offset2 
         hid_act2 = get_act(hid_value2) 
         out_value = np.dot(w3,hid_act2) + out_offset            
         out_act = get_act(out_value)               
      
         e = t_label - out_act                        
         losslist.append(e.sum())  
         out_delta = e * out_act * (1-out_act)             
         hid_delta2 = hid_act2 * (1-hid_act2) * np.dot(w3.T,out_delta) 
         hid_delta1 = hid_act1 * (1-hid_act1) * np.dot(w2.T,hid_delta2) 
         for i in range(0, n_y):
             w_temp=w3[i,:].reshape((1,len(w3[i,:])))
             w_temp += hid_lrate * out_delta[i] * (hid_act2.T)   
             w3[i,:]=w_temp
         for i in range(0, hid_num):
             w_temp1=w1[i,:].reshape((1,len(w1[i,:])))
             w_temp1 += inp_lrate * hid_delta1[i] * tra_data[count]
             w1[i,:]=w_temp1  
             w_temp2=w2[i,:].reshape((1,len(w2[i,:])))
             w_temp2 += hid_lrate * hid_delta2[i] * hid_act1.T   
             w2[i,:]=w_temp2
         
         out_offset += hid_lrate * out_delta
         hid_offset1 += inp_lrate * hid_delta1
         hid_offset2 += hid_lrate * hid_delta2

    return w1,w2,w3,hid_offset1,hid_offset2,out_offset,losslist,corretinlist
def test():
    
    filename = 'mnist_test.mat'
    test = sio.loadmat(filename)
    test_s = test["mnist_test"]
    test_s /= 256.0
    filename = 'mnist_test_labels.mat'
    testlabel = sio.loadmat(filename)
    test_l = testlabel["mnist_test_labels"]
    right = np.zeros(10)   
    numbers = np.zeros(10)  

    loss = []
    pre_correction = []
    w1,w2,w3,hid_offset1,hid_offset2,out_offset,loss,pre_correction=propagation(loss,pre_correction)
    for i in test_l:
         numbers[i] += 1

    for count in range(len(test_s)):
         tra=test_s[count].reshape((len(test_s[count]),1))
         hid_value1 = np.dot(w1,tra) + hid_offset1      
         hid_act1 = get_act(hid_value1)                
         hid_value2 = np.dot(w2,hid_act1) + hid_offset2
         hid_act2 = get_act(hid_value2)
         out_value = np.dot(w3,hid_act2) + out_offset            
         out_act = get_act(out_value)                
         if np.argmax(out_act) == test_l[count]:
             right[test_l[count]] += 1
    print (right)
    print (numbers)
    result = right/numbers #各数字正确率
    sum = right.sum()
    print (result)
    print (sum/len(test_s)) #正确率

    ##draw pic
    plt.plot(range(60000),loss)
    #plt.plot(range(4000),pre_correction)
    plt.show()
    return loss,pre_correction
loss, pre_correction = test()  
