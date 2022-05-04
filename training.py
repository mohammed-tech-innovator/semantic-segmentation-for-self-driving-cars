import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import torch.optim as optim
import os


from drive.MyDrive.semantic_code import unet_model

import torch

device =  torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')


training_set_size = 0.8
test_set_size = 1.0 - training_set_size # percentage
LR = 0.002
Epochs = 100
load_from_check_point = True
data_path = 'drive/MyDrive/data_1.db'
results_path = 'drive/MyDrive/semantic_results/'
data = []
training_loss = []
validation_loss = []
net = unet_model.UNET()


#optimizer = optim.RMSprop(net.parameters(), lr= LR, weight_decay=1e-8, momentum=0.9)


if load_from_check_point :
	status = torch.load(os.path.join(results_path  , 'status.s'))
	training_loss = status['training_loss']
	validation_loss = status['validation_loss']
	net = torch.load(results_path + 'semantic.model',map_location=torch.device('cpu'))
	net.eval()
	optimizer = optim.Adam(net.parameters(), lr= LR)
	


net = net.to(device)


content = torch.load(data_path)
data = content['data']
del content
data = data[:200]


print("data loaded number of batches is : " , len(data))
num_of_training_patches = int(training_set_size*len(data))
num_of_test_patches = len(data) - num_of_training_patches
randkey = list(range(len(data)))




for epoch in range(Epochs) :
	random.shuffle(randkey) # random selection of the training and validation patches
	
	training_index = randkey[:num_of_training_patches]
	test_index = randkey[num_of_training_patches:]
	tain_total_loss = 0
	test_total_loss = 0
	print("############################################")
	print("Epoch : ",epoch," start training over ",num_of_training_patches,"patches.")
	for index in tqdm(training_index) :

		
		net , loss = unet_model.train(net , data[index] ,optimizer)
	
		tain_total_loss = tain_total_loss + loss.cpu().detach().numpy()
		


	print()
	print("training phase is over with total loss : ",tain_total_loss ,"Avg loss : ",(tain_total_loss/num_of_training_patches))
	print("start testing over ",num_of_test_patches,"patches.")
	print()

	for index in tqdm(test_index) :
		
		loss = unet_model.test(net , data[index])
		test_total_loss = test_total_loss + loss.cpu().detach().numpy()
			

	print()
	print("testing phase is over with total loss : ",test_total_loss ,"Avg loss : ",(test_total_loss/num_of_test_patches))
	training_loss.append((tain_total_loss/num_of_training_patches))
	validation_loss.append((test_total_loss/num_of_test_patches))


	unet_model.show_example(net , data[randkey[0]])






	#save check a point
	
	torch.save(net,os.path.join(results_path ,'semantic.model'))
	torch.save(optimizer ,os.path.join(results_path ,'optimizer.op'))
	status = {
	'training_loss' : training_loss , 
	'validation_loss' : validation_loss,
	}
	torch.save(status , os.path.join(results_path ,'status.s'))
	print('check point saved !')
		


plt.plot(list(range(len(training_loss))) , training_loss , 'r')
plt.plot(list(range(len(validation_loss))) , validation_loss , 'b')
plt.show()