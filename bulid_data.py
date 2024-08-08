import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image


image_size = (120,480)

path = 'data/semantic/Semantic-Data/'
images_path = 'vkitti_1.3.1_rgb/'
targits_path = 'vkitti_1.3.1_scenegt/'
sup_path = ['0001/','0002/','0006/','0018/','0020/']
sup_sup_path = ['15-deg-left/','15-deg-right/','30-deg-left/','30-deg-right/'
,'clone/','fog/','morning/','overcast/','rain/','sunset/']

class Data :
	def __init__(self , batch_size = 8 ,data_path = path , loading_path = path , num = 100):
		self.batch_size = batch_size
		self.data_path = data_path
		self.loading_path = path 
		self.is_bulid = False
		self.num_of_images_per_file = num
		self.num_of_batches = 0
		self.data = []
		return



	def map(self , image):
		data = np.array(image).T
		red,green,blue = data

		Car_class = (((red > 199) & (green > 199) & (blue > 199)).astype(int)).T #0
		Terrain_class = (((red == 210) & (green == 0) & (blue == 200)).astype(int)).T #1
		Sky_class = (((red == 90) & (green == 200) & (blue == 255)).astype(int)).T #2
		Tree_class = (((red == 0) & (green == 199) & (blue == 0)).astype(int)).T #3
		Vegetation_class = (((red == 90) & (green == 240) & (blue == 0)).astype(int)).T #4
		Building_class = (((red == 140) & (green == 140) & (blue == 140)).astype(int)).T #5
		Road_class = (((red == 100) & (green == 60) & (blue == 100)).astype(int)).T #6
		GuardRail_class = (((red == 255) & (green == 100) & (blue == 255)).astype(int)).T #7
		TrafficSign_class = (((red == 255) & (green == 255) & (blue == 0)).astype(int)).T #8
		TrafficLight_class = (((red == 200) & (green == 200) & (blue == 0)).astype(int)).T #9
		Pole_class = (((red == 255) & (green == 130) & (blue == 0)).astype(int)).T #10
		Misc_class = (((red == 80) & (green == 80) & (blue == 80)).astype(int)).T #11
		Truck_class = (((red == 160) & (green == 60) & (blue == 60)).astype(int)).T #12


		result = Truck_class*12 + Misc_class*11 + Pole_class*10 + TrafficLight_class*9 + TrafficSign_class*8 
		result = result + GuardRail_class*7 + Road_class*6 + Building_class*5 + Vegetation_class*4 + Tree_class*3  
		result = result + Sky_class*2 +Terrain_class*1 + Car_class*0


		return (result)



	def bulid(self):


		if self.is_bulid :
			print('data is allready bulid !!')
			return

		x = np.array([])
		y = np.array([])
		for i in tqdm(range(self.num_of_images_per_file)):
		
			x1 = np.array([])
			y1 = np.array([])
			
			for sup in (sup_path) :

				x2 = np.array([])
				y2 = np.array([])

				
				for sup_sup in(sup_sup_path) :

					image_path = os.path.join(path + images_path , sup + sup_sup + str(50 + i).zfill(5) + '.png')
					image = cv2.imread(image_path)
					image = cv2.resize(image , image_size)
					image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
					image = np.reshape(image,(1,3,image.shape[0],image.shape[1]))
					image = image.astype(np.byte)
					
					if len(x2) == 0 :
						x2 = image
					else:
						x2 = np.append(x2,image,axis = 0)

					targit_path = os.path.join(path + targits_path , sup + sup_sup + str(50 + i).zfill(5) + '.png')
					targit = Image.open(targit_path)
					targit = targit.resize(image_size)
					targit = self.map(targit)
					targit = targit.astype(np.byte)


					if len(y2) == 0 :
						y2 = [targit]
					else:
						y2 = np.append(y2,[targit],axis = 0)

				
				if len(x1) == 0 :
					x1 = x2
				else :
					x1 = np.append(x1,x2,axis = 0)
				if len(y1) == 0 :
					y1 = y2
				else :
					y1 = np.append(y1,y2,axis = 0)

				del y2
				del x2


			if len(x) == 0 :
				x =x1
			else :
				x = np.append(x,x1,axis = 0)
			if len(y) == 0 :
				y =y1
			else :
				y = np.append(y,y1,axis = 0)

			del y1
			del x1


		data_length = len(x)
		self.num_of_batches = data_length//self.batch_size

		data = []
		for batch in tqdm(range(self.num_of_batches)) :
			x_ = x[batch*self.batch_size : batch*self.batch_size + self.batch_size]
			y_ = y[batch*self.batch_size : batch*self.batch_size + self.batch_size]


			x_ = torch.tensor(x_).byte()
			y_ = torch.tensor(y_).byte()

			data.append((x_,y_))



		print("num of batches : " +str(self.num_of_batches) + " batch size : " + str(self.batch_size))


		self.data = data




	def load(self , data_loading_path = ''):


		try:
			datapath = os.path.join(data_loading_path , 'data.db')
			content = torch.load(datapath)
			self.data = content['data']
			self.data_path = content['data_path']
			self.loading_path = content['loading path']
			self.is_bulid = content['is_bulid']
			self.num_of_images_per_file = content['images_per_file']

			print('data was loaded !!')

		except:

			print('cannot find data !!')


		return


	def save(self , data_loading_path = ''):

		content = {
		'data' : self.data ,
		'data_path' : self.data_path ,
		'loading path' : self.loading_path ,
		'is_bulid' : self.is_bulid ,
		'images_per_file' : self.num_of_images_per_file
		}
		try :
			torch.save(content , os.path.join(data_loading_path , 'data_2.db' ))
		except :
			print('data saving failed please check if the path exist !')


		return
	 

d = Data()
d.bulid()
d.save( data_loading_path = 'drive/MyDrive/')
