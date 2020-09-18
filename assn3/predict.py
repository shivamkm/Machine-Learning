import numpy as np
from keras.preprocessing.image import load_img,img_to_array
import os
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

def most_frequent(List):
	return max(set(List), key = List.count)
# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a numpy array (not numpy matrix or scipy matrix) and a list of strings.
# Make sure that the length of the array and the list is the same as the number of filenames that
# were given. The evaluation code may give unexpected results if this convention is not followed.

def decaptcha( filenames ): #output (numpy array of num_chars, list of strings)
	# numChars = 3 * np.ones( (len( filenames ),) )
	# # The use of a model file is just for sake of illustration
	# file = open( "model.txt", "r" )
	# codes = file.read().splitlines()
	# file.close()
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(100,100,3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(26, activation='softmax'))
	model.load_weights('model.h5')

	numChars=[]
	codes=[]
	ct = 0
	for image in filenames:
		ct +=1
		path = image
		img  = Image.open(path)
		pix = img.load()
		corners = [pix[0,0],pix[img.size[0]-1,0],pix[0,img.size[1]-1],pix[img.size[0]-1,img.size[1]-1]]
		bg = most_frequent(corners)


		bg2_avail = []
		for i in range(img.size[0]):
			for j in range(img.size[1]):
				if pix[i,j]== (bg[0],bg[1],bg[2],254):
					bg2_avail.append(i)
					break

		ints = []
		ints.append(bg2_avail[0])
		for i in range(1,len(bg2_avail)):
			if (bg2_avail[i]!=bg2_avail[i-1]+1):
				ints.append(bg2_avail[i-1])
				ints.append(bg2_avail[i])

		ints.append(bg2_avail[-1])
		bins = []
		i = 1
		while(i < len(ints)):
			if (ints[i]-ints[i-1]) < 50:
				pass
			elif (ints[i]-ints[i-1])<=139:
				bins.append((ints[i-1],ints[i]))
			elif ((ints[i]-ints[i-1]) > 139) and ((ints[i]-ints[i-1]) < 280):
				bins.append((ints[i-1],ints[i-1]+139))
				bins.append((ints[i]-139,ints[i]))
			elif ((ints[i]-ints[i-1]) > 139) and ((ints[i]-ints[i-1]) < 420):
				bins.append((ints[i-1],ints[i-1]+139))
				tem = ((ints[i]-ints[i-1])-140)/2
				bins.append((ints[i-1]+tem,ints[i-1]+tem+139))
				bins.append((ints[i]-139,ints[i]))
			elif ((ints[i]-ints[i-1]) > 139) and ((ints[i]-ints[i-1]) < 560):
				bins.append((ints[i-1],ints[i-1]+139))
				tem = ((ints[i]-ints[i-1])-270)/2
				bins.append((ints[i-1]+tem,ints[i-1]+tem+139))
				tem2 = ints[i]-tem
				bins.append((tem2-139,tem2))
				bins.append((ints[i]-139,ints[i]))
			i+=2

		for i in range(img.size[0]): # for every pixel:
			for j in range(img.size[1]):
				if pix[i,j]==bg:
					pix[i,j] = (255, 255, 255, 255)
				elif pix[i,j]== (bg[0],bg[1],bg[2],254):
					pix[i,j] = (255, 255, 255, 255)

		top = 0
		bottom = img.size[1]-1
		numChars.append(len(bins))
		output_str = ""
		for i in bins:
			left = i[0]
			right = i[1]
			im1 = img.crop((left, top, right, bottom))
			im1.save('testing.png')
			image_pre = load_img('testing.png',target_size=(100,100,3))
			image_pre = img_to_array(image_pre)
			to_pre = []
			to_pre.append(image_pre)
			image_pre = np.array(to_pre)
			prediction = model.predict(image_pre)[0]
			output_str += chr(65+prediction.argmax())
			os.system('rm testing.png')
		codes.append(output_str)
		#print(len(bins)," ",output_str)

	numChars = np.array(numChars)
	return (numChars, codes)
