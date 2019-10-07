import requests 

with open('Indian_Number_plates.json') as file:
	data = file.readline()
	while data:
	    data=eval(data.split(',')[0][11:])
	    
	    file_name = data.split('/')[-1]
	    print("----------------------FILE NAME--------------------------", file_name)
	    r = requests.get(data) # create HTTP resposnse object 
	    with open("Images/" + file_name,'wb') as f:
	        f.write(r.content)
	        data= file.readline()




	