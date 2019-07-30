#versao 2: cores primarias, secundarias, etc com total de 12 cores
import os
import cv2
import numpy as np
import sklearn
from sklearn.neighbors import DistanceMetric
import time

start_time = time.time()

csv = []	#cria array para csv saida
saida= open("saida.csv", "w")
saida.write('arquivo,black,red,green,blue,yellow,magenta,cyan,maroon,purple,orange,gray,white\n')
contaimagens=0
for file in sorted(os.listdir("imagens/")): #laco para carregar todas as imagens da pasta
	contaimagens=contaimagens+1
	print('\n Imagem: ',contaimagens,file)
	if file.endswith(".jpg"): #formato das imagens
		aux1 = "imagens/" + file	#path entrada
		aux2 = "modificadas/" + file	#path saida
		image = cv2.imread(aux1) #Carrega imagem
		dist = DistanceMetric.get_metric('euclidean') #Define qual distancia usar(euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobis)
		centroids = [[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[128,0,0],[128,0,128],[255,128,0],[128,128,128],[255,255,255]] #array com centroides

		v = [] #vetor para armazenar distancias
		v2 = [] #vetor para armazenar quantas vezes apareceu cada indice
		image_out = np.asarray(image) #vetor para imagem de saida

		for i in range(image.shape[0]):	#laco para largura
			for j in range(image.shape[1]): #laco para altura
				b,g,r = image[i][j]	#extrai b,g,r do pixel
				pixel=[r,g,b] #converte para rgb
				for k in range(len(centroids)):	#laco para centroides
					d = [pixel,centroids[k]]	#par para comparacao
					v.append(np.amax(sklearn.metrics.pairwise.pairwise_distances(d, metric='euclidean', n_jobs=-1))) #compara os dois vetores, pega o maior valor da matriz e adiciona ao vetor v
					#print ("Pixel " + str(i) + "," + str(j) + " para centroid " + str(centroids[k]) + " Calculado") 
					ind_min = np.argmin(v)	#Extrai o indice do menor valor dentro de v
					image_out[i][j]=centroids[ind_min]	#Escreve na imagem de saida a cor correspondente ao centroide mais proximo			
				v2.append(ind_min) #Adiciona registro do ind_min ao vetor v2
				v = []	#reseta o vetor v

			#Calcula a porcentagem de cada cor
			total = float(len(v2))
			p_black = (v2.count(0)*100)/total
			p_red = (v2.count(1)*100)/total
			p_green = (v2.count(2)*100)/total
			p_blue = (v2.count(3)*100)/total
			p_yellow = (v2.count(4)*100)/total
			p_magenta = (v2.count(5)*100)/total
			p_cyan = (v2.count(6)*100)/total
			p_maroon = (v2.count(7)*100)/total
			p_purple = (v2.count(8)*100)/total
			p_orange = (v2.count(9)*100)/total
			p_gray = (v2.count(10)*100)/total
			p_white = (v2.count(11)*100)/total

		#csv.append([file,p_black,p_red,p_green,p_blue,p_yellow,p_magenta,p_cyan,p_maroon,p_purple,p_orange,p_gray,p_white])	#cria vetor de porcentagens para cada imagem
		saida.write(file+","+str(p_black)+","+str(p_red)+","+str(p_green)+","+str(p_blue)+","+str(p_yellow)+","+str(p_magenta)+","+str(p_cyan)+","+str(p_maroon)+","+str(p_purple)+","+str(p_orange)+","+str(p_gray)+","+str(p_white)+"\n")		
		saida.flush()
		#np.savetxt("colors.csv", csv, delimiter=",",  newline='\r\n', header='arquivo,%_black,%_red,%_green,%_blue,%_yellow,%_magenta,%_cyan,%_maroon,%_purple,%_orange,%_gray,%_white')	#adiciona vetor de % dentro do csv

		cv2.imwrite(aux2, cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR)) #salva imagem de saida
		print("Tempo de processamento acumulado: %10.2f segundos ---" % (time.time() - start_time))

saida.close()










