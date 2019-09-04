#!/usr/bin/env python

#versao 2: cores primarias, secundarias, etc com total de 12 cores
import os
import cv2
import numpy as np
from sklearn.neighbors import DistanceMetric
import time
from multiprocessing import Pool, current_process
import argparse
import signal
import tqdm
import string
from PIL import ImageEnhance
import PIL

# Variaveis globais
INPUT_DIR = None
OUTPUT_DIR = None
SIZE = None
SCALE = None
WORKERS = None
OUTPUT_FILE = None
FILE_FORMAT = None
VERBOSE = None


# Funcao principal para calcular as cores e gerar a imagem de saida para cada imagem de entrada
def pixel_color(file):

	start_time = time.time()

	if file.endswith(FILE_FORMAT): #formato das imagens

		aux1 = str(INPUT_DIR) + str(file)	#path entrada
		aux2 = str(OUTPUT_DIR) + str(file)	#path saida

		"""
		image = PIL.Image.open(aux1)
		converter = PIL.ImageEnhance.Color(image)
		image2 = converter.enhance(1.5)
		image = np.array(image2)
		image = image[:, :, ::-1].copy()
		"""

		image = cv2.imread(aux1, cv2.IMREAD_UNCHANGED)

		if FILE_FORMAT == 'png':
			image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
		else:
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


		if SIZE != None:
			image = cv2.resize(image, SIZE)
		elif SCALE != None:
			image = cv2.resize(image, (int(image.shape[0] * SCALE / 100), int(image.shape[1] * SCALE / 100)))

		#image = cv2.medianBlur(image,5) # Aplica filtro 5x5 na imagem

		#Define qual distancia usar(euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobis)
		dist = DistanceMetric.get_metric('euclidean')

		centroids_labels = ['red'] * 3 + ['orange'] * 3 + ['yellow'] * 3 + ['green'] * 3 + ['blue'] * 3 + ['purple'] * 3 + ['pink'] * 3 + ['black'] * 3 + ['gray'] * 3 + ['white'] * 3

		centroids = [[128, 0, 0],[178, 34, 34],[255, 0, 0], # red: maroon, firebrick e  red.
		[255, 69, 0],[255, 140, 0],[210, 105, 30], 			# orange: orange red, dark orange e chocolate.
		[255, 215, 0],[218, 165, 32],[255, 255, 0],			# yellow: gold, golden rod e yellow.
		[154, 205, 50],[0, 100, 0],[0, 51, 25],				# green: yellow green, dark green e lime.
		[0, 191, 255],[0, 0, 139],[70, 130, 180],			# blue: deep sky blue, dark blue e steel blue.
		#[138, 43, 226], [148, 0, 211],[128, 0, 128],		# purple: blue violet, dark violet e purple.
		[51, 0, 102], [51, 0, 51],[40, 20, 50],				# purple: blue violet, dark violet e purple.
		[255, 0, 255],[255, 20, 147],[199, 21, 133],		# pink: magenta/fuchsia, deep pink e medium violet red.
		[0, 0, 0],[20, 20, 20], [40, 40, 40], 				# black
		[105, 105, 105],[128, 128, 128],[169, 169, 169], 	# gray: dim gray, gray e dark gray
		[211, 211, 211], [220, 220, 220], [245, 245, 245]]  # white: light gray, gainsboro, white smoke

		colors_dict = {'red': 0, 'orange': 0, 'yellow': 0, 'green': 0, 'blue': 0, 'purple': 0 , 'pink': 0, 'black': 0, 'gray': 0, 'white': 0}

		# Se a imagem for .png, descarta os pixels transparentes para nao considerar pixels que nao sao relevantes, alterando o resultado
		# Isso e necessario pois um pixel transparente ainda tem um valor RGB
		relevantes = list()
		if FILE_FORMAT == 'png':
			for i in range(image.shape[0]):
				for j in range(image.shape[1]):
					if (image[i][j][3] > 32):
						relevantes.append([image[i][j][0], image[i][j][1], image[i][j][2]])
			image = relevantes.copy()
			image = np.float32(image)
		else:
			image = np.reshape(image, (-1, 3))

		for pixel in image:
			aux = list()
			for k in range(len(centroids)):	#laco para centroides
				d = [pixel,centroids[k]]	#par para comparacao
				aux.append(np.amax(dist.pairwise(d)))
			ind_min = aux.index(min(aux))
			colors_dict[centroids_labels[ind_min]] += 1 # incrementa o numero de ocorrencias da cor na imagem

		total = len(image)

		csv = [file]

		#Calcula a porcentagem de cada cor
		for k,v in colors_dict.items():
			csv.append(float(v / total) * 100)


		if VERBOSE is True:
			print('Worker {:02}: file {} finished in {:.3f} seconds'.format(current_process()._identity[0], file, time.time() - start_time))

		return csv



def get_args():
	# Argumentos para serem passados ao script
	parser = argparse.ArgumentParser(prog='PixelColor')
	parser.add_argument('path', metavar='PATH',  help='path to directory to read images from')
	parser.add_argument('-c', '--cpus', help='determines how many CPUs to use. default=all', default='all')
	parser.add_argument('-sc', '--scale', help='scale image by a factor of SCALE per cent. default=no scaling', default=None, type=int)
	parser.add_argument('-s', '--size', help='resizes all images to SIZE (WIDTHxHEIGHT). overrides --scale. default=no resizing', default=None)
	parser.add_argument('-of', '--output_file', help='file to write output to. default=output.csv', default='output.csv')
	parser.add_argument('-ff', '--file_format', help='format of images that will be read. default=png', default='png')
	parser.add_argument('-v', '--verbose', help='displays information of time taken to process each individual image', action='store_true')
	#parser.add_argument('-od', '--output_dir', help='directory to write modified images to. default=pixelcolor_output/', default='pixelcolor_output/')

	# Recebe argumentos
	args = parser.parse_args()

	return args

# Funcao principal
def main():

	global INPUT_DIR
	global OUTPUT_DIR
	global SIZE
	global SCALE
	global WORKERS
	global OUTPUT_FILE
	global FILE_FORMAT
	global VERBOSE

	args = get_args()


	INPUT_DIR = args.path

	if args.cpus == 'all':
		WORKERS = os.cpu_count()
	else:
		WORKERS = int(args.cpus)

	if args.verbose is True:
		VERBOSE = True
	else:
		VERBOSE = False

	if args.size != None:
		args.size.lower()
		x = args.size.index('x')
		width = int(args.size[:x])
		height = int(args.size[x+1:])
		SIZE = (width, height)
		print('Images will be of size {}x{}\n'.format(SIZE[0], SIZE[1]))
	else:
		SIZE = None

	if args.scale != None:
		SCALE = args.scale
		print('Images will be resized to {}% of their original size\n'.format(SCALE))
	else:
		SCALE = None
	"""
	if args.output_dir == 'pixelcolor_output/':
		OUTPUT_DIR = 'pixelcolor_output/'
	else:
		OUTPUT_DIR = args.output_dir
	"""

	if args.output_file == 'output.csv':
		OUTPUT_FILE = 'output.csv'
	else:
		OUTPUT_FILE = args.output_file

	FILE_FORMAT = args.file_format


	# Cria diretorio que contem as imagens de saida
	try:
		os.mkdir(OUTPUT_DIR)
	except:
		pass

	saida = open(OUTPUT_FILE, 'w')
	saida.write('arquivo,red,orange,yellow,green,blue,purple,pink,black,gray,white\n')

	total_time = time.time()

	try:
		filelist = sorted(os.listdir(INPUT_DIR)) # Lista com todos os arquivos no diretorio
	except FileNotFoundError:
		print("Invalid directory!")
		return

	if (len(filelist) == 0):
		print("Empty directory!")
		return

	sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

	# Cria a pool e coloca os workers pra trabalhar
	print("Initializing {} workers..\n".format(WORKERS))

	try:
		with Pool(WORKERS) as p:
			signal.signal(signal.SIGINT, sigint_handler)
			if VERBOSE is True:
				result = p.map(pixel_color, filelist)
			else:
				result = list(tqdm.tqdm(p.imap(pixel_color, filelist), total=len(filelist), bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{elapsed}, {rate_fmt}{postfix}]'))
	except KeyboardInterrupt:
		print("\nCaught KeyboardInterrupt, terminating workers")
		p.terminate()
		sys.exit(0)
	else:
		p.close()
	p.join()

	# Salva os resultados no arquivo OUTPUT_FILE
	for i in result:
		saida.write('{},'.format(str(i[0])))
		for j in i[1:-1]:
			saida.write('{:.3f},'.format(j))
		saida.write('{:.3f}\n'.format(i[-1]))

	saida.close()

	print("\nAll done! Total time taken: %.2f seconds" % (time.time() - total_time))



if __name__ == '__main__':
	main()
