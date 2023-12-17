import fitz
from PIL import Image, ImageDraw
import time
import cv2
import numpy as np
import tempfile
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy import stats


def recortar_imagen(imagen):
    ancho, alto = imagen.size
    imagen = imagen.crop((0,0,ancho,alto - 50)) #Quitamos la parte inferior negra, que puede confundir el algoritmo
    izquierdo, derecho, superior, inferior = encontrar_limites_cuadricula(imagen)
    imagen_recortada = imagen.crop((izquierdo, superior, derecho, inferior))
    return imagen_recortada

def encontrar_limites_cuadricula(imagen):
    datos_pixeles = imagen.load()

    # Encontrar límites izquierdo, derecho, superior e inferior de la cuadrícula. Esquina sup izq coords 0,0
    ancho, alto = imagen.size
    izquierdo = ancho
    derecho = 0
    superior = alto
    inferior = 0

    #Encontrar limite superior
    dic = {}
    for x in range(ancho):
        for y in range(alto):
            #print("x: ", x, " y: ", y, " col: ", datos_pixeles[x,y])
            if datos_pixeles[x, y][0] < 250 and datos_pixeles[x, y][2] < 250 and datos_pixeles[x, y][1] > 50:
                #print(datos_pixeles[x,y])
                if y in dic:
                    dic[y] += 1
                else:
                    dic[y] = 1
                break;
    maxv = 0
    for k, v in dic.items():
        #print("La y ", k, " es repeteix ", v, " vegades pel limit sup")
        if v > maxv:
            maxv = v
            superior = k
        

    #Encontrar limite inferior
    dic = {}
    for x in range(ancho):
        for y in range(alto-1,-1,-1):
            if datos_pixeles[x, y][0] > 150 and datos_pixeles[x, y][0] < 250 and datos_pixeles[x, y][2] > 150 and datos_pixeles[x, y][2] < 250 and datos_pixeles[x, y][1] > 50:
                #print(datos_pixeles[x,y])                
                if y in dic:
                    dic[y] += 1
                else:
                    dic[y] = 1
                break;
    maxv = 0
    for k, v in dic.items():
        #print("La y ", k, " es repeteix ", v, " vegades pel limit inf")
        if v > maxv:
            maxv = v
            inferior = k
                
    #Encontrar limite izquierdo
    dic = {}
    for y in range(alto):
        for x in range(ancho):
            if datos_pixeles[x, y][0] < 250 and datos_pixeles[x, y][2] < 250 and datos_pixeles[x, y][1] > 50:
                if x in dic:
                    dic[x] += 1
                else:
                    dic[x] = 1
                break;
    maxv = 0
    for k, v in dic.items():
        #print("La x ", k, " es repeteix ", v, " vegades pel limit dre")
        if v > maxv:
            maxv = v
            izquierdo = k

    #Encontrar limite derecho
    dic = {}
    for y in range(alto):
        for x in range(ancho-1,-1,-1):
            if datos_pixeles[x, y][0] < 250 and datos_pixeles[x, y][2] < 250 and datos_pixeles[x, y][1] > 50:
                if x in dic:
                    dic[x] += 1
                else:
                    dic[x] = 1
                break;
    maxv = 0
    for k, v in dic.items():
        #print("La x ", k, " es repeteix ", v, " vegades pel limit esq")
        if v > maxv:
            maxv = v
            derecho = k

    
    return izquierdo, derecho, superior, inferior-2

def get_all_image_from_pdf(path):
    doc = fitz.open(path)
    arr = []
    for i in range(1,doc.page_count):
        img = get_image_from_pdf(path,i)
        arr.append(img)

    return arr

def get_image_from_pdf(path, num):
    doc = fitz.open(path)
    page = doc.load_page(num)
    pix = page.get_pixmap()
    img_bytes = pix.tobytes('pbm')
    img = Image.frombytes("RGB", (pix.width, pix.height), img_bytes)
    return img

def smooth_data(data, window_size):
    valid_data = np.array([x if x is not None else np.nan for x in data], dtype=float)
    smoothed_data = np.convolve(valid_data, np.ones(window_size) / window_size, mode='same')
    return smoothed_data

path = "/home/adria/Escritorio/PC/bitsxlamarato/RCTG/RCTG1.pdf"

imgs = get_all_image_from_pdf(path)
print("Extretes ", len(imgs), " imatges")

cut = []
for x in imgs:
    cut.append(recortar_imagen(x))



ancho_total = sum(imagen.width for imagen in cut)
alto_maximo = max(imagen.height for imagen in cut)

# Crea una nueva imagen vacía con el tamaño total
super_imagen = Image.new('RGB', (ancho_total, alto_maximo))

# Inicializa una variable para el desplazamiento horizontal
desplazamiento_horizontal = 0

# Pega cada imagen en la superimagen
for imagen in cut:
    super_imagen.paste(imagen, (desplazamiento_horizontal, 0))
    desplazamiento_horizontal += imagen.width


super_imagen.save(tempfile.gettempdir() + "/dnttouch.png")
image = cv2.imread(tempfile.gettempdir() + "/dnttouch.png")
lin = []

def trobar_punt(cord_x):
    for i in range(LIMIT_AMUN,LIMIT_ABAIX):
            b, g, r = image[i, cord_x]
            if b <= COLOR[0] and g <= COLOR[1] and r <= COLOR[2]:
                image[i, cord_x] = (255, 0, 0)
                return i
    return None

def trobar_seguent(cord_x, cord_y):
        best = 255+255+255
        best_point= None
        b = cord_y + DISPERSIO
        if b > LIMIT_ABAIX:
            b = LIMIT_ABAIX
        a = cord_y - DISPERSIO
        if a < LIMIT_AMUN:
            a=LIMIT_AMUN
        for j in range(a, b):
            b, g, r = image[j, cord_x]
            if b <= COLOR[0] and g <= COLOR[1] and r <= COLOR[2]:
                if best > g+b+r:
                    best = g+b+r
                    best_point = j
        if best_point != None:
            image[best_point, cord_x] = (255,0,0)

        return best_point

def main():
    y = None
    for x in range (image.shape[1]):
        if y == None:
            y = trobar_punt(x)
        else:
            y = trobar_seguent(x,y)
        if y != None:
            lin.append(((294-y)/28)*20)
        else:
            lin.append(None)


###Contracciones####
DISPERSIO = 10
COLOR = [225,225,225]
LIMIT_AMUN = image.shape[0]-100
LIMIT_ABAIX = image.shape[0]
main()

lin = []
DISPERSIO = 10
COLOR = [195, 195, 255]
LIMIT_AMUN = 0
LIMIT_ABAIX = 200
main()

estado = ""
temps_reducida = 0
normal = 0
reducida = 0



for i in range(len(lin)-30):
    aux=[]
    for j in range(30):
        if lin[i+j] != None:
            aux.append(lin[i+j])
    variance = stats.tstd(aux)
    if len(aux) > 20:
        if variance < 25 and variance > 5:
            estado = "normal"
            temps_reducida = 0
        elif variance > 1 and variance < 5:
            temps_reducida += 1
            if temps_reducida/30*14 == 300:
                estado = "reducida"
                if lin[i] != None:
                    cv2.circle(image, (i+15,int(294-((lin[i]/20)*28))), 15, (255, 182, 193), 1)
                print(estado)
        elif variance < 1:
            estado = "ausente"
            if lin[i] != None:
                cv2.circle(image, (i+15,int(294-((lin[i]/20)*28))), 15, (255, 215, 0), 1)
            print(estado)
        elif variance > 25:
            estado = "aumentada"
            if lin[i] != None:
                cv2.circle(image, (i+15,int(294-((lin[i]/20)*28))), 15, (30, 114, 255), 1)
            print(estado)

#20 lpm son 28 pixels. El 0 està a y = 294 (començant a dalt).      62 lineas.  14 pixeles de ancho son 30s

ilin = []
for x in lin:
    if x != None:
        ilin.append(250 - x)
    else:
        ilin.append(None)

# Encontrar picos
peaks, _ = find_peaks(lin, height=20, distance=40)
ipeaks, _ = find_peaks(ilin, height=20, distance=40)
        
#peaks, _ = find_peaks(lin, height=20, distance=50, width=7,wlen=20)
#ipeaks, _ = find_peaks(ilin, height=20, distance=50, width=7, wlen=20)

# Convertir peaks a una lista
peaks_list = list(peaks)
ipeaks_list = list(ipeaks)

datos_validos = np.array([x if x is not None else np.nan for x in lin], dtype=float)
# Calcular la media y la varianza
med = np.nanmean(datos_validos)
lin2=[]
for i in lin:
    if i != None:
        lin2.append(i)
var = stats.tstd(lin2)

print("M: ", med, " V: ", var)

spks = peaks_list + ipeaks_list
sorted(spks)



for x in range(image.shape[1]):
    image[int(294-(((med+var)/20)*28)),x] = (0,255,0)
    image[int(294-(((med-var)/20)*28)),x]  = (0,255,0)


pks = []
for pos in spks:
    if (lin[pos] > (med + var) or lin[pos] < (med-var)):
        #print("Afegit punt ", lin[pos])
        pks.append(pos)

print("Detectades ", len(pks), " anomalies")

for it in pks:
    #print("Poniendo imagen en x: ", it, " y: ", int(294-((lin[it]/20)*28)))
    cv2.circle(image, (it,int(294-((lin[it]/20)*28))), 5, (255, 0, 0), 2)




#cv2.imshow('Imagen', image)
cv2.imwrite('fotofinish.png', image)
cv2.waitKey(0)
