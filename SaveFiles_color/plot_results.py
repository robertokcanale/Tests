import numpy as np
import matplotlib.pyplot as plt


iterations=500
from numpy import loadtxt
elapsed = loadtxt("SaveFiles/JetsonNano/test_HN2_keras.txt", delimiter=" ", unpack=False)


x=range(iterations)
y=elapsed

plt.figure(figsize=(8, 8))
plt.plot(x, y, label='HN1 Processing Time')
plt.xlabel("Image")
plt.ylabel("Seconds")
plt.title('HN2 Processing Time /per Image')

plt.show()
