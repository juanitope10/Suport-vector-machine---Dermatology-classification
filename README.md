# Suport-vector-machine-Dermatology-classification

Este trabajo presenta la implementación desde cero en R de una Máquina de Soporte
Vectorial (SVM) multiclase (One-vs-Rest) para clasificar seis patologías en el conjunto de
datos Dermatology. Tras aplicar una normalización min-max sobre las 34 variables clínicas e
histopatológicas, se evaluó el desempeño de los kernels RBF, Lineal, Sigmoidal y Polinómico
mediante un algoritmo de gradiente proyectado. Los resultados demuestran la superioridad
del enfoque no paramétrico con el kernel RBF, el cual alcanzó una exactitud del 98.65
