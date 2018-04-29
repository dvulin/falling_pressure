# -*- coding: utf-8 -*-
"""
Falling Pressure Permeability (FPP)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
dimenzije uzorka
"""

def loadExcel(sheet='Izmjereni podatci', fileName='MASTER_EXCELICA.xlsx', firstRow=6, columns='B:C'):
    data = pd.read_excel(fileName, 
                         sheetname=sheet,
                         indexcol=0,
                         skiprows = firstRow, 
                         parse_cols = columns)
    data=data.dropna(axis=0, how='any')
    return (data)

d=18.8              # promjer uzorka, d (cm)
L=5.5               # duljina uzorka, L (cm)
p_atm=101.325       # atmosferski tlak, kPa
mi=0.0000176        # viskoznost fluida, Pas
Vch=1644.           # volumen spremnika, cm3 (ili ml)
# Vch=1868.177      # volumen spremnika, cm3 (ili ml)
t1=0                # pocetno vrijeme za analizu (interval cut - start)
t2=40.              # krajnje vrijeme za analizu (interval cut - end)

d=d*0.01            # promjer uzorka, d (m)
A=0.25*np.pi*d**2   # povrsina uzorka, A (m2)
L=5.5*0.01          # duljina uzorka, L (m)
Vt=L*A              # volumen uzorka, m3
Vch=Vch*10**(-6)

df=loadExcel(sheet='falling pressure', 
             fileName='proracuni.xlsx', 
             firstRow=5, 
             columns='H:I')


df.columns=['t', 'p']               # preimenuj kolone u dataframeu u t i p

higher=df['t']>= t1                 # boolean lista provjere celija iznad t1
lower=df['t']<= t2                  # boolean lista provjere celija iznad t1

fp=df[lower & higher].copy()               # dataframe unutar gornje i donje granice
fp=fp.reset_index()
           
fp['p_t']=fp.p+p_atm                        # apsolutni tlak u mbar
p0=fp.p_t[0]                                # pocetni tlak
c=(p0+p_atm)/(p0-p_atm)                     # faktor c
fp['lnf']=np.log(c*(fp.p/(fp.p_t+p_atm)))   # ln funkcija (provjeriti zbrajanje apsolutnih tlakova)

x=fp['t'].as_matrix()
y=fp['lnf'].as_matrix()
coeff,residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
slope, intercept=coeff[0], coeff[1]             # koeficijenti linearne regresije
R2=residuals[0]/len(fp['t'])                    # R^2 pogreska
k=Vch*L*mi*slope/(A*p_atm*1000)                 # propusnost, m2
k_mD=1.013249966*k*10**15                       # propusnost, mD


fig=plt.figure()                            # kreiraj novu sliku
x1, y1 =x, fp['p'].as_matrix()
ax1 = fig.add_subplot(1,2,1)                # kreiraj subplot height, width, 
ax1.scatter(x1,y1)                            # kreiraj osi na subplotu
ax1.set_xlabel('t, s')
ax1.set_ylabel('p, mbar')

x1, y1=x, y
ax2 = fig.add_subplot(1,2,2)
fitted_info= "s = " + str(round(slope, 5)) + "\n R$^2$ = " + str(round(R2, 4))
ax2.scatter(x1,y1, label=fitted_info)                            
ax2.set_xlabel('t, s')
ax2.set_ylabel('ln funkcija')

ax2 = fig.add_subplot(1,2,2)
ax2.legend(loc='best')
plt.plot(x1, x1*slope+intercept, lw=1)

plt.tight_layout()
plt.show()
# plt.savefig('ime.png', dpi=300)

print ("propusnost, k = %s m2  (%s mD)" %(k, round(k_mD, 2)))




