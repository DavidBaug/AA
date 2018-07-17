# Proyecto Final

#### Información dataset

Usar técnicas lineales, en caso de que no funcionen debemos usar modelos no lineales.

**Attribute Information:**

1. 0) The binary result of quality assessment. 0 = bad quality 1 = sufficient quality. 

2. 1) The binary result of pre-screening, where 1 indicates severe retinal abnormality and 0 its lack. 

3. 2-7) The results of MA detection. Each feature value stand for the 

   number of MAs found at the confidence levels alpha = 0.5, . . . , 1, respectively. 

4. 8-15) contain the same information as 2-7) for exudates. However,  as exudates are represented by a set of points rather than the number of  pixels constructing the lesions, these features are normalized by dividing the number of lesions with the diameter of the ROI to compensate different image sizes. 

5. 16) The euclidean distance of the center of the macula and the center of the optic disc to provide important information regarding the patient's condition. This feature is also normalized with the diameter of the ROI. 

6. 17) The diameter of the optic disc. 

7. 18) The binary result of the AM/FM-based classification. 

8. 19) Class label. 1 = contains signs of DR (Accumulative label for the Messidor classes 1, 2, 3), 0 = no signs of DR.

# laksdk

[^pojoih+7]: ñlmñlm,

Si tenemos variables categóricas

string: casa, piso, adosado...

Labelizer (casa= 0, piso = 1...)

Transformar a binario

casa (0,1)

piso(0,1)

...

N nuevas características

