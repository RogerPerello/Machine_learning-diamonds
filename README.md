# Diamond price predictor
<p align="center">
    <img src="/.streamlit/images/front_picture.png" width="1000">
</p>

## El resultado: Diamond APPraiser

Para predecir el precio de tu diamante a partir de una foto o de sus características [haz clic aquí](https://rogerperello-machine-learning-diamon--streamlitfilesmain-bke5k8.streamlit.app/)

## Los objetivos
- Se tiene como objetivo inicial de este proyecto superar la cifra de la predicción ganadora de [una competición de Kaggle ya cerrada](https://www.kaggle.com/competitions/diamonds-part-datamad0122/) y ser capaz de predecir el precio de cualquier diamante a partir de sus características

- Se usa un segundo modelo en lugar del de la competición para obtener los precios a partir de esas características. Ese otro modelo se entrena con [el "dataset" original de diamantes](https://www.kaggle.com/datasets/swatikhedekar/price-prediction-of-diamond), pues su variable "target" es más precisa que la del torneo, dado que no se ha escalado ni redondeado

- El objetivo secundario es predecir el precio a partir de una imagen. Esto se hace con una combinación de modelos. [El "dataset" con las imágenes de los diamantes y sus características se obtiene, también, de Kaggle](https://www.kaggle.com/datasets/harshitlakhani/natural-diamonds-prices-images)

- Con los modelos que obtienen el precio a partir de una foto y el que predice los precios a partir de sus características se crea, además, [una "app" de Streamlit que les saca partido](https://rogerperello-machine-learning-diamon--streamlitfilesmain-bke5k8.streamlit.app/)

- La fuente del "dataframe" de competición y el original, según señala el autor en Kaggle, es esta: "[Tiffany & Co's snapshot pricelist from 2017](https://www.chegg.com/homework-help/questions-and-answers/data-source-tiffany-co-s-snapshot-price-list-2017-data-detail-ofiginal-dataset-peblishod-t-q103429046)" 

- En cuanto al autor de las imágenes de los diamantes, también en Kaggle, [las ha obtenido con "webscrapping"](https://capitalwholesalediamonds.com/)

## FAQ

#### ¿Que problema hay que resolver?

- Evaluar el precio de un diamante no es cosa fácil, y cuensta entre 50 y 150 dólares si se deja en manos de un profesional. Si el diamante es pequeño, eso puede ser un porcentaje importante de su valor

### ¿Que solución se aporta?

- Se ofrece una aproximación del precio de un diamante a partir de sus características o de su imagen, siempre y cuando este no supere los 2 quilates. Para los que son más grandes es recomendable acudir a un profesional; como el precio de venta del diamante será alto, saldrá a cuenta el servicio

### ¿Qué modelos se han probado?

- Muchísimos. Para predecir el precio a partir de las características:

1) Se ha hecho un baseline y se han tuneado ligeramente "LinearRegression", "Ridge", k vecinos, "SVR", "DecisionTree", "RandomForest" y "XGBRegressor"

2) Se ha hecho un "stacking" con los modelos optimizados de "XGBRegressor" y k vecinos, unidos por una regresión lineal

Para predecir el precio a partir de las imágenes:

1) Se han probado varios modelos de "transfer learning" para obtener un precio provisional (VGG16, Resnet50, mobilenet...), así como varias versiones de los mismos. Se ha elegido, al final, MobilenetV3Large

2) Se han probado varios modelos de "machine learning" para mejorar esa predicción a partir del peso de los diamantes en quilates. El baseline ha incluido "LinearRegression", "Ridge", k vecinos, "SVR", "DecisionTree", "RandomForest" y "XGBRegressor". Se ha elegido "SVR" por su buen rendimiento y se ha tuneado con el "kernel" lineal, entre otros hiperparámetros

### ¿Qué resultados y conclusiones se han obtenido?

- Es muy complicado obtener una predicción del precio de un diamante, pues hay muchos factores que no pueden tenerse en cuenta, como la moda actual (por ejemplo, es posible que un determinado color suba puntualmente de precio) o el criterio del vendedor. Además, los datos de los que se dispone son limitados, y no incluyen factores como la fluorescencia o la simetría. Con todo, es factible obtener una predicción que no sea escandalosa para diamantes que no sean exageradamente grandes

### ¿Cuáles han sido las variables de mayor impacto?

- Con diferencia, el principal factor que determina el precio de un diamante es el peso (y por extensión, el conjunto de sus dimensiones)

### ¿Qué consecuencias tiene en términos de negocio?

- La predicción obtenida probablemente no sirva para tasar directamente el diamante, pero sí para determinar, en función del criterio del usuario, si supera un varemo que justifique su venta, o para corroborar una estimación previa

### ¿Cómo podría mejorarse?

- Sin recurrir a Kaggle, ignorando la competición, haciendo el "webscrapping" directamente para obtener fotos y características. Se elegiría una página como [bluenile](https://www.bluenile.com/es/en/diamond-details/LD21034945?track=FCOM&slot=1&type=1), donde los diamantes pueden observarse en diferentes perspectivas, para poder evaluar también la profundidad y detectar más imperfecciones. Además, se extraerían características de las que actualmente no se dispone, como la simetría y el "culet"

- Obteniendo una muestra mayor, que incluyese más diamantes con precios altos

## La guía de carpetas
### [.streamlit](/.streamlit)
- Conjunto de carpetas y materiales necesarios para el funcionamiento de la "app". Puede lanzarse en local solo con ejecutar el "[launcher](/.streamlit/launcher.py)"
### [src](/src)
- [data](/src/data), que contiene todos los documentos "csv" e imágenes utilizados, tanto en bruto como procesados
- [kaggle_submission](/src/kaggle_submission), con la predicción que supera la ganadora de la competición y una captura que lo demuestra
- [models](/src/models), donde se guardan cuatro modelos en formato comprimido en sus respectivas carpetas: [predict_from_variables](/src/models/predict_from_variables), que contiene el que se usa para el torneo (competition_only.pkl) y el que predice los precios en función de sus características para la "app" (price_prediction.pkl), y [predict_from_images](/src/models/predict_from_images), con el modelo de "transfer learning" que predice un precio provisional a partir de una imagen (price_prediction_images.h5) y el que mejora el resultado a partir de esa estimación y el peso de los diamantes (price_image_prediction.pkl)
- [notebooks](/src/notebooks), con los "notebooks" de Jupiter numerados y documentados donde se lleva a cabo todo el proceso (carpeta project). Además, a modo de resumen completo, contiene un ["notebook" adicional](src/notebooks/project_resume.ipynb)
- [utils](/src/utils), donde se almacenan las clases y funciones utilizadas
- [train_competition_only.py](/src/train_competition_only.py), que entrena el mejor modelo para la competición
- [train_price_prediction.py](/src/train_price_prediction.py), que entrena el mejor modelo para la predicción de precios a partir de características en la "app"
- [train_image_provisional_prediction.py](/src/train_image_provisional_prediction.py), que entrena el mejor modelo para la predicción provisional de un precio de un diamante a partir de su foto
- [train_image_final_price_prediction.py](/src/train_image_final_price_prediction.py), que entrena el mejor modelo para la mejora del resultado provisional al tener en cuenta el peso de los diamantes
