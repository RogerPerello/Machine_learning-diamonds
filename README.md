# Diamond price predictor
<p align="center">
    <img src="/.streamlit/images/front_picture.png" width="1000">
</p>

## El resultado: Diamond APPraiser

Para predecir el precio de tu diamante a partir de una foto o de sus características [haz clic aquí](https://rogerperello-machine-learning-diamon--streamlitfilesmain-bke5k8.streamlit.app/)

## Los objetivos
- Se tiene como objetivo inicial de este proyecto superar la cifra de la predicción ganadora de [una competición de Kaggle ya cerrada](https://www.kaggle.com/competitions/diamonds-part-datamad0122/).

- El objetivo secundario es, mediante un segundo modelo que saque las características de un diamante a partir de su foto, obtener los datos necesarios para predecir el precio. [El "dataset" con las imágenes de los diamantes y sus características se obtiene, también, de Kaggle](https://www.kaggle.com/datasets/harshitlakhani/natural-diamonds-prices-images).

- Se usa un tercer modelo en lugar del de la competición para obtener los precios a partir de esas características. Ese otro modelo se entrena con [el "dataset" original de diamantes](https://www.kaggle.com/datasets/swatikhedekar/price-prediction-of-diamond), pues su variable "target" es más precisa que la del torneo, dado que no se ha escalado ni redondeado.

- Con el modelo que obtiene las características de los diamantes a partir de una foto y el modelo que predice los precios se crea, finalmente, la "app".

## La guía de carpetas
### [.streamlit](/.streamlit)
- Conjunto de carpetas y materiales necesarios para el funcionamiento de la "app". Puede lanzarse en local solo con ejecutar el "[launcher](/.streamlit/launcher.py)".
### [src](/src)
- [data](/src/data), que contiene todos los documentos "csv" e imágenes utilizados, tanto en bruto como procesados.
- [kaggle_submission](/src/kaggle_submission), con la predicción que supera la ganadora de la competición y una captura que lo demuestra.
- [models](/src/models), donde se guardan tres modelos en formato "pickle": "[competition_only.pkl](/src/models/competition_only.pkl)", el que se usa para el torneo; "[price_prediction.pkl](/src/models/price_prediction.pkl)", el que predice los precios mejor; e "[image_prediction.pkl](/src/models/image_prediction.pkl)", el que detecta las características de un diamante a partir de la imagen correspondiente.
- [notebooks](/src/notebooks), con los "notebooks" de Jupiter numerados y documentados donde se lleva a cabo todo el proceso. Además, a modo de resumen completo, contiene un ["notebook" adicional](src/notebooks/project_resume.ipynb).
- [utils](/src/utils), donde se almacenan las clases y funciones utilizadas.
- [train_competition_only.py](/src/train_competition_only.py), que entrena el mejor modelo para la competición.
- [train_price_prediction.py](/src/train_price_prediction.py), que entrena el mejor modelo para la predicción de precios en la "app".
- [train_image_prediction.py](/src/train_image_prediction.py), que entrena el mejor modelo para la predicción de características de un diamante a partir de su foto.
