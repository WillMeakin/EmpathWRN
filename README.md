Ensure "image_data_format": "channels_last", in .keras/keras.json
put fer2013.csv data in datasets folder

To train: run modelBuildFer.py N k or modelBuildCifar.py N k

n=(#ConvLayers-4)/6, k=widening factor

WRN-28-4: n=4 k=4

To run camera: python3 modelLoadEg.py WRN-28-4-Fer-Fin.h5

