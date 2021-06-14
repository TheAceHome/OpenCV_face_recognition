# OpenCV for face recognition
В работе используется:

 **Haarcascade_frontalface_default.xml** предобученная модель OpenCV
  
  **LBPHFaceRecognizer** модель распознавания лиц для добавления новых фотографий для будующего распознования
  
  **FaceMesh** необходим для предсказания лицевых точек
  
  ## Installation and running

1. Clone the repo
```
$ git clone 
```

2. Create a Python virtual environment named 'test' and activate it
```
$ virtualenv test
```
```
$ source test/bin/activate
```

3. Run the following command in your Terminal/Command Prompt to install the libraries required
```
$ pip3 install -r requirements.txt
```

4. To recognize face masks in real-time video streams type the following command:

```
$ python3 face_recognition.py
```