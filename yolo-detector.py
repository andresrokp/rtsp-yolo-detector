from ultralytics import YOLO
import cv2
import cvzone
import math
import requests
import base64
import numpy as np
from dotenv import dotenv_values

CONFIG = dotenv_values
# URL a la que se hará envío por POST
TB_DEVICE_TELTRY_ENDPOINT = CONFIG.get('TB_DEVICE_TELTRY_ENDPOINT')
# URL para atributo de la imagen
TB_DEVICE_ATTS_ENDPOINT = CONFIG.get('TB_DEVICE_ATTS_ENDPOINT')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara o video.")
    exit()

# Nombres de clases
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
# clases no deseadas bypass
unwanted_classes = ['chair', 'tvmonitor', 'boat']
unwanted_indices = [classNames.index(cls) for cls in unwanted_classes if cls in classNames]




model = YOLO("../Yolo-Weights/yolov8n.pt")


def send_image_as_base64(url, image):
    # Convertir la imagen a formato JPEG en memoria
    retval, buffer = cv2.imencode('.jpg', image)
    # Codificar la imagen como Base64
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Crear el payload con la imagen codificada en Base64
    data = {
        'fotogenica': f"data:image/jpeg;base64,{image_base64}"
    }

    # Realizar la solicitud POST
    response = requests.post(url, json=data)

    # Retornar la respuesta (por si deseas hacer algo con ella)
    return response


# Registro inicial de objetos detectados
last_object_counts = {}


while True:

    success, img = cap.read()

    if not success:
        print("Error: No se pudo leer el frame.")
        break

    results = model(img, stream=True)

    current_object_counts = {}
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            bbox = x1, y1, w, h
            cvzone.cornerRect(img, bbox)
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            if cls not in unwanted_indices:  # Solo procesa la detección si la clase no es no deseada
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2)

                # Actualizar el conteo de objetos detectados
                if classNames[cls] in current_object_counts:
                    current_object_counts[classNames[cls]] += 1
                else:
                    current_object_counts[classNames[cls]] = 1

            # Si hay un cambio en la detección de objetos, enviar información
       # if current_object_counts != last_object_counts+1:
            print("Cambios detectados en los objetos.")
            
            # Imprimir el payload antes de enviar
            print("Payload to be sent:", current_object_counts)

            # Enviar el diccionario como JSON
            response = requests.post(TB_DEVICE_TELTRY_ENDPOINT, json=current_object_counts)
            # Y se envia la imagen donde hubo el cambio.
            image_response_test = send_image_as_base64(TB_DEVICE_ATTS_ENDPOINT,img)
            print(f"Response Code: {image_response_test.status_code}")
            print(image_response_test.text)

            # Imprimiendo el código de respuesta y el contenido de la respuesta
            print(f"Response Code: {response.status_code}")
            print(f"Response Content: {response.text}")

            # Por ahora, solo imprimiré los cambios detectados en la consola
            print("Objetos en el frame anterior:", last_object_counts)
            print("Objetos en el frame actual:", current_object_counts)




            # Mostrar el video en tiempo real
            cv2.imshow("Video en tiempo real", img)

            # Actualizar el registro de objetos detectados
        last_object_counts = current_object_counts

        # Espera 1 ms y verifica si se presionó la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release(1)
cv2.destroyAllWindows()
