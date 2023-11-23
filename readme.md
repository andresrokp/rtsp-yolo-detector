##Workflow

---
Init project: set folders and main files
---
Organize .env variables
---
Install packages
+ What a heavy sh**t... almost 6 GB
+ ultralytics bring it all
+ just cvzone appart
---
Paste Pollo's work >> Error
---
Investigate Error
`ImportError: libGL.so.1: cannot open shared object file: No such file or directory`
LEARN
* ``libGL.so.1`` is a shared binary code file (like a ddl in windows) . `libGL.so.1` is put in system by the ``libgl1`` linux package. `libgl1` is an all-terrain massive graphics library called OpenGL . Ubuntu 22 requires `libgl1-mesa-dev` . Seems ``libgl1-mesa-dev`` is a 'dummy' library wrapper to translate real libgl1
    * chatGPT
* install ``libgl1-mesa-dev``
    * https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
    * https://github.com/ultralytics/ultralytics/issues/1270
* 
* It is posible to run CUDA models in AMD chips!!!
    * https://github.com/ultralytics/yolov5/issues/6735
    * https://github.com/ultralytics/yolov5/issues/2995#issuecomment-1737833987
    * https://stackoverflow.com/questions/74539306/what-platform-to-use-for-yolo-output-when-using-amd-gpu
---
Try play >> Error: No se pudo abrir la cámara o video.
---
Investigate web cam not accesible
LEARN
+ wsl maybe has not hardware permissions
+ linux video I/O connection files are in `/dev/video*`
+ ... unconcluded/unnecesary
+ ... web cam wont be the stream in the server
---
FEAT: Connect to a youtube stream
+ Possible tu pafy lib
    + https://stackoverflow.com/questions/37555195/is-it-possible-to-stream-video-from-https-e-g-youtube-into-python-with-ope
+ gpt: "I don't want to use an external library. I do want to do it with Just plain python. How to achieve that? Please be conciese in the script"
    + Offered pytube
    + https://github.com/pytube/pytube
+ DO: choose pytube over pafy
    + much more maintained
    + much more githubs stars
---
Error: requests.exceptions.ConnectionError: requests.exceptions.ConnectionError: HTTPConnectionPool(host='192.168.1.39', port=80): Max retries exceeded with url: /api/v1/xxxxxxxxxxxxxxxxxxx/telemetry/ (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7fdab565e8c0>: Failed to establish a new connection: [Errno 113] No route to host'))
+ Test restman > no
+ ..duhh.. is the local ip...
+ DO: fix DNS
---
``Error: qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/afroklx/.local/lib/python3.10/site-packages/cv2/qt/plugins" even though it was found. This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.``
+ Seems hardcore
    + https://stackoverflow.com/questions/60042568/this-application-failed-to-start-because-no-qt-platform-plugin-could-be-initiali
+ Is graphic rendering related
    + Qt-5 is and engine for GUIs
    + https://www.linuxfromscratch.org/blfs/view/svn/x/qt5.html
+ `cv2.imshow()` triggers it
+ ...ignore by now. Not necesary to test YOLO detection
+ DO: comment `imshow()` line
+ TODO: make it work
---
Code worked :)
+ Detection images reach platform
+ caveat: the program allways dowload the video first
+ TODO: Pull a cache or download the vid for real
---
FEAT: connect to an IP camera
+ read Tapo C200 docu
+ standar URL is: `rtsp://cam_name:cam_pass@192.168.XX.XXX:554/stream2` or stream1 for HQ
+ Insert URL as cv2 stream source
+ refactor: rect draw inside condition
Worked :)
+ problem: the TB image post is failing too often



