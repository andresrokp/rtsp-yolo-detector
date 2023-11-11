##Workflow

---
Init project: set folders and main files
---
Organize .env variables
---
Paste Pollo's work >> doesnt work
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

