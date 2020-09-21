#app.py

from flask import Flask, request
import tempfile
from db import *
from flask_cors import CORS
import time
import report_generator

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
import numpy as np
from matplotlib import pyplot as plt
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

fb = firebase.FirebaseApplication('https://digiathero---med.firebaseio.com', None)

def dcm2png(path):
    dcm = pydicom.read_file(path)
    # extracting image (pixels data)
    img = apply_voi_lut(apply_modality_lut(dcm.pixel_array, dcm), dcm)
    if not (("PhotometricInterpretation" in dcm) and (dcm.PhotometricInterpretation == 'MONOCHROME2')):
        img = np.invert(img)
    img -= img.min()
    img = img / img.max()
    img = (img * 255)
    img = img.astype(np.uint8)
    path = str(path)+'.png'
    plt.imsave(path, img)
    return path


@app.route('/get_docx', methods = ['GET', 'POST'])
def get_docx():
    path = report_generator.generate_docx(
        patient_name=request.get_json()["patient_name"],
                  doctor_name=request.get_json()["doctor_name"],
                  date=str(datetime.now()),
                  description=request.get_json()["description"],
                  pathologies= [1,2] if sum(list(request.get_json()["pathologies"].values())) > 0 else []
                 )

    return {'path': upload_img_to_firebase('demo.docx', name = 'demo.docx', name_salt=str(time.time()).split('.')[-1]) } # send file to front

@app.route('/get_sr', methods = ['GET', 'POST'])
def get_sr():
    path = report_generator.generate_sr(patient_name=request.get_json()["patient_name"],
                  doctor_name=request.get_json()["doctor_name"],
                  date=str(datetime.now()),
                  description=request.get_json()["description"],
                  pathologies=list(request.get_json()["pathologies"].keys())
                                        )
    return {'path': upload_img_to_firebase(path, name = 'sr.dcm', name_salt=str(time.time()).split('.')[-1]) } # send file to front

@app.route('/uploader', methods = ['POST'])
def upload_file():
   if request.method == 'POST':
      # print(request)
      uploaded_files = request.files.getlist("file[]")
      print(uploaded_files)
      for file in uploaded_files:
          # file.save(file.filename)

          key = create_record_in_fb(fb, filename=file.filename)     # create empty record
          temp = tempfile.NamedTemporaryFile(delete=False)
          file.save(temp.name)
          if str(file.filename).endswith('.dcm'):
              dcm2png(temp.name)
          url1 = upload_img_to_firebase(temp.name, name = file.filename, name_salt=key)   # upload to FS

          # input_img = plt.imread(temp.name)  # read img to feed into the model later
          os.remove(temp.name)
          upd_source_url(url1, fb, key)     # upd URL for input image in earlier created empty record

          # get model prediction and visualisation
          #
          # pred = model(input_img)
          # plt.imwrite('visualisation_img.png', img)

          # url2 = upload_img_to_firebase('visualisation_img.png', name = str('visual_'+file.filename), name_salt=key)
          # upd_visualisation_url(url2, fb, key)
          # upd_prediction(pred, fb, key)



      return 'file uploaded successfully'


@app.route('/')
def index():
    return f'''
    <form method="POST" enctype="multipart/form-data" action="/uploader">
        <input type="file" name="file[]" multiple="">
        <input type="submit" value="add">
    </form>
    '''

if __name__ == '__main_':
    app.run(debug=True, port=5000)  #run app in debug mode on port 5000
