#app.py

from flask import Flask, render_template, request
import json
import os
import tempfile
from db import *
app = Flask(__name__)


fb = firebase.FirebaseApplication('https://digiathero---med.firebaseio.com', None)

@app.route('/uploader', methods = ['POST'])
def upload_file():
   if request.method == 'POST':
      uploaded_files = request.files.getlist("file[]")
      print(uploaded_files)
      for file in uploaded_files:
          # file.save(file.filename)

          key = create_record_in_fb(fb)     # create empty record
          temp = tempfile.NamedTemporaryFile(delete=False)
          file.save(temp.name)
          url1 = upload_img_to_firebase(temp.name, name_salt=key)   # upload to FS

          # input_img = plt.imread(temp.name)  # read img to feed into the model later
          os.remove(temp.name)
          upd_source_url(url1, fb, key)     # upd URL for input image in earlier created empty record

          # get model prediction and visualisation
          #
          # pred = model(input_img)
          # plt.imwrite('visualisation_img.png', img)

          # url2 = upload_img_to_firebase('visualisation_img.png', name_salt=key)
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
