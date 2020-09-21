from firebase import firebase
from datetime import datetime
from gcloud import storage
import os


def create_record_in_fb(fb):
    data = {'Date': datetime.now().timestamp() * 1000,
            'Predicted': False,
            'Source_url': 'some-url',
            'Visualisation_url': 'NEWURL_@22',
            'Predictions': {'Atelectasis': 0,
                     'Cardiomegaly': 0,
                     'Effusion': 0,
                     'Infiltration': 0,
                     'Mass': 0,
                     'Nodule': 0,
                     'Pneumonia': 0,
                     'Pneumothorax': 0,
                     'Consolidation': 0,
                     'Edema': 0,
                     'Emphysema': 0,
                     'Fibrosis': 0,
                     'Pleural_Thickening': 0,
                     'Hernia': 0 }
              }
    result = fb.post('/History', data)
    return result['name']



def upd_source_url(url, fb, key):
    fb.put(f'/History/{key}',"Source_url", url)

def upd_visualisation_url(url, fb, key):
    fb.put(f'/History/{key}',"Visualisation_url", url)
    
def upd_prediction(pred, fb, key):
    fb.put(f'/History/{key}',"Predicitons", pred)
    
    
def upload_img_to_firebase(imagePath, name='default_name', name_salt='salt', remote_save_folder = 'images'):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= "digiathero---med-firebase-adminsdk-e8553-48e828bbd0.json"
    os.environ["GCLOUD_PROJECT"] = "digiathero---med"
    client = storage.Client()

    bucket = client.get_bucket('digiathero---med.appspot.com')

    new_name = str(name_salt + '_' + name)
    remote_image_path = os.path.join(remote_save_folder, new_name)
    
    imageBlob = bucket.blob(remote_image_path)
    imageBlob.upload_from_filename(imagePath)
    return imageBlob.generate_signed_url(1000*3600*1000)

