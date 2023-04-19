import os
from flask import *
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'csv'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
filename = ''
df = None
X = None
y = None
X_train = None
X_test = None
y_train = None
y_test = None
model = None
y_pred = None
target = None
feature = None


def mae_mape_r2(y_test, y_pred):
    return mean_absolute_error(y_test, y_pred), mean_absolute_percentage_error(y_test, y_pred), r2_score(y_test, y_pred)


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def savemodel() :
    global model
    modelname = str(datetime.now()).replace(' ', '').replace('.', '').replace(':', '')
    fullModelFilename = './static/models/AAM-' + modelname + '.pkl'
    pickle.dump(model, open(fullModelFilename, "wb"))
    return fullModelFilename


def accuracygraph():
    # print(target)
    # print(type(y_test[target[0]].to_numpy()),y_test[target[0]].to_numpy())
    # print(type(y_pred), y_pred)

    accuracygraphname = str(datetime.now()).replace(' ', '').replace('.', '').replace(':', '')
    fullaccuracygraphFilename = './static/' + accuracygraphname + '.jpeg'

    plt.subplots(figsize=(15, 5))
    # y_test plot
    plt.subplot(1, 3, 1)
    plt.plot(y_test[target[0]].to_numpy(), color='r', label='Actual')
    plt.legend()
    # y_pred plot
    plt.subplot(1, 3, 2)
    plt.plot(y_pred, color='g', label='Predicted')
    plt.legend()
    # y_pred and y_test plot
    plt.subplot(1, 3, 3)
    plt.plot(y_test[target[0]].to_numpy(), color='r', label='Actual')
    plt.plot(y_pred, color='g', label='Predicted')
    plt.legend()
    plt.savefig(fullaccuracygraphFilename)
    return fullaccuracygraphFilename



def listToOrderedDict(lst):
    orderedDict = {}
    for i in lst:
        if orderedDict.get(i) is None:
            orderedDict[i] = len(orderedDict)
    return orderedDict


# NOTE: ADD TRY CATCH BLOCK AS THIS FUNCTION RAISES ERROR WHEN THERE IS NO NON-NUMERIC COLUMN IN THE DATASET
# This function converts non numeric term to numeric
def nonNumColToNumCol(df):
    non_numeric_cols = list(set(df.columns).difference(set(df.select_dtypes(include=np.number).columns)))
    if non_numeric_cols == []:
        return df
    else:
        new_values = {}
        for i in non_numeric_cols:
            unqValues = df[i].unique()
            orderedDict = listToOrderedDict(unqValues)
            df[i] = df[i].map(orderedDict)
            new_values[i]= orderedDict
        return df,new_values


@app.route('/savefile', methods=['POST'])
def savefile():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        global filename
        filename = './uploads/' + f.filename
        # print('Uploaded File Name : (/savefile)',filename)
        return Response(status=200)

@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/upload')
def upload_file():
    return render_template('upload.html')


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        global filename
        filename = './uploads/' + f.filename
        # print('Uploaded File Name : (/success)',filename)
        return render_template("success.html", name=f.filename)


@app.route('/info', methods=['GET'])
def info():
    global df
    text = ''
    dataframe = pd.read_csv(filename)
    df = dataframe
    _ = dataframe.info(buf=open('dfinfo.txt', 'w'))  # save to txt
    with open('dfinfo.txt', 'r') as contents:
        for lines in contents.readlines():
            text += "<pre>" + lines + "</pre>\n"
    os.remove('dfinfo.txt')
    corr = dataframe.corr().to_html()
    describe = dataframe.describe().to_html()
    non_numeric_cols = list(set(dataframe.columns).difference(set(dataframe.select_dtypes(include=np.number).columns)))
    # print('colnames : ',non_numeric_cols)
    colnamesobject = {}
    for i in non_numeric_cols:
        colnamesobject[i] = list(dataframe[i].unique())
    # print(colnamesobject)
    return jsonify({'info': text, 'corr': corr, 'describe': describe, 'categoricalcolnames' : colnamesobject})


@app.route('/columns', methods=['GET'])
def columns():
    global df
    df = pd.read_csv(filename)
    return jsonify(list(df.columns))


@app.route('/pairplot', methods=['GET'])
def pairplot():
    sns.pairplot(df)
    fname = str(datetime.now()).replace(' ', '').replace('.', '').replace(':', '')
    fullFilename = './static/' + fname + '.jpeg'
    plt.savefig(fullFilename)
    return jsonify(fullFilename)


@app.route('/colvalues', methods=['POST', 'GET'])
def colvalues():
    if request.method == 'POST':
        global df, feature, target, X, y, X_train, X_test, y_train, y_test
        df.dropna(inplace=True)
        data = request.get_json()
        feature = data['feature']
        target = data['target']
        X = df[feature]
        y = df[target]
        xAndYdf = {'X': X.to_html(), 'Y': y.to_html()}
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                            random_state=np.random.randint(100, 9999))
        return jsonify(xAndYdf)


@app.route('/encodedatacol', methods=['GET', 'POST'])
def encodedatacol():
    if request.method == 'POST':
        mappingData = request.get_json()
        mapcolname = mappingData['columnname']
        del mappingData['columnname']
        global df
        df[mapcolname].replace(mappingData, inplace=True)
        return jsonify('success')


@app.route('/main', methods=['GET', 'POST'])
def main():
    return render_template('main.html')


@app.route('/linearRegression', methods=['GET', 'POST'])
def linearRegression():
    global df, feature, target, X, y, X_train, X_test, y_train, y_test, model, y_pred
    # print('linreg',df)
    # print(target)
    linreg = LinearRegression()
    model = linreg.fit(X_train, y_train)
    intercept = linreg.intercept_
    coefficient = linreg.coef_
    y_pred = linreg.predict(X_test)
    mae, mape, r2 = mae_mape_r2(y_test, y_pred)
    graphnames = []
    try:
        for i in feature:
            graphname = str(datetime.now()).replace(' ', '').replace('.', '').replace(':', '')
            fullGraphFilename = './static/' + graphname + '.jpeg'
            sns.lmplot(x=i, y=target[0], data=df)
            plt.savefig(fullGraphFilename)
            graphnames.append(fullGraphFilename)
    except :
        graphnames = 'ErroR Occured'

    respond = {'intercept': str(intercept), 'coefficient': str(coefficient), 'mae': mae, 'mape': mape, 'r2': r2,
               'regLine': graphnames,'modelfilename' : savemodel() ,'accuracygraphFilename': accuracygraph()}
    # print(respond)
    return jsonify(respond)


@app.route('/decisionTree', methods=['GET', 'POST'])
def decisionTree():
    global df, feature, target, X, y, X_train, X_test, y_train, y_test, model, y_pred
    dtree = DecisionTreeClassifier(random_state=np.random.randint(100, 9999))
    model = dtree.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_pred, y_test)
    treename = str(datetime.now()).replace(' ', '').replace('.', '').replace(':', '')
    fullTreeFilename = './static/' + treename + '.jpeg'
    # print([str(i) for i in list(pd.unique(y.squeeze()))])  # sueeze() converts Dataframe to Series
    # print(feature)
    plt.subplots(nrows=1, ncols=1, dpi=800)
    tree.plot_tree(dtree,
                   feature_names=feature,
                   # y.squeeze().unique()
                   class_names=[str(i) for i in list(pd.unique(y.squeeze()))],
                   filled=True)
    plt.savefig(fullTreeFilename)
    # print(score)
    # print(type(y_test),y_test)
    # print(type(y_pred), y_pred)
    return jsonify({'score': score, 'treeGraphName': fullTreeFilename,'modelfilename' : savemodel(),'accuracygraphFilename': accuracygraph()})


@app.route('/naiveBayes', methods=['GET', 'POST'])
def naiveBayes():
    global df, feature, target, X, y, X_train, X_test, y_train, y_test, model, y_pred
    # print(df)
    # print(type(df))
    new_df , new_values = nonNumColToNumCol(df)
    # print(new_df)
    # print(type(new_df))
    # print(new_values)
    X = new_df[feature]
    y = new_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                        random_state=np.random.randint(100, 9999))
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae, mape, r2 = mae_mape_r2(y_test, y_pred)
    respond = { 'mae': mae, 'mape': mape, 'r2': r2, 'modelfilename': savemodel(),'accuracygraphFilename': accuracygraph()}
    # print(respond)
    return jsonify(respond)



@app.route('/predictdata', methods=['POST', 'GET'])
def predictdata():
    if request.method == 'POST':
        postdata = dict(request.get_json())
        predictfor = []
        for key in list(postdata.keys()):
            if postdata[key].isdecimal():
                predictfor.append(int(postdata[key]))
            elif isfloat(postdata[key]):
                predictfor.append(float(postdata[key]))
            elif postdata[key].isalpha():
                predictfor.append(postdata[key])
        predictedval = model.predict([predictfor])
        return jsonify(str(predictedval))




# ********************************* OCR SECTION *********************************

import pytesseract
from PIL import Image
from werkzeug.utils import secure_filename

pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR\\tesseract.exe'

def ocr_core(filename):
    text = pytesseract.image_to_string(Image.open(filename),lang='eng')
    return text


@app.route('/ocr')
def ocr():
        return render_template('ocrIndex.html')


@app.route('/capimgupload', methods=['GET', 'POST'])
def capimgupload():
    if request.method == 'POST':
        # fs = request.files['snap'] # it raise error when there is no `snap` in form
        fs = request.files.get('snap')
        if fs:
            now = datetime.now()
            p = os.path.sep.join(['static/shots', "shot_{}.png".format(str(now).replace(":", ''))])
            # print('FileStorage:', fs)
            # print('filename:', fs.filename)
            fs.save(p)
            ocr_text = ocr_core(p)
            # print(ocr_text)
            return jsonify({'filename':p,'ocr_text': ocr_text})
        else:
            return 'You forgot Snap!'

    return 'Hello World!'


@app.route('/ocr-uploader', methods = ['GET', 'POST'])
def upload_ocr_file():
    UPLOAD_FOLDER = './static/shots'
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        fullFileName = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        ocr_text = ocr_core(fullFileName)
        # print(ocr_text)
    return jsonify({'filename': fullFileName, 'ocr_text': ocr_text})




# ********************************* OCR SECTION *********************************

# ********************************* SPEECH TRANSCRIBE SECTION *********************************
import speech_recognition as sr

@app.route('/speechtotext')
def speechtotext():
        return render_template('speechtotext.html')

@app.route("/speechtranscribe", methods=["POST"])
def speechtranscribe():
    audio_file = request.files["file"]
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    text = recognizer.recognize_google(audio)
    # print('Did you say : ', text)
    return text


# ********************************* SPEECH TRANSCRIBE SECTION *********************************

import cv2
from cv2 import dnn_superres

@app.route("/imageupscaler", methods=['GET',"POST"])
def imageupscaler():
    if request.method == 'POST':
        image_file = request.files["file"]
        upscaleBy = request.form.get('upscaleBy')
        now = datetime.now()
        ipfilep = os.path.sep.join(['static/imgupscale/input', "ipimg_{}.png".format(str(now).replace(":", ''))])
        image_file.save(ipfilep)
        # print(upscaleBy)
        sr = dnn_superres.DnnSuperResImpl_create()
        if upscaleBy == '2x':
            path = "./image upscaler models/FSRCNN_x2.pb"
            sr.readModel(path)
            sr.setModel("fsrcnn", 2)
        elif upscaleBy == '3x':
            path = "./image upscaler models/ESPCN_x3.pb"
            sr.readModel(path)
            sr.setModel("espcn", 3)
        elif upscaleBy == '4x':
            path = "./image upscaler models/EDSR_x4.pb"
            sr.readModel(path)
            sr.setModel("edsr", 4)
        elif upscaleBy == '8x':
            path = "./image upscaler models/LapSRN_x8.pb"
            sr.readModel(path)
            sr.setModel("lapsrn", 8)
        # print(ipfilep)
        readipimage = cv2.imread(ipfilep)
        result = sr.upsample(readipimage)
        opfilep = os.path.sep.join(['static/imgupscale/output', "opimg_{}.png".format(str(datetime.now()).replace(":", ''))])
        cv2.imwrite(opfilep, result)
        return opfilep
    elif request.method == 'GET':
        return render_template('imageupscaler.html')






import cvzone
from cvzone.HandTrackingModule import HandDetector
from tensorflow import keras
import math

frame = None
videoprediction = ''
def gen_frames():
    global videoprediction,frame
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    offset = 20
    imagesize = 224
    model = keras.models.load_model('sign language model/new_sign_language_model.h5')

    class_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
                    12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X',
                    23: 'Y'}
    while True:
        try:
            success, img = cap.read()
            img_copy = img.copy()
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((imagesize, imagesize, 3), np.uint8) * 255
                imgCrop = img_copy[y - offset:y + h + offset, x - offset:x + w + offset]
                imgCropShape = imgCrop.shape
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imagesize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imagesize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imagesize - wCal) / 2)
                    imgWhite[0:imgResizeShape[0], wGap:wCal + wGap] = imgResize
                    img = cv2.resize(imgWhite, (imagesize, imagesize))
                    img = np.array(img)
                    img = np.expand_dims(img, axis=0)
                    img = img / 255.0
                    prediction = model.predict(img)
                    class_label = np.argmax(prediction)
                    class_text = class_labels[class_label]
                    videoprediction = class_text
                    cv2.putText(imgWhite, class_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,cv2.LINE_AA)
                else:
                    k = imagesize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imagesize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imagesize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    img = cv2.resize(imgWhite, (imagesize, imagesize))
                    img = np.array(img)
                    img = np.expand_dims(img, axis=0)
                    img = img / 255.0
                    prediction = model.predict(img)
                    class_label = np.argmax(prediction)
                    class_text = class_labels[class_label]
                    videoprediction = class_text
                    cv2.putText(imgWhite, class_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                cv2.LINE_AA)
                # encode the frame in JPEG format
                cvzone.cornerRect(img_copy, (x-offset, y-offset, w++2*offset, h+offset), 20, rt=2)
                cv2.putText(img_copy, class_text, (x + 10, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', img_copy)
                frame = buffer.tobytes()

                # yield the output frame in byte format
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                ret, buffer = cv2.imencode('.jpg', img_copy)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(e)
        # finally:
        #     global frame
        #     yield (b'--frame\r\n'
        #            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/signlangdetection')
def signlangdetection():
    return render_template('signlangdetection.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_prediction')
def video_feed_prediction():
    global videoprediction
    return videoprediction






# if __name__ == '__main__':
#     app.run(debug=True)

