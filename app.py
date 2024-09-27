from flask import Flask,render_template, request
import os
import torch
import model as m

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mnist',methods=['get'])
def mnist_get():
    return render_template('mnist_upload.html')

@app.route('/mnist', methods=['post'])
def mnist_post():
    f = request.files['imgfile']
    print(f)
    img_path = os.path.dirname(__file__)+'/static/upload/' + f.filename
    print(img_path)
    f.save(img_path)
    image_tensor = m.preprocess_image(img_path)
    model = m.CNN()
    model.load_state_dict(torch.load('mnist_model.pth', weights_only=True))
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        predicted = torch.argmax(output)
    print(predicted)
    return render_template('mnist_result.html', data = predicted.item(), img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True, port=8089)
