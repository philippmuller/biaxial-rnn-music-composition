from flask import Flask
import main
app = Flask(__name__)

@app.route('/')
def hello_world():
    main.web_endpoint()
    return 'OOLLAA'

if __name__ == '__main__':
    app.run()
