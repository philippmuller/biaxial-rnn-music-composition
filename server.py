from flask import Flask
import main
app = Flask(__name__)

@app.route('/')
def hello_world():
    main.web_endpoint(m,pcs)
    return 'OOLLAA'

m,pcs = web_endpoint_create()

if __name__ == '__main__':
    app.run()

