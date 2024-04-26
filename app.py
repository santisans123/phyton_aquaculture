from flask import Flask, jsonify
from main import metrics 

app = Flask(__name__)

@app.route("/", methods=["GET"])
def get_metrics():
  metricsData = metrics()
  return jsonify({"metrics": metricsData})

if __name__ == "__main__":
  app.run(debug=True)