from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# 创建 Flask 应用
app = Flask(__name__, template_folder='C:\\Users\\18638\\templates')

# 加载模型和标准化器
model_rf = joblib.load(r'C:\Users\18638\random_forest_model.pkl')
scaler = joblib.load(r'C:\Users\18638\scaler.pkl')

# Flask路由：主页
@app.route("/")
def home():
    return render_template("CCAR.html")  # 确保这个文件在 templates 文件夹里

# Flask路由：预测
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 获取前端传来的数据
        data = request.get_json()
        crp = float(data['CRP'])
        albumin = float(data['Albumin'])
        cr = float(data['Cr'])

        # 计算 CAR
        car = crp / albumin

        # 标准化数据
        input_data = np.array([[car, cr]])
        input_scaled = scaler.transform(input_data)

        # 预测 CCAR
        ccar_prediction = model_rf.predict(input_scaled)[0]

        return jsonify({
            "CCAR": round(ccar_prediction, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5002, use_reloader=False)
