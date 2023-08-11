from flask import Flask, render_template, request, flash, session, redirect, url_for
from flask_mysqldb import MySQL, MySQLdb
import bcrypt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
# db
app.config['MYSQL_HOST']     = 'localhost'
app.config['MYSQL_USER']     = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB']       = 'klasifikasi_kualitas_air'
mysql = MySQL(app)

# Tentukan ekstensi file yang diizinkan
ALLOWED_EXTENSIONS = {'csv'}

model       = tf.keras.models.load_model("model_klasifikasi_air.h5")  # model
class_names = ["Tidak Memenuhi Syarat", "Memenuhi Syarat"]  # to convert class

# Fungsi untuk memeriksa ekstensi file yang diunggah
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    if 'username' in session:
        cursor = mysql.connection.cursor()
        cursor.execute(
        "SELECT Potability, COUNT(*) as jumlah FROM dataset GROUP BY Potability")
        data = cursor.fetchall()

        # Memisahkan label dan nilai dari hasil query
        labels = [row[0] for row in data]
        values = [row[1] for row in data]

        plt.pie(values, labels=labels)
        plt.title("Jumlah Data Berdasarkan label")


        cursor.execute("SELECT COUNT(*) FROM data_training")
        data1 = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM data_testing")
        data2 = cursor.fetchone()[0]
        # cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM dataset")
        data3 = cursor.fetchone()[0]
        cursor.close()

        return render_template("pages/index.html", data=data,data1=data1,data2=data2,data3=data3)
    else:
        return redirect(url_for('login'))
        
# PAGE HOME
@app.route("/home")
def home():
    if 'username' in session:
        return render_template("pages/index.html")
    else:
        return redirect(url_for('login'))
        
# PAGE LOGIN
@app.route("/login", methods=['POST', 'GET'])
def login():
    if 'username' in session:
        session.clear()
        return redirect(url_for('login'))
    else:
        if request.method == "POST":
            username = request.form['username']
            password = request.form['password'].encode('utf-8')
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute(
                "SELECT * FROM user WHERE username = (%s)", (username,))
            data = cursor.fetchone()
            cursor.close()

            if data is not None and len(data) > 0:
                hashed_password = bcrypt.hashpw(
                    password, data['password'].encode('utf-8'))
                if (hashed_password == data['password'].encode('utf-8')):
                    session['username'] = data['username']
                    return redirect(url_for('home'))
                else:
                    flash('username/password invalid', 'error')
                    return redirect(url_for('login'))
            else:
                flash('invalid, data not found', 'error')
                return redirect(url_for('login'))
        else:
            return render_template("pages/login.html")  # page login

# PAGE REGISTER
@app.route("/register", methods=['POST', 'GET'])
def register():
    if 'username' in session:
        session.clear()
        return redirect(url_for('register'))
    else:
        if request.method == "GET":
            return render_template("pages/register.html")
        else:
            nama     = request.form['nama']
            email    = request.form['email']
            username = request.form['username']
            password = request.form['password'].encode('utf-8')
            password = bcrypt.hashpw(password, bcrypt.gensalt())

            cursor = mysql.connection.cursor()
            cursor.execute(
                """INSERT INTO
                user (nama, email, username, password)
                VALUES (%s,%s,%s,%s)""",
                (nama, email, username, password))
            mysql.connection.commit()
            session['username'] = request.form['username']
            flash('Register Berhasil', 'success')
            return render_template("pages/register.html")  # page register

# PAGE DATATABLE
@app.route("/datatable")
def datatable():
    if 'username' in session:
    # get data form
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM dataset")
        data = cursor.fetchall()
        cursor.close()
        return render_template("pages/datatable.html", data=data)  # page datatable
    else:
        return redirect(url_for('login'))

@app.route("/datamodel")
def datamodel():
    if 'username' in session:
    # get data form
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM data_training")
        data1 = cursor.fetchall()
        cursor.execute("SELECT * FROM data_testing")
        data2 = cursor.fetchall()
        cursor.close()
        return render_template("pages/dataset_model.html", datatraining=data1,datatest=data2)  # page datatable
    else:
        return redirect(url_for('login'))
    
#uploaddata
@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Periksa apakah file ada dalam request
        file = request.files['file']
        if file.filename == '':
            flash('Tidak ada file yang dipilih!', 'error')
            return redirect(url_for('form_klasifikasi'))
        # Periksa apakah file yang diunggah memiliki ekstensi yang diizinkan
        if file and allowed_file(file.filename):
            col_name = ['ph','Hardness','Solids','Chloramines','Sulfate', 
                        'Conductivity','Organic_carbon','Trihalomethanes','Turbidity']
            df = pd.read_csv(file, names=col_name, header=0,
                             sep=',', skipinitialspace=True)
            # Cek dan mengganti nilai nan dengan None
            df.replace({np.nan: None}, inplace=True)

            # Load the trained model
            model = tf.keras.models.load_model("model_klasifikasi_air.h5")  # model
            class_names = ["Tidak Memenuhi Syarat", "Memenuhi Syarat"]

            for i, row in df.iterrows():
                # Memastikan tidak ada nilai nan sebelum menyimpan ke database
                if any(pd.isnull(row)):
                    flash('Data file tidak Valid', 'error')
                    return redirect(url_for('form_klasifikasi'))
                row = row.astype(float)
                # Preprocess the input data
                input_data = row.values.reshape(1, -1)

                # Predict using the loaded model
                prediction = model.predict(input_data)
                predicted_class_index = np.argmax(prediction)
                predicted_class = class_names[predicted_class_index]

                # Insert the predicted class into the database
                cursor = mysql.connection.cursor()
                cursor.execute(
                  """INSERT INTO
                  dataset (ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity,Potability)
                  VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                  (row.ph,row.Hardness,row.Solids,
                   row.Chloramines,row.Sulfate,row.Conductivity,
                   row.Organic_carbon,row.Trihalomethanes,row.Turbidity,
                   predicted_class))
                mysql.connection.commit()
                cursor.close()

            flash('Data Berhasil di Upload', 'success')
            return redirect(url_for('datatable'))
        else:
            flash('Ekstensi File Salah!', 'error')
            return redirect(url_for('form_klasifikasi'))
    return render_template("pages/datatable.html")

#EDIT DATA
@app.route("/editdata/<int:id_dataset>", methods=['GET'])
def editdata(id_dataset):
    if 'username' in session:
        # get edit data
        cursor = mysql.connection.cursor()
        cursor.execute(
            "SELECT * FROM dataset WHERE id_dataset = (%s)", (id_dataset,))
        data = cursor.fetchall()
        # cursor.close()
        # page edit data form
        return render_template("pages/form_editdata.html", data=data)  # page form edit
    else:
        return redirect(url_for('login'))

#UPDATE DATA
@app.route("/updatedata/<int:id_dataset>", methods=['POST'])
def updatedata(id_dataset):
    if request.method == "POST":
        # insert data form
        ph          = request.form['ph']
        Hardness    = request.form['Hardness']
        Solids      = request.form['Solids']
        Chloramines  = request.form['Chloramines']
        Sulfate     = request.form['Sulfate']
        Conductivity    = request.form['Conductivity']
        Organic_carbon  = request.form['Organic_carbon']
        Trihalomethanes = request.form['Trihalomethanes']
        Turbidity       = request.form['Turbidity']
        # Potability = request.form['Potability']
        
        if not all(value.replace('.', '', 1).isdigit() for value in [ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]):
            flash('Invalid Data', 'error')
            return render_template("pages/form_editdata.html",  ph=ph, Hardness=Hardness,Solids=Solids, Chloramines=Chloramines,Sulfate=Sulfate,
                                                            Conductivity=Conductivity, Organic_carbon=Organic_carbon, Trihalomethanes=Trihalomethanes,
                                                            Turbidity=Turbidity)
      
        ph          = float(ph)
        Hardness    = float(Hardness)
        Solids      = float(Solids)
        Chloramines  = float(Chloramines)
        Sulfate     = float(Sulfate)
        Conductivity    = float(Conductivity)
        Organic_carbon  = float(Organic_carbon)
        Trihalomethanes = float(Trihalomethanes)
        Turbidity       = float(Turbidity)
        # Potability      = float(Potability)
        
        data = {
                'ph': [ph],
                'Hardness': [Hardness],
                'Solids': [Solids],
                'Chloramines': [Chloramines],
                'Sulfate': [Sulfate],
                'Conductivity': [Conductivity],
                'Organic_carbon': [Organic_carbon],
                'Trihalomethanes': [Trihalomethanes],
                'Turbidity': [Turbidity]
                # 'Potability': [Potability]
                }
        
        df              = pd.DataFrame(data)
        prediction      = model.predict(df)
        predicted_index = np.argmax(prediction)
        Potability      = class_names[predicted_index]
        cursor          = mysql.connection.cursor()
        cursor.execute(
            """UPDATE dataset SET 
            ph=%s, Hardness=%s, Solids=%s, Chloramines=%s, Sulfate=%s, Conductivity=%s, Organic_carbon=%s,
            Trihalomethanes=%s, Turbidity=%s, Potability=%s  WHERE id_dataset = %s""",
            (ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes,
            Turbidity, Potability, id_dataset))
        mysql.connection.commit()
        cursor.close()
        flash('Data berhasil di edit', 'success')
        return redirect(url_for('datatable'))
                            
    return render_template("pages/form_editdata.html", ph=ph, Hardness=Hardness,Solids=Solids, Chloramines=Chloramines,Sulfate=Sulfate,
                            Conductivity=Conductivity, Organic_carbon=Organic_carbon, Trihalomethanes=Trihalomethanes,
                            Turbidity=Turbidity, Potability=Potability)
                
#DELETE DATA          
@app.route("/deletedata/<int:id_dataset>", methods=['POST'])
def deletedata(id_dataset):
    if 'username' in session:
        # delete data
        cursor = mysql.connection.cursor()
        cursor.execute(
            "DELETE FROM dataset WHERE id_dataset = (%s)", (id_dataset,))
        mysql.connection.commit()
        cursor.close()
        return render_template("pages/datatable.html")  # page formdata
    else:
        return redirect(url_for('login'))

#KLASIFIKASI DATA
@app.route("/form_klasifikasi")
def klasifikasi():
    if 'username' in session:   
        return render_template("pages/form_klasifikasi.html")  # page datatable
    else:
        return redirect(url_for('login'))
    
@ app.route("/klasifikasi", methods=['POST'])
def predict():
    if 'username' in session:
        # insert data form
        ph          = request.form['ph']
        Hardness    = request.form['Hardness']
        Solids      = request.form['Solids']
        Chloramines  = request.form['Chloramines']
        Sulfate     = request.form['Sulfate']
        Conductivity    = request.form['Conductivity']
        Organic_carbon  = request.form['Organic_carbon']
        Trihalomethanes = request.form['Trihalomethanes']
        Turbidity       = request.form['Turbidity']
        
        if not all(value.replace('.', '', 1).isdigit() for value in [ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]):
            flash('Invalid data!!', 'error')
            return render_template("pages/form_klasifikasi.html",  ph=ph, Hardness=Hardness,Solids=Solids, Chloramines=Chloramines,Sulfate=Sulfate,
                                                            Conductivity=Conductivity, Organic_carbon=Organic_carbon, Trihalomethanes=Trihalomethanes,
                                                            Turbidity=Turbidity)
            
        ph          = float(ph)
        Hardness    = float(Hardness)
        Solids      = float(Solids)
        Chloramines  = float(Chloramines)
        Sulfate     = float(Sulfate)
        Conductivity    = float(Conductivity)
        Organic_carbon  = float(Organic_carbon)
        Trihalomethanes = float(Trihalomethanes)
        Turbidity       = float(Turbidity)
        
        data = {'ph': [ph],
                'Hardness': [Hardness],
                'Solids': [Solids],
                'Chloramines': [Chloramines],
                'Sulfate': [Sulfate],
                'Conductivity': [Conductivity],
                'Organic_carbon': [Organic_carbon],
                'Trihalomethanes': [Trihalomethanes],
                'Turbidity': [Turbidity]}

        df              = pd.DataFrame(data)
        prediction      = model.predict(df)
        predicted_index = np.argmax(prediction)
        Potability      = class_names[predicted_index]

        cursor = mysql.connection.cursor()
        cursor.execute(
            """INSERT INTO
            dataset (ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity,Potability)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity,Potability))
        # db.commit()
        mysql.connection.commit()
        cursor.close()
        flash('Data has been saved successfully', 'success')
        # page formdata predict
        return render_template("pages/form_klasifikasi.html", ph=ph, Hardness=Hardness,Solids=Solids, Chloramines=Chloramines,Sulfate=Sulfate,
                            Conductivity=Conductivity, Organic_carbon=Organic_carbon, Trihalomethanes=Trihalomethanes,
                            Turbidity=Turbidity, Potability=Potability)
    else:
        return redirect(url_for('index'))



# PAGE LOGOUT
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login')) # page login


if __name__ == "__main__":
    app.run(debug=True, port=8005)
    

