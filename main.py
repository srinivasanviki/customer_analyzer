"""`main` is the top level module for your Flask application."""

# Import the Flask Framework
from flask import render_template
from flask import Flask,request
from forms import FileForm
import StringIO
import urllib,base64

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd
from matplotlib import style
from matplotlib import pyplot as plt
style.use("ggplot")
colours=['g.','r.']
app = Flask(__name__)
app.config.from_object('config')
# Note: We don't need to call run() since our application is embedded within
# the App Engine WSGI application server.


def handle_non_numerical_data(df):
   columns = df.columns.values

   for column in columns:
       text_digit_vals = {}
       def convert_to_int(val):
           return text_digit_vals[val]

       if df[column].dtype != np.int64 and df[column].dtype != np.float64:
           column_contents = df[column].values.tolist()
           unique_elements = set(column_contents)
           x = 0
           for unique in unique_elements:
               if unique not in text_digit_vals:
                   text_digit_vals[unique] = x
                   x+=1

           df[column] = list(map(convert_to_int, df[column]))

   return df

def predictCustomerEngagement(df):
	correct = 0
	X = np.array(df.drop(['events_plan'], 1).astype(float))
	X = preprocessing.scale(X)
	y = np.array(df['events_plan'])
	X_pca = PCA(n_components=2, whiten=True).fit_transform(X)
	clf = KMeans(n_clusters=2,max_iter=100,n_init=16,n_jobs=-1)
	clf.fit(X_pca)
	count_paid=0
	count_free=0
	centroids = clf.cluster_centers_
	lables = clf.labels_
	for i in range(len(X_pca)):
		predict_me = np.array(X_pca[i].astype(float))
		predict_me = predict_me.reshape(-1, len(predict_me))
		prediction = clf.predict(predict_me)
		plt.plot(X_pca[i][0], X_pca[i][1],colours[lables[i]],markersize=10)		    
		if prediction[0] == 0:
			count_paid += 1
		elif prediction[0] == 1:
			count_free += 1
		if prediction[0] == y[i]:
			correct += 1

	plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", c=('g','r'), s=150 ,zorder=10)
	imgdata = StringIO.StringIO()
	plt.savefig(imgdata, format='png')
	imgdata.seek(0)

	fig = plt.figure()
	ax = fig.gca()
	ax.pie((count_free,count_paid),colors=('r', 'g'),radius=0.25, center=(0.5, 0.5), frame=True)
	pieData = StringIO.StringIO()
	plt.savefig(pieData,format='png')
	pieData.seek(0)
	pieUri = 'data:image/png;base64,' + urllib.quote(base64.b64encode(pieData.buf))
	uri = 'data:image/png;base64,' + urllib.quote(base64.b64encode(imgdata.buf))
	return {'accuracy':(float(correct) / len(X_pca))*100,'centroids':centroids, 'points':X_pca ,'total':count_free+count_paid,'count_paid':count_paid,'count_free':count_free,'plot':uri,'pieUri':pieUri}

@app.route('/',methods=['GET','POST'])
def hello():
	"""Return a friendly HTTP greeting."""
	form = FileForm()
	if request.method =="POST":
		print request.files
		if 'csv_file' not in request.files:
			return 'No file part'
		file = request.files['csv_file']
		# if user does not select file, browser also
		# submit a empty part without filename
		if file.filename == '':
			return 'No selected file'
		json_data={}
		count_paid=0
		count_free=0
		data=pd.read_csv(file)
		data.fillna(0, inplace=True)
		df = handle_non_numerical_data(data)
		a = predictCustomerEngagement(df)
		print a
		return render_template('index.html',plot= a['plot'],form=form,pieUri=a['pieUri'],accuracy=a['accuracy'],count_paid=a['count_paid'],count_free=a['count_free'])
	return render_template('index.html',form=form,plot="",pieUri="",accuracy="",count_paid="",count_free="")


@app.errorhandler(404)
def page_not_found(e):
    """Return a custom 404 error."""
    return 'Sorry, Nothing at this URL.', 404


@app.errorhandler(500)
def application_error(e):
    """Return a custom 500 error."""
    return 'Sorry, unexpected error: {}'.format(e), 500

if __name__=="__main__":
	app.run()
