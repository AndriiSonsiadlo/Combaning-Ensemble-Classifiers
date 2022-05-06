from model import Model


class classifier_knn(Model):

	def __init__(self, model, path_model, weight=None, n_neighbor=None):
		super().__init__()

		self.train_persons = []
		self.test_persons = []
		self.count_train_persons = 0
		self.count_test_persons = 0

		self.Z_test = []
		self.Z_train = []

		self.knn_clf = None
		self.n_neighbor = n_neighbor
		if (weight == 'distance' or weight == 'uniform'):
			self.weight = weight
		else:
			self.weight = 'distance'
		self.knn_algo = 'ball_tree'
		self.accuracy = 0

		self.verbose = True

		self.model_full_path = path_model
		self.model = model

	def parting(self, xs, parts):
		part_len = math.ceil(len(xs) / parts)
		return [xs[part_len * k:part_len * (k + 1)] for k in range(parts)]

	def load_image(self, person_names):
		for name in person_names:
			is_trained = False
			is_tested = False
			dir_path_person_photo = os.path.join(self.train_dir, name, self.folder_photo)

			for index, img_path in enumerate(image_files_in_folder(dir_path_person_photo)):

				image = face_recognition.load_image_file(img_path)
				face_bounding_boxes = face_recognition.face_locations(image)

				if len(face_bounding_boxes) != 1:
					# If there are no people (or too many people) in a training image, skip the image.
					if self.verbose:
						print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
							face_bounding_boxes) < 1 else "Found more than one face"))
				else:
					# Add face encoding for current image to the training set
					encoded = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
					if (index > 5) or (index == 1):
						self.Z_test.append([encoded, name])
						print(f"Added to test - {name}")
						is_tested = True
						continue
					self.Z_train.append([encoded, name])
					is_trained = True
			if not is_trained:
				self.train_persons.remove(name)
			if is_tested:
				self.test_persons.append(name)
		return 1

	def load_model(self):
		try:
			with open(self.model_full_path, 'rb') as f:
				self.knn_clf = pickle.load(f)
			return True
		except BaseException:
			print(f"Error: Model have not loaded\nPath: {self.model_full_path}")
			return False

	def pre_train_knn(self):
		"""
		Trains a k-nearest neighbors classifier for face recognition.

		:param train_dir: directory that contains a sub-directory for each known person, with its name.

		 (View in source code to see train_dir example tree structure)

		 Structure:
			<train_dir>/
			├── <person1>/
			│   ├── <somename1>.jpeg
			│   ├── <somename2>.jpeg
			│   ├── ...
			├── <person2>/
			│   ├── <somename1>.jpeg
			│   └── <somename2>.jpeg
			└── ...

		:param model_save_path: (optional) path to save model on disk
		:param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
		:param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
		:param verbose: verbosity of training
		:return: returns knn classifier that was trained on the given data.
		"""

		# Loop through each person in the training set
		for person_name in os.listdir(self.train_dir):
			dir_path_person_photo = os.path.join(self.train_dir, person_name, self.folder_photo)
			if not os.path.isdir(dir_path_person_photo) or person_name == "temp":
				continue
			else:
				self.train_persons.append(person_name)

		parts_person_names = self.parting(self.train_persons, parts=8)
		print(parts_person_names)
		executor = ThreadPoolExecutor(max_workers=8)
		future1 = executor.submit(self.load_image, parts_person_names[0])
		future2 = executor.submit(self.load_image, parts_person_names[1])
		future3 = executor.submit(self.load_image, parts_person_names[2])
		future4 = executor.submit(self.load_image, parts_person_names[3])
		future5 = executor.submit(self.load_image, parts_person_names[4])
		future6 = executor.submit(self.load_image, parts_person_names[5])
		future7 = executor.submit(self.load_image, parts_person_names[6])
		future8 = executor.submit(self.load_image, parts_person_names[7])
		try:
			test = future1.result() + future2.result() + future3.result() + future4.result() + future5.result() + future6.result() + future7.result() + future8.result()
		except BaseException:
			pass
		self.count_train_persons = len(self.train_persons)
		self.count_test_persons = len(self.test_persons)

	def train(self):
		self.pre_train_knn()
		if (self.count_train_persons):

			X = []
			y = []
			for x_i, y_i in self.Z_train:
				X.append(x_i)
				y.append(y_i)

			# Determine how many neighbors to use for weighting in the KNN classifier
			if self.Z_train:
				if self.n_neighbor is None or self.n_neighbor < 1:
					self.n_neighbor = int(round(math.sqrt(len(X))))
					if self.verbose:
						print("Chose n_neighbors automatically:", self.n_neighbor)
				self.model.n_neighbor = self.n_neighbor

				# Create and train the KNN classifier
				self.knn_clf = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbor, algorithm=self.knn_algo,
				                                              weights=self.weight)

				self.knn_clf.fit(X, y)
				if self.Z_test:
					self.test_predict()
				else:
					print("No data for testing")
				return (self.save_model())
			else:
				return (False, "Not found encodings of persons photos and label with names")
		else:
			return (False, "Not found person with one facial photo in database")

	def test_predict(self):
		X_test = []
		y_test = []

		for x_i, y_i in self.Z_test:
			X_test.append(x_i)
			y_test.append(y_i)

		y_pred = self.knn_clf.predict(X_test)

		all_count_photo = len(self.Z_train) + len(self.Z_test)
		perc_test = (len(self.Z_test) * 100) / all_count_photo
		perc_train = 100 - perc_test

		print(f"Test in %: {perc_train}")
		print(f"Train in %: {perc_test}")

		try:
			# Model Accuracy: how often is the classifier correct?
			self.accuracy = round(metrics.accuracy_score(y_test, y_pred), 6)
			print("Accuracy:", self.accuracy)
		# Model Precision: what percentage of positive tuples are labeled as such?
		#			print("Precision:", metrics.precision_score(y_test, y_pred))
		# Model Recall: what percentage of positive tuples are labelled as such?
		#			print("Recall:", metrics.recall_score(y_test, y_pred))
		except BaseException:
			pass

	def save_model(self):
		# Save the trained KNN classifier
		print(self.model_full_path)
		if self.model_full_path is not None:
			try:
				with open(self.model_full_path, 'wb') as f:
					pickle.dump(self.knn_clf, f)
				return (True, "")
			except BaseException:
				return (False, "Cannot to save a model knn_model.clf")

	def predict_webcam(self, frame):

		X_face_landmarks_list = face_recognition.face_landmarks(frame)
		X_face_locations = face_recognition.face_locations(frame)
		# If no faces are found in the image, return an empty result.
		if len(X_face_locations) == 0:
			print("No face found")
			self.counter_frame = 0
			return frame, self.counter_frame, ""

		# Find encodings for faces in the test image
		faces_encodings = face_recognition.face_encodings(frame, known_face_locations=X_face_locations)

		# Use the KNN model to find the best matches for the test face
		closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=1)

		are_matches = [closest_distances[0][i][0] <= self.model.threshold for i in range(len(X_face_locations))]

		for i in range(len(X_face_locations)):
			print(closest_distances[0][i][0])

		# Predict classes and remove classifications that aren't within the threshold
		predictions = [(pred, loc) if rec else (text_unknown, loc) for pred, loc, rec in
		               zip(self.knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

		return self.show_prediction_labels_webcam(frame=frame, predictions=predictions,
		                                          X_face_landmarks_list=X_face_landmarks_list)

	def show_prediction_labels_webcam(self, frame, predictions, X_face_landmarks_list):

		frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		draw = ImageDraw.Draw(frame)

		for name, (top, right, bottom, left) in predictions:
			draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

			# There's a bug in Pillow where it blows up with non-UTF-8 text
			# when using the default bitmap font
			name = str(name)

			# Draw a label with a name below the face
			text_width, text_height = draw.textsize(name)
			draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
			draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

		""" for face_landmarks in X_face_landmarks_list:
			# Print the location of each facial feature in this image
			for facial_feature in face_landmarks.keys():
				print("The {} in this face has the following points: {}".format(facial_feature,
				                                                                face_landmarks[facial_feature]))
			# Let's trace out each facial feature in the image with a line!
			for facial_feature in face_landmarks.keys():
				draw.line(face_landmarks[facial_feature], width=2) """

		del draw

		frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

		if len(predictions) == 1:
			if self.identificated_name == name:
				self.counter_frame += 1
			elif (name == text_unknown):
				pass
			else:
				self.identificated_name = name
				self.counter_frame = 0
		else:
			self.counter_frame = 0

		return frame, self.counter_frame, self.identificated_name

	# image predict
	def predict_image(self, X_img_path):
		"""
		Recognizes faces in given image using a trained KNN classifier

		:param X_img_path: path to image to be recognized
		:param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
		:param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
		:param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
			   of mis-classifying an unknown person as a known one.
		:return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
			For faces of unrecognized persons, the name 'unknown' will be returned.
		"""

		# Load image file and find face locations
		X_img = face_recognition.load_image_file(X_img_path)

		if (X_img.shape[1] < X_img.shape[0]):
			basewidth = 400
			wpercent = (basewidth / float(X_img.shape[1]))
			hsize = int(float(X_img.shape[0]) * float(wpercent))
			dsize = (basewidth, hsize)
			# resize image
			X_img = cv2.resize(X_img, dsize)
		else:
			basewidth = 400
			wpercent = (basewidth / float(X_img.shape[0]))
			hsize = int(float(X_img.shape[1]) * float(wpercent))
			dsize = (hsize, basewidth)
			# resize image
			X_img = cv2.resize(X_img, dsize)

		X_face_landmarks_list = face_recognition.face_landmarks(X_img)
		X_face_locations = face_recognition.face_locations(X_img)

		# If no faces are found in the image, return an empty result.
		if len(X_face_locations) == 0:
			return X_img, text_unknown

		# Find encodings for faces in the test image
		faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

		# Use the KNN model to find the best matches for the test face
		closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=1)
		are_matches = [closest_distances[0][i][0] <= self.model.threshold for i in range(len(X_face_locations))]

		# Predict classes and remove classifications that aren't within the threshold
		predictions = [(pred, loc) if rec else (text_unknown, loc) for pred, loc, rec in
		               zip(self.knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

		return self.show_prediction_labels_on_image(frame=X_img, predictions=predictions,
		                                            X_face_landmarks_list=X_face_landmarks_list)

	def show_prediction_labels_on_image(self, frame, predictions, X_face_landmarks_list):
		"""
		Shows the face recognition results visually.

		:param img_path: path to image to be recognized
		:param predictions: results of the predict function
		:return:
		"""

		frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		draw = ImageDraw.Draw(frame)

		for name, (top, right, bottom, left) in predictions:
			draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0))

			# There's a bug in Pillow where it blows up with non-UTF-8 text
			# when using the default bitmap font

			name = name.encode().decode("utf-8")

			# Draw a label with a name below the face
			text_width, text_height = draw.textsize(name)
			print(text_height)
			print(text_width)

			draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(255, 0, 0), outline=(0, 0, 0))
			draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

		""" for face_landmarks in X_face_landmarks_list:

			# Print the location of each facial feature in this image
			for facial_feature in face_landmarks.keys():
				print("The {} in this face has the following points: {}".format(facial_feature,
				                                                                face_landmarks[facial_feature]))

			# Let's trace out each facial feature in the image with a line!
			for facial_feature in face_landmarks.keys():
				draw.line(face_landmarks[facial_feature], width=2) """

		del draw

		frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

		if (len(predictions) == 1):
			return frame, name
		else:
			return frame, text_unknown

	def create_plot(self):
		n_neighbors = 15

		# import some data to play with
		iris = datasets.load_iris()

		# we only take the first two features. We could avoid this ugly
		# slicing by using a two-dim dataset
		X = iris.data[:, :2]
		y = iris.target
		print(X)
		print(y)

		h = .02  # step size in the mesh

		# Create color maps
		cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
		cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

		for weights in ['uniform', 'distance']:
			# we create an instance of Neighbours Classifier and fit the data.
			clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
			clf.fit(X, y)

			# Plot the decision boundary. For that, we will assign a color to each
			# point in the mesh [x_min, x_max]x[y_min, y_max].
			x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
			y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
			xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
			                     np.arange(y_min, y_max, h))
			Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

			# Put the result into a color plot
			Z = Z.reshape(xx.shape)
			plt.figure()
			plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

			# Plot also the training points
			plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
			            edgecolor='k', s=20)
			plt.xlim(xx.min(), xx.max())
			plt.ylim(yy.min(), yy.max())
			plt.title("3-Class classification (k = %i, weights = '%s')"
			          % (n_neighbors, weights))

		plt.show()

	def knn_comparison(self):

		#		print(self.X)
		dataset = np.array(self.X).tolist()

		# calculate the Euclidean distance between two vectors
		def euclidean_distance(row1, row2):
			distance = 0.0
			for i in range(len(row1) - 1):
				distance += (row1[i] - row2[i]) ** 2
			return math.sqrt(distance)

		row0 = dataset[0]
		for row in dataset:
			distance = euclidean_distance(row0, row)
			print(distance)

		euclidean_distance()

		# A = kneighbors_graph(self.X, self.n_neighbor, mode='distance', include_self=True)
		# print(A)
		# b = A.nonzero()
		# c = np.log10(np.array(A[b[0], b[1]]))
		# mean = c[0].mean()
		# std = c[0].std()
		# pc = np.percentile(c[0], 70)
		#
		# n, bins, patches = plt.hist(c[0], 50)
		plt.show()