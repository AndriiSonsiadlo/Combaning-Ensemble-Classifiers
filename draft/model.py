from get_time import get_time, get_date


class Model:
    def __init__(self, name: str = "Unnamed", author: str = "Unknown", comment: str = "",
                 created_time: str = get_time(), created_date: str = get_date(),
                 learning_time: str = 0, train_dataset_Y=None,
                 count_train_Y=0,
                 test_dataset_Y=None, count_test_Y=0, accuracy=0, threshold=None, path_dataset=None):

        if train_dataset_Y is None:
            train_dataset_Y = []
        if test_dataset_Y is None:
            test_dataset_Y = []

        self.name = name
        self.created_time = created_time
        self.created_date = created_date
        self.created = f"{self.created_date} {self.created_time}"
        self.author = author
        self.comment = comment
        self.selected = False

        self.path_dataset = path_dataset
        self.threshold = threshold
        self.learning_time = learning_time
        self.accuracy = accuracy

        self.count_train_Y = count_train_Y
        self.count_test_Y = count_test_Y
        self.train_dataset_Y = train_dataset_Y
        self.test_dataset_Y = test_dataset_Y

        self.init_paths()

    def init_paths(self):
        self.path_model_data = os.path.join(folder_models_data, self.name)
        if (self.algorithm == algorithm_knn):
            self.path_file_model = os.path.join(folder_models_data, self.name, filename_knn_model)
        else:
            self.path_file_model = os.path.join(folder_models_data, self.name, filename_svm_model)

    def edit(self, new_name: str):
        self.name = new_name

    # what should be editable? what about creation date?

    def begin_learning(self, algorithm, weight, gamma, n_neighbor=None):
        learningTimeStart = time.time()
        self.algorithm = algorithm
        self.init_paths()

        if (algorithm == algorithm_knn):
            algorithm_object = KNN_classifier(self, path_model=self.path_file_model, n_neighbor=n_neighbor,
                                              weight=weight)
            self.weight = algorithm_object.weight
        else:
            algorithm_object = SVM_classifier(self, path_model=self.path_file_model, gamma=gamma)
            self.gamma = algorithm_object.gamma
        try:
            succes_learned, title_warn = algorithm_object.train()
        except BaseException:
            return (False, "Cannot to train a model")

        if (succes_learned):
            learningTimeStop = time.time() - learningTimeStart
            self.learning_time = round(learningTimeStop, 2)
            self.train_dataset_Y = algorithm_object.train_persons
            self.test_dataset_Y = algorithm_object.test_persons
            self.count_train_Y = algorithm_object.count_train_persons
            self.count_test_Y = algorithm_object.count_test_persons
            self.accuracy = algorithm_object.accuracy

            self.save_to_json()
            return (True, "")
        else:
            return (succes_learned, title_warn)

    def save_to_json(self):
        filepath_model_json = os.path.join(folder_models_data, self.name, file_model_json)
        dataJSON = {
            model_name_k: self.name,
            author_k: self.author,
            comment_k: self.comment,
            p_date_k: self.created_date,
            p_time_k: self.created_time,
            learning_time_k: self.learning_time,
            algorithm_k: self.algorithm,
            n_neighbor_k: self.n_neighbor,
            gamma_k: self.gamma,
            weights_k: self.weight,
            threshold_k: self.threshold,
            accuracy_k: self.accuracy,
            # test_size_k: test_size,
            # train_size_k: (1.0 - test_size),
            count_train_dataset_k: self.count_train_Y,
            count_test_dataset_k: self.count_test_Y,
            train_dataset_k: self.train_dataset_Y,
            test_dataset_k: self.test_dataset_Y
        }
        print("[INFO] saving data of model to .json...")
        with open(filepath_model_json, "w") as write_file:
            json.dump(dataJSON, write_file)
            json.dumps(dataJSON, indent=4)
