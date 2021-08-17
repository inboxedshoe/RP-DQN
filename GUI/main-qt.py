# This Python file uses the following encoding: utf-8
import sys
import os
import PySide2
from os.path import expanduser
import traceback
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from PySide2.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog
from PySide2 import QtWidgets, QtCore
from PySide2.QtCore import QFile, QObject
from PySide2.QtUiTools import QUiLoader
from Utilities.model_loader import load_model_config, load_model

import torch
from DataGenerator import read_test_set
import time


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')


dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


class Worker(QtCore.QRunnable):
    """Worker thread for running background tasks."""

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self):
        try:
            result = self.fn(
                *self.args, **self.kwargs,
            )
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class WorkerSignals(QtCore.QObject):
    """
    Defines the signals available from a running worker thread.
    Supported signals are:
    finished
        No data
    error
        `tuple` (exctype, value, traceback.format_exc() )
    result
        `object` data returned from processing, anything
    """
    finished = QtCore.Signal()
    error = QtCore.Signal(tuple)
    result = QtCore.Signal(object)
    progress = QtCore.Signal(int)


class loader(QWidget):
    def __init__(self):
        super(loader, self).__init__()
        self.load_ui()
        self.threadpool = QtCore.QThreadPool()

    def load_ui(self):
        loader = QUiLoader()
        path = os.path.join(os.path.dirname(__file__), "form-qt.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.window = loader.load(ui_file, self)
        ui_file.close()

        self.browse_path = "~"
        self.model_path = ""
        self.model_folders = []
        self.selected_model = ""
        self.directory_model = ""
        self.selected_valid = ""
        self.output = []
        self.sampling_output = []

        btn = self.window.findChild(QPushButton, 'model_butt')
        btn2 = self.window.findChild(QPushButton, 'vali_butt')
        self.btn3 = self.window.findChild(QPushButton, 'test_button')
        self.samples_butt = self.window.findChild(QPushButton, 'samples_butt')
        self.model_folders_box = self.window.findChild(QtWidgets.QComboBox, 'model_box')
        self.model_list_view = self.window.findChild(QtWidgets.QListWidget, 'mod_time_list')
        self.time_togg = self.window.findChild(QtWidgets.QCheckBox, 'time_togg')
        self.load_conf = self.window.findChild(QtWidgets.QCheckBox, 'load_conf')

        self.batch_size = self.window.findChild(QtWidgets.QLineEdit, 'batch_size')
        self.number_depots = self.window.findChild(QtWidgets.QLineEdit, 'number_depots')
        self.number_customers = self.window.findChild(QtWidgets.QLineEdit, 'number_customers')
        self.inter_dim = self.window.findChild(QtWidgets.QLineEdit, 'inter_dim')
        self.valid_size = self.window.findChild(QtWidgets.QLineEdit, 'valid_size')
        self.inner_mask = self.window.findChild(QtWidgets.QLineEdit, 'inner_mask')
        self.normalization = self.window.findChild(QtWidgets.QLineEdit, 'normalization')
        self.embed_dim = self.window.findChild(QtWidgets.QLineEdit, 'embed_dim')
        self.type = self.window.findChild(QtWidgets.QLineEdit, 'type')
        self.temp_box = self.window.findChild(QtWidgets.QLineEdit, 'temp_box')
        self.sample_num_solutions = self.window.findChild(QtWidgets.QLineEdit, 'sample_num_solutions')

        self.valid_text = self.window.findChild(QtWidgets.QLabel, 'set_name')
        self.mean_text = self.window.findChild(QtWidgets.QLabel, 'Mean')
        self.var_text = self.window.findChild(QtWidgets.QLabel, 'Var')
        self.sample_mean_text = self.window.findChild(QtWidgets.QLabel, 'sample_mean')
        self.sample_var_text = self.window.findChild(QtWidgets.QLabel, 'sample_var')
        self.time_text = self.window.findChild(QtWidgets.QLabel, 'time')
        self.model_text = self.window.findChild(QtWidgets.QLabel, 'model_text')
        self.sample_time = self.window.findChild(QtWidgets.QLabel, 'sample_time')

        btn.clicked.connect(self.browseSlot)
        btn2.clicked.connect(self.browse_valid)
        self.btn3.clicked.connect(self.test)
        self.samples_butt.clicked.connect(self.sample_start)
        self.model_folders_box.activated.connect(self.model_folder_chosen)
        self.model_list_view.itemClicked.connect(self.set_current_model)

        self.window.show()

    def browseSlot(self):
        ''' Called when the user presses the Browse button
        '''
        self.model_path = QFileDialog.getExistingDirectory(None, 'Select a folder:', expanduser(self.browse_path))
        self.browse_path = self.model_path
        list_model_folders = [f.name for f in os.scandir(self.model_path) if f.is_dir()]
        self.model_folders= [f.path for f in os.scandir(self.model_path) if f.is_dir()]

        self.model_folders_box.clear()
        self.model_folders_box.addItems(list_model_folders)

    def model_folder_chosen(self):
        ''' Called when the user presses the Browse button
        '''
        self.model_list_view.clear()
        self.directory_model = self.model_folders[self.model_folders_box.currentIndex()]
        models = sorted([f for f in os.listdir(self.directory_model) if f.endswith('.pt')])
        self.model_list_view.addItems(models)

    def set_current_model(self, item):
        self.selected_model = os.path.join(self.directory_model, item.text())
        self.model_text.setText(os.path.basename(self.selected_model))

        config_file = os.path.join(self.directory_model, "config.json")

        with open(config_file, 'r') as fp:
            dic = json.load(fp)

        self.batch_size.setText(str(dic["batchsize"]))
        self.number_depots.setText(str(dic["num_depots"]))
        self.number_customers.setText(str(dic["num_cust_node"]))
        self.inter_dim.setText(str(dic["intermediate_dimension"]))
        #self.valid_size.setText(str(dic["batchsize"]))
        self.inner_mask.setText(str(dic["inner_masking"]))
        self.normalization.setText(dic["normalization"])
        self.embed_dim.setText(str(dic["embedding_dimension"]))
        self.type.setText(dic["problem_type"])


    def browse_valid(self):
        self.selected_valid = QFileDialog.getOpenFileName(None, 'Select a validation set:', expanduser(self.browse_path), '*.pkl')[0]
        self.browse_path = self.selected_valid
        self.valid_text.setText(os.path.basename(self.selected_valid))

    def run_threaded_process(self, process, on_complete):
        """Execute a function in the background with a worker"""

        worker = Worker(fn=process)
        self.threadpool.start(worker)
        worker.signals.finished.connect(on_complete)
        self.progressbar.setRange(0,0)
        return

    def test(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setWindowTitle("Missing settings")

        if self.model_text.text() == "-":
            msg.setText('No model specified')
            msg.exec_()
            self.btn3.setEnabled(True)
            self.samples_butt.setEnabled(True)
            return

        if self.valid_text.text() == "-":
            msg.setText('No validation set specified')
            msg.exec_()
            self.btn3.setEnabled(True)
            self.samples_butt.setEnabled(True)
            return

        self.btn3.setEnabled(False)
        self.samples_butt.setEnabled(False)

        worker = Worker(fn = self.validate)
        self.threadpool.start(worker)
        worker.signals.finished.connect(self.test_complete)

    def validate(self):

        #model = load_model_config(self.selected_model, os.path.join(self.directory_model, "config.json"), device)

        self.model = load_model(model_filename= self.selected_model,
                            embedding_dimension= int(self.embed_dim.text()),
                            problem_type=self.type.text(),
                            normalization=self.normalization.text(),
                            inner_masking=self.inner_mask.text() == ["true", "True"],
                            num_depots=int(self.number_depots.text()),
                            valid_size = int(self.valid_size.text()),
                            num_cust_node=int(self.number_customers.text()),
                            device=device,
                            intermediate_dim=int(self.inter_dim.text()))

        if self.selected_valid.find("MDVRP") == -1:
            test_set = read_test_set(self.selected_valid, 'cvrp')
        else:
            test_set = read_test_set(self.selected_valid, 'mdvrp')

        if self.time_togg.isChecked():
            start = time.time()
            a, b = self.model.validate(data=test_set)
            self.time_text.setText("%.2f s" %(time.time()-start))
        else:
            a, b = self.model.validate(data=test_set)
            self.time_text.setText=("-")

        self.output = [a, b]

    def test_complete(self):
        self.btn3.setEnabled(True)
        self.samples_butt.setEnabled(True)
        self.mean_text.setText("%.4f" % torch.mean(self.output[1]).item())
        self.var_text.setText("%.4f" % torch.var(self.output[1]).item())

    def sample_start(self):
        self.samples_butt.setEnabled(False)
        self.btn3.setEnabled(False)

        #start = time.time()
        #a, b = self.model.sampling(softmax_temperature=float(self.temp_box.text()))
        #self.sampling_output = [a, b]
        #self.sample_mean_text.setText("%.4f" % torch.mean(self.sampling_output[1]).item())
        #self.sample_var_text.setText("%.4f" % torch.var(self.sampling_output[1]).item())
        #self.sample_time.setText("%.2f" % (time.time() - start))

        #return

        worker = Worker(fn = self.sample)
        self.threadpool.start(worker)
        worker.signals.finished.connect(self.sample_complete)

    def sample(self):
        start = time.time()

        if self.selected_valid.find("MDVRP") == -1:
            test_set = read_test_set(self.selected_valid, 'cvrp')
        else:
            test_set = read_test_set(self.selected_valid, 'mdvrp')

        a, b = self.model.sampling(data = test_set, softmax_temperature=float(self.temp_box.text()), num_solutions=int(self.sample_num_solutions.text()))
        self.sampling_output = [a, b]
        self.sample_time.setText("%.2f" % (time.time() - start))

    def sample_complete(self):
        self.samples_butt.setEnabled(True)
        self.btn3.setEnabled(True)
        self.sample_mean_text.setText("%.4f" % torch.mean(self.sampling_output[1]).item())
        self.sample_var_text.setText("%.4f" % torch.var(self.sampling_output[1]).item())


if __name__ == "__main__":
    app = QApplication([])
    widget = loader()
    widget.show()
    sys.exit(app.exec_())
