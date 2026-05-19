"""
Dialog views for the Cellpose GUI.
"""

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from . import io as gui_io


class TrainWindow(QDialog):
    def __init__(self, parent, model_strings):
        super().__init__(parent)
        self.main_window = parent
        self.setGeometry(100, 100, 800, 480)
        self.setWindowTitle("train settings")
        self.l0 = QHBoxLayout()
        self.setLayout(self.l0)
        self.l0.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        left_column = QVBoxLayout()
        left_column.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        qlabel = QLabel("train data folder")
        qlabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        train_data_layout = QHBoxLayout()
        train_data_layout.addWidget(qlabel)
        self.train_folder = QLineEdit(
            parent.training_params.get(
                "train_data_folder", parent.training_params.get("model_save_folder", "")
            )
        )
        self.train_folder.editingFinished.connect(self._refresh_train_folder_preview)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(lambda: self.browse_train_folder())
        train_data_layout.addWidget(self.train_folder)
        train_data_layout.addWidget(browse_btn)
        left_column.addLayout(train_data_layout)

        self.ModelChoose = QComboBox()
        self.ModelChoose.addItems(model_strings)
        self.ModelChoose.setCurrentIndex(parent.training_params["model_index"])
        qlabel = QLabel("initial model: ")
        qlabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        model_layout = QHBoxLayout()
        model_layout.addWidget(qlabel)
        model_layout.addWidget(self.ModelChoose)
        left_column.addLayout(model_layout)

        labels = ["learning_rate", "weight_decay", "n_epochs", "model_name"]
        self.edits = []
        for label in labels:
            param_layout = QHBoxLayout()
            qlabel = QLabel(label)
            qlabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            param_layout.addWidget(qlabel)
            self.edits.append(QLineEdit())
            self.edits[-1].setText(str(parent.training_params[label]))
            param_layout.addWidget(self.edits[-1])
            left_column.addLayout(param_layout)

        self.use_norm = QCheckBox("use restored/filtered image")
        self.use_norm.setChecked(True)

        qbtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(qbtn)
        self.buttonBox.accepted.connect(lambda: self.accept(parent))
        self.buttonBox.rejected.connect(self.reject)
        left_column.addWidget(self.buttonBox)
        left_column.addStretch(1)

        self.train_files_table = QTableWidget(0, 2, self)
        self.train_files_table.setHorizontalHeaderLabels(["filenames", "# of masks"])
        self.train_files_table.verticalHeader().setVisible(False)
        self.train_files_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.train_files_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.train_files_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.train_files_table.horizontalHeader().setStretchLastSection(True)
        self.train_files_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.train_files_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.train_files_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.train_files_table.setAlternatingRowColors(True)
        self.train_files_table.setWordWrap(False)

        right_column = QVBoxLayout()
        right_column.addWidget(self.train_files_table)
        right_column.setContentsMargins(0, 0, 0, 0)
        self.train_files_table.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        self.l0.addLayout(left_column)
        self.l0.addLayout(right_column)
        self._refresh_train_folder_preview()

    def accept(self, parent):
        parent.training_params = {
            "model_index": self.ModelChoose.currentIndex(),
            "learning_rate": float(self.edits[0].text()),
            "weight_decay": float(self.edits[1].text()),
            "n_epochs": int(self.edits[2].text()),
            "model_name": self.edits[3].text(),
            "train_data_folder": self.train_folder.text().strip(),
        }
        self.done(1)

    def _refresh_train_folder_preview(self):
        self.train_files_table.setRowCount(0)
        folder = self.train_folder.text().strip()
        if not folder:
            self._add_train_preview_message("(no folder selected)")
            return

        try:
            image_names = self.main_window.get_training_image_files(folder, nested=True)
            _, train_labels, train_files, _, _ = gui_io._get_train_set(image_names)
        except Exception as e:
            self._add_train_preview_message(str(e))
            return

        if not train_files:
            self._add_train_preview_message("no _seg.npy files found")
            return

        for i, train_file in enumerate(train_files):
            label = os.path.split(train_file)[-1]
            nmasks = str(train_labels[i].max())
            self._add_train_preview_row(label, nmasks)

    def _add_train_preview_row(self, filename, nmasks):
        row = self.train_files_table.rowCount()
        self.train_files_table.insertRow(row)
        self.train_files_table.setItem(row, 0, QTableWidgetItem(filename))
        self.train_files_table.setItem(row, 1, QTableWidgetItem(str(nmasks)))

    def _add_train_preview_message(self, message):
        self._add_train_preview_row(message, "")
        self.train_files_table.item(0, 0).setTextAlignment(
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )

    def browse_train_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Train images folder", self.train_folder.text()
        )
        if folder:
            self.train_folder.setText(folder)
            self._refresh_train_folder_preview()
