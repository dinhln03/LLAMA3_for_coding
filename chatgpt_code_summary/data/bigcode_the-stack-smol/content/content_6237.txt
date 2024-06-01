from PySide import QtGui, QtCore
from AttributeWidgetImpl import AttributeWidget


class ScalarWidget(AttributeWidget):

    def __init__(self, attribute, parentWidget=None, addNotificationListener = True):
        super(ScalarWidget, self).__init__(attribute, parentWidget=parentWidget, addNotificationListener = addNotificationListener)

        hbox = QtGui.QHBoxLayout()

        self._widget = QtGui.QLineEdit(self)
        validator = QtGui.QDoubleValidator(self)
        validator.setDecimals(3)
        self._widget.setValidator(validator)
        hbox.addWidget(self._widget, 1)

        hbox.addStretch(0)
        hbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(hbox)
        self.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)

        self.updateWidgetValue()
        if self.isEditable():
            self._widget.editingFinished.connect(self._invokeSetter)
        else:
            self._widget.setReadOnly(True)


    def getWidgetValue(self):
        return float(self._widget.text())

    def setWidgetValue(self, value):
        self._widget.setText(str(round(value, 4)))

    @classmethod
    def canDisplay(cls, attribute):
        return(
                attribute.getDataType() == 'Scalar' or
                attribute.getDataType() == 'Float32' or
                attribute.getDataType() == 'Float64'
            )

ScalarWidget.registerPortWidget()
