from model import *
from UI import *
from Rent import *

if __name__ == '__main__':
    # models = model()
    # models.PreProcessing()
    # models.PredictRent()

    
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())

