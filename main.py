import os, warnings, logging, faulthandler, sys, signal
faulthandler.enable(all_threads=True)
faulthandler.dump_traceback_later(10, repeat=True)

# Make Python noisier
if sys.flags.dev_mode == 0:
    os.environ.setdefault("PYTHONWARNINGS", "default")
warnings.simplefilter("default")

# Verbose logging to console
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
)
logging.getLogger("matplotlib").setLevel(logging.INFO)  # too noisy at DEBUG

from PySide6.QtWidgets import QApplication
from i20_xes.main_gui import MainWindow

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()