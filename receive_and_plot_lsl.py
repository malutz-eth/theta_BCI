"""
ReceiveAndPlot example for LSL

This example shows data from all found outlets in realtime.
It illustrates the following use cases:
- efficiently pulling data, re-using buffers
- automatically discarding older samples
- online postprocessing
"""
import numpy as np
import math
import pylsl
import pyqtgraph as pg
import datetime
import pickle
import sys
import GUI as gui
import keyboard
import os
import window_buttns as btns

from pyqtgraph.Qt import QtCore, QtGui
from typing import List
from scipy.signal import butter, lfilter

class Inlet:
    """Base class to represent a plottable inlet"""
    # Basic parameters for the plotting window

    plot_duration = 10 # how many seconds of data to show
    update_interval = 10  # ms between screen updates
    pull_interval = 10  # ms between each pull operation

    def __init__(self, info: pylsl.StreamInfo):
        # create an inlet and connect it to the outlet we found earlier.
        # max_buflen is set so data older the plot_duration is discarded
        # automatically and we only pull data new enough to show it

        # Also, perform online clock synchronization so all streams are in the
        # same time domain as the local lsl_clock()
        # (see https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/enums.html#_CPPv414proc_clocksync)
        # and dejitter timestamps
        self.inlet = pylsl.StreamInlet(info, max_buflen=self.plot_duration,
                                       processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)
        # store the name and channel count
        self.name = info.name()
        self.channel_count = info.channel_count()

    def pull_and_plot(self, plot_time: float, plt: pg.PlotItem):
        """Pull data from the inlet and add it to the plot.
        :param plot_time: lowest timestamp that's still visible in the plot
        :param plt: the plot the data should be shown on
        """
        # We don't know what to do with a generic inlet, so we skip it.
        pass

class DataInlet(Inlet):
    """A DataInlet represents an inlet with continuous, multi-channel data that
    should be plotted as multiple lines."""
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]
    data_buffer_list = []
    time_buffer_list = []

    data_buffer = np.asarray(data_buffer_list)
    time_buffer = np.empty(time_buffer_list)

    timelist = []
    timelist.append(pylsl.local_clock())

    if gui.freq_option == 1:
        freq_band = "alpha"
        hcut = 14
        lcut = 8
    elif gui.freq_option == 2:
        freq_band = "theta"
        hcut = 8
        lcut = 4

    filter_order = 5

    #sampling rate of the stream
    sr = 500

    def __init__(self, info: pylsl.StreamInfo, plt: pg.PlotItem):
        super().__init__(info)
        # calculate the size for our buffer, i.e. two times the displayed data
        bufsize = (2 * math.ceil(info.nominal_srate() * Inlet.plot_duration), info.channel_count())
        self.buffer = np.empty(bufsize, dtype=self.dtypes[info.channel_format()])
        empty = np.array([])
        # create one curve object for each channel/line that will handle displaying the data
        self.curves = [pg.PlotCurveItem(x=empty, y=empty, autoDownsample=True) for _ in range(self.channel_count)]
        self.curve_filt = [pg.PlotCurveItem(x=empty, y=empty, autoDownsample=True,pen=pg.mkPen('g', width=5))]

        plt.addItem(self.curves[0])
        plt.addItem(self.curve_filt[0])

    def pull_and_plot(self, plot_time, plt):

        def filter_function(data,cutfreq,fs,order,btype):
            nyq = fs*0.5
            cutfreq= cutfreq/nyq
            b, a = butter(order, cutfreq, btype=btype, analog=False)
            filt = lfilter(b,a,data)
            return filt

        def band_pass_filter(data, l, h, fs, order):
            data_filt_lp = filter_function(data, h, fs, order, "low")
            data_filt_hp = filter_function(data_filt_lp, l, fs, order, "high")
            return data_filt_hp

        # pull the data
        samples, ts = self.inlet.pull_chunk()
        if ts:
            samples = np.asarray(samples)
            samples = samples.flatten()

        # ts will be empty if no samples were pulled, a list of timestamps otherwise

            ts = np.asarray(ts)
            ts = ts.flatten()

            self.data_buffer = np.append(self.data_buffer, samples)
            self.time_buffer = np.append(self.time_buffer, ts)
            if len(self.data_buffer) < 10000:

                filt = band_pass_filter(self.data_buffer[-(self.sr*Inlet.plot_duration)-1:-1], self.lcut, self.hcut, self.sr, self.filter_order)
                raw = band_pass_filter(self.data_buffer[-(self.sr*Inlet.plot_duration)-1:-1], 1, 35, self.sr, self.filter_order)

                self.curves[0].setData(self.time_buffer[-len(raw)-1:-1], raw)
                self.curve_filt[0].setData(self.time_buffer[-len(raw)-1:-1], filt)

            else:
                filt = band_pass_filter(self.data_buffer[-(self.sr*Inlet.plot_duration)*2-1:-1], self.lcut, self.hcut, self.sr, self.filter_order)
                raw = band_pass_filter(self.data_buffer[-(self.sr*Inlet.plot_duration)*2-1:-1], 1, 35, self.sr, self.filter_order)

                raw = raw[-round(0.5*(len(raw))):-1]
                filt = filt[-round(0.5*(len(filt))):-1]

                self.curves[0].setData(self.time_buffer[-len(raw)-1:-1], raw)
                self.curve_filt[0].setData(self.time_buffer[-len(raw)-1:-1], filt)

            if keyboard.is_pressed("Enter"):
                btns.main(plot_time, self.timelist)

def main():
    """
    print("name of the participant?: ")
    name = input("")
    print("Which frequnecy is filtered?: ")
    freq_type = input("")
    """

    # firstly resolve all streams that could be shown
    inlets: List[Inlet] = []
    print("looking for streams")
    streams = pylsl.resolve_streams()

    # Create the pyqtgraph window
    pw = pg.plot(title='LSL Plot')
    pw.showMaximized()
    plt = pw.getPlotItem()
    plt.enableAutoRange(x=False, y=False)
    pw.showGrid(x=True,y=True)

    # iterate over found streams, creating specialized inlet objects that will
    # handle plotting the data
    for info in streams:
        print('Adding data inlet: ' + info.name())
        inlets.append(DataInlet(info, plt))

    def scroll():
        """Move the view so the data appears to scroll"""
        # We show data only up to a timepoint shortly before the current time
        # so new data doesn't suddenly appear in the middle of the plot
        fudge_factor = Inlet.pull_interval * .002
        plot_time = pylsl.local_clock()
        pw.setXRange(plot_time - Inlet.plot_duration + fudge_factor, plot_time - fudge_factor)

    def update():
        # Read data from the inlet. Use a timeout of 0.0 so we don't block GUI interaction.
        mintime = pylsl.local_clock() - Inlet.plot_duration

        # call pull_and_plot for each inlet.
        # Special handling of inlet types (markers, continuous data) is done in
        # the different inlet classes.

        for inlet in inlets:
            inlet.pull_and_plot(mintime, plt)

    # create a timer that will move the view every update_interval ms
    update_timer = QtCore.QTimer()
    update_timer.timeout.connect(scroll)
    update_timer.start(Inlet.update_interval)

    # create a timer that will pull and add new data occasionally
    pull_timer = QtCore.QTimer()
    pull_timer.timeout.connect(update)
    pull_timer.start(Inlet.pull_interval)

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

    def data_saving(list, session_title, name, freq_type, interact):
        os.chdir(gui.path)
        date = (datetime.datetime.now())
        date_time = date.strftime("%m%d%Y%H%M")
        data = {"data": list,"session_title": session_title , "name": name,"frequeny_range": freq_type,
                    "datetime": date, "interactions": interact}
        a_file = open(session_title+name+date_time+freq_type+".pkl", "wb")
        pickle.dump(data, a_file)
        a_file.close()

    data_saving(inlets[0].data_buffer, gui.session_name, gui.name, DataInlet.freq_band, DataInlet.timelist)

if __name__ == '__main__':
        main()
