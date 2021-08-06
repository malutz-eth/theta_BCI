import numpy as np
import math
import pylsl
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
from typing import List
from scipy.signal import butter, lfilter

## These are "global" variables. Just because python lets you do this, doesn't mean you should. Instead, pass these numbers as inputs when initiating the Inlet class
plot_duration = 5  # how many seconds of data to show
update_interval = 9  # ms between screen updates
pull_interval = 8  # ms between each pull operation

sr = 11025


class Inlet:
    """Base class to represent a plottable inlet"""

    def __init__(self, info: pylsl.StreamInfo):
        # create an inlet and connect it to the outlet we found earlier.
        # max_buflen is set so data older the plot_duration is discarded
        # automatically and we only pull data new enough to show it

        # Also, perform online clock synchronization so all streams are in the
        # same time domain as the local lsl_clock()
        # (see https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/enums.html#_CPPv414proc_clocksync)
        # and dejitter timestamps
        self.inlet = pylsl.StreamInlet(
            info,
            max_buflen=plot_duration,
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter,
        )  ## do you really need this as a class? might work as well to just do this inside DataInlet
        # store the name and channel count
        self.name = info.name()
        self.channel_count = info.channel_count()
        self.sampling_rate = info.nominal_srate()

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

    dtypes = [
        [],
        np.float32,
        np.float64,
        None,
        np.int32,
        np.int16,
        np.int8,
        np.int64,
    ]  ## does this actually become part of the "self" pool of variables?

    def __init__(self, info: pylsl.StreamInfo, plt: pg.PlotItem):
        super().__init__(info)
        # calculate the size for our buffer, i.e. two times the displayed data
        bufsize = (
            2 * math.ceil(info.nominal_srate() * plot_duration),
            info.channel_count(),
        )
        self.buffer = np.empty(bufsize, dtype=self.dtypes[info.channel_format()])
        empty = np.array([])

        # create one curve object for each channel/line that will handle displaying the data
        self.curves = [
            pg.PlotCurveItem(x=empty, y=empty, autoDownsample=True)
            for _ in range(self.channel_count)
        ]
        self.curve_filt = [
            pg.PlotCurveItem(x=empty, y=empty, autoDownsample=True, pen=(3))
        ]

        for (
            curve
        ) in (
            self.curves
        ):  ## no need to go through this loop? we know we'll only have 1 channel stream
            plt.addItem(curve)
        for curve in self.curve_filt:
            plt.addItem(curve)

    def pull_and_plot(self, plot_time, plt): 

        # define the filter functions
        def low_pass_filter(data, cutoff, fs, order):
            nyq = fs * 0.5
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype="low", analog=False)
            filt = lfilter(b, a, data)
            return filt

        def high_pass_filter(
            data, cutoff, fs, order=5
        ):  ## why does high-pass have a default order but not low pass?
            nyq = fs * 0.5
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype="high", analog=False)
            filt = lfilter(b, a, data)
            return filt

        """ ## Try just one function, since they're identical except for the bandtype
        def _filter(data, cutoff, fs, order, bandtype) ## convention is to have little _ for functions that are only used within the class, but you choose
            nyq = fs*0.5
            normal_cutoff = cutoff/nyq
            b, a = butter(order, normal_cutoff, btype=bandtype, analog=False)
            filt = lfilter(b, a, data)
            return filt
        
        def _bandpass(data, hp, lp, fs, order_hp, order_lp)
            data_filthp = _filter(data, hp, fs, order, "high") 
            data_filtbp = _filter(data_filthp, lp, fs, order, "low") 
            return data_filtbp
        """

        # pull the data
        _, ts = self.inlet.pull_chunk(
            timeout=0.0, max_samples=self.buffer.shape[0], dest_obj=self.buffer
        )  ## doesn't this just pull the timestamps?

        # ts will be empty if no samples were pulled, a list of timestamps otherwise
        if ts:
            ts = np.asarray(ts)
            y = self.buffer[0 : ts.size, :]

            this_x = None ## this is a meaningless variable name
            old_offset = 0
            new_offset = 0
            for ch_ix in range(self.channel_count):
                # we don't pull an entire screen's worth of data, so we have to
                # trim the old data and append the new data to it
                old_x, old_y = self.curves[ch_ix].getData() ## like this, you are using the PLOTTING buffer as your data butter! So you keep filtering it

                # the timestamps are identical for all channels, so we need to do
                # this calculation only once
                if ch_ix == 0:
                    # find the index of the first sample that's still visible,
                    # i.e. newer than the left border of the plot
                    old_offset = old_x.searchsorted(plot_time)
                    # same for the new data, in case we pulled more data than
                    # can be shown at once
                    new_offset = ts.searchsorted(plot_time)
                    # append new timestamps to the trimmed old timestamps
                    this_x = np.hstack((old_x[old_offset:], ts[new_offset:]))
                
                # append new data to the trimmed old data
                this_y = np.hstack((old_y[old_offset:], y[new_offset:, ch_ix] - ch_ix)) ## this should be the UNFILTERED data from the buffer + the new unfiltered data

                # replace the old data and filter it before
                y_lowfiltered = low_pass_filter(this_y, 40, sr, 5) ## specify these variables in the initialization of the class, dont ever write numbers/strings deep inside code
                y_filtered = high_pass_filter(y_lowfiltered, 4, sr, 5) 

                self.curves[ch_ix].setData(this_x, this_y)
                self.curve_filt[ch_ix].setData(this_x, y_filtered)

                """
                raw_y =  np.hstack((old_y[old_offset:], y[new_offset:, ch_ix] - ch_ix))

                lightfilt_y = _bandpass(raw_y, self.hp_light, self.lp_light, self.fs, self.order_hp, self.order_lp) ## white curve

                heavyfilt_y = _bandpass(raw_y, self.hp_heavy, self.lp_heavy, self.fs, self.order_hp, self.order_lp) ## green curve

                ## save raw data to buffer

                ## set filtered data to plotting curves

                """


"""
class MarkerInlet(Inlet):
    #A MarkerInlet shows events that happen sporadically as vertical lines
    def __init__(self, info: pylsl.StreamInfo):
        super().__init__(info)

    def pull_and_plot(self, plot_time, plt):
        # TODO: purge old markers
        strings, timestamps = self.inlet.pull_chunk(0)
        if timestamps:
            for string, ts in zip(strings, timestamps):
                plt.addItem(pg.InfiniteLine(ts, angle=90, movable=False, label=string[0]))
"""


def main():
    # firstly resolve all streams that could be shown
    inlets: List[Inlet] = []
    print("looking for streams")
    streams = pylsl.resolve_streams()
    print(*streams, sep="\n")

    # Create the pyqtgraph window
    pw = pg.plot(title="LSL Plot")
    plt = pw.getPlotItem()
    plt.enableAutoRange(x=False, y=True)

    # iterate over found streams, creating specialized inlet objects that will
    # handle plotting the data
    for info in streams:
        if info.type() == "Markers":
            """
            if info.nominal_srate() != pylsl.IRREGULAR_RATE \
                    or info.channel_format() != pylsl.cf_string:
                print('Invalid marker stream ' + info.name())
            print('Adding marker inlet: ' + info.name())
            inlets.append(MarkerInlet(info))
            """
        elif (
            info.nominal_srate() != pylsl.IRREGULAR_RATE
            and info.channel_format() != pylsl.cf_string
        ):
            print("Adding data inlet: " + info.name())
            inlets.append(DataInlet(info, plt))
        """
        else:
            print('Don\'t know what to do with stream ' + info.name())
        """

    def scroll():
        """Move the view so the data appears to scroll"""
        # We show data only up to a timepoint shortly before the current time
        # so new data doesn't suddenly appear in the middle of the plot
        fudge_factor = pull_interval * 0.002
        plot_time = pylsl.local_clock()
        pw.setXRange(plot_time - plot_duration + fudge_factor, plot_time - fudge_factor)

    def update():
        # Read data from the inlet. Use a timeout of 0.0 so we don't block GUI interaction.
        mintime = pylsl.local_clock() - plot_duration

        # call pull_and_plot for each inlet.
        # Special handling of inlet types (markers, continuous data) is done in
        # the different inlet classes.
        for inlet in inlets:
            inlet.pull_and_plot(mintime, plt)

    # create a timer that will move the view every update_interval ms
    update_timer = QtCore.QTimer()
    update_timer.timeout.connect(scroll)
    update_timer.start(update_interval)

    # create a timer that will pull and add new data occasionally
    pull_timer = QtCore.QTimer()
    pull_timer.timeout.connect(update)
    pull_timer.start(pull_interval)

    import sys

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtGui.QApplication.instance().exec_()


if __name__ == "__main__":
    main()
