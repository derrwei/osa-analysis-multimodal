import multimodal.signal_config as signal_config
from multimodal.respiratory import RespiratoryAnalyser

if __name__ == "__main__":
    # test the code
    dataroot = "/export/catch2/data/osa-brahms"
    sig_config = signal_config.SignalConfig(name="Flow", sampling_rate=256, dataroot=dataroot)
    analyser = RespiratoryAnalyser(sig_config)
    sig = analyser.get_signal("001/2019-09-06")
    sig = sig[256*600:256*700] # first 1 minute
    # plot the signal
    analyser.plot_signal(sig, title="Test Signal Plot Flow")
    print(sig.shape)