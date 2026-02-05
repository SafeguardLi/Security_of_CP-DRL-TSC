import random

from src.trafficsignalcontroller import TrafficSignalController

class ActuatedTSC(TrafficSignalController):
    def __init__(self, conn, tsc_id, mode, netdata, red_t, yellow_t, green_t, detect_r):
        super().__init__(conn, tsc_id, mode, netdata, red_t, yellow_t, detect_r)
        self.data = None
        self.uv_data = None
        self.tsc_type = 'actuated'

    def next_phase(self):
        return None

    def next_phase_duration(self):
        return 1

    def update(self, data, cv_data,uv_data, mask):
        if mask:
            self.data = cv_data
            self.uv_data = uv_data
        else:
            self.data = data
